# producers/eonet_producer.py
import json
import time
import requests
from datetime import datetime, timezone
from kafka import KafkaProducer

KAFKA_BROKER = "localhost:9092"
TOPIC        = "nasa-eonet-events"
EONET_URL    = "https://eonet.gsfc.nasa.gov/api/v3/events"

CATEGORY_MAP = {
    "wildfires":    "wildfire",
    "severeStorms": "storm",
    "volcanoes":    "volcano",
    "floods":       "flood",
    "seaLakeIce":   "ice",
    "landslides":   "landslide",
    "drought":      "drought",
    "dustHaze":     "dust_haze",
    "manmade":      "manmade",
    "snow":         "snow"
}

def get_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None
    )

def fetch_eonet_events(days=30, status="all"):
    params = {"days": days, "status": status, "limit": 500}
    try:
        r = requests.get(EONET_URL, params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("events", [])
    except requests.RequestException as e:
        print(f"[EONET] Request error: {e}")
        return []

def parse_eonet_event(event):
    categories = event.get("categories", [])
    cat_id     = categories[0].get("id", "unknown") if categories else "unknown"
    geometries = event.get("geometry", [])
    latest_geo = geometries[-1] if geometries else {}
    coords     = latest_geo.get("coordinates", [None, None])
    if isinstance(coords[0], list):
        coords = coords[0]
    return {
        "event_id":       event.get("id"),
        "source":         "NASA_EONET",
        "type":           CATEGORY_MAP.get(cat_id, cat_id),
        "title":          event.get("title"),
        "description":    event.get("description"),
        "status":         event.get("status"),
        "categories":     [c.get("title") for c in categories],
        "longitude":      coords[0] if len(coords) > 0 else None,
        "latitude":       coords[1] if len(coords) > 1 else None,
        "geometry_count": len(geometries),
        "start_date":     geometries[0].get("date") if geometries else None,
        "last_date":      latest_geo.get("date"),
        "sources":        [s.get("url") for s in event.get("sources", [])],
        "ingested_at":    datetime.now(timezone.utc).isoformat()
    }

def load_historical_eonet(producer, seen_ids):
    print("[EONET] Loading full archive (closed events, paginated)...")

    total  = 0
    offset = 0
    limit  = 500

    # Paginate through all closed (resolved) events to get historical data
    while True:
        params = {
            "status": "closed",
            "limit":  limit,
            "offset": offset
        }
        try:
            r      = requests.get(EONET_URL, params=params, timeout=30)
            r.raise_for_status()
            events = r.json().get("events", [])
        except requests.RequestException as e:
            print(f"[EONET] Error at offset {offset}: {e}")
            break

        if not events:
            print(f"[EONET] No more events at offset {offset}. Stopping.")
            break

        for e in events:
            parsed = parse_eonet_event(e)
            if parsed["event_id"] and parsed["event_id"] not in seen_ids:
                producer.send(TOPIC, key=parsed["event_id"], value=parsed)
                seen_ids.add(parsed["event_id"])
                total += 1

        producer.flush()
        print(f"  [EONET] offset {offset} → {len(events)} events (running total: {total})")

        if len(events) < limit:
            break

        offset += limit
        time.sleep(0.5)

    # Load all currently open/active events
    print("[EONET] Loading currently open events...")
    open_events = fetch_eonet_events(days=365, status="open")
    for e in open_events:
        parsed = parse_eonet_event(e)
        if parsed["event_id"] and parsed["event_id"] not in seen_ids:
            producer.send(TOPIC, key=parsed["event_id"], value=parsed)
            seen_ids.add(parsed["event_id"])
            total += 1
    producer.flush()
    print(f"  [EONET] {len(open_events)} open events added.")

    print(f"[EONET] Historical load complete. {total} total events published.")
    return seen_ids

def run_producer():
    producer = get_producer()
    seen_ids = set()
    print("[EONET Producer] Starting...")

    # Full archive load
    seen_ids = load_historical_eonet(producer, seen_ids)

    print("[EONET] Starting real-time polling every 5 minutes...")
    while True:
        try:
            events    = fetch_eonet_events(days=3, status="open")
            new_count = 0
            for e in events:
                parsed = parse_eonet_event(e)
                producer.send(TOPIC, key=parsed["event_id"], value=parsed)
                new_count += 1
            producer.flush()
            print(f"[EONET] {datetime.now(timezone.utc).isoformat()} — {new_count} active events published.")
        except Exception as ex:
            print(f"[EONET] Error: {ex}")
        time.sleep(300)

if __name__ == "__main__":
    run_producer()