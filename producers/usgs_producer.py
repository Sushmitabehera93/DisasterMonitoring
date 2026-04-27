# producers/usgs_producer.py
import json
import time
import requests
from datetime import datetime, timedelta, timezone
from kafka import KafkaProducer

KAFKA_BROKER = "localhost:9092"
TOPIC        = "usgs-earthquakes"
USGS_URL     = "https://earthquake.usgs.gov/fdsnws/event/1/query"

def get_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None
    )

def fetch_earthquakes(start_time, end_time, min_magnitude=2.5):  
    params = {
        "format":       "geojson",
        "starttime":    start_time,
        "endtime":      end_time,
        "minmagnitude": min_magnitude,
        "orderby":      "time"
    }
    try:
        response = requests.get(USGS_URL, params=params, timeout=30)
        response.raise_for_status()
        return response.json().get("features", [])
    except requests.RequestException as e:
        print(f"[USGS] Request error: {e}")
        return []

def parse_earthquake(feature):
    props  = feature.get("properties", {})
    coords = feature.get("geometry", {}).get("coordinates", [None, None, None])
    return {
        "event_id":    feature.get("id"),
        "source":      "USGS",
        "type":        "earthquake",
        "magnitude":   props.get("mag"),
        "place":       props.get("place"),
        "time":        datetime.fromtimestamp(
                           props["time"] / 1000, tz=timezone.utc
                       ).isoformat() if props.get("time") else None,
        "updated": datetime.fromtimestamp(
                           props["updated"] / 1000, tz=timezone.utc
                       ).isoformat() if props.get("updated") else None,               
        "longitude":   coords[0],
        "latitude":    coords[1],
        "depth_km":    coords[2],
        "status":      props.get("status"),
        "tsunami":     props.get("tsunami"),
        "alert":       props.get("alert"),
        "felt":        props.get("felt"),
        "url":         props.get("url"),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "significance": props.get("sig")  
    }

def load_historical_earthquakes(producer, seen_ids):
    print("[USGS] Loading historical data (15 years, mag >= 2.5)") 

    total      = 0
    end_date = datetime.now(timezone.utc)
    start_date = datetime(2010, 1, 1, tzinfo=timezone.utc) 

    # Chunk into 3 month intervals
    current = start_date
    while current < end_date:
        chunk_end = min(current + timedelta(days=90), end_date)

        features = fetch_earthquakes(
            current.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
            min_magnitude=2.5  
        )

        for f in features:
            event = parse_earthquake(f)
            if event["event_id"] and event["event_id"] not in seen_ids:
                producer.send(TOPIC, key=event["event_id"], value=event)
                seen_ids.add(event["event_id"])
                total += 1

        producer.flush()
        print(f"  [USGS] {current.strftime('%Y-%m-%d')} to "
              f"{chunk_end.strftime('%Y-%m-%d')} "
              f"→ {len(features)} events (running total: {total})")

        current = chunk_end + timedelta(days=1)
        time.sleep(1) 

    print(f"[USGS] Historical load complete. {total} total events published.")
    return seen_ids

def run_producer():
    producer = get_producer()
    seen_ids = set()
    print("[USGS Producer] Starting")

    # Historical load
    seen_ids = load_historical_earthquakes(producer, seen_ids)

    print("[USGS] Starting real-time streaming from 2 days ago")
    poll_interval = 60  
    start_time = datetime.now(timezone.utc) - timedelta(days=2)

    while True:
        try:
            end_time = datetime.now(timezone.utc)
            features = fetch_earthquakes(
                start_time.strftime("%Y-%m-%dT%H:%M:%S"),
                end_time.strftime("%Y-%m-%dT%H:%M:%S"),
                min_magnitude=2.5  
            )
            new_count = 0
            for f in features:
                event = parse_earthquake(f)
                if event["event_id"] and event["event_id"] not in seen_ids:
                    producer.send(TOPIC, key=event["event_id"], value=event)
                    seen_ids.add(event["event_id"])
                    new_count += 1
            producer.flush()
            print(f"[USGS] {end_time.isoformat()} — {new_count} new events.")
            start_time = end_time
        except Exception as e:
            print(f"[USGS] Error: {e}")
        time.sleep(poll_interval)

if __name__ == "__main__":
    run_producer()