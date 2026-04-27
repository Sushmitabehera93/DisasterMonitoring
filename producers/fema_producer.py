# producers/fema_producer.py
import json
import time
import requests
from datetime import datetime, timedelta, timezone
from kafka import KafkaProducer

KAFKA_BROKER     = "localhost:9092"
TOPIC            = "fema-disasters"
FEMA_URL         = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
FEMA_DETAILS_URL = "https://www.fema.gov/api/open/v1/FemaWebDisasterDeclarations"

def get_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None
    )

def fetch_fema_summaries(skip=0, top=1000, since_date=None):
    params = {
        "$skip": skip,
        "$top": top,
        "$format": "json"
    }
    if since_date:
        params["$filter"] = f"declarationDate gt '{since_date}'"
    try:
        response = requests.get(FEMA_URL, params=params, timeout=30)
        response.raise_for_status()
        return response.json().get("DisasterDeclarationsSummaries", [])
    except requests.RequestException as e:
        print(f"[FEMA] Request error: {e}")
        return []

def parse_fema_summary(record):
    return {
        "event_id": record.get("disasterNumber"),
        "state": record.get("state"),
        "county": record.get("designatedArea"),
        "incident_type": record.get("incidentType"),
        "declaration_date": record.get("declarationDate"),
        "title": record.get("declarationTitle"),
        "ih_program_declared": record.get("ihProgramDeclared"),
        "ia_program_declared": record.get("iaProgramDeclared"),
        "pa_program_declared": record.get("paProgramDeclared"),
        "hm_program_declared": record.get("hmProgramDeclared"),
        "ingested_at": datetime.now(timezone.utc).isoformat()
    }

def load_historical_fema(producer, seen_ids):
    print("[FEMA] Loading historical data (since 2010)...")
    skip = 0
    top = 1000
    total = 0
    while True:
        records = fetch_fema_summaries(skip=skip, top=top)
        if not records:
            break
        for rec in records:
            parsed = parse_fema_summary(rec)
            key = f"{parsed['event_id']}-{parsed.get('county', '')}-summaries"
            if key not in seen_ids:
                producer.send(TOPIC, key=key, value=parsed)
                seen_ids.add(key)
                total += 1
        producer.flush()
        print(f"  [FEMA] {skip} to {skip+top} → {len(records)} events (running total: {total})")
        if len(records) < top:
            break
        skip += top
        time.sleep(1)
    print(f"[FEMA] Historical load complete. {total} total events published.")
    return seen_ids

def run_producer():
    producer = get_producer()
    seen_ids = set()
    print("[FEMA Producer] Starting...")

    # Historical load
    seen_ids = load_historical_fema(producer, seen_ids)

    print("[FEMA] Starting real-time polling every 5 minutes (window: 7 days)...")
    poll_interval = 300  
    window_days = 7      

    while True:
        try:
            since = (datetime.now(timezone.utc) - timedelta(days=window_days)).strftime("%Y-%m-%dT%H:%M:%S")
            records = fetch_fema_summaries(since_date=since)
            new_count = 0
            for rec in records:
                parsed = parse_fema_summary(rec)
                key = f"{parsed['event_id']}-{parsed.get('county', '')}-summaries"
                if key not in seen_ids:
                    producer.send(TOPIC, key=key, value=parsed)
                    seen_ids.add(key)
                    new_count += 1
            producer.flush()
            print(f"[FEMA] {datetime.now(timezone.utc).isoformat()} — {new_count} new declarations.")
        except Exception as e:
            print(f"[FEMA] Error: {e}")
        time.sleep(poll_interval)

if __name__ == "__main__":
    run_producer()