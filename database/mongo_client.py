# database/mongo_client.py
from pymongo import MongoClient, GEOSPHERE, ASCENDING, DESCENDING
from datetime import datetime, timezone

MONGO_URI = "mongodb://localhost:27017"
DB_NAME   = "disaster_monitor"

# Create a single client and db instance
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

def setup_indexes():
    # Use the shared db instance
    db.earthquakes.create_index([("location", GEOSPHERE)])
    db.earthquakes.create_index([("time", DESCENDING)])
    db.earthquakes.create_index([("magnitude", DESCENDING)])
    db.earthquakes.create_index("event_id", unique=True)

    db.eonet_events.create_index([("location", GEOSPHERE)])
    db.eonet_events.create_index([("last_date", DESCENDING)])
    db.eonet_events.create_index("event_id", unique=True)

    db.fema_disasters.create_index([("declaration_date", DESCENDING)])
    db.fema_disasters.create_index("state")
    db.fema_disasters.create_index("incident_type")
    db.fema_disasters.create_index("event_id")

    db.risk_scores.create_index([("location", GEOSPHERE)])
    db.risk_scores.create_index([("computed_at", DESCENDING)])

    print("MongoDB indexes created.")

# Upsert functions for producers

def upsert_earthquake(doc: dict):
    if doc.get("latitude") and doc.get("longitude"):
        doc["location"] = {
            "type": "Point",
            "coordinates": [doc["longitude"], doc["latitude"]]
        }
    db.earthquakes.update_one(
        {"event_id": doc["event_id"]},
        {"$set": doc},
        upsert=True
    )

def upsert_eonet(doc: dict):
    if doc.get("latitude") and doc.get("longitude"):
        doc["location"] = {
            "type": "Point",
            "coordinates": [doc["longitude"], doc["latitude"]]
        }
    db.eonet_events.update_one(
        {"event_id": doc["event_id"]},
        {"$set": doc},
        upsert=True
    )

def upsert_fema(doc: dict):
    db.fema_disasters.update_one(
        {"event_id": doc["event_id"], "county": doc.get("county")},
        {"$set": doc},
        upsert=True
    )

# Dashboard query functions

def get_recent_earthquakes(limit=500, min_mag=0):
    return list(
        db.earthquakes.find(
            {"magnitude": {"$gte": min_mag}},
            {"_id": 0}
        ).sort("time", DESCENDING).limit(limit)
    )

def get_recent_eonet(limit=200, event_types=None):
    filt = {}
    if event_types:
        filt["type"] = {"$in": event_types}
    return list(
        db.eonet_events.find(filt, {"_id": 0})
        .sort("last_date", DESCENDING).limit(limit)
    )

def get_fema_by_state():
    pipeline = [
        {"$group": {
            "_id":   "$state",
            "count": {"$sum": 1},
            "types": {"$addToSet": "$incident_type"}
        }},
        {"$sort": {"count": DESCENDING}}
    ]
    return list(db.fema_disasters.aggregate(pipeline))

def get_disaster_trend(days=30):
    from datetime import timedelta
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    pipeline = [
        {"$match": {"time": {"$gte": since}}},
        {"$group": {
            "_id": {
                "date": {"$dateToString": {
                    "format": "%Y-%m-%d",
                    "date":   {"$dateFromString": {"dateString": "$time"}}
                }}
            },
            "count": {"$sum": 1},
            "avg_mag": {"$avg": "$magnitude"}
        }},
        {"$sort": {"_id.date": ASCENDING}}
    ]
    return list(db.earthquakes.aggregate(pipeline))

if __name__ == "__main__":
    setup_indexes()