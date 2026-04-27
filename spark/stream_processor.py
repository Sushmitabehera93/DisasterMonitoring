# spark/stream_processor.py
import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_json, udf, current_timestamp,
    when, lit, round as spark_round
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, TimestampType
)
from database.mongo_client import upsert_earthquake, upsert_eonet, upsert_fema

KAFKA_BROKER = "localhost:9092"

# Schemas

EARTHQUAKE_SCHEMA = StructType([
    StructField("event_id",    StringType()),
    StructField("source",      StringType()),
    StructField("type",        StringType()),
    StructField("magnitude",   DoubleType()),
    StructField("place",       StringType()),
    StructField("time",        StringType()),
    StructField("longitude",   DoubleType()),
    StructField("latitude",    DoubleType()),
    StructField("depth_km",    DoubleType()),
    StructField("status",      StringType()),
    StructField("tsunami",     IntegerType()),
    StructField("alert",       StringType()),
    StructField("felt",        IntegerType()),
    StructField("url",         StringType()),
    StructField("ingested_at", StringType()),
    StructField("significance", DoubleType())
])

EONET_SCHEMA = StructType([
    StructField("event_id",       StringType()),
    StructField("source",         StringType()),
    StructField("type",           StringType()),
    StructField("title",          StringType()),
    StructField("status",         StringType()),
    StructField("longitude",      DoubleType()),
    StructField("latitude",       DoubleType()),
    StructField("last_date",      StringType()),
    StructField("start_date",     StringType()),
    StructField("geometry_count", IntegerType()),
    StructField("ingested_at",    StringType())
])

FEMA_SCHEMA = StructType([
    StructField("event_id",         StringType()),
    StructField("source",           StringType()),
    StructField("incident_type",    StringType()),
    StructField("title",            StringType()),
    StructField("state",            StringType()),
    StructField("county",           StringType()),
    StructField("declaration_date", StringType()),
    StructField("incident_begin",   StringType()),
    StructField("incident_end",     StringType()),
    StructField("fema_region",      IntegerType()),
    StructField("longitude",        DoubleType()),
    StructField("latitude",         DoubleType()),
    StructField("ingested_at",      StringType())
])

# Risk scoring

def compute_risk_score(magnitude, tsunami, alert, depth_km):
    if magnitude is None:
        return 0.0
    score = min(magnitude * 10, 60.0)         
    if tsunami:
        score += 20
    alert_bonus = {"red": 20, "orange": 15, "yellow": 10, "green": 5}
    score += alert_bonus.get(alert or "", 0)
    if depth_km and depth_km < 70:             
        score += 5
    return round(min(score, 100.0), 2)

risk_udf = udf(compute_risk_score, DoubleType())

# Spark session 

def create_spark():
    return (SparkSession.builder
        .appName("DisasterMonitor")
        .config("spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
                "org.mongodb.spark:mongo-spark-connector_2.12:10.2.1")
        .config("spark.mongodb.write.connection.uri",
                "mongodb://localhost:27017/disaster_monitor")  
        .getOrCreate())

# For each batch 

def write_earthquakes(df, epoch_id):
    count = 0
    for row in df.toLocalIterator():
        doc = row.asDict()
        doc["risk_score"] = compute_risk_score(
            doc.get("magnitude"), doc.get("tsunami"),
            doc.get("alert"),     doc.get("depth_km")
        )
        # Assign risk_tier based on risk_score
        score = doc["risk_score"]
        if score >= 90:
            doc["risk_tier"] = "Critical"
        elif score >= 70:
            doc["risk_tier"] = "High"
        elif score >= 40:
            doc["risk_tier"] = "Moderate"
        else:
            doc["risk_tier"] = "Low"
        upsert_earthquake(doc)
        count += 1
    if count:
        print(f"[Spark/EQ] epoch {epoch_id}: wrote {count} earthquakes")

def write_eonet(df, epoch_id):
    count = 0
    for row in df.toLocalIterator():
        upsert_eonet(row.asDict())
        count += 1
    if count:
        print(f"[Spark/EONET] epoch {epoch_id}: wrote {count} events")

def write_fema(df, epoch_id):
    count = 0
    for row in df.toLocalIterator():
        upsert_fema(row.asDict())
        count += 1
    if count:
        print(f"[Spark/FEMA] epoch {epoch_id}: wrote {count} declarations")

# Main streaming job 

def run():
    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    def kafka_stream(topic):
        return (spark.readStream
            .format("kafka")
            .option("kafka.bootstrap.servers", KAFKA_BROKER)
            .option("subscribe", topic)
            .option("startingOffsets", "earliest")
            .option("failOnDataLoss", "false")  
            .option("maxOffsetsPerTrigger", 1000)
            .load()
            .selectExpr("CAST(value AS STRING) as json_str"))

    # Earthquakes
    eq_df = (kafka_stream("usgs-earthquakes")
        .select(from_json(col("json_str"), EARTHQUAKE_SCHEMA).alias("d"))
        .select("d.*"))

    eq_query = (eq_df.writeStream
        .foreachBatch(write_earthquakes)
        .option("checkpointLocation", "/tmp/checkpoints/eq")
        .trigger(processingTime="30 seconds")
        .start())

    # EONET
    eonet_df = (kafka_stream("nasa-eonet-events")
        .select(from_json(col("json_str"), EONET_SCHEMA).alias("d"))
        .select("d.*"))

    eonet_query = (eonet_df.writeStream
        .foreachBatch(write_eonet)
        .option("checkpointLocation", "/tmp/checkpoints/eonet")
        .trigger(processingTime="60 seconds")
        .start())

    # FEMA
    fema_df = (kafka_stream("fema-disasters")
        .select(from_json(col("json_str"), FEMA_SCHEMA).alias("d"))
        .select("d.*"))

    fema_query = (fema_df.writeStream
        .foreachBatch(write_fema)
        .option("checkpointLocation", "/tmp/checkpoints/fema")
        .trigger(processingTime="120 seconds")
        .start())

    print("Spark Structured Streaming started on all 3 topics.")
    spark.streams.awaitAnyTermination()

if __name__ == "__main__":
    run()