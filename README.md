# Disaster Monitoring Pipeline

A real-time disaster monitoring system that ingests, processes, and analyzes natural disaster data from multiple sources using Apache Kafka, Apache Spark,
and MongoDB.

---

## Architecture
Data Sources → Kafka Producers → Kafka Topics → Spark Streaming → MongoDB → ML Training → Dashboard

---

## Data Sources

- **USGS** — Earthquake events (magnitude, depth, location, tsunami alerts)
- **NASA EONET** — Natural events (wildfires, floods, storms)
- **FEMA** — Disaster declarations across US states

---

## Project Structure

|-- producers/
│   |-- usgs_producer.py       # Fetches earthquake data → Kafka
│   |-- eonet_producer.py      # Fetches NASA EONET events → Kafka
│   |-- fema_producer.py       # Fetches FEMA declarations → Kafka
|-- spark/
│   |-- stream_processor.py    # Spark Structured Streaming → MongoDB
|-- ml/
│   |-- batch_ml_training.py   # ML model training pipeline
|-- database/
│   |-- mongo_client.py        # MongoDB upsert helpers
|-- dashboard/
│   |-- app.py                 # Streamlit dashboard
|-- setup_topics.py            # Kafka topic setup
|-- docker-compose.yml         # MongoDB + Kafka containers
| requirements.txt

---

## Pipeline

1. **Producers** fetch data from USGS, NASA EONET, and FEMA APIs and publish
   to Kafka topics (`usgs-earthquakes`, `nasa-eonet-events`, `fema-disasters`)
2. **Spark Structured Streaming** consumes from all 3 Kafka topics, computes
   risk scores, and upserts records into MongoDB
3. **Batch ML Training** reads from MongoDB and trains 4 models:
   - Random Forest risk score predictor (R² = 0.90)
   - Anomaly detection using daily-rate baseline across 515 grid cells
   - KMeans regional clustering (5 clusters via elbow method)
   - Seasonal trend analysis
4. **Streamlit Dashboard** visualizes live data and ML insights

---

## ML Models

| Model | Description | Result |
|---|---|---|
| Random Forest | Predicts earthquake risk score | R² = 0.90, RMSE = 3.57 |
| Anomaly Detection | Flags spike events vs historical baseline | 3x threshold |
| KMeans Clustering | Groups regions by seismic activity | 5 clusters |
| Seasonal Analysis | Monthly/yearly event rate trends | 10,970 events |

**Top features for risk scoring:** magnitude (49%), mag_squared (39%), depth (3%)

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Apache Kafka | Real-time event streaming |
| Apache Spark (PySpark) | Distributed stream + batch processing |
| MongoDB | NoSQL storage for disaster events |
| Streamlit | Interactive dashboard |
| Docker | Containerized Kafka + MongoDB |
| Python | Core language |

---

## Prerequisites

- Python 3.8+
- Java JDK 11
- Apache Spark 3.5.0
- Docker Desktop

---

## How to Run

**1. Start infrastructure**
```bash
docker-compose up -d
python setup_topics.py
```

**2. Run producers**
```bash
python producers/usgs_producer.py
python producers/eonet_producer.py
python producers/fema_producer.py
```

**3. Start stream processor**
```bash
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
  --jars <mongo-connector-jars> spark/stream_processor.py
```

**4. Train ML models**
```bash
spark-submit --jars <mongo-connector-jars> ml/batch_ml_training.py
```

**5. Launch dashboard**
```bash
streamlit run dashboard/app.py
```

---

## Key Findings

- 5,988 earthquake records and 509 EONET events ingested in first run
- Identified 3 Critical Seismic Zones and 2 high-activity clusters globally
- Extreme Cold accounts for the most common extreme weather classification
- Vancouver has the mildest climate; Winnipeg experiences the harshest winters

