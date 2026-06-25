# Disaster Monitoring Pipeline

A real-time disaster monitoring system that ingests live event streams from 3 public APIs (USGS, NASA EONET, FEMA), processes them through Apache Kafka and Spark Structured Streaming, and stores enriched records in MongoDB. Trained 4 ML models for risk scoring, anomaly detection, regional clustering, and seasonal trend analysis. Visualized insights through an interactive Streamlit dashboard.

**Scale:** 5,988 earthquake records, 509 EONET events, and FEMA disaster declarations ingested and processed in first run.

---

## Dashboard Preview

<!-- Add 1-2 screenshots of your Streamlit dashboard here -->
<!-- Example: -->
<!-- ![Dashboard Overview](screenshots/dashboard_overview.png) -->
<!-- ![Risk Map](screenshots/risk_map.png) -->

> **TODO:** Add screenshots of your Streamlit dashboard showing the map, risk scores, and ML insights.

---

## Architecture

```
Data Sources (USGS, NASA EONET, FEMA)
        │
        ▼
  Kafka Producers ──► Kafka Topics ──► Spark Structured Streaming
                                                │
                                                ▼
                                            MongoDB
                                           ╱        ╲
                                          ▼          ▼
                                   ML Training    Streamlit
                                   Pipeline       Dashboard
```

---

## Data Sources

| Source | Data | Key Fields |
|--------|------|------------|
| **USGS** | Earthquake events | Magnitude, depth, location, tsunami alerts |
| **NASA EONET** | Natural events | Wildfires, floods, storms, coordinates |
| **FEMA** | Disaster declarations | US state-level declarations, disaster types |

---

## Pipeline

1. **Ingestion** — Three Kafka producers fetch data from USGS, NASA EONET, and FEMA APIs and publish to dedicated Kafka topics (`usgs-earthquakes`, `nasa-eonet-events`, `fema-disasters`).
2. **Stream Processing** — Spark Structured Streaming consumes from all 3 topics, computes risk scores, and upserts enriched records into MongoDB.
3. **ML Training** — Batch pipeline reads from MongoDB and trains 4 models (see below).
4. **Dashboard** — Streamlit app visualizes live data, risk maps, and ML-generated insights.

---

## ML Models

| Model | Purpose | Result |
|-------|---------|--------|
| Random Forest | Predict earthquake risk score from magnitude, depth, location | R² = 0.90, RMSE = 3.57 |
| Anomaly Detection | Flag spike events vs. historical daily-rate baseline across 515 grid cells | 3× threshold trigger |
| KMeans Clustering | Group regions by seismic activity patterns | 5 clusters (elbow method) |
| Seasonal Analysis | Identify monthly/yearly event rate trends | 10,970 events analyzed |

**Top features for risk scoring:** magnitude (49%), mag_squared (39%), depth (3%)

---

## Key Findings

- Identified **3 critical seismic zones** and **2 high-activity clusters** globally
- Anomaly detection flagged regional spike events exceeding 3× the historical baseline
- Seasonal analysis revealed cyclical patterns in event frequency across years
- Random Forest achieved strong predictive performance (R² = 0.90) with magnitude and depth as dominant features

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Apache Kafka | Real-time event streaming |
| Apache Spark (PySpark) | Distributed stream and batch processing |
| MongoDB | NoSQL storage for disaster events |
| Streamlit | Interactive dashboard |
| Docker | Containerized Kafka and MongoDB |
| scikit-learn | ML model training |
| Python | Core language |

---

## Project Structure

```
├── producers/
│   ├── usgs_producer.py          # USGS earthquake data → Kafka
│   ├── eonet_producer.py         # NASA EONET events → Kafka
│   └── fema_producer.py          # FEMA declarations → Kafka
├── spark/
│   └── stream_processor.py       # Spark Structured Streaming → MongoDB
├── ml/
│   └── batch_ml_training.py      # ML model training pipeline
├── database/
│   └── mongo_client.py           # MongoDB upsert helpers
├── dashboard/
│   └── app.py                    # Streamlit dashboard
├── setup_topics.py               # Kafka topic setup
├── docker-compose.yml            # MongoDB + Kafka containers
└── requirements.txt
```

---

## How to Run

**Prerequisites:** Python 3.8+, Java JDK 11, Apache Spark 3.5.0, Docker Desktop

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
