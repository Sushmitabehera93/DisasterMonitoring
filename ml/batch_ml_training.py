# ml/batch_ml_training.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, lit, count, avg, max as spark_max, min as spark_min,
    month, dayofweek, hour, year, datediff, to_timestamp,
    round as spark_round, log1p, stddev, expr, abs as spark_abs
)
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.regression import RandomForestRegressor          
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator            

import json
from datetime import datetime, timezone

# Paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), "trained_models")
os.makedirs(MODEL_DIR, exist_ok=True)

RF_MODEL_PATH            = os.path.join(MODEL_DIR, "risk_score_rf")
KMEANS_MODEL_PATH        = os.path.join(MODEL_DIR, "region_kmeans")
IF_STATS_PATH            = os.path.join(MODEL_DIR, "anomaly_stats.json")
SEASONAL_STATS_PATH      = os.path.join(MODEL_DIR, "seasonal_stats.json")   

MONGO_URI      = "mongodb://localhost:27017/disaster_monitor"
TRAINING_START = "2010-01-01"   

#  Spark Session

def create_spark():
    return (SparkSession.builder
        .appName("DisasterMonitor_BatchML")
        .config("spark.jars.packages",
                "org.mongodb.spark:mongo-spark-connector_2.12:10.2.1")
        .config("spark.mongodb.read.connection.uri", MONGO_URI)
        .getOrCreate())


def load_collection(spark, collection):
    return (spark.read
        .format("mongodb")  
        .option("spark.mongodb.read.connection.uri", MONGO_URI)  
        .option("spark.mongodb.read.database", "disaster_monitor")
        .option("spark.mongodb.read.collection", collection)
        .load())

#  1. Random Forest — Risk Score Prediction

def train_risk_model(eq_df_cached):
    print("\n" + "=" * 70)
    print("  TRAINING: Random Forest Risk Score Model")
    print("=" * 70)

    df = eq_df_cached.select(
        col("magnitude").cast(DoubleType()),
        col("depth_km").cast(DoubleType()),
        col("tsunami").cast(IntegerType()),
        col("alert").cast("string"),
        col("significance").cast(DoubleType()),   
        col("time").cast("string"),
    ).na.drop(subset=["magnitude", "significance"])   

    total = df.count()
    print(f"  Records with significance label: {total:,}")

    df = (df
        .fillna({"depth_km": 33.0, "tsunami": 0, "alert": "none"})
        .withColumn("depth_km", when(col("depth_km") < 0, 0).otherwise(col("depth_km")))
    )

    # Derived features
    df = (df
        .withColumn("is_shallow",      when(col("depth_km") < 70, 1).otherwise(0))
        .withColumn("is_intermediate", when(
            (col("depth_km") >= 70) & (col("depth_km") < 300), 1).otherwise(0))
        .withColumn("log_depth",   log1p(col("depth_km")))
        .withColumn("mag_squared", col("magnitude") * col("magnitude"))
    )

    # Time features
    df = df.withColumn("ts", to_timestamp(col("time")))
    df = (df
        .withColumn("month",       month(col("ts")))
        .withColumn("hour_of_day", hour(col("ts")))
        .withColumn("day_of_week", dayofweek(col("ts")))
    ).fillna({"month": 6, "hour_of_day": 12, "day_of_week": 4})


    df = df.withColumn("sig_norm",
        spark_round(
            when(col("significance") > 1000, 100.0)
            .when(col("significance") < 0,   0.0)
            .otherwise(col("significance") / 10.0),
        2)
    )

    df = df.withColumn(
        "risk_tier",
            when(col("sig_norm") >= 90, "Critical")
            .when(col("sig_norm") >= 70, "High")
            .when(col("sig_norm") >= 40, "Moderate")
            .otherwise("Low")
    )

    mag_low    = df.filter(col("magnitude") <  4.0)  
    mag_mid    = df.filter((col("magnitude") >= 4.0) & (col("magnitude") < 5.0))  
    mag_high   = df.filter(col("magnitude") >= 5.0)  

    df_balanced = (mag_low.sample(fraction=0.05, seed=42)
                   .union(mag_mid.sample(fraction=0.30, seed=42))
                   .union(mag_high))

    bal_count = df_balanced.count()
    print(f"  Balanced training set: {bal_count:,} records "
          f"(from {total:,} after stratified sampling)")

    df_balanced.cache()

    alert_indexer = StringIndexer(
        inputCol="alert", outputCol="alert_idx", handleInvalid="keep"
    )

    feature_cols = [
        "magnitude", "depth_km", "tsunami", "alert_idx",
        "is_shallow", "is_intermediate", "log_depth", "mag_squared",
        "month", "hour_of_day", "day_of_week"
    ]

    assembler = VectorAssembler(
        inputCols=feature_cols, outputCol="features", handleInvalid="skip"
    )

    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="sig_norm",       
        numTrees=100,
        maxDepth=8,
        seed=42
    )

    pipeline = Pipeline(stages=[alert_indexer, assembler, rf])

    train_df, test_df = df_balanced.randomSplit([0.8, 0.2], seed=42)
    train_df.cache()
    test_df.cache()
    print(f"  Train: {train_df.count():,} | Test: {test_df.count():,}")

    model = pipeline.fit(train_df)

    predictions = model.transform(test_df)
    rmse = RegressionEvaluator(
        labelCol="sig_norm", predictionCol="prediction", metricName="rmse"
    ).evaluate(predictions)
    r2 = RegressionEvaluator(
        labelCol="sig_norm", predictionCol="prediction", metricName="r2"
    ).evaluate(predictions)

    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}  (meaningful — target is independent of features)")

    rf_model = model.stages[-1]
    importances = rf_model.featureImportances.toArray()
    print("  Feature importances:")
    for name, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
        print(f"    {name:20s} {imp:.4f}")

    model.write().overwrite().save(RF_MODEL_PATH)
    print(f"Model saved to {RF_MODEL_PATH}")

    df_balanced.unpersist()
    train_df.unpersist()
    test_df.unpersist()

    return model

#  2. Anomaly Detection — Statistical Baseline per Grid Cell

def train_anomaly_detector(eq_df_cached):
    print("\n" + "=" * 70)
    print("  TRAINING: Anomaly Detection (Daily-Rate Baseline)")
    print("=" * 70)

    df = (eq_df_cached
        .select(
            col("latitude").cast(DoubleType()),
            col("longitude").cast(DoubleType()),
            col("magnitude").cast(DoubleType()),
            col("time").cast("string")
        )
        .na.drop(subset=["latitude", "longitude", "magnitude"])
    )

    df = df.withColumn("ts", to_timestamp(col("time")))

    span_row = df.select(
        expr("datediff(max(ts), min(ts))").alias("span_days")
    ).collect()[0]
    total_span_days = max(int(span_row["span_days"] or 1), 1)
    print(f"  Historical window: {total_span_days} days")

    df = (df
        .withColumn("grid_lat", spark_round(col("latitude")  / 5, 0) * 5)
        .withColumn("grid_lon", spark_round(col("longitude") / 5, 0) * 5)
    )

    grid_stats = (df.groupBy("grid_lat", "grid_lon")
        .agg(
            count("*").alias("event_count"),
            avg("magnitude").alias("avg_mag"),
            spark_max("magnitude").alias("max_mag"),
            stddev("magnitude").alias("std_mag"),
        )
    ).toPandas()

    cell_baselines = {}
    for _, row in grid_stats.iterrows():
        key = f"{row['grid_lat']},{row['grid_lon']}"
        event_count = int(row["event_count"])
        daily_rate  = round(event_count / total_span_days, 6)
        cell_baselines[key] = {
            "historical_count": event_count,
            "daily_rate":       daily_rate,          
            "avg_mag":          round(float(row["avg_mag"]), 3),
            "max_mag":          round(float(row["max_mag"]), 2),
            "std_mag":          round(float(row["std_mag"]) if row["std_mag"] else 0, 3),
        }

    # Global fallback thresholds 
    global_mean = float(grid_stats["event_count"].mean())
    global_std  = float(grid_stats["event_count"].std())

    stats = {
        "global_mean_count":     round(global_mean, 2),
        "global_std_count":      round(global_std, 2),
        "global_daily_rate":     round(global_mean / total_span_days, 6),
        "spike_multiplier":      3.0,   
        "threshold_moderate":    round(global_mean + 2 * global_std, 2),
        "threshold_high":        round(global_mean + 3 * global_std, 2),
        "total_cells":           len(grid_stats),
        "historical_span_days":  total_span_days,
        "cell_baselines":        cell_baselines,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(IF_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  Global mean events/cell: {global_mean:.2f}")
    print(f"  Global daily rate/cell:  {global_mean / total_span_days:.4f}")
    print(f"  Spike threshold:         3x per-cell daily_rate")
    print(f"  Total grid cells:        {len(grid_stats)}")
    print(f"Anomaly stats saved to {IF_STATS_PATH}")

    return stats

#  3. KMeans Clustering — Regional Disaster Profiles

def train_regional_clusters(eq_df_cached):
    print("\n" + "=" * 70)
    print("  TRAINING: KMeans Regional Clustering (elbow method)")
    print("=" * 70)

    df = (eq_df_cached
        .select(
            col("latitude").cast(DoubleType()),
            col("longitude").cast(DoubleType()),
            col("magnitude").cast(DoubleType()),
            col("depth_km").cast(DoubleType()),
        )
        .na.drop(subset=["latitude", "longitude", "magnitude"])
        .fillna({"depth_km": 33.0})
    )

    df = (df
        .withColumn("grid_lat", spark_round(col("latitude")  / 10, 0) * 10)
        .withColumn("grid_lon", spark_round(col("longitude") / 10, 0) * 10)
    )

    regional = (df.groupBy("grid_lat", "grid_lon")
        .agg(
            count("*").alias("eq_count"),
            avg("magnitude").alias("avg_mag"),
            spark_max("magnitude").alias("max_mag"),
            avg("depth_km").alias("avg_depth"),
            stddev("magnitude").alias("mag_variability"),
        )
        .fillna({"mag_variability": 0.0})
    )
    regional.cache()

    feature_cols = ["eq_count", "avg_mag", "max_mag", "avg_depth",
                    "mag_variability", "grid_lat", "grid_lon"]

    assembler = VectorAssembler(
        inputCols=feature_cols, outputCol="raw_features", handleInvalid="skip"
    )
    scaler = StandardScaler(
        inputCol="raw_features", outputCol="features", withStd=True, withMean=True
    )
    prep_pipeline = Pipeline(stages=[assembler, scaler])
    prepped = prep_pipeline.fit(regional).transform(regional)
    prepped.cache()

    print("  Running elbow loop k=3..8 ...")
    wssse_by_k = {}
    for k in range(3, 9):
        km = KMeans(featuresCol="features", predictionCol="cluster_id",
                    k=k, seed=42, maxIter=100)
        km_model = km.fit(prepped)
        wssse = km_model.summary.trainingCost
        wssse_by_k[k] = round(wssse, 2)
        print(f"    k={k}  WSSSE={wssse:.2f}")

    best_k = 5 
    max_drop = 0
    ks = sorted(wssse_by_k.keys())
    for i in range(1, len(ks)):
        drop = wssse_by_k[ks[i - 1]] - wssse_by_k[ks[i]]
        if drop > max_drop:
            max_drop = drop
            best_k = ks[i]
    print(f"  Selected k={best_k} (largest WSSSE drop = {max_drop:.2f})")

    kmeans = KMeans(featuresCol="features", predictionCol="cluster_id",
                    k=best_k, seed=42, maxIter=100)
    final_pipeline = Pipeline(stages=[assembler, scaler, kmeans])
    model = final_pipeline.fit(regional)

    clustered = model.transform(regional)
    cluster_profiles = (clustered.groupBy("cluster_id")
        .agg(
            count("*").alias("region_count"),
            avg("eq_count").alias("avg_eq_count"),
            avg("avg_mag").alias("cluster_avg_mag"),
            spark_max("max_mag").alias("cluster_max_mag"),
            avg("avg_depth").alias("cluster_avg_depth"),
        )
    ).toPandas()

    profile_labels = {}
    for _, row in cluster_profiles.iterrows():
        cid = int(row["cluster_id"])
        if row["cluster_max_mag"] >= 7.0 and row["avg_eq_count"] > 50:
            label = "Critical Seismic Zone"
        elif row["cluster_avg_mag"] >= 4.5:
            label = "High Seismic Activity"
        elif row["avg_eq_count"] >= 20:
            label = "Frequent Moderate Activity"
        elif row["cluster_avg_depth"] > 200:
            label = "Deep Seismic Zone"
        else:
            label = "Low Activity"
        profile_labels[cid] = label
        print(f"  Cluster {cid}: {label}")
        print(f"    Regions: {int(row['region_count'])}, "
              f"Avg events: {row['avg_eq_count']:.0f}, "
              f"Avg mag: {row['cluster_avg_mag']:.2f}, "
              f"Max mag: {row['cluster_max_mag']:.1f}")

    model.write().overwrite().save(KMEANS_MODEL_PATH)

    kmeans_meta = {
        "k":             best_k,           
        "wssse_by_k":    wssse_by_k,       
        "profile_labels": {str(k): v for k, v in profile_labels.items()},
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(MODEL_DIR, "kmeans_meta.json"), "w") as f:
        json.dump(kmeans_meta, f, indent=2)

    regional.unpersist()
    prepped.unpersist()

    print(f"KMeans model saved to {KMEANS_MODEL_PATH}")
    return model

#  4. Seasonal Trend Analysis

def train_seasonal_baseline(eq_df_cached):
    print("\n" + "=" * 70)
    print("  TRAINING: Seasonal Trend Analysis")     
    print("=" * 70)

    df = (eq_df_cached
        .select(
            col("magnitude").cast(DoubleType()),
            col("time").cast("string"),
        )
        .na.drop(subset=["magnitude", "time"])
    )

    df = df.withColumn("ts", to_timestamp(col("time")))
    df = (df
        .withColumn("year",        year(col("ts")))
        .withColumn("month",       month(col("ts")))
        .withColumn("day_of_week", dayofweek(col("ts")))
    )

    # Monthly seasonal pattern 
    monthly = (df.groupBy("month")
        .agg(count("*").alias("total_count"), avg("magnitude").alias("avg_mag"))
    ).toPandas()

    yearly_monthly = (df.groupBy("year", "month")
        .agg(count("*").alias("count"), avg("magnitude").alias("avg_mag"))
        .orderBy("year", "month")
    ).toPandas()

    # Day-of-week pattern
    dow = (df.groupBy("day_of_week")
        .agg(count("*").alias("total_count"), avg("magnitude").alias("avg_mag"))
    ).toPandas()

    # Rates
    span_row     = df.select(expr("datediff(max(ts), min(ts))").alias("d")).collect()[0]
    total_days   = max(int(span_row["d"] or 1), 1)
    total_events = df.count()
    daily_rate   = total_events / total_days

    recent = (df
        .filter(col("ts") >= expr("current_timestamp() - interval 30 days"))
        .agg(count("*").alias("rc"), avg("magnitude").alias("rm"))
    ).toPandas()
    recent_daily   = float(recent["rc"].iloc[0])  / 30 if not recent.empty else daily_rate
    recent_avg_mag = float(recent["rm"].iloc[0])       if not recent.empty else 0.0

    stats = {
        "daily_event_rate":     round(daily_rate, 2),
        "recent_daily_rate":    round(recent_daily, 2),
        "recent_avg_magnitude": round(recent_avg_mag, 3),
        "monthly_pattern": {
            int(r["month"]): {
                "count":   int(r["total_count"]),
                "avg_mag": round(float(r["avg_mag"]), 3)
            } for _, r in monthly.iterrows()
        },
        "yearly_monthly_pattern": [
            {
                "year":    int(r["year"]),
                "month":   int(r["month"]),
                "count":   int(r["count"]),
                "avg_mag": round(float(r["avg_mag"]), 3)
            } for _, r in yearly_monthly.iterrows()
        ],
        "dow_pattern": {
            int(r["day_of_week"]): {
                "count":   int(r["total_count"]),
                "avg_mag": round(float(r["avg_mag"]), 3)
            } for _, r in dow.iterrows()
        },
        "total_events": total_events,
        "total_days":   total_days,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(SEASONAL_STATS_PATH, "w") as f:   
        json.dump(stats, f, indent=2)

    print(f"  Daily event rate:    {daily_rate:.2f}")
    print(f"  Recent 30-day rate:  {recent_daily:.2f}")
    print(f"  Total events:        {total_events:,}")
    print(f"  Yearlyxmonthly rows: {len(yearly_monthly)}")
    print(f"  Seasonal stats saved to {SEASONAL_STATS_PATH}")

    return stats

#  Run all training

def main():
    print("=" * 70)
    print("  DISASTER MONITOR — BATCH ML TRAINING PIPELINE")
    print(f"  Started: {datetime.now(timezone.utc).isoformat()} UTC")
    print("=" * 70)

    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    try:
        print(f"\n[INIT] Loading earthquake collection (>= {TRAINING_START}) ...")
        raw_eq = load_collection(spark, "earthquakes")
        raw_eq.printSchema()  # add this temporarily
        raw_eq.show(2)        # and this
        eq_df  = (raw_eq
            .withColumn("_ts", to_timestamp(col("time")))
            .filter(col("_ts") >= TRAINING_START)  
            .drop("_ts")
            .cache()
        )
        eq_count = eq_df.count()
        print(f"[INIT] Cached {eq_count:,} earthquake records for training.\n")

        # 1. Random Forest
        train_risk_model(eq_df)

        # 2. Anomaly Detection
        train_anomaly_detector(eq_df)

        # 3. KMeans — Regional Clustering 
        train_regional_clusters(eq_df)

        # 4. Seasonal Trend Analysis 
        train_seasonal_baseline(eq_df)

        print("ALL MODELS TRAINED SUCCESSFULLY")
        print(f"  Models saved to: {MODEL_DIR}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
