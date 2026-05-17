import builtins
import numpy as np
import pandas as pd
import mlflow
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count, max as spark_max, min as spark_min, col
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# MLflow config
TRACKING_URI = "sqlite:////home/hadoop/hpo_project/mlflow.db"
mlflow.set_tracking_uri(TRACKING_URI)
experiment = mlflow.get_experiment_by_name("HPO_Bayesian_Optimization")
experiment_id = experiment.experiment_id

# Start Spark
spark = SparkSession.builder \
    .appName("HPO Recommendation Engine") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print("=" * 60)
print("🔍 HPO Query-Driven Recommendation Engine")
print("=" * 60)

# Load real data from HDFS
df_spark = spark.read.parquet(
    "hdfs://localhost:9000/hpo/processed/hpob_real_parquet"
)
df_spark.createOrReplaceTempView("hpo_runs_real")

# Step 1: Query Hive-style — get dataset meta-features
print("\n📊 Step 1: Mining dataset meta-features via SparkSQL...")
dataset_stats = spark.sql("""
    SELECT
        search_space_id,
        dataset_id,
        COUNT(*) as n_runs,
        ROUND(AVG(val_accuracy), 4) as avg_acc,
        ROUND(MAX(val_accuracy), 4) as best_acc,
        ROUND(STDDEV(val_accuracy), 4) as std_acc,
        ROUND(AVG(hp_0), 4) as avg_hp0,
        ROUND(AVG(hp_1), 4) as avg_hp1,
        ROUND(AVG(hp_2), 4) as avg_hp2,
        ROUND(AVG(hp_3), 4) as avg_hp3
    FROM hpo_runs_real
    GROUP BY search_space_id, dataset_id
    HAVING COUNT(*) >= 10
    ORDER BY best_acc DESC
""")
dataset_stats_df = dataset_stats.toPandas()
print(f"✅ Mined {len(dataset_stats_df)} dataset profiles")
print(dataset_stats.show(5))

# Step 2: Query top configs per search space
print("\n📊 Step 2: Querying top configurations per space...")
top_configs = spark.sql("""
    SELECT
        search_space_id,
        ROUND(AVG(hp_0), 6) as best_hp0,
        ROUND(AVG(hp_1), 6) as best_hp1,
        ROUND(AVG(hp_2), 6) as best_hp2,
        ROUND(AVG(hp_3), 6) as best_hp3,
        ROUND(AVG(hp_4), 6) as best_hp4,
        ROUND(MAX(val_accuracy), 4) as best_acc,
        COUNT(*) as top_runs
    FROM hpo_runs_real
    WHERE val_accuracy >= 0.95
    GROUP BY search_space_id
    ORDER BY best_acc DESC
""")
top_configs_df = top_configs.toPandas()
print(f"✅ Found top configs for {len(top_configs_df)} search spaces")
top_configs.show()

spark.stop()

# Step 3: Build similarity-based recommender
print("\n🧠 Step 3: Building similarity-based recommender...")
meta_features = ['avg_acc', 'best_acc', 'std_acc', 'avg_hp0', 'avg_hp1', 'avg_hp2', 'avg_hp3']
dataset_matrix = dataset_stats_df[meta_features].fillna(0).values

def recommend_configs(query_dataset_idx, top_k=5):
    """
    Given a new dataset, find similar past datasets
    and recommend their best hyperparameter configs
    """
    query_vec = dataset_matrix[query_dataset_idx].reshape(1, -1)
    similarities = cosine_similarity(query_vec, dataset_matrix)[0]
    top_indices = np.argsort(similarities)[::-1][1:top_k+1]

    recommendations = []
    for idx in top_indices:
        row = dataset_stats_df.iloc[idx]
        sim_score = similarities[idx]
        space_id = int(row['search_space_id'])

        # Get best config for this space
        matching = top_configs_df[top_configs_df['search_space_id'] == space_id]
        if len(matching) > 0:
            config = matching.iloc[0]
            recommendations.append({
                'similar_dataset': int(row['dataset_id']),
                'search_space': space_id,
                'similarity': round(float(sim_score), 4),
                'best_acc': float(row['best_acc']),
                'recommended_hp0': float(config['best_hp0']),
                'recommended_hp1': float(config['best_hp1']),
                'recommended_hp2': float(config['best_hp2']),
                'recommended_hp3': float(config['best_hp3']),
            })
    return recommendations

# Test recommendation for first dataset
print("\n🎯 Step 4: Testing recommender on sample datasets...")
for test_idx in [0, 10, 50]:
    target = dataset_stats_df.iloc[test_idx]
    recs = recommend_configs(test_idx, top_k=3)
    print(f"\n📌 Query: Dataset {int(target['dataset_id'])} "
          f"(space={int(target['search_space_id'])}, "
          f"avg_acc={target['avg_acc']})")
    print(f"   Top {len(recs)} recommendations:")
    for r in recs:
        print(f"   → Space {r['search_space']} | "
              f"Similarity={r['similarity']} | "
              f"Best acc={r['best_acc']} | "
              f"HP0={r['recommended_hp0']:.4f}")

# Step 5: Evaluate recommendation quality
print("\n📊 Step 5: Evaluating recommendation quality...")
hits = 0
total = __builtins__["min"](100, len(dataset_stats_df)) if isinstance(__builtins__, dict) else min(100, len(dataset_stats_df))
for i in range(total):
    target_best = dataset_stats_df.iloc[i]['best_acc']
    recs = recommend_configs(i, top_k=3)
    if recs:
        rec_best = builtins.max(r['best_acc'] for r in recs)
        if rec_best >= target_best * 0.95:
            hits += 1

hit_rate = hits / total * 100
print(f"✅ Recommendation Hit Rate (95% threshold): {hit_rate:.1f}%")
print(f"   ({hits}/{total} datasets matched within 5% of best)")

# Log to MLflow
with mlflow.start_run(experiment_id=experiment_id, run_name="recommendation_engine"):
    mlflow.log_param("n_datasets_profiled", len(dataset_stats_df))
    mlflow.log_param("n_spaces_with_top_configs", len(top_configs_df))
    mlflow.log_param("similarity_metric", "cosine")
    mlflow.log_param("top_k", 5)
    mlflow.log_metric("recommendation_hit_rate", round(hit_rate, 2))
    print("✅ Logged to MLflow!")

print("\n🎉 Day 10 Complete — Query-Driven Recommender working!")
