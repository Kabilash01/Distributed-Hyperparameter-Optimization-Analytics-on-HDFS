import numpy as np
import pandas as pd
import mlflow
import optuna
from pyspark.sql import SparkSession
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# MLflow config
TRACKING_URI = "sqlite:////home/hadoop/hpo_project/mlflow.db"
mlflow.set_tracking_uri(TRACKING_URI)
experiment = mlflow.get_experiment_by_name("HPO_Bayesian_Optimization")
experiment_id = experiment.experiment_id

# Load real data from HDFS
spark = SparkSession.builder \
    .appName("Real HPO Optimizer") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print("📂 Loading REAL HPO-B data from HDFS...")
df_spark = spark.read.parquet(
    "hdfs://localhost:9000/hpo/processed/hpob_real_parquet"
)

# Use search space 6794 (highest avg accuracy — 591K runs)
TARGET_SPACE = 6794
df_space = df_spark.filter(
    df_spark.search_space_id == TARGET_SPACE
).toPandas()

spark.stop()
print(f"✅ Loaded {len(df_space):,} real runs for space {TARGET_SPACE}")

# Feature columns — use non-null hp columns
hp_cols = [c for c in df_space.columns if c.startswith('hp_')]
df_space[hp_cols] = df_space[hp_cols].fillna(0)

# Find which hp columns have variance
valid_hp = [c for c in hp_cols if df_space[c].std() > 0]
print(f"✅ Valid hyperparameter columns: {valid_hp}")

X = df_space[valid_hp].values
y = df_space['val_accuracy'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✅ Train: {len(X_train):,} | Test: {len(X_test):,}")

# Train surrogate on REAL data
print("🔧 Training surrogate on REAL HPO-B data...")
surrogate = RandomForestRegressor(
    n_estimators=100, random_state=42, n_jobs=-1
)
surrogate.fit(X_train, y_train)
r2_train = surrogate.score(X_train, y_train)
r2_test = surrogate.score(X_test, y_test)
print(f"✅ Surrogate R² train={r2_train:.4f} test={r2_test:.4f}")

# Feature importance
importance = dict(zip(valid_hp, surrogate.feature_importances_))
print("\n📊 Real Hyperparameter Importance:")
for k, v in sorted(importance.items(), key=lambda x: -x[1]):
    print(f"   {k}: {v:.4f}")

n_hp = len(valid_hp)

# Optuna objective on real surrogate
def objective(trial):
    config = [
        trial.suggest_float(f"hp_{i}", 0.0, 1.0)
        for i in range(n_hp)
    ]
    return surrogate.predict([config])[0]

print("\n🚀 Running optimization on REAL data...")

# Random search baseline
random_study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.RandomSampler(seed=42)
)
random_study.optimize(objective, n_trials=200)
random_best = random_study.best_value
print(f"✅ Random Search best: {random_best:.4f}")

# Bayesian TPE
tpe_study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)
tpe_study.optimize(objective, n_trials=200)
tpe_best = tpe_study.best_value
print(f"✅ Bayesian TPE best:  {tpe_best:.4f}")

improvement = ((tpe_best - random_best) / random_best) * 100
print(f"✅ Improvement: {improvement:.2f}%")

# Compare vs dataset best
dataset_best = y.max()
dataset_avg = y.mean()
print(f"\n📊 Real Dataset Stats:")
print(f"   Dataset best:    {dataset_best:.4f}")
print(f"   Dataset avg:     {dataset_avg:.4f}")
print(f"   Our TPE best:    {tpe_best:.4f}")
print(f"   Gap to optimal:  {((dataset_best - tpe_best)/dataset_best*100):.2f}%")

# Log to MLflow
with mlflow.start_run(experiment_id=experiment_id, run_name="real_data_bayesian"):
    mlflow.log_param("dataset", "HPO-B Real")
    mlflow.log_param("search_space", TARGET_SPACE)
    mlflow.log_param("n_real_runs", len(df_space))
    mlflow.log_param("n_trials", 200)
    mlflow.log_param("surrogate", "RandomForest")
    mlflow.log_metric("surrogate_r2_train", round(r2_train, 4))
    mlflow.log_metric("surrogate_r2_test", round(r2_test, 4))
    mlflow.log_metric("random_search_best", round(random_best, 4))
    mlflow.log_metric("bayesian_tpe_best", round(tpe_best, 4))
    mlflow.log_metric("improvement_pct", round(improvement, 2))
    mlflow.log_metric("dataset_best", round(float(dataset_best), 4))
    print("✅ Logged to MLflow!")

print("\n🎉 Day 9 Complete — Real Data Bayesian Optimization!")
