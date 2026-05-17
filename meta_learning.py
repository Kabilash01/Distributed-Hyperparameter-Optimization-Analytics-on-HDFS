import numpy as np
import pandas as pd
import mlflow
import optuna
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, stddev, count
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# MLflow config
TRACKING_URI = "sqlite:////home/hadoop/hpo_project/mlflow.db"
mlflow.set_tracking_uri(TRACKING_URI)
experiment = mlflow.get_experiment_by_name("HPO_Bayesian_Optimization")
experiment_id = experiment.experiment_id

# Load data
spark = SparkSession.builder \
    .appName("Meta-Learning HPO") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print("✅ Loading data from HDFS...")
df_spark = spark.read.parquet(
    "hdfs://localhost:9000/hpo/processed/hpo_runs_parquet"
)

# Query top configs per search space using Hive-style SparkSQL
df_spark.createOrReplaceTempView("hpo_runs")

print("\n📊 Top configs per search space (from HDFS via SparkSQL):")
top_configs = spark.sql("""
    SELECT search_space_id, optimizer,
           ROUND(AVG(val_accuracy), 4) as avg_acc,
           ROUND(AVG(learning_rate), 6) as avg_lr,
           ROUND(AVG(dropout), 4) as avg_dropout,
           COUNT(*) as runs
    FROM hpo_runs
    GROUP BY search_space_id, optimizer
    ORDER BY avg_acc DESC
    LIMIT 15
""")
top_configs.show()

df = df_spark.toPandas()
spark.stop()

# Encode
le_opt = LabelEncoder()
le_space = LabelEncoder()
df["optimizer_enc"] = le_opt.fit_transform(df["optimizer"])
df["space_enc"] = le_space.fit_transform(df["search_space_id"].astype(str))

feature_cols = [
    "learning_rate", "batch_size", "dropout",
    "num_layers", "hidden_units", "weight_decay",
    "epochs", "optimizer_enc", "space_enc"
]

# Meta-Learning: train per search space surrogates
print("\n🧠 Training per-search-space surrogate models...")
search_spaces = df["search_space_id"].unique()
surrogates = {}

for space in search_spaces:
    space_df = df[df["search_space_id"] == space]
    X = space_df[feature_cols].values
    y = space_df["val_accuracy"].values
    model = RandomForestRegressor(
        n_estimators=50, random_state=42, n_jobs=-1
    )
    model.fit(X, y)
    surrogates[space] = model
    print(f"   Space {space}: {len(space_df)} runs, R²={model.score(X,y):.4f}")

print(f"✅ {len(surrogates)} surrogate models trained!")

# Transfer HPO: use best configs from similar spaces
# to warm-start optimization on target space
TARGET_SPACE = 5906
print(f"\n🎯 Transfer HPO for target space: {TARGET_SPACE}")

# Get top 10 configs from similar spaces as warm-start
similar_spaces = [s for s in search_spaces if s != TARGET_SPACE]
warm_start_configs = []

for space in similar_spaces[:3]:
    space_df = df[df["search_space_id"] == space]
    top = space_df.nlargest(3, "val_accuracy")[feature_cols + ["val_accuracy"]]
    warm_start_configs.append(top)

warm_df = pd.concat(warm_start_configs)
print(f"✅ Warm-start pool: {len(warm_df)} configs from similar spaces")

# Optimize target space with warm-start
surrogate_target = surrogates[TARGET_SPACE]

def objective_transfer(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "num_layers": trial.suggest_int("num_layers", 1, 5),
        "hidden_units": trial.suggest_categorical("hidden_units", [64, 128, 256, 512]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "epochs": trial.suggest_int("epochs", 10, 200),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop", "adagrad"]),
    }
    try:
        opt_enc = le_opt.transform([params["optimizer"]])[0]
        space_enc_val = le_space.transform([str(TARGET_SPACE)])[0]
    except:
        return 0.0

    X_pred = np.array([[
        params["learning_rate"], params["batch_size"],
        params["dropout"], params["num_layers"],
        params["hidden_units"], params["weight_decay"],
        params["epochs"], opt_enc, space_enc_val
    ]])
    return surrogate_target.predict(X_pred)[0]

print("\n🚀 Running Transfer HPO optimization...")

# Vanilla TPE (no warm start)
vanilla_study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)
vanilla_study.optimize(objective_transfer, n_trials=100, show_progress_bar=False)

# Transfer TPE (with warm start from similar spaces)
transfer_study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)

# Enqueue warm-start configs
for _, row in warm_df.head(10).iterrows():
    try:
        opt_name = le_opt.inverse_transform([int(row['optimizer_enc'])])[0]
        transfer_study.enqueue_trial({
            "learning_rate": float(row["learning_rate"]),
            "batch_size": int(row["batch_size"]),
            "dropout": float(row["dropout"]),
            "num_layers": int(row["num_layers"]),
            "hidden_units": int(row["hidden_units"]),
            "weight_decay": float(row["weight_decay"]),
            "epochs": int(row["epochs"]),
            "optimizer": opt_name
        })
    except:
        pass

transfer_study.optimize(objective_transfer, n_trials=100, show_progress_bar=False)

# Results
vanilla_best = vanilla_study.best_value
transfer_best = transfer_study.best_value
improvement = ((transfer_best - vanilla_best) / vanilla_best) * 100

print(f"\n📊 Results for search space {TARGET_SPACE}:")
print(f"   Vanilla TPE best:   {vanilla_best:.4f}")
print(f"   Transfer TPE best:  {transfer_best:.4f}")
print(f"   Improvement:        {improvement:.2f}%")

print(f"\n🏆 Best config (Transfer TPE):")
for k, v in transfer_study.best_params.items():
    print(f"   {k}: {v}")

# Log to MLflow
with mlflow.start_run(experiment_id=experiment_id, run_name="meta_learning_transfer"):
    mlflow.log_param("target_space", TARGET_SPACE)
    mlflow.log_param("n_source_spaces", len(similar_spaces[:3]))
    mlflow.log_param("warm_start_configs", len(warm_df))
    mlflow.log_param("n_trials", 100)
    mlflow.log_metric("vanilla_tpe_best", round(vanilla_best, 4))
    mlflow.log_metric("transfer_tpe_best", round(transfer_best, 4))
    mlflow.log_metric("transfer_improvement_pct", round(improvement, 2))
    for k, v in transfer_study.best_params.items():
        mlflow.log_param(f"best_{k}", v)
    print("\n✅ Results logged to MLflow!")

print("\n🎉 Day 8 Complete — Meta-Learning Transfer HPO working!")
