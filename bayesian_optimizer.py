import optuna
import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
import warnings
warnings.filterwarnings("ignore")

# MLflow config
TRACKING_URI = "sqlite:////home/hadoop/hpo_project/mlflow.db"
mlflow.set_tracking_uri(TRACKING_URI)
experiment = mlflow.get_experiment_by_name("HPO_Bayesian_Optimization")
experiment_id = experiment.experiment_id

# Load historical data from HDFS via PySpark
spark = SparkSession.builder \
    .appName("HPO Bayesian Optimizer") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print("✅ Loading historical HPO data from HDFS...")
df = spark.read.parquet(
    "hdfs://localhost:9000/hpo/processed/hpo_runs_parquet"
).toPandas()
print(f"✅ Loaded {len(df):,} historical runs")

# Build surrogate model from historical data
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

print("🔧 Training surrogate model on historical runs...")

le_opt = LabelEncoder()
le_space = LabelEncoder()

df["optimizer_enc"] = le_opt.fit_transform(df["optimizer"])
df["space_enc"] = le_space.fit_transform(df["search_space_id"].astype(str))

feature_cols = [
    "learning_rate", "batch_size", "dropout",
    "num_layers", "hidden_units", "weight_decay",
    "epochs", "optimizer_enc", "space_enc"
]

X = df[feature_cols].values
y = df["val_accuracy"].values

surrogate = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
surrogate.fit(X, y)
print(f"✅ Surrogate model trained! R² score: {surrogate.score(X, y):.4f}")

spark.stop()

# Define Optuna objective using surrogate
def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "num_layers": trial.suggest_int("num_layers", 1, 5),
        "hidden_units": trial.suggest_categorical("hidden_units", [64, 128, 256, 512]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "epochs": trial.suggest_int("epochs", 10, 200),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop", "adagrad"]),
        "search_space_id": "5859"
    }

    # Encode for surrogate
    try:
        opt_enc = le_opt.transform([params["optimizer"]])[0]
        space_enc = le_space.transform([params["search_space_id"]])[0]
    except:
        return 0.0

    X_pred = np.array([[
        params["learning_rate"], params["batch_size"],
        params["dropout"], params["num_layers"],
        params["hidden_units"], params["weight_decay"],
        params["epochs"], opt_enc, space_enc
    ]])

    predicted_accuracy = surrogate.predict(X_pred)[0]
    return predicted_accuracy

# Run Bayesian optimization
print("\n🚀 Running Bayesian Optimization...")
print("=" * 50)

with mlflow.start_run(experiment_id=experiment_id, run_name="bayesian_optimization"):

    # Baseline — random search
    random_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=42)
    )
    random_study.optimize(objective, n_trials=200, show_progress_bar=False)
    random_best = random_study.best_value
    print(f"✅ Random Search best: {random_best:.4f}")

    # Bayesian optimization with TPE
    tpe_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    tpe_study.optimize(objective, n_trials=200, show_progress_bar=False)
    tpe_best = tpe_study.best_value
    print(f"✅ Bayesian (TPE) best: {tpe_best:.4f}")

    # Improvement
    improvement = ((tpe_best - random_best) / random_best) * 100
    print(f"✅ Improvement over random: {improvement:.2f}%")

    # Log to MLflow
    mlflow.log_param("n_trials", 50)
    mlflow.log_param("sampler", "TPE")
    mlflow.log_param("target_search_space", "5859")
    mlflow.log_metric("random_search_best", round(random_best, 4))
    mlflow.log_metric("bayesian_best", round(tpe_best, 4))
    mlflow.log_metric("improvement_pct", round(improvement, 2))

    # Log best hyperparameters
    best_params = tpe_study.best_params
    for k, v in best_params.items():
        mlflow.log_param(f"best_{k}", v)

    print("\n🏆 Best Hyperparameters Found:")
    for k, v in best_params.items():
        print(f"   {k}: {v}")

print("\n🎉 Day 7 Complete — Bayesian Optimizer working!")
