import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
import optuna
import pandas as pd
import numpy as np

# Configure MLflow to store artifacts in HDFS
MLFLOW_TRACKING_URI = "sqlite:///home/hadoop/hpo_project/mlflow.db"
ARTIFACT_ROOT = "hdfs://localhost:9000/hpo/mlflow_artifacts"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print(f"✅ MLflow tracking URI: {MLFLOW_TRACKING_URI}")
print(f"✅ Artifact root: {ARTIFACT_ROOT}")

# Create experiment
experiment_name = "HPO_Bayesian_Optimization"
try:
    experiment_id = mlflow.create_experiment(
        experiment_name,
        artifact_location=ARTIFACT_ROOT
    )
    print(f"✅ Created experiment: {experiment_name} (ID: {experiment_id})")
except:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    print(f"✅ Using existing experiment: {experiment_name}")

# Test logging a sample run
with mlflow.start_run(experiment_id=experiment_id):
    # Log hyperparameters
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 64)
    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("dropout", 0.3)
    mlflow.log_param("num_layers", 3)

    # Log metrics
    mlflow.log_metric("val_accuracy", 0.89)
    mlflow.log_metric("train_loss", 0.12)
    mlflow.log_metric("val_loss", 0.15)

    print("✅ Test run logged to MLflow!")

print("\n✅ MLflow setup complete!")
print("Run: mlflow ui --backend-store-uri sqlite:///home/hadoop/hpo_project/mlflow.db")
