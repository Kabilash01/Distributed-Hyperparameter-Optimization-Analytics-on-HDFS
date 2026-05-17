from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
import mlflow

# MLflow config — fix path with 4 slashes for absolute path
TRACKING_URI = "sqlite:////home/hadoop/hpo_project/mlflow.db"
mlflow.set_tracking_uri(TRACKING_URI)

# Get or create experiment
experiment_name = "HPO_Bayesian_Optimization"
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    print(f"✅ Created experiment ID: {experiment_id}")
else:
    experiment_id = experiment.experiment_id
    print(f"✅ Found experiment ID: {experiment_id}")

# Start Spark
spark = SparkSession.builder \
    .appName("HPO Feature Engineering") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Load from HDFS
df = spark.read.parquet(
    "hdfs://localhost:9000/hpo/processed/hpo_runs_parquet"
)
print(f"✅ Loaded {df.count():,} rows")

# Feature Engineering
indexer = StringIndexer(inputCol="optimizer", outputCol="optimizer_idx")
space_indexer = StringIndexer(inputCol="search_space_id", outputCol="space_idx")
dataset_indexer = StringIndexer(inputCol="dataset_id", outputCol="dataset_idx")

assembler = VectorAssembler(
    inputCols=[
        "learning_rate", "batch_size", "dropout",
        "num_layers", "hidden_units", "weight_decay",
        "epochs", "optimizer_idx", "space_idx", "dataset_idx"
    ],
    outputCol="features"
)

pipeline = Pipeline(stages=[
    indexer, space_indexer, dataset_indexer, assembler
])

model = pipeline.fit(df)
df_features = model.transform(df)
print("✅ Feature vector created!")

# Save to HDFS
df_features.select(
    "run_id", "search_space_id", "dataset_id",
    "features", "val_accuracy", "optimizer"
).write.mode("overwrite").parquet(
    "hdfs://localhost:9000/hpo/processed/hpo_features"
)
print("✅ Feature matrix saved to HDFS!")

# Log to MLflow
avg_acc = df_features.agg(avg("val_accuracy")).collect()[0][0]
with mlflow.start_run(experiment_id=experiment_id, run_name="feature_engineering"):
    mlflow.log_param("num_features", 10)
    mlflow.log_param("num_rows", 100000)
    mlflow.log_metric("avg_val_accuracy", float(f"{avg_acc:.4f}"))
    print(f"✅ Logged to MLflow! avg_val_accuracy={round(avg_acc,4)}")

spark.stop()
print("\n🎉 Day 6 Complete!")
