from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("HPO-B Real Data") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("📂 Loading real HPO-B from HDFS...")
df = spark.read.csv(
    "hdfs://localhost:9000/hpo/raw/hpob/hpob_real_train.csv",
    header=True,
    inferSchema=True
)

print(f"✅ Loaded {df.count():,} real runs")
df.printSchema()

# Write Parquet
df.write.mode("overwrite").parquet(
    "hdfs://localhost:9000/hpo/processed/hpob_real_parquet"
)
print("✅ Real Parquet written to HDFS!")
spark.stop()
