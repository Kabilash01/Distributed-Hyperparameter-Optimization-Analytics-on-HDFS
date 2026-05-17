from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("HPO-B CSV to Parquet") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .config("spark.sql.warehouse.dir", "hdfs://localhost:9000/user/hive/warehouse") \
    .config("hive.metastore.uris", "thrift://localhost:9083") \
    .enableHiveSupport() \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("✅ Spark started!")

# Read CSV from HDFS
df = spark.read.csv(
    "hdfs://localhost:9000/hpo/raw/hpob/hpo_runs.csv",
    header=True,
    inferSchema=True
)
print(f"✅ Loaded {df.count():,} rows")
df.printSchema()

# Write Parquet to HDFS
df.write.mode("overwrite").parquet(
    "hdfs://localhost:9000/hpo/processed/hpo_runs_parquet"
)
print("✅ Parquet written to HDFS!")

# Register Hive table via Beeline instead
print("✅ Done! Now create Hive table manually via Beeline.")
spark.stop()
