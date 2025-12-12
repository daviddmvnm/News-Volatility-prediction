# silver_news.py

from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, trim, length, lower, current_timestamp,
    input_file_name, from_json, to_date, to_timestamp, when
)
from pyspark.sql.types import StructType, StructField, StringType
from delta import configure_spark_with_delta_pip


# ---------------------------------------------------------
# Build Spark (Delta-enabled)
# ---------------------------------------------------------
def build_spark():
    builder = (
        SparkSession.builder
        .appName("silver_transform")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.driver.memory", "3g")
        .config("spark.executor.memory", "3g")
        .config("spark.sql.shuffle.partitions", "16")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


spark = build_spark()


# ---------------------------------------------------------
# Stable paths
# ---------------------------------------------------------
FILE_DIR = Path(__file__).resolve().parent
ROOT = FILE_DIR.parent
BRONZE_DELTA = ROOT / "bronze" / "delta_news"
SILVER_DELTA = FILE_DIR / "delta_news_silver"

print("Bronze Delta:", BRONZE_DELTA)
print("Silver Delta:", SILVER_DELTA)


# ---------------------------------------------------------
# Load Bronze table
# ---------------------------------------------------------
bronze_df = spark.read.format("delta").load(str(BRONZE_DELTA))


# ---------------------------------------------------------
# JSON schema for extra metadata
# ---------------------------------------------------------
extra_schema = StructType([
    StructField("publication", StringType()),
    StructField("author", StringType()),
    StructField("url", StringType()),
    StructField("text_type", StringType()),
    StructField("time_precision", StringType()),
    StructField("dataset_source", StringType()),
    StructField("dataset", StringType()),
    StructField("source", StringType()),
    StructField("raw_type", StringType()),
    StructField("tz_hint", StringType()),
    StructField("date_raw", StringType()),
    StructField("date_trading", StringType()),
    StructField("anchor_policy", StringType())
])


# ---------------------------------------------------------
# Parse JSON → struct + extract fields
# ---------------------------------------------------------
df = bronze_df.withColumn("extra", from_json(col("extra_fields"), extra_schema))

for field in extra_schema.fieldNames():
    df = df.withColumn(field, col(f"extra.{field}"))


# ---------------------------------------------------------
# Core cleaning
# ---------------------------------------------------------
df = df.withColumn("text", trim(col("text")))
df = df.withColumn("len_text", length(col("text")))

df = (
    df.where(col("text").isNotNull())
      .where(length(trim(col("text"))) > 50)
      .where(col("date").isNotNull())
)

df = df.withColumn("publication", lower(trim(col("publication"))))


# ---------------------------------------------------------
# Date normalization (★ NEW ★)
# ---------------------------------------------------------
df = df.withColumn(
    "date",
    when(col("date").rlike("T"), to_timestamp("date"))
    .otherwise(to_date("date"))
)

df = df.withColumn(
    "date_raw",
    when(col("date_raw").rlike("T"), to_timestamp("date_raw"))
    .otherwise(to_date("date_raw"))
)

df = df.withColumn(
    "date_trading",
    to_timestamp("date_trading")
)


# ---------------------------------------------------------
# Metadata passthrough
# ---------------------------------------------------------
if "source_file" not in df.columns:
    df = df.withColumn("source_file", input_file_name())

df = df.withColumn("silver_ingestion_ts", current_timestamp())


# ---------------------------------------------------------
# Cleanup
# ---------------------------------------------------------
df = df.drop("extra", "extra_fields")


# ---------------------------------------------------------
# Write Silver
# ---------------------------------------------------------
(
    df.write
      .format("delta")
      .mode("overwrite")
      .save(str(SILVER_DELTA))
)

print("\nSilver ETL completed")
print("Rows written:", df.count())
print("Saved to:", SILVER_DELTA)
