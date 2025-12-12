from pathlib import Path
from math import ceil

from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, input_file_name
from delta import configure_spark_with_delta_pip


# -------------------------------------------------------------
# Paths
# -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raws_split"
BRONZE_DELTA = BASE_DIR / "delta_news_split"


# -------------------------------------------------------------
# Build Spark with Delta (PyPI version only)
# -------------------------------------------------------------
def build_spark():
    builder = (
        SparkSession.builder
        .appName("bronze_news_ingestion")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

        # Reasonable configs for your laptop
        .config("spark.driver.memory", "6g")
        .config("spark.executor.memory", "6g")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.files.maxPartitionBytes", "64m")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


# -------------------------------------------------------------
# Write a single parquet file in safe randomSplit batches
# -------------------------------------------------------------
def process_file(spark, file_path, batch_size=200_000):
    print(f"\nProcessing: {file_path.name}")

    # Load data with metadata
    df = (
        spark.read.parquet(str(file_path))
        .withColumn("ingestion_ts", current_timestamp())
        .withColumn("source_file", input_file_name())
    )

    total_rows = df.count()
    print(f"  → Total rows: {total_rows:,}")

    # Determine number of chunks
    num_batches = ceil(total_rows / batch_size)
    print(f"  → Creating {num_batches} batches (~{batch_size:,} rows each)\n")

    # Create evenly sized random splits
    # (Fast, stable, no shuffle warnings)
    weights = [1.0] * num_batches
    splits = df.randomSplit(weights, seed=42)

    for i, chunk in enumerate(splits, 1):
        print(f"    Writing batch {i}/{num_batches}")
        chunk.write.format("delta").mode("append").save(str(BRONZE_DELTA))


# -------------------------------------------------------------
# Process all parquet files in raw directory
# -------------------------------------------------------------
def ingest_all(spark):
    files = sorted(RAW_DIR.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No parquet files found in {RAW_DIR}")

    print(f"Found {len(files)} parquet files.\n")

    for idx, path in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] {path.name}")
        process_file(spark, path)


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    spark = build_spark()
    ingest_all(spark)
    spark.stop()
    print("\n✅ Bronze ingestion completed.\n")


if __name__ == "__main__":
    main()
