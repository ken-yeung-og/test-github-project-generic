import argparse
import logging
import os
import pathlib
import sys
import subprocess


# Ensure DuckDB is installed
def install_duckdb():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "duckdb==1.0.0"])


install_duckdb()

import boto3
import duckdb
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Environment variables with fallbacks
S3_PATH = os.getenv(
    "S3_PATH",
    "s3://og-407-temp-test-data/austin-traffic/Radar_Traffic_Counts_20240528.csv",
)
DB_PATH = os.getenv("DB_PATH", "data/traffic.duckdb")

TARGET_COLUMN = "volume"
FEATURE_COLUMNS = [
    "intersection_name",
    "month",
    "day",
    "year",
    "day_of_week",
    "lane_nb_out",
    "lane_sb",
    "lane_nb",
    "lane_eb_mid",
    "lane_nb_4",
    "lane_sb_out",
    "lane_wb_in",
    "lane_wb_out",
    "lane_nb_mid",
    "lane_sb_mid",
    "lane_sb_1",
    "lane_nb_in",
    "lane_sb_rturn",
    "lane_sb_lturn",
    "lane_eb_out",
    "lane_sb_2",
    "lane_nb_3",
    "lane_sb_in",
    "lane_eb_in",
    "direction_sb2",
    "direction_wb",
    "direction_nb3",
    "direction_eb",
    "direction_nb4",
    "direction_sb",
    "direction_nb",
    "direction_sb1",
]


def clean_directions(table: duckdb.DuckDBPyRelation) -> duckdb.DuckDBPyRelation:
    """Remove rows with invalid travel direction."""
    logging.debug("Cleaning rows with invalid travel directions.")
    return table.filter("direction <> '%None%'")


def clean_lanes(table: duckdb.DuckDBPyRelation) -> duckdb.DuckDBPyRelation:
    """Remove rows with invalid lane values."""
    logger.debug("Cleaning rows with invalid lane values.")
    return table.filter("lane not like '%Lane%'")


def normalize_lanes(table: duckdb.DuckDBPyRelation) -> duckdb.DuckDBPyRelation:
    """Normalize lane values to a consistent snake_case format."""
    logger.debug("Normalizing lane values to consistent format.")
    return table.select(
        duckdb.StarExpression(exclude=["lane"]),
        duckdb.FunctionExpression(
            "regexp_replace",
            duckdb.ColumnExpression("lane"),
            duckdb.ConstantExpression(r"(\d+$)"),
            duckdb.ConstantExpression(r"_\1"),
        ).alias("lane"),
    )


def apply_dummies(
    table: duckdb.DuckDBPyRelation, column: str, exclude_original: bool = True
) -> duckdb.DuckDBPyRelation:
    """Convert a categorical column into dummy columns (i.e. one-hot encoding)."""
    logger.debug(f"Applying one-hot encoding to column '{column}'.")
    unique_values = table.select(column).distinct().fetchall()
    logger.debug(f"Unique values in column '{column}': {unique_values}")

    dummy_expressions = [
        duckdb.CaseExpression(
            condition=duckdb.ColumnExpression(column)
            == duckdb.ConstantExpression(val[0]),
            value=duckdb.ConstantExpression(1),
        )
        .otherwise(duckdb.ConstantExpression(0))
        .alias(f"{column}_{val[0]}")
        for val in unique_values
    ]

    return table.select(
        duckdb.StarExpression(exclude=[column] if exclude_original else []),
        *dummy_expressions,
    )


def downcase_columns(table: duckdb.DuckDBPyRelation) -> duckdb.DuckDBPyRelation:
    """Convert column names to lowercase and replace spaces with underscores."""
    return table.select(
        *[
            duckdb.ColumnExpression(col).alias("_".join(col.lower().split()))
            for col in table.columns
        ]
    )


def main():
    """Command line interface for preprocessing script."""
    logging.info("Starting preprocessing.")

    parser = argparse.ArgumentParser(
        description="Preprocess austin traffic data for XGBoost."
    )
    parser.add_argument(
        "--input-data",
        type=str,
        required=True,
        default=S3_PATH,
        help="Path to input data.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed for random number generation.",
    )
    args = parser.parse_args()

    input_data = args.input_data
    random_seed = args.random_seed

    base_dir = os.path.join("/", "opt", "ml", "processing")
    data_dir = os.path.join(base_dir, "data")
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)

    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logging.info(f"Downloading data from bucket: {bucket}, key: {key}")

    file_name = os.path.join(data_dir, "traffic-dataset.csv")
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, file_name)

    logging.debug(f"Reading downloaded data from {file_name}")
    table = duckdb.read_csv(file_name, header=True)

    logger.debug("Applying transformations.")
    table = clean_directions(table)
    table = clean_lanes(table)
    table = normalize_lanes(table)
    table = apply_dummies(table, "direction")
    table = apply_dummies(table, "lane")
    table = downcase_columns(table)

    y = table.select(TARGET_COLUMN).fetchdf()
    X_pre = table.select(*FEATURE_COLUMNS).fetchdf().to_numpy()
    y_pre = y.to_numpy().reshape(len(y), 1)

    X = np.concatenate((X_pre, y_pre), axis=1)
    X = pd.DataFrame(X)
    X = pd.get_dummies(X)
    logger.info("Splitting data into train, validation, and test sets.")
    X = X.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    train, val, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])

    logger.debug(f"train: {train.shape}, val: {val.shape}, test: {test.shape}")

    logger.debug(f"Writing processed data to {base_dir}.")

    train_path = os.path.join(base_dir, "train")
    val_path = os.path.join(base_dir, "validation")
    test_path = os.path.join(base_dir, "test")
    pathlib.Path(train_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(val_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(test_path).mkdir(parents=True, exist_ok=True)

    train.to_csv(os.path.join(train_path, "train.csv"), header=False, index=False)
    val.to_csv(os.path.join(val_path, "validation.csv"), header=False, index=False)
    test.to_csv(os.path.join(test_path, "test.csv"), header=False, index=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
