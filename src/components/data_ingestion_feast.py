"""Feast-aware data ingestion for the Sparkov pipeline.

Loads fraudTrain.csv + fraudTest.csv, runs the same FeatureEngineering
as the production pipeline, splits the resulting features into the
three Feast feature-view tables, and writes them to:

  data/sparkov_cc_features.csv
  data/sparkov_merchant_features.csv
  data/sparkov_transaction_features.csv

These CSVs are the offline-store sources referenced in
feast/feature_repo/features.py. After `feast apply` + `feast materialize`
they land in Postgres (offline) and Redis (online).
"""
import os

import pandas as pd

from src.components.feature_engineering import FeatureEngineering
from src.utils.logger import logging


def _read_sparkov(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    return df


def _to_event_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Feast needs an explicit `event_timestamp` column on every source."""
    df = df.copy()
    df["event_timestamp"] = df["trans_date_trans_time"]
    df["created"]         = df["trans_date_trans_time"]
    return df


def build_feature_tables(train_path: str = "data/raw/fraudTrain.csv",
                          test_path:  str = "data/raw/fraudTest.csv",
                          out_dir:    str = "data") -> dict:
    """Build the three feature tables from the raw CSVs.

    Returns: dict of {view_name: output_csv_path}
    """
    logging.info("Feast data ingestion started")
    train_df = _read_sparkov(train_path)
    test_df  = _read_sparkov(test_path)
    logging.info(f"Loaded train={len(train_df):,}  test={len(test_df):,}")

    # Build features on the union (the velocity/TEs need history)
    fe = FeatureEngineering()
    combined = pd.concat([train_df, test_df]).sort_values("trans_date_trans_time").reset_index(drop=True)
    combined_fe = fe.fit_transform(combined, compute_velocity=True)
    logging.info(f"Feature engineering: {len(combined_fe.columns)} columns")

    # Split back into train + test for the per-view tables
    is_train = combined["trans_date_trans_time"] < train_df["trans_date_trans_time"].max()
    _train_fe = combined_fe[is_train].reset_index(drop=True)
    _test_fe  = combined_fe[~is_train].reset_index(drop=True)
    all_fe    = combined_fe  # for the per-entity tables we need all rows
    _features = fe.feature_list

    # ── View 1: per-transaction (one row per trans_num) ──────────────────────
    txn_cols = [
        "trans_num", "trans_date_trans_time",
        "hour", "dow", "month", "is_night", "age", "distance_km",
        "amt", "amt_log", "amt_is_round",
        "category_FE", "city_FE", "state_FE", "job_FE", "zip_FE",
        "category_te", "city_te", "state_te", "job_te",
        "amt_per_category_mean", "amt_per_category_std",
        "is_fraud",
    ]
    txn_view = _to_event_timestamp(all_fe[txn_cols])
    txn_view = txn_view[[c for c in txn_cols if c != "trans_date_trans_time"] + ["event_timestamp", "created"]]
    txn_view = txn_view.rename(columns={"trans_num": "trans_num"})
    txn_view["trans_num"] = txn_view["trans_num"].astype(str)
    txn_view["is_fraud"]  = txn_view["is_fraud"].astype(float)
    txn_path = os.path.join(out_dir, "sparkov_transaction_features.csv")
    txn_view.to_csv(txn_path, index=False)
    txn_parquet = os.path.join(out_dir, "sparkov_transaction_features.parquet")
    txn_view.to_parquet(txn_parquet, index=False)
    logging.info(f"Wrote {txn_path} ({len(txn_view):,} rows) + {txn_parquet}")

    # ── View 2: per-cc_num (one row per card, max-timestamped) ───────────────
    cc_cols = [
        "cc_num", "trans_date_trans_time",
        "cc_num_FE", "txn_last_1h", "txn_last_24h", "txn_last_168h",
        "amt_sum_last_1h", "amt_sum_last_24h", "amt_sum_last_168h",
        "amt_per_cc_num_mean", "amt_per_cc_num_std",
    ]
    cc_view = _to_event_timestamp(all_fe[cc_cols])
    cc_view = cc_view[[c for c in cc_cols if c != "trans_date_trans_time"] + ["event_timestamp", "created"]]
    cc_view["cc_num"] = cc_view["cc_num"].astype("int64")
    cc_view = cc_view.sort_values("event_timestamp").drop_duplicates("cc_num", keep="last")
    cc_path = os.path.join(out_dir, "sparkov_cc_features.csv")
    cc_view.to_csv(cc_path, index=False)
    cc_parquet = os.path.join(out_dir, "sparkov_cc_features.parquet")
    cc_view.to_parquet(cc_parquet, index=False)
    logging.info(f"Wrote {cc_path} ({len(cc_view):,} unique cards) + {cc_parquet}")

    # ── View 3: per-merchant (one row per merchant) ──────────────────────────
    mer_cols = [
        "merchant", "trans_date_trans_time",
        "merchant_FE", "merchant_te",
        "amt_per_merchant_mean", "amt_per_merchant_std",
    ]
    mer_view = _to_event_timestamp(all_fe[mer_cols])
    mer_view = mer_view[[c for c in mer_cols if c != "trans_date_trans_time"] + ["event_timestamp", "created"]]
    mer_view["merchant"] = mer_view["merchant"].astype(str)
    mer_view = mer_view.sort_values("event_timestamp").drop_duplicates("merchant", keep="last")
    mer_path = os.path.join(out_dir, "sparkov_merchant_features.csv")
    mer_view.to_csv(mer_path, index=False)
    mer_parquet = os.path.join(out_dir, "sparkov_merchant_features.parquet")
    mer_view.to_parquet(mer_parquet, index=False)
    logging.info(f"Wrote {mer_path} ({len(mer_view):,} unique merchants) + {mer_parquet}")

    return {
        "transaction_features": txn_path,
        "cc_num_features":      cc_path,
        "merchant_features":    mer_path,
    }


def main():
    paths = build_feature_tables()
    print("Feast source CSVs ready:")
    for view, path in paths.items():
        print(f"  {view:<25} {path}")


if __name__ == "__main__":
    main()
