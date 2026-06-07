"""Feast client wrapper for the Sparkov fraud-detection pipeline.

Two read paths:
  - get_online_features() : Redis → for inference (low-latency lookups)
  - get_historical_features(): Postgres (file-based offline store) → for
    point-in-time correct training data

Online features are keyed by:
  - cc_num → cc_num_features (per-card aggregates: velocity, amt stats)
  - merchant → merchant_features (per-merchant aggregates: FE, TE, amt stats)

Per-transaction features (hour, amt, distance, etc.) are NOT in the
online store (trans_num is unique per transaction) — they come from the
raw row at inference time.
"""
import os
import sys

import pandas as pd

from feast import FeatureStore
from src.utils.exception import CustomException
from src.utils.logger import logging

FEAST_REPO_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "feast", "feature_repo"
))


class FeatureStoreClient:
    """Thin wrapper around the Feast FeatureStore for online reads."""

    _instance = None

    def __init__(self, repo_path: str = FEAST_REPO_PATH):
        self.repo_path = repo_path
        self.store = FeatureStore(repo_path=repo_path)
        logging.info(f"Feast FeatureStore ready: project={self.store.project}")

    @classmethod
    def get(cls) -> "FeatureStoreClient":
        """Lazy singleton — load Feast on first call, reuse after."""
        if cls._instance is None:
            cls._instance = FeatureStoreClient()
        return cls._instance

    def get_online_features_for_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fetch online features for a batch of transactions.

        Args:
            df: DataFrame with columns `cc_num` (int) and `merchant` (str)

        Returns:
            DataFrame with the original `trans_num` + cc_num + merchant +
            all online feature columns. Missing features are NaN.
        """
        if len(df) == 0:
            return df
        try:
            cc_nums    = df["cc_num"].astype("int64").tolist()
            merchants  = df["merchant"].astype(str).tolist()
            entity_rows = [
                {"cc_num": int(c), "merchant": str(m)}
                for c, m in zip(cc_nums, merchants)
            ]
            feature_refs = [
                "cc_num_features:cc_num_FE",
                "cc_num_features:txn_last_1h",
                "cc_num_features:txn_last_24h",
                "cc_num_features:txn_last_168h",
                "cc_num_features:amt_sum_last_1h",
                "cc_num_features:amt_sum_last_24h",
                "cc_num_features:amt_sum_last_168h",
                "cc_num_features:amt_per_cc_num_mean",
                "cc_num_features:amt_per_cc_num_std",
                "merchant_features:merchant_FE",
                "merchant_features:merchant_te",
                "merchant_features:amt_per_merchant_mean",
                "merchant_features:amt_per_merchant_std",
            ]
            response = self.store.get_online_features(
                features=feature_refs,
                entity_rows=entity_rows,
            )
            online = response.to_df()
            # Re-attach the original `trans_num` so we can join back to the row
            # (Feast already includes cc_num + merchant in the response as entity keys)
            online.insert(0, "trans_num", df["trans_num"].values)
            logging.info(f"Fetched online features for {len(online)} rows")
            return online
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    fsc = FeatureStoreClient.get()
    # Sanity: lookup a few online features (real Sparkov cards/merchants)
    df = pd.DataFrame({
        "trans_num": ["t1", "t2"],
        "cc_num":    [4110266553600176127, 4110266553600176127],
        "merchant":  ["fraud_Kertzmann and Sons", "Hudson, Davis and Copeland"],
    })
    out = fsc.get_online_features_for_batch(df)
    print(out.to_string())
