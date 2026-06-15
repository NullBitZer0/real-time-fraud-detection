"""Inference pipeline for the Sparkov fraud detection project.

Two source paths for features:
  1. Per-transaction features (hour, amt, distance, etc.) — computed from
     the row at inference time using the fitted FeatureEngineering
  2. Per-entity features (cc_num_FE, txn_last_*, merchant_FE, ...) —
     fetched from Feast's online store (Redis) at inference time

The 6 velocity features and 4 merchant features come from Redis.
Missing values (entity not in Redis yet) → 0 fallback.

Loads the trained CatBoost model from models/catboost.cbm and applies
the 3-tier logic (auto_block / review_queue / soft_signal / approve).
Predictions are stored permanently in the Postgres audit log via
api/app.py:insert_decision().
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier

from src.components.feature_store import FeatureStoreClient
from src.utils.exception import CustomException
from src.utils.logger import logging

# Features that come from Feast's online store (Redis) at inference time
ONLINE_FEATURES = [
    "cc_num_FE", "txn_last_1h", "txn_last_24h", "txn_last_168h",
    "amt_sum_last_1h", "amt_sum_last_24h", "amt_sum_last_168h",
    "amt_per_cc_num_mean", "amt_per_cc_num_std",
    "merchant_FE", "merchant_te",
    "amt_per_merchant_mean", "amt_per_merchant_std",
]


class PredictionPipeline:
    """Loads model + Feast client; predicts fraud probability + 3-tier action."""

    def __init__(self, model_dir: str = "models", use_feast: bool = True):

        model_file = Path(model_dir) / "catboost.cbm"
        if not model_file.exists():
            self._download_model(model_dir)

        # Load trained CatBoost model
        self.model = CatBoostClassifier()
        self.model.load_model(f"{model_dir}/catboost.cbm")

        with open(f"{model_dir}/metadata.json") as f:
            self.metadata = json.load(f)
        self.tier_thresholds = self.metadata["tier_thresholds"]
        self.features        = self.metadata["feature_list"]
        self.fe              = joblib.load(f"{model_dir}/feature_engineering.pkl")

        # Connect to Feast (online store = Redis)
        self.use_feast = use_feast
        self.fsc = FeatureStoreClient.get() if use_feast else None

        logging.info(
            f"PredictionPipeline ready — model={self.metadata['model_name']} | "
            f"features={len(self.features)} | "
            f"feast={'on' if use_feast else 'off'} | "
            f"tiers={self.tier_thresholds}"
        )

    @staticmethod
    def _download_model(model_dir: str) -> None:
        """Download model artifacts from MLflow/DagsHub if not present locally."""
        try:
            from scripts.download_model import download_production_model
            download_production_model(model_dir)
        except Exception as e:
            logging.error(f"Failed to download model from MLflow: {e}")
            raise RuntimeError(
                f"Model not found at {model_dir}/catboost.cbm and download failed: {e}"
            )

    def _enrich_with_online_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fetch per-cc_num + per-merchant features from Redis (Feast)."""
        if not self.use_feast or self.fsc is None:
            for c in ONLINE_FEATURES:
                df[c] = 0.0
            return df
        try:
            online = self.fsc.get_online_features_for_batch(df)
            online = online.set_index("trans_num")
            for c in ONLINE_FEATURES:
                if c in online.columns:
                    df[c] = df["trans_num"].map(online[c]).fillna(0.0).astype("float32")
                else:
                    df[c] = 0.0
            return df
        except Exception as e:
            logging.warning(f"Feast online lookup failed: {e} — using zero fallback")
            for c in ONLINE_FEATURES:
                df[c] = 0.0
            return df

    def get_cached_decision(self, trans_num: str) -> dict:
        """Legacy method — decision cache was removed.
        Returns None; predictions are stored permanently in Postgres audit log."""
        return None

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict fraud probability + 3-tier action for each row.

        Returns:
            Original df with extra columns: proba, tier, action, threshold.
        """
        try:
            df = df.copy()
            if "trans_num" not in df.columns:
                df["trans_num"] = [f"txn_{i}" for i in range(len(df))]
            ids = df["trans_num"].values

            if not pd.api.types.is_datetime64_any_dtype(df["trans_date_trans_time"]):
                df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

            # 1. Per-transaction features
            fe_df = self.fe.transform(df, compute_velocity=False)

            # 2. Per-entity features from Feast (Redis) — 13 columns
            fe_df = self._enrich_with_online_features(fe_df)

            # 3. Build the model input
            X = fe_df[self.features].values.astype("float32")

            # 4. Predict
            probs = self.model.predict_proba(X)[:, 1]

            # 5. Map to 3-tier
            tiers, actions, thresholds = [], [], []
            for trans_num, p in zip(ids, probs):
                t = self.classify_tier(float(p))
                tiers.append(t["tier"])
                actions.append(t["action"])
                thresholds.append(t["threshold"])
                # Decisions stored permanently in Postgres audit log
                # (insert_decision in api/app.py — no Redis cache)

            out = df.copy()
            out["transaction_id"] = ids
            out["proba"]      = probs.round(4)
            out["tier"]       = tiers
            out["action"]     = actions
            out["threshold"]  = thresholds
            return out
        except Exception as e:
            raise CustomException(e, sys)

    def classify_tier(self, proba: float) -> dict:
        """Map a single probability to a 3-tier action."""
        if proba >= self.tier_thresholds["tier1_auto_block"]:
            return {"tier": 1, "action": "auto_block",   "threshold": self.tier_thresholds["tier1_auto_block"]}
        if proba >= self.tier_thresholds["tier2_review_queue"]:
            return {"tier": 2, "action": "review_queue", "threshold": self.tier_thresholds["tier2_review_queue"]}
        if proba >= self.tier_thresholds["tier3_soft_signal"]:
            return {"tier": 3, "action": "soft_signal",  "threshold": self.tier_thresholds["tier3_soft_signal"]}
        return     {"tier": 0, "action": "approve",      "threshold": 0.0}


if __name__ == "__main__":
    from src.components.data_ingestion import read_sparkov_split
    df = read_sparkov_split("data/raw/fraudTest.csv").head(5)
    pipe = PredictionPipeline()
    out = pipe.predict(df)
    print(out[["trans_num", "proba", "tier", "action"]].to_string())
    # Legacy cache check (always None now — predictions are in Postgres audit log)
    for tn in out["trans_num"].head(3):
        cached = pipe.get_cached_decision(tn)
        print(f"  cached {tn[:8]}… → {cached}")
