"""Feature engineering for the Sparkov fraud-detection dataset.

Builds the 34 features used by the CatBoost baseline. The pipeline is the
same as notebooks/features.ipynb cell 4 — kept here as a single Python
class so it can be called from the training pipeline AND the prediction
pipeline (where it must apply encoders fit on the training set).

Three groups of engineered features:
1. Base features — derived from timestamps and raw fields (8 features)
2. Encoded features — frequency + target + amount-stats (19 features)
3. Velocity features — per-cc_num rolling counts/sums (6 features)

Note: velocity features at inference time are computed from the training
data history (Phase 5 deferred); for now the training pipeline computes
them from each row's full history.
"""
import sys
import numpy as np
import pandas as pd

from src.utils.logger import logging
from src.utils.exception import CustomException


DROP = [
    "trans_date_trans_time", "first", "last", "street", "dob",
    "trans_num", "unix_time", "lat", "long", "merch_lat", "merch_long",
    "cc_num", "merchant", "category", "city", "state", "job", "gender", "zip",
    "is_fraud",
]

SMOOTHING = 50.0


class FeatureEngineering:

    def __init__(self):
        # Encoder state — fit on train, persisted for inference
        self.freq_maps   = {}  # col -> {value: count}
        self.te_maps     = {}  # col -> {value: smoothed mean fraud}
        self.stat_maps   = {}  # col -> {value: mean/std amt}
        self.global_mean = None
        self.amt_mean    = None
        self.amt_std     = None
        self.window_hours = [1, 24, 168]

    # ── Base features (no leakage) ──────────────────────────────────────────
    @staticmethod
    def build_base_features(df):
        df = df.copy()
        df["hour"]        = df["trans_date_trans_time"].dt.hour.astype("int8")
        df["dow"]         = df["trans_date_trans_time"].dt.dayofweek.astype("int8")
        df["month"]       = df["trans_date_trans_time"].dt.month.astype("int8")
        df["is_night"]    = df["hour"].isin([0, 1, 2, 3, 4, 22, 23]).astype("int8")
        df["dob"]         = pd.to_datetime(df["dob"])
        df["age"]         = ((df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25).astype("float32")
        df["amt_log"]     = np.log1p(df["amt"]).astype("float32")
        df["amt_is_round"] = (df["amt"] == df["amt"].round(0)).astype("int8")
        R = 6371.0
        lat1, lon1 = np.radians(df["lat"]),     np.radians(df["long"])
        lat2, lon2 = np.radians(df["merch_lat"]), np.radians(df["merch_long"])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        df["distance_km"] = (2 * R * np.arcsin(np.sqrt(a))).astype("float32")
        return df

    # ── Frequency encodings (count-based, no leakage) ──────────────────────
    def _fit_frequency(self, train_df):
        # Use train only to avoid using future info
        for col in ["cc_num", "merchant", "category", "city", "state", "job", "zip"]:
            self.freq_maps[col] = train_df[col].value_counts().to_dict()

    def _apply_frequency(self, df):
        for col, mapping in self.freq_maps.items():
            df[f"{col}_FE"] = df[col].map(mapping).fillna(0).astype("float32")
        return df

    # ── Target encodings (fit on train only) ───────────────────────────────
    def _fit_target(self, train_df):
        self.global_mean = float(train_df["is_fraud"].mean())
        for col in ["merchant", "category", "city", "state", "job"]:
            stats = train_df.groupby(col)["is_fraud"].agg(["mean", "count"])
            smoothed = (stats["mean"] * stats["count"] + self.global_mean * SMOOTHING) / (stats["count"] + SMOOTHING)
            self.te_maps[col] = smoothed.to_dict()

    def _apply_target(self, df):
        for col, mapping in self.te_maps.items():
            df[f"{col}_te"] = df[col].map(mapping).fillna(self.global_mean).astype("float32")
        return df

    # ── Amount stat features (per-group, fit on train) ─────────────────────
    def _fit_amount_stats(self, train_df):
        self.amt_mean = float(train_df["amt"].mean())
        self.amt_std  = float(train_df["amt"].std())
        for col in ["merchant", "category", "cc_num"]:
            stats = train_df.groupby(col)["amt"].agg(["mean", "std"])
            self.stat_maps[col] = {"mean": stats["mean"].to_dict(), "std": stats["std"].to_dict()}

    def _apply_amount_stats(self, df):
        for col, maps in self.stat_maps.items():
            df[f"amt_per_{col}_mean"] = df[col].map(maps["mean"]).fillna(self.amt_mean).astype("float32")
            df[f"amt_per_{col}_std"]  = df[col].map(maps["std"]) .fillna(self.amt_std) .astype("float32")
        return df

    # ── Velocity features (count of PRIOR transactions per cc_num) ─────────
    @staticmethod
    def _add_velocity(df, window_hours_list):
        df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)
        df["ts"] = df["trans_date_trans_time"].astype("int64") // 10 ** 9
        for h in window_hours_list:
            window_sec = h * 3600
            df[f"txn_last_{h}h"] = (
                df.groupby("cc_num")["ts"]
                  .transform(lambda s: s.searchsorted(s.values - h * 3600, side="right") - 1)
                  .astype("float32")
            )
            amt_sums = np.zeros(len(df), dtype="float32")
            for _, g in df.groupby("cc_num", sort=False):
                ts  = g["ts"].values
                amt = g["amt"].values
                idx = g.index.values
                n   = len(g)
                cum_amt = np.concatenate([[0.0], np.cumsum(amt, dtype="float64")])
                for i in range(n):
                    j = np.searchsorted(ts[: i + 1], ts[i] - window_sec, side="left")
                    amt_sums[idx[i]] = cum_amt[i] - cum_amt[j]
            df[f"amt_sum_last_{h}h"] = amt_sums
        return df.drop(columns=["ts"])

    # ── Public API ─────────────────────────────────────────────────────────
    def fit(self, train_df):
        """Fit all encoders on the training set. Call once before transform()."""
        self._fit_frequency(train_df)
        self._fit_target(train_df)
        self._fit_amount_stats(train_df)

    def transform(self, df, compute_velocity=True):
        """Apply fitted encoders + base features to any split."""
        df = self.build_base_features(df)
        df = self._apply_frequency(df)
        df = self._apply_target(df)
        df = self._apply_amount_stats(df)
        if compute_velocity:
            df = self._add_velocity(df, self.window_hours)
        return df

    def fit_transform(self, train_df, compute_velocity=True):
        self.fit(train_df)
        return self.transform(train_df, compute_velocity=compute_velocity)

    @property
    def feature_list(self):
        """List of the 34 feature names produced by the pipeline."""
        return [
            "hour", "dow", "month", "is_night", "age", "amt_log", "amt_is_round", "distance_km",
            "cc_num_FE", "merchant_FE", "category_FE", "city_FE", "state_FE", "job_FE", "zip_FE",
            "merchant_te", "category_te", "city_te", "state_te", "job_te",
            "amt_per_merchant_mean", "amt_per_merchant_std",
            "amt_per_category_mean", "amt_per_category_std",
            "amt_per_cc_num_mean",   "amt_per_cc_num_std",
            "txn_last_1h",  "txn_last_24h",  "txn_last_168h",
            "amt_sum_last_1h", "amt_sum_last_24h", "amt_sum_last_168h",
        ]
