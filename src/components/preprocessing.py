import os
import sys
import joblib

import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler

from src.utils.logger import logging
from src.utils.exception import CustomException


# Columns to scale — engineered features + Amount_log
SCALE_COLS = [
    "Amount_log",
    "Hour_sin",
    "Hour_cos",
    "V_mean",
    "V_std",
    "V_max",
    "V_min",
]


class DataPreprocessing:

    def __init__(self, scaler_type: str = "standard", target_col: str = "Class"):
        """
        Args:
            scaler_type: "standard" or "robust"
            target_col : target column name — excluded from imputer/scaler fit
        """
        if scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()

        self.scaler_type = scaler_type
        self.target_col  = target_col

    def initiate_preprocessing(self, train_path, is_train: bool = True):
        """
        Args:
            train_path : path to CSV to preprocess.
            is_train   : if True, fit scaler/imputer on this data and save.
                         if False, only transform (load saved artifacts).
        """
        try:

            df = pd.read_csv(train_path)

            logging.info(
                f"Loaded {train_path} — shape: {df.shape}"
            )

            # ── Separate target so imputer is fit on features only ────────────
            # This prevents the "feature names mismatch" error at inference time
            target_series = None
            if self.target_col in df.columns:
                target_series = df[self.target_col].copy()
                df = df.drop(columns=[self.target_col])

            # ── Drop high-missing columns ─────────────────────────────────────
            missing_pct = df.isnull().mean() * 100
            drop_cols = missing_pct[missing_pct > 90].index
            df.drop(columns=drop_cols, inplace=True)

            # ── Encode categorical columns ────────────────────────────────────
            categorical_cols = df.select_dtypes(include=["object"]).columns
            label_encoders = {}

            for col in categorical_cols:
                df[col] = df[col].astype(str)
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le

            # ── Impute missing values (features only — no target) ─────────────
            imputer = SimpleImputer(strategy="median")

            if is_train:
                df_imputed = pd.DataFrame(
                    imputer.fit_transform(df),
                    columns=df.columns
                )
            else:
                imputer = joblib.load("models/imputer.pkl")
                df_imputed = pd.DataFrame(
                    imputer.transform(df),
                    columns=df.columns
                )

            # ── Add target back (needed for train/test CSVs) ──────────────────
            if target_series is not None:
                df_imputed[self.target_col] = target_series.values

            # ── Scale engineered features ─────────────────────────────────────
            # Only scale columns that exist in this dataframe
            cols_to_scale = [
                c for c in SCALE_COLS
                if c in df_imputed.columns
            ]

            if cols_to_scale:

                if is_train:
                    df_imputed[cols_to_scale] = (
                        self.scaler.fit_transform(
                            df_imputed[cols_to_scale]
                        )
                    )
                else:
                    scaler = joblib.load("models/scaler.pkl")
                    df_imputed[cols_to_scale] = (
                        scaler.transform(
                            df_imputed[cols_to_scale]
                        )
                    )

            # ── Save artifacts (train only) ───────────────────────────────────
            if is_train:
                os.makedirs("models", exist_ok=True)

                joblib.dump(imputer,        "models/imputer.pkl")
                joblib.dump(self.scaler,    "models/scaler.pkl")
                joblib.dump(label_encoders, "models/label_encoders.pkl")

                logging.info(
                    f"Saved scaler ({self.scaler_type}), "
                    "imputer, label_encoders to models/"
                )

            logging.info("Preprocessing completed")

            return df_imputed

        except Exception as e:
            raise CustomException(e, sys)