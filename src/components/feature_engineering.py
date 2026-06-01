import sys
import pandas as pd
import numpy as np
from src.utils.logger import logging
from src.utils.exception import CustomException


class FeatureEngineering:

    def initiate_feature_engineering(self, df):

        try:

            logging.info(
                "Feature engineering started"
            )

            # Log transform on Amount (skewed distribution)
            if "Amount" in df.columns:

                df["Amount_log"] = np.log1p(
                    df["Amount"]
                )

            # Time-based features: hour of day and day cycle
            if "Time" in df.columns:

                df["Hour_sin"] = np.sin(
                    2 * np.pi * (df["Time"] // 3600) / 24
                )

                df["Hour_cos"] = np.cos(
                    2 * np.pi * (df["Time"] // 3600) / 24
                )

            logging.info(
                "Feature engineering completed"
            )

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def remove_outliers(self, df, target_col="Class"):

        try:

            logging.info("Outlier removal started")

            v_cols = [
                col for col in df.columns
                if col.startswith("V")
            ]

            if len(v_cols) == 0 or target_col not in df.columns:
                logging.info("No V-features or target column — skipping outlier removal")
                return df

            y = df[target_col].values
            fraud_mask = y == 1

            if fraud_mask.sum() == 0:
                logging.info("No fraud samples — skipping outlier removal")
                return df

            fraud_data = df.loc[fraud_mask, v_cols]
            mu = fraud_data.mean()
            sg = fraud_data.std()

            extreme = (
                df[v_cols].sub(mu).abs() > 3 * sg
            ).astype(int).sum(axis=1)

            rows_to_drop = extreme[extreme >= 5].index
            df_clean = df.drop(rows_to_drop)

            logging.info(
                f"Removed {len(rows_to_drop)} extreme transactions "
                f"({fraud_mask.sum()} fraud before, "
                f"{df_clean[target_col].sum()} after)"
            )

            return df_clean

        except Exception as e:
            raise CustomException(e, sys)