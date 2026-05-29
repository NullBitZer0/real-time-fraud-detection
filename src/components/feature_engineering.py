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

                # Convert seconds to hour of day (cyclic)
                df["Hour"] = (df["Time"] // 3600) % 24

                df["Hour_sin"] = np.sin(
                    2 * np.pi * df["Hour"] / 24
                )

                df["Hour_cos"] = np.cos(
                    2 * np.pi * df["Hour"] / 24
                )

            # Summary stats across PCA V-features
            v_cols = [
                col for col in df.columns
                if col.startswith("V")
            ]

            if len(v_cols) > 0:

                df["V_mean"] = df[v_cols].mean(axis=1)

                df["V_std"] = df[v_cols].std(axis=1)

                df["V_max"] = df[v_cols].max(axis=1)

                df["V_min"] = df[v_cols].min(axis=1)

            logging.info(
                "Feature engineering completed"
            )

            return df

        except Exception as e:
            raise CustomException(e, sys)