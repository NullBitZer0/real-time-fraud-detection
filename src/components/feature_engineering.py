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

            # Transaction amount log transform
            if "TransactionAmt" in df.columns:

                df["TransactionAmt_log"] = np.log1p(
                    df["TransactionAmt"]
                )

            # Email matching feature
            if (
                "P_emaildomain" in df.columns
                and
                "R_emaildomain" in df.columns
            ):

                df["email_match"] = (
                    df["P_emaildomain"]
                    ==
                    df["R_emaildomain"]
                ).astype(int)

            # Card features
            card_cols = [
                col for col in df.columns
                if "card" in col.lower()
            ]

            if len(card_cols) > 0:

                df["card_feature_count"] = (
                    df[card_cols]
                    .notnull()
                    .sum(axis=1)
                )

            logging.info(
                "Feature engineering completed"
            )

            return df

        except Exception as e:
            raise CustomException(e, sys)