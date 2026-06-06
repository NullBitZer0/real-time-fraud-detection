"""Data ingestion for the Sparkov fraud-detection dataset.

Loads fraudTrain.csv and fraudTest.csv (separate time-based splits),
performs the 80/20 time split of fraudTrain (last 20% as val), and
saves the three splits to artifacts/.

Differences from the legacy IBM pipeline:
- Two input files instead of one (train_full + test_full)
- Time-based split instead of stratified random split
- Sparkov schema (cc_num, merchant, category, etc.) instead of (V1-V28, Time, Amount)
"""
import os
import sys
import pandas as pd

from src.utils.logger import logging
from src.utils.exception import CustomException


class DataIngestion:

    def __init__(self):
        self.train_path = "data/raw/fraudTrain.csv"
        self.test_path  = "data/raw/fraudTest.csv"
        self.out_train  = "artifacts/sparkov_train.csv"
        self.out_val    = "artifacts/sparkov_val.csv"
        self.out_test   = "artifacts/sparkov_test.csv"
        self.val_quantile = 0.80  # last 20% of fraudTrain as val

    def initiate_data_ingestion(self):
        logging.info("Sparkov data ingestion started")
        try:
            train_full = pd.read_csv(self.train_path).drop(columns=["Unnamed: 0"])
            test_full  = pd.read_csv(self.test_path ).drop(columns=["Unnamed: 0"])

            train_full["trans_date_trans_time"] = pd.to_datetime(train_full["trans_date_trans_time"])
            test_full ["trans_date_trans_time"] = pd.to_datetime(test_full ["trans_date_trans_time"])

            logging.info(f"fraudTrain: {train_full.shape[0]:>9,} rows | fraud: {train_full.is_fraud.sum():>5,} ({train_full.is_fraud.mean():.4%})")
            logging.info(f"fraudTest : {test_full.shape[0]:>9,} rows | fraud: {test_full.is_fraud.sum():>5,} ({test_full.is_fraud.mean():.4%})")

            # Time-based split: last 20% of fraudTrain as val
            train_full = train_full.sort_values("trans_date_trans_time").reset_index(drop=True)
            cutoff = train_full["trans_date_trans_time"].quantile(self.val_quantile)
            val_df   = train_full[train_full["trans_date_trans_time"] >= cutoff].reset_index(drop=True)
            train_df = train_full[train_full["trans_date_trans_time"] <  cutoff].reset_index(drop=True)
            test_df  = test_full.reset_index(drop=True)

            logging.info(f"train: {len(train_df):>9,} | fraud: {train_df.is_fraud.sum():>4,} ({train_df.is_fraud.mean():.4%})")
            logging.info(f"val  : {len(val_df):>9,} | fraud: {val_df.is_fraud.sum():>4,} ({val_df.is_fraud.mean():.4%})")
            logging.info(f"test : {len(test_df):>9,} | fraud: {test_df.is_fraud.sum():>4,} ({test_df.is_fraud.mean():.4%})")

            os.makedirs("artifacts", exist_ok=True)
            train_df.to_csv(self.out_train, index=False)
            val_df  .to_csv(self.out_val,   index=False)
            test_df .to_csv(self.out_test,  index=False)

            return self.out_train, self.out_val, self.out_test
        except Exception as e:
            raise CustomException(e, sys)


def read_sparkov_split(path: str) -> pd.DataFrame:
    """Read a Sparkov split CSV with the datetime column parsed."""
    df = pd.read_csv(path)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    return df
