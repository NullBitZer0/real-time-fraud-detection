import os
import sys

import pandas as pd

from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact


class DataIngestion:

    def __init__(self):

        self.config = DataIngestionConfig(
            raw_data_path="data/processed/merged_data.parquet",
            train_path="artifacts/train.csv",
            test_path="artifacts/test.csv"
        )

    def initiate_data_ingestion(self):

        logging.info("Starting data ingestion")

        try:

            df = pd.read_parquet(
                self.config.raw_data_path
            )

            logging.info(
                f"Dataset loaded with shape: {df.shape}"
            )

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df["isFraud"]
            )

            train_set.to_csv(
                self.config.train_path,
                index=False
            )

            test_set.to_csv(
                self.config.test_path,
                index=False
            )

            logging.info("Train and test saved")

            return DataIngestionArtifact(
                train_path=self.config.train_path,
                test_path=self.config.test_path
            )

        except Exception as e:
            raise CustomException(e, sys)