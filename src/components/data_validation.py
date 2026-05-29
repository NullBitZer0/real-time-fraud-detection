import sys
import pandas as pd

from src.utils.logger import logging
from src.utils.exception import CustomException


class DataValidation:

    def validate_columns(self, df):

        required_columns = [
            "Time",
            "Amount",
            "Class"
        ]

        missing_columns = []

        for col in required_columns:

            if col not in df.columns:
                missing_columns.append(col)

        if len(missing_columns) > 0:

            raise Exception(
                f"Missing columns: {missing_columns}"
            )

        logging.info("Data validation successful")

        return True

    def initiate_validation(self, train_path):

        try:

            df = pd.read_csv(train_path)

            self.validate_columns(df)

            return True

        except Exception as e:
            raise CustomException(e, sys)