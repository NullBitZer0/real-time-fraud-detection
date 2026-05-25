import os
import sys
import joblib

import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from src.utils.logger import logging
from src.utils.exception import CustomException


class DataPreprocessing:

    def initiate_preprocessing(self, train_path):

        try:

            df = pd.read_csv(train_path)

            missing_percentage = (
                df.isnull().mean() * 100
            )

            drop_cols = missing_percentage[
                missing_percentage > 90
            ].index

            df.drop(
                columns=drop_cols,
                inplace=True
            )

            categorical_cols = df.select_dtypes(
                include=["object"]
            ).columns

            label_encoders = {}

            for col in categorical_cols:

                df[col] = df[col].astype(str)

                le = LabelEncoder()

                df[col] = le.fit_transform(df[col])

                label_encoders[col] = le

            imputer = SimpleImputer(
                strategy="median"
            )

            df_imputed = pd.DataFrame(
                imputer.fit_transform(df),
                columns=df.columns
            )

            os.makedirs("models", exist_ok=True)

            joblib.dump(
                imputer,
                "models/imputer.pkl"
            )

            joblib.dump(
                label_encoders,
                "models/label_encoders.pkl"
            )

            logging.info(
                "Preprocessing completed"
            )

            return df_imputed

        except Exception as e:
            raise CustomException(e, sys)