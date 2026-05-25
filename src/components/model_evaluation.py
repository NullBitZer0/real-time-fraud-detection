import sys
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from src.utils.logger import logging
from src.utils.exception import CustomException


class ModelEvaluation:

    def evaluate_model(
        self,
        model,
        X_test,
        y_test
    ):

        try:

            predictions = model.predict(X_test)

            cm = confusion_matrix(
                y_test,
                predictions
            )

            report = classification_report(
                y_test,
                predictions
            )

            logging.info(
                f"Confusion Matrix:\n{cm}"
            )

            logging.info(
                f"Classification Report:\n{report}"
            )

            return {
                "confusion_matrix": cm,
                "classification_report": report
            }

        except Exception as e:
            raise CustomException(e, sys)