import sys
import joblib

from lightgbm import LGBMClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

from src.utils.logger import logging
from src.utils.exception import CustomException


class ModelTrainer:

    def initiate_model_training(
        self,
        train_df,
        test_df
    ):

        try:

            logging.info(
                "Model training started"
            )

            X_train = train_df.drop(
                "isFraud",
                axis=1
            )

            y_train = train_df["isFraud"]

            X_test = test_df.drop(
                "isFraud",
                axis=1
            )

            y_test = test_df["isFraud"]

            model = LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=10,
                num_leaves=64,
                random_state=42
            )

            model.fit(
                X_train,
                y_train
            )

            probs = model.predict_proba(
                X_test
            )[:, 1]

            roc_auc = roc_auc_score(
                y_test,
                probs
            )

            pr_auc = average_precision_score(
                y_test,
                probs
            )

            logging.info(
                f"ROC AUC: {roc_auc}"
            )

            logging.info(
                f"PR AUC: {pr_auc}"
            )

            joblib.dump(
                model,
                "models/model.pkl"
            )

            logging.info(
                "Model saved successfully"
            )

            return {
                "roc_auc": roc_auc,
                "pr_auc": pr_auc
            }

        except Exception as e:
            raise CustomException(e, sys)