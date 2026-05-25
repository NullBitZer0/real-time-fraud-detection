import sys
import joblib

import mlflow
import mlflow.sklearn
import dagshub

from lightgbm import LGBMClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

from src.utils.logger import logging
from src.utils.exception import CustomException

from src.mlflow_tracking import *


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

            # Split features and target
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

            # Model
            model = LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=10,
                num_leaves=64,
                random_state=42
            )

            # MLflow tracking
            with mlflow.start_run():

                logging.info(
                    "MLflow run started"
                )

                # Log parameters
                mlflow.log_param(
                    "model",
                    "LightGBM"
                )

                mlflow.log_param(
                    "n_estimators",
                    300
                )

                mlflow.log_param(
                    "learning_rate",
                    0.05
                )

                mlflow.log_param(
                    "max_depth",
                    10
                )

                mlflow.log_param(
                    "num_leaves",
                    64
                )

                # Train model
                model.fit(
                    X_train,
                    y_train
                )

                logging.info(
                    "Model training completed"
                )

                # Predictions
                probs = model.predict_proba(
                    X_test
                )[:, 1]

                # Metrics
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

                # Log metrics
                mlflow.log_metric(
                    "roc_auc",
                    roc_auc
                )

                mlflow.log_metric(
                    "pr_auc",
                    pr_auc
                )

                # Log model
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="lightgbm_model"
                )

                logging.info(
                    "Model logged to MLflow"
                )

            # Save local model
            joblib.dump(
                model,
                "models/model.pkl"
            )

            logging.info(
                "Model saved locally"
            )

            return {
                "roc_auc": roc_auc,
                "pr_auc": pr_auc
            }

        except Exception as e:
            raise CustomException(e, sys)