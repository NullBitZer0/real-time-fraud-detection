import sys
import joblib

import mlflow
import mlflow.sklearn
import dagshub

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

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
        test_df,
        config
    ):

        try:

            logging.info(
                "Model training started"
            )

            X_train = train_df.drop(
                "Class",
                axis=1
            )

            y_train = train_df["Class"]

            X_test = test_df.drop(
                "Class",
                axis=1
            )

            y_test = test_df["Class"]

            # Model Selection

            if config.model.model_name == "lightgbm":

                model = LGBMClassifier(
                    n_estimators=config.model.n_estimators,
                    learning_rate=config.model.learning_rate,
                    max_depth=config.model.max_depth,
                    num_leaves=config.model.num_leaves,
                    subsample=config.model.subsample,
                    colsample_bytree=config.model.colsample_bytree,
                    class_weight=config.model.class_weight,
                    verbose=config.model.verbose,
                    random_state=config.model.random_state
                )

            elif config.model.model_name == "xgboost":

                model = XGBClassifier(
                    n_estimators=config.model.n_estimators,
                    learning_rate=config.model.learning_rate,
                    max_depth=config.model.max_depth,
                    subsample=config.model.subsample,
                    colsample_bytree=config.model.colsample_bytree,
                    eval_metric="logloss",
                    random_state=config.model.random_state
                )

            elif config.model.model_name == "catboost":

                model = CatBoostClassifier(
                    iterations=config.model.iterations,
                    learning_rate=config.model.learning_rate,
                    depth=config.model.depth,
                    random_seed=config.model.random_seed,
                    verbose=config.model.verbose
                )

            else:
                raise Exception(
                    f"Unsupported model: {config.model.model_name}"
                )

            logging.info(
                f"Selected model: {config.model.model_name}"
            )

            # MLflow Autologging
            mlflow.autolog()

            with mlflow.start_run():

                mlflow.set_tag(
                    "model_type",
                    config.model.model_name
                )

                # Log threshold used for inference
                threshold = getattr(
                    getattr(config, "data", None),
                    "inference_threshold",
                    0.5
                )
                mlflow.log_param(
                    "inference_threshold",
                    threshold
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

                # Custom fraud metrics
                mlflow.log_metric(
                    "custom_roc_auc",
                    roc_auc
                )

                mlflow.log_metric(
                    "custom_pr_auc",
                    pr_auc
                )

                logging.info(
                    f"ROC AUC: {roc_auc}"
                )

                logging.info(
                    f"PR AUC: {pr_auc}"
                )

            # Save model locally
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