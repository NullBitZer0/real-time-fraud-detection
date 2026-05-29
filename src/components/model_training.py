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
        params: dict
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

            # ── Model selection ──────────────────────────────────────────
            mp = params["model"]                 # shorthand
            model_name = mp["model_name"]

            if model_name == "lightgbm":

                model = LGBMClassifier(
                    n_estimators=mp["n_estimators"],
                    learning_rate=mp["learning_rate"],
                    max_depth=mp["max_depth"],
                    num_leaves=mp["num_leaves"],
                    random_state=mp["random_state"]
                )

            elif model_name == "xgboost":

                model = XGBClassifier(
                    n_estimators=mp["n_estimators"],
                    learning_rate=mp["learning_rate"],
                    max_depth=mp["max_depth"],
                    subsample=mp["subsample"],
                    colsample_bytree=mp["colsample_bytree"],
                    eval_metric="logloss",
                    random_state=mp["random_state"]
                )

            elif model_name == "catboost":

                model = CatBoostClassifier(
                    iterations=mp["iterations"],
                    learning_rate=mp["learning_rate"],
                    depth=mp["depth"],
                    random_seed=mp["random_seed"],
                    verbose=mp["verbose"]
                )

            else:
                raise Exception(
                    f"Unsupported model: {model_name}"
                )

            logging.info(
                f"Selected model: {model_name}"
            )

            # ── MLflow Autologging ───────────────────────────────────────
            mlflow.autolog()

            with mlflow.start_run():

                mlflow.set_tag(
                    "model_type",
                    model_name
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

            # ── Save model locally ───────────────────────────────────────
            model_path = params["model"].get("model_path", "models/model.pkl")

            joblib.dump(
                model,
                model_path
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