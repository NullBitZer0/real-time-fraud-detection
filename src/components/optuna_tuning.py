import sys
import optuna
import mlflow
import pandas as pd

from lightgbm import LGBMClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

from src.utils.logger import logging
from src.utils.exception import CustomException


class OptunaTuner:

    def objective(
        self,
        trial,
        X_train,
        y_train,
        X_test,
        y_test
    ):

        try:

            params = {

                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    100,
                    1000
                ),

                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.3
                ),

                "max_depth": trial.suggest_int(
                    "max_depth",
                    3,
                    15
                ),

                "num_leaves": trial.suggest_int(
                    "num_leaves",
                    20,
                    200
                ),

                "subsample": trial.suggest_float(
                    "subsample",
                    0.5,
                    1.0
                ),

                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree",
                    0.5,
                    1.0
                ),

                "random_state": 42
            }

            with mlflow.start_run(
                nested=True
            ):

                mlflow.log_params(params)

                model = LGBMClassifier(
                    **params
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

                mlflow.log_metric(
                    "roc_auc",
                    roc_auc
                )

                mlflow.log_metric(
                    "pr_auc",
                    pr_auc
                )

                return pr_auc

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_optuna(
        self,
        train_df,
        test_df,
        n_trials=10
    ):

        try:

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

            study = optuna.create_study(
                direction="maximize"
            )

            study.optimize(
                lambda trial: self.objective(
                    trial,
                    X_train,
                    y_train,
                    X_test,
                    y_test
                ),
                n_trials=n_trials
            )

            logging.info(
                f"Best Trial: {study.best_trial.params}"
            )

            logging.info(
                f"Best Score: {study.best_value}"
            )

            return study

        except Exception as e:
            raise CustomException(e, sys)