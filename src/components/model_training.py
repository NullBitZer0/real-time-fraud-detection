import sys
import joblib

import mlflow
import mlflow.sklearn
import dagshub

from omegaconf import DictConfig

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
        cfg: DictConfig       # Hydra DictConfig — use dot-access
    ):

        try:

            logging.info("Model training started")

            target_col = cfg.data.target_column
            threshold  = cfg.data.inference_threshold

            X_train = train_df.drop(target_col, axis=1)
            y_train = train_df[target_col]

            X_test  = test_df.drop(target_col, axis=1)
            y_test  = test_df[target_col]

            # ── Model selection ───────────────────────────────────────────────
            if cfg.model.model_name == "lightgbm":

                model = LGBMClassifier(
                    n_estimators=cfg.model.n_estimators,
                    learning_rate=cfg.model.learning_rate,
                    max_depth=cfg.model.max_depth,
                    num_leaves=cfg.model.num_leaves,
                    subsample=cfg.model.subsample,
                    colsample_bytree=cfg.model.colsample_bytree,
                    min_child_samples=cfg.model.min_child_samples,
                    reg_alpha=cfg.model.reg_alpha,
                    reg_lambda=cfg.model.reg_lambda,
                    class_weight=cfg.model.class_weight,
                    verbose=cfg.model.verbose,
                    random_state=cfg.model.random_state
                )

            elif cfg.model.model_name == "xgboost":

                model = XGBClassifier(
                    n_estimators=cfg.model.n_estimators,
                    learning_rate=cfg.model.learning_rate,
                    max_depth=cfg.model.max_depth,
                    subsample=cfg.model.subsample,
                    colsample_bytree=cfg.model.colsample_bytree,
                    eval_metric="logloss",
                    random_state=cfg.model.random_state
                )

            elif cfg.model.model_name == "catboost":

                model = CatBoostClassifier(
                    iterations=cfg.model.iterations,
                    learning_rate=cfg.model.learning_rate,
                    depth=cfg.model.depth,
                    random_seed=cfg.model.random_seed,
                    verbose=cfg.model.verbose
                )

            else:
                raise ValueError(
                    f"Unsupported model: {cfg.model.model_name}"
                )

            logging.info(f"Selected model: {cfg.model.model_name}")

            # ── MLflow Autologging ────────────────────────────────────────────
            mlflow.autolog()

            with mlflow.start_run(run_name=f"train_{cfg.model.model_name}"):

                mlflow.set_tag("model_type", cfg.model.model_name)
                mlflow.log_param("inference_threshold", threshold)

                model.fit(X_train, y_train)

                probs   = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, probs)
                pr_auc  = average_precision_score(y_test, probs)

                mlflow.log_metric("custom_roc_auc", roc_auc)
                mlflow.log_metric("custom_pr_auc",  pr_auc)

                logging.info(f"ROC AUC : {roc_auc:.4f}")
                logging.info(f"PR AUC  : {pr_auc:.4f}")

            # ── Save model ────────────────────────────────────────────────────
            joblib.dump(model, "models/model.pkl")

            logging.info("Model saved → models/model.pkl")

            return {
                "roc_auc": roc_auc,
                "pr_auc":  pr_auc
            }

        except Exception as e:
            raise CustomException(e, sys)