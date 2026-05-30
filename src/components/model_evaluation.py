import sys
import joblib

import pandas as pd
import mlflow

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils.logger import logging
from src.utils.exception import CustomException


class ModelEvaluation:

    def evaluate_model(
        self,
        model,
        X_test,
        y_test,
        threshold: float = 0.35
    ):
        """
        Evaluates the model using PR-AUC as primary metric.
        Uses `threshold` instead of 0.5 for binary predictions.

        Args:
            model     : fitted classifier
            X_test    : test features
            y_test    : true labels
            threshold : decision threshold from params.yaml → data.inference_threshold
        """
        try:

            # ── Probabilities ─────────────────────────────────────────────────
            probs = model.predict_proba(X_test)[:, 1]

            # ── Apply optimised threshold ─────────────────────────────────────
            preds = (probs >= threshold).astype(int)

            # ── Core metrics ──────────────────────────────────────────────────
            pr_auc  = average_precision_score(y_test, probs)
            roc_auc = roc_auc_score(y_test, probs)
            f1      = f1_score(y_test, preds, zero_division=0)
            prec    = precision_score(y_test, preds, zero_division=0)
            rec     = recall_score(y_test, preds, zero_division=0)

            cm     = confusion_matrix(y_test, preds)
            report = classification_report(y_test, preds, zero_division=0)

            # ── Log to MLflow ─────────────────────────────────────────────────
            mlflow.log_metric("eval_pr_auc",   pr_auc)
            mlflow.log_metric("eval_roc_auc",  roc_auc)
            mlflow.log_metric("eval_f1",       f1)
            mlflow.log_metric("eval_precision", prec)
            mlflow.log_metric("eval_recall",   rec)
            mlflow.log_param("eval_threshold", threshold)

            # ── Log to console ────────────────────────────────────────────────
            logging.info(
                f"[Evaluation @ threshold={threshold}]"
            )
            logging.info(f"  PR-AUC    : {pr_auc:.4f}  ← primary metric")
            logging.info(f"  ROC-AUC   : {roc_auc:.4f}")
            logging.info(f"  F1        : {f1:.4f}")
            logging.info(f"  Precision : {prec:.4f}")
            logging.info(f"  Recall    : {rec:.4f}")
            logging.info(f"Confusion Matrix:\n{cm}")
            logging.info(f"Classification Report:\n{report}")

            return {
                "pr_auc":           pr_auc,
                "roc_auc":          roc_auc,
                "f1":               f1,
                "precision":        prec,
                "recall":           rec,
                "confusion_matrix": cm,
                "classification_report": report,
                "threshold":        threshold,
            }

        except Exception as e:
            raise CustomException(e, sys)