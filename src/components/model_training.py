"""Model training for the Sparkov pipeline.

Trains a single CatBoost classifier with the baseline hyperparameters
(iterations=500, depth=8, learning_rate=0.05). The model is fit on the
full training set and evaluated on val + test. The trained model and
metadata (feature list, tier thresholds, encoder state) are saved to
models/ for the prediction pipeline to use.
"""
import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    fbeta_score, f1_score, precision_score, recall_score,
    confusion_matrix,
)

from src.utils.logger import logging
from src.utils.exception import CustomException


# 3-tier thresholds from the OOF run in notebooks/experiments.ipynb
TIER_THRESHOLDS = {
    "tier1_auto_block":   0.5613,  # precision >= 0.95
    "tier2_review_queue": 0.1198,  # F2-optimal
    "tier3_soft_signal":  0.0040,  # recall >= 0.95
}


class ModelTrainer:

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "catboost.cbm")
        self.meta_path  = os.path.join(model_dir, "metadata.json")
        self.feat_path  = os.path.join(model_dir, "feature_engineering.pkl")
        os.makedirs(model_dir, exist_ok=True)

    def initiate_model_training(self, X_train, y_train, X_val, y_val, X_test, y_test,
                                  feature_engineer=None):
        """
        Args:
            X_train, y_train : training features and labels (numpy arrays or DataFrame)
            X_val, y_val     : validation features and labels
            X_test, y_test   : test features and labels
            feature_engineer : fitted FeatureEngineering instance (persisted for inference)

        Returns:
            dict with metrics + paths to saved artifacts
        """
        logging.info("CatBoost training started (baseline, 500 iter, depth=8)")
        try:
            t0 = time.time()
            model = CatBoostClassifier(
                iterations=500, depth=8, learning_rate=0.05,
                eval_metric="PRAUC", random_seed=42, verbose=0,
                early_stopping_rounds=50, task_type="CPU",
            )
            model.fit(X_train, y_train, eval_set=(X_val, y_val))
            t_fit = time.time() - t0

            # Evaluate on val + test
            val_proba  = model.predict_proba(X_val) [:, 1]
            test_proba = model.predict_proba(X_test)[:, 1]
            val_pr  = average_precision_score(y_val,  val_proba)
            val_roc = roc_auc_score(y_val,  val_proba)
            test_pr  = average_precision_score(y_test, test_proba)
            test_roc = roc_auc_score(y_test, test_proba)

            logging.info(f"CatBoost fit {t_fit:.1f}s | val PR-AUC={val_pr:.4f} | test PR-AUC={test_pr:.4f}")

            # 3-tier analysis on the test set
            tier_summary = self._three_tier_analysis(test_proba, y_test)

            # Save artifacts
            model.save_model(self.model_path)
            logging.info(f"Saved model: {self.model_path}")

            metadata = {
                "model_name":        "catboost",
                "model_path":        self.model_path,
                "feature_list":      feature_engineer.feature_list if feature_engineer else list(X_train.columns),
                "tier_thresholds":   TIER_THRESHOLDS,
                "val_pr_auc":        float(val_pr),
                "val_roc_auc":       float(val_roc),
                "test_pr_auc":       float(test_pr),
                "test_roc_auc":      float(test_roc),
                "fit_time_seconds":  float(t_fit),
                "tier_summary":      tier_summary,
            }
            with open(self.meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logging.info(f"Saved metadata: {self.meta_path}")

            if feature_engineer is not None:
                joblib.dump(feature_engineer, self.feat_path)
                logging.info(f"Saved feature engineering state: {self.feat_path}")

            return metadata
        except Exception as e:
            raise CustomException(e, sys)

    def _three_tier_analysis(self, proba, y_true, n_thresh=1000):
        """Compute the 3-tier operating points and return a summary dict."""
        thresholds = np.linspace(0.001, 0.99, n_thresh)
        precisions = np.array([precision_score(y_true, (proba >= t).astype(int), zero_division=0) for t in thresholds])
        recalls    = np.array([recall_score   (y_true, (proba >= t).astype(int), zero_division=0) for t in thresholds])

        # Tier 1: smallest t with precision >= 0.95
        mask_p = precisions >= 0.95
        t_tier1 = float(thresholds[np.argmax(mask_p)]) if mask_p.any() else None
        # Tier 2: F2-optimal
        f2_scores = [fbeta_score(y_true, (proba >= t).astype(int), beta=2, zero_division=0) for t in thresholds]
        t_tier2 = float(thresholds[int(np.argmax(f2_scores))])
        # Tier 3: largest t with recall >= 0.95
        mask_r = recalls >= 0.95
        t_tier3 = float(thresholds[len(thresholds) - 1 - np.argmax(mask_r[::-1])]) if mask_r.any() else None

        summary = {}
        for label, t in [("tier1", t_tier1), ("tier2", t_tier2), ("tier3", t_tier3)]:
            if t is None:
                summary[label] = None
                continue
            preds = (proba >= t).astype(int)
            tp = int(((preds == 1) & (y_true == 1)).sum())
            fp = int(((preds == 1) & (y_true == 0)).sum())
            fn = int(((preds == 0) & (y_true == 1)).sum())
            tn = int(((preds == 0) & (y_true == 0)).sum())
            summary[label] = {
                "threshold": float(t),
                "precision": float(precision_score(y_true, preds, zero_division=0)),
                "recall":    float(recall_score   (y_true, preds, zero_division=0)),
                "f1":        float(f1_score       (y_true, preds, zero_division=0)),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            }
        return summary

    def classify_tier(self, proba):
        """Map a single probability to a 3-tier action."""
        if proba >= TIER_THRESHOLDS["tier1_auto_block"]:
            return {"tier": 1, "action": "auto_block",    "threshold": TIER_THRESHOLDS["tier1_auto_block"]}
        if proba >= TIER_THRESHOLDS["tier2_review_queue"]:
            return {"tier": 2, "action": "review_queue",  "threshold": TIER_THRESHOLDS["tier2_review_queue"]}
        if proba >= TIER_THRESHOLDS["tier3_soft_signal"]:
            return {"tier": 3, "action": "soft_signal",   "threshold": TIER_THRESHOLDS["tier3_soft_signal"]}
        return     {"tier": 0, "action": "approve",       "threshold": 0.0}
