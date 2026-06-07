"""Model training for the Sparkov pipeline.

Trains a CatBoost classifier with the tuned hyperparameters from
`models/optuna_best.json` (or whatever params are passed in).
The model is fit on the full training set, evaluated on val + test,
and the trained model + metadata (feature list, tier thresholds,
encoder state) are saved to models/ for the prediction pipeline.

Metrics and the model artifact are also logged to MLflow (DAGsHub-backed).
"""
import json
import os
import sys
import time

import joblib
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.exception import CustomException
from src.utils.logger import logging

# 3-tier thresholds from the OOF run in notebooks/experiments.ipynb
TIER_THRESHOLDS = {
    "tier1_auto_block":   0.5613,  # precision >= 0.95
    "tier2_review_queue": 0.1198,  # F2-optimal
    "tier3_soft_signal":  0.0040,  # recall >= 0.95
}


def build_catboost_params(cfg_model: dict) -> dict:
    """Build a CatBoostClassifier kwargs dict from the Hydra config.

    Only includes params that are valid for `CatBoostClassifier.__init__`.
    `early_stopping` is excluded — it belongs to `fit()`, not the constructor.
    """
    keys = [
        "iterations", "depth", "learning_rate", "l2_leaf_reg",
        "random_strength", "bagging_temperature", "border_count",
        "eval_metric", "random_seed", "task_type",
    ]
    params = {k: cfg_model[k] for k in keys if k in cfg_model}
    params.setdefault("verbose", 0)
    return params


class ModelTrainer:

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "catboost.cbm")
        self.meta_path  = os.path.join(model_dir, "metadata.json")
        self.feat_path  = os.path.join(model_dir, "feature_engineering.pkl")
        os.makedirs(model_dir, exist_ok=True)

    def initiate_model_training(self,
                                  X_train, y_train, X_val, y_val, X_test, y_test,
                                  feature_engineer=None,
                                  catboost_params: dict = None,
                                  mlflow_tracker = None):
        """
        Args:
            X_train, y_train : training features and labels
            X_val, y_val     : validation features and labels
            X_test, y_test   : test features and labels
            feature_engineer : fitted FeatureEngineering instance
            catboost_params  : kwargs for CatBoostClassifier
                               (default: tuned from configs/model/catboost.yaml)
            mlflow_tracker   : optional MLflowTracker (logs params/metrics/artifacts)

        Returns:
            dict with metrics + paths to saved artifacts
        """
        if catboost_params is None:
            catboost_params = build_catboost_params({
                "iterations": 307, "depth": 10,
                "learning_rate": 0.14757878491291962,
                "l2_leaf_reg": 19.613363614465985,
                "random_strength": 1.2157708813939343,
                "bagging_temperature": 1.1713021441934048,
                "border_count": 192,
                "eval_metric": "PRAUC",
                "early_stopping": 50,
                "random_seed": 42,
                "task_type": "CPU",
            })

        logging.info(f"CatBoost training started — {catboost_params}")
        try:
            t0 = time.time()
            model = CatBoostClassifier(**catboost_params)
            # early_stopping is a fit-time kwarg in CatBoost, not a constructor arg
            es = catboost_params.get("early_stopping", 50)
            model.fit(X_train, y_train, eval_set=(X_val, y_val),
                       early_stopping_rounds=es)
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

            # MLflow logging (if a tracker is provided)
            if mlflow_tracker is not None:
                self._log_to_mlflow(mlflow_tracker, catboost_params, metadata, model,
                                     tier_summary, val_pr, val_roc, test_pr, test_roc, t_fit)

            return metadata
        except Exception as e:
            raise CustomException(e, sys)

    def _log_to_mlflow(self, tracker, catboost_params, metadata, model, tier_summary,
                        val_pr, val_roc, test_pr, test_roc, t_fit):
        """Push params + metrics + artifacts to the active MLflow run."""
        import mlflow
        try:
            # Params
            mlflow.log_params({k: v for k, v in catboost_params.items()
                                if isinstance(v, (str, int, float, bool))})
            # Metrics
            mlflow.log_metrics({
                "val_pr_auc":   val_pr,
                "val_roc_auc":  val_roc,
                "test_pr_auc":  test_pr,
                "test_roc_auc": test_roc,
                "fit_time_seconds": t_fit,
            })
            for tier_name, info in tier_summary.items():
                if info is None:
                    continue
                for k, v in info.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"{tier_name}_{k}", v)
            # Artifacts — log native CatBoost model (so it can be registered)
            _catboost_logged = False
            try:
                import mlflow.catboost
                # MLflow 3.x: use `name` (artifact_path is deprecated)
                try:
                    mlflow.catboost.log_model(
                        model,
                        name="catboost",
                        registered_model_name=None,
                    )
                    _catboost_logged = True
                except TypeError:
                    # MLflow 2.x fallback
                    mlflow.catboost.log_model(
                        model,
                        artifact_path="catboost",
                    )
                    _catboost_logged = True
            except Exception as e:
                logging.warning(f"mlflow.catboost.log_model failed: {e}")
            # Always also log the raw .cbm file as a backup artifact so the
            # model is downloadable directly from the MLflow run UI
            mlflow.log_artifact(self.model_path)
            # Supporting artifacts
            mlflow.log_artifact(self.meta_path)
            mlflow.log_artifact(self.feat_path)
            # Tags
            mlflow.set_tag("model",        "catboost")
            mlflow.set_tag("dataset",      "sparkov")
            mlflow.set_tag("dvc_stage",    "train")
            mlflow.set_tag("tuned",        "true")
            logging.info("MLflow: params + metrics + artifacts logged")
        except Exception as e:
            logging.warning(f"MLflow logging failed: {e}")

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
