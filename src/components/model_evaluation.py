"""Model evaluation for the Sparkov pipeline.

Generates the metrics + 3-tier table that goes into the demo dashboard.
Used by the training pipeline to summarize the model, and by the
prediction pipeline to compare against ground truth.
"""

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test, tier_thresholds=None, threshold=None):
    """Evaluate a fitted model on a test set. Returns a dict of metrics
    including 3-tier operating points.

    Args:
        model            : fitted classifier with predict_proba
        X_test, y_test   : test features and labels
        tier_thresholds  : dict with tier1/tier2/tier3 thresholds (defaults to F2-opt)
        threshold        : binary decision threshold for `fraud_prediction`
    """
    if threshold is None:
        threshold = tier_thresholds.get("tier2_review_queue", 0.5) if tier_thresholds else 0.5

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    metrics = {
        "pr_auc":  float(average_precision_score(y_test, probs)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "threshold": float(threshold),
        "f1":     float(f1_score(y_test, preds, zero_division=0)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall":    float(recall_score(y_test, preds, zero_division=0)),
        "f2":     float(fbeta_score(y_test, preds, beta=2, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "n_test": int(len(y_test)),
        "n_fraud_test": int(y_test.sum()),
    }

    if tier_thresholds:
        metrics["tier_thresholds"] = tier_thresholds
        metrics["tier_confusion"] = _tier_confusion(probs, y_test, tier_thresholds)

    return metrics


def _tier_confusion(probs, y_true, tier_thresholds):
    """For each tier, count how many fraud and legit are caught (TP/FP)."""
    out = {}
    for tier_name, t in tier_thresholds.items():
        preds = (probs >= t).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        out[tier_name] = {"threshold": t, "tp": tp, "fp": fp, "fn": fn,
                          "precision": float(precision_score(y_true, preds, zero_division=0)),
                          "recall":    float(recall_score   (y_true, preds, zero_division=0))}
    return out
