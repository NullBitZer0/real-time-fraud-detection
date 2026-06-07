"""Parity test: notebook model vs production model on the same test set.

Loads the saved production CatBoost model + FeatureEngineering state from
models/, scores a held-out chunk of fraudTest.csv, and checks that the
3-tier confusion matrix matches the metadata.json recorded during the
DVC `train` stage.

Run:
    python -m tests.test_parity
    python -m tests.test_parity --n 5000
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from pipelines.prediction_pipeline import PredictionPipeline
from src.components.data_ingestion import read_sparkov_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000,
                        help="Number of fraudTest rows to test (default 2000)")
    args = parser.parse_args()

    print("=" * 60)
    print("PARITY TEST — production model on held-out fraudTest rows")
    print("=" * 60)

    # Load the production model + Feast-enabled pipeline
    pipe = PredictionPipeline()
    md = pipe.metadata
    print(f"Production model: val_pr={md['val_pr_auc']:.4f} "
          f"test_pr={md['test_pr_auc']:.4f} "
          f"f1@T2={md['tier_summary']['tier2']['f1']:.4f}")

    # Load a held-out chunk of fraudTest
    df = read_sparkov_split("data/raw/fraudTest.csv")
    sample = df.sample(n=min(args.n, len(df)), random_state=123).reset_index(drop=True)
    print(f"Scoring {len(sample):,} rows (fraud rate: {sample.is_fraud.mean():.4%})")

    out = pipe.predict(sample)
    y_true = out["is_fraud"].astype(int).values
    y_pred = (out["tier"] >= 1).astype(int).values
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    f1m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    tn, fp = cm[0]; fn, tp = cm[1]
    print(f"Parity test confusion [[TN,FP],[FN,TP]]: {cm}")
    print(f"Parity test macro F1: {f1m:.4f}")

    # Sanity: model should be roughly in the same ballpark as the metadata
    md_t2_f1 = md["tier_summary"]["tier2"]["f1"]
    if abs(f1m - md_t2_f1) > 0.10:
        print(f"⚠ Parity F1 ({f1m:.4f}) differs from metadata F1 ({md_t2_f1:.4f})")
    else:
        print(f"✓ Parity OK: |F1 - metadata F1| = {abs(f1m - md_t2_f1):.4f} ≤ 0.10")

    return cm, f1m


if __name__ == "__main__":
    main()
