"""Standalone Optuna tuning script — run via `dvc repro optuna_tune`.

Uses the parquet sources built by data_ingestion_feast.py (which has
both train + test rows with features already computed). Trains on the
train rows, evaluates on the val rows.

Output: models/optuna_best.json with the best params found.
"""
import os
import sys
import json
import argparse
import optuna
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit

from src.utils.logger import logging
from src.utils.exception import CustomException


def load_data():
    """Read the per-transaction parquet, split by timestamp into train/val."""
    df = pd.read_parquet("data/sparkov_transaction_features.parquet")
    df = df.sort_values("event_timestamp").reset_index(drop=True)
    cutoff = df["event_timestamp"].quantile(0.80)
    val_df  = df[df.event_timestamp >= cutoff].reset_index(drop=True)
    train_df = df[df.event_timestamp <  cutoff].reset_index(drop=True)

    feat_cols = [c for c in df.columns if c not in (
        "trans_num", "event_timestamp", "created", "is_fraud",
    )]
    X_train = train_df[feat_cols].astype("float32").values
    y_train = train_df["is_fraud"].astype(int).values
    X_val   = val_df[feat_cols].astype("float32").values
    y_val   = val_df["is_fraud"].astype(int).values
    return X_train, y_train, X_val, y_val, feat_cols


def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective: maximize val PR-AUC."""
    params = {
        "iterations":    trial.suggest_int("iterations", 200, 1000, step=100),
        "depth":         trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
        "random_seed":   42,
        "eval_metric":   "PRAUC",
        "verbose":       0,
        "early_stopping_rounds": 50,
        "task_type":     "CPU",
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    val_proba = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, val_proba)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models/optuna_best.json")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--timeout-minutes", type=int, default=0,
                        help="If > 0, stop after N minutes")
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.info(f"Optuna tuning started — n_trials={args.n_trials} timeout={args.timeout_minutes}m")

    X_train, y_train, X_val, y_val, feat_cols = load_data()
    logging.info(f"Train: {X_train.shape}  Val: {X_val.shape}  features={len(feat_cols)}")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda t: objective(t, X_train, y_train, X_val, y_val),
        n_trials=args.n_trials,
        timeout=(args.timeout_minutes * 60) if args.timeout_minutes else None,
        show_progress_bar=False,
    )

    best = {
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "n_trials":    len(study.trials),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(best, f, indent=2)
    logging.info(f"Best val PR-AUC: {best['best_value']:.4f}")
    logging.info(f"Best params: {best['best_params']}")
    logging.info(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
