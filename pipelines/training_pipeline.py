"""End-to-end training pipeline for the Sparkov fraud detection project.

Runs:
  1. Data Ingestion       — read from Postgres (or CSV), time-based 80/20 split
  2. Data Validation      — schema + row-overlap checks
  3. Feature Engineering  — fit on train, apply to all splits
  4. Preprocessing        — no-op for tree model
  5. Model Training       — CatBoost baseline (500 iter, depth=8, lr=0.05)
  6. Model Evaluation     — PR-AUC + 3-tier analysis

Usage:
    python -m pipelines.training_pipeline
    python -m pipelines.training_pipeline data.source=csv   # fallback
    python -m pipelines.training_pipeline model.iterations=1000 model.depth=10
"""
import os
import sys
import json

import joblib
import hydra
import numpy as np
import pandas as pd
import psycopg2
from omegaconf import DictConfig

from src.components.data_ingestion import DataIngestion, read_sparkov_split
from src.components.data_validation import DataValidation
from src.components.feature_engineering import FeatureEngineering
from src.components.preprocessing import DataPreprocessing
from src.components.model_training import ModelTrainer, build_catboost_params
from src.components.model_evaluation import evaluate_model
from src.components.mlflow_tracking import from_hydra as mlflow_from_hydra

from src.utils.logger import logging


def load_from_postgres() -> tuple:
    """Read fraud_detection.raw_transactions from Postgres + time-split.

    Returns: (train_df, val_df, test_df) where train/val come from
    fraudTrain (last 20% as val) and test comes from fraudTest.

    We distinguish train vs test rows by trans_date_trans_time: fraudTest
    covers 2020-06-21 onward, so rows with timestamp >= 2020-06-21 are
    treated as the held-out test set. Everything earlier is fraudTrain.
    """
    conn = psycopg2.connect(
        host="localhost", port=5432, user="feast", password="feast", dbname="feast"
    )
    sql = """
        SELECT
            trans_num, trans_date_trans_time, cc_num, merchant, category,
            amt, first, last, gender, street, city, state, zip,
            lat, long, city_pop, job, dob,
            merch_lat, merch_long, is_fraud, unix_time
        FROM fraud_detection.raw_transactions
        ORDER BY trans_date_trans_time
    """
    df = pd.read_sql(sql, conn)
    conn.close()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"]                    = pd.to_datetime(df["dob"])

    # Sparkov: fraudTrain ends 2020-06-20, fraudTest starts 2020-06-21
    cutoff = pd.Timestamp("2020-06-21")
    train_full = df[df.trans_date_trans_time <  cutoff].reset_index(drop=True)
    test_df    = df[df.trans_date_trans_time >= cutoff].reset_index(drop=True)

    # 80/20 time-based split of fraudTrain
    cutoff2 = train_full["trans_date_trans_time"].quantile(0.80)
    val_df   = train_full[train_full.trans_date_trans_time >= cutoff2].reset_index(drop=True)
    train_df = train_full[train_full.trans_date_trans_time <  cutoff2].reset_index(drop=True)
    return train_df, val_df, test_df


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config"
)
def run_training_pipeline(cfg: DictConfig):
    """Full end-to-end Sparkov training pipeline driven by Hydra config."""

    logging.info("=" * 60)
    logging.info("TRAINING PIPELINE STARTED")
    logging.info(f"  Model     : {cfg.model.model_name}")
    logging.info(f"  Threshold : {cfg.data.tier_thresholds.tier2_review_queue}")
    logging.info("=" * 60)

    # ── Step 1: Data Ingestion ────────────────────────────────────────────────
    source = cfg.data.get("source", "postgres")
    logging.info(f"Step 1/5 — Data Ingestion (source={source})")
    if source == "postgres":
        train_df, val_df, test_df = load_from_postgres()
    else:
        ingestion = DataIngestion()
        train_path, val_path, test_path = ingestion.initiate_data_ingestion()
        train_df = read_sparkov_split(train_path)
        val_df   = read_sparkov_split(val_path)
        test_df  = read_sparkov_split(test_path)

    # ── Step 2: Data Validation ───────────────────────────────────────────────
    logging.info("Step 2/5 — Data Validation")
    validator = DataValidation()
    validator.validate_columns(train_df)
    validator.validate_split(train_df, val_df, test_df)
    logging.info(
        f"Splits OK | train={len(train_df):,} ({train_df['is_fraud'].mean():.4%}) | "
        f"val={len(val_df):,} ({val_df['is_fraud'].mean():.4%}) | "
        f"test={len(test_df):,} ({test_df['is_fraud'].mean():.4%})"
    )

    # ── Step 3: Feature Engineering (fit on train, apply to all) ──────────────
    logging.info("Step 3/5 — Feature Engineering")
    feature_engineer = FeatureEngineering()
    train_fe = feature_engineer.fit_transform(train_df, compute_velocity=True)
    val_fe   = feature_engineer.transform  (val_df,   compute_velocity=True)
    test_fe  = feature_engineer.transform  (test_df,  compute_velocity=True)
    features = feature_engineer.feature_list
    logging.info(f"Built {len(features)} features: {features}")

    X_train = train_fe[features].values.astype("float32")
    y_train = train_fe["is_fraud"].values
    X_val   = val_fe  [features].values.astype("float32")
    y_val   = val_fe  ["is_fraud"].values
    X_test  = test_fe [features].values.astype("float32")
    y_test  = test_fe ["is_fraud"].values

    # ── Step 4: Optuna Tuning (skipped for demo — run separately) ────────────
    logging.info("Step 4/5 — Optuna Tuning (skipped — run `dvc repro optuna_tune` separately)")
    if cfg.optuna.enabled:
        # The actual tuning is done in the separate optuna_tune DVC stage.
        # Here we just load the best params if they exist.
        optuna_path = "models/optuna_best.json"
        if os.path.exists(optuna_path):
            import json as _json
            with open(optuna_path) as f:
                best = _json.load(f)
            logging.info(f"Loaded Optuna best params: {best['best_params']}")
            # Apply by overriding the trainer
            for k, v in best["best_params"].items():
                if k in cfg.model:
                    cfg.model[k] = v
        else:
            logging.warning("Optuna enabled but models/optuna_best.json not found — using baseline params")
    else:
        logging.info("Optuna disabled. Using CatBoost baseline params.")

    # ── Step 5: Model Training + Evaluation (wrapped in MLflow run) ──────────
    logging.info("Step 5/5 — Model Training + Evaluation")
    catboost_params = build_catboost_params(dict(cfg.model))
    tracker = mlflow_from_hydra(cfg, run_name=cfg.mlflow.run_name)
    with tracker as t:
        trainer = ModelTrainer()
        metadata = trainer.initiate_model_training(
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_engineer=feature_engineer,
            catboost_params=catboost_params,
            mlflow_tracker=t,
        )
        # Register the model in the MLflow registry
        model_uri = f"runs:/{t.run.info.run_id}/catboost"
        t.register_model(model_uri, cfg.mlflow.registered_name, stage="Staging")

    # Re-evaluate with the 3-tier threshold for confusion-matrix output
    model = joblib.load if False else None  # placeholder (model already saved)
    from catboost import CatBoostClassifier
    loaded = CatBoostClassifier()
    loaded.load_model(metadata["model_path"])
    eval_metrics = evaluate_model(
        loaded, X_test, y_test,
        tier_thresholds=metadata["tier_thresholds"],
        threshold=cfg.data.tier_thresholds.tier2_review_queue,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    logging.info("=" * 60)
    logging.info("TRAINING PIPELINE COMPLETED")
    logging.info("=" * 60)
    logging.info(f"  val  PR-AUC : {metadata['val_pr_auc']:.4f}")
    logging.info(f"  val  ROC-AUC: {metadata['val_roc_auc']:.4f}")
    logging.info(f"  test PR-AUC : {metadata['test_pr_auc']:.4f}")
    logging.info(f"  test ROC-AUC: {metadata['test_roc_auc']:.4f}")
    logging.info(f"  F1 @ T2     : {eval_metrics['f1']:.4f}")
    logging.info("")
    logging.info("3-tier analysis (test set):")
    for tier_name, info in metadata["tier_summary"].items():
        if info is None:
            continue
        logging.info(
            f"  {tier_name:<10} t={info['threshold']:.4f}  "
            f"P={info['precision']:.4f}  R={info['recall']:.4f}  F1={info['f1']:.4f}  "
            f"TP={info['tp']:>5}  FP={info['fp']:>5}  FN={info['fn']:>5}"
        )
    logging.info("")
    logging.info(f"Artifacts:")
    logging.info(f"  model       : {metadata['model_path']}")
    logging.info(f"  metadata    : models/metadata.json")
    logging.info(f"  feature_eng : models/feature_engineering.pkl")

    # ── DVC metrics (tracked via `dvc metrics show`) ─────────────────────────
    os.makedirs("metrics", exist_ok=True)
    dvc_metrics = {
        "val_pr_auc":   float(metadata["val_pr_auc"]),
        "val_roc_auc":  float(metadata["val_roc_auc"]),
        "test_pr_auc":  float(metadata["test_pr_auc"]),
        "test_roc_auc": float(metadata["test_roc_auc"]),
        "f1_at_t2":     float(eval_metrics["f1"]),
        "tier1_precision": float(metadata["tier_summary"]["tier1"]["precision"]) if metadata["tier_summary"].get("tier1") else None,
        "tier1_recall":    float(metadata["tier_summary"]["tier1"]["recall"])    if metadata["tier_summary"].get("tier1") else None,
        "tier2_precision": float(metadata["tier_summary"]["tier2"]["precision"]) if metadata["tier_summary"].get("tier2") else None,
        "tier2_recall":    float(metadata["tier_summary"]["tier2"]["recall"])    if metadata["tier_summary"].get("tier2") else None,
    }
    with open("metrics/train_metrics.json", "w") as f:
        json.dump(dvc_metrics, f, indent=2)
    logging.info(f"DVC metrics → metrics/train_metrics.json")

    return eval_metrics


if __name__ == "__main__":
    run_training_pipeline()
