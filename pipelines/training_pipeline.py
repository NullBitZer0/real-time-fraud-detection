"""End-to-end training pipeline for the Sparkov fraud detection project.

Two data paths:
  1. feast (default) — reads pre-computed features from Postgres via Feast
     `get_historical_features()`, skips redundant FeatureEngineering
  2. csv — reads raw CSVs, runs FeatureEngineering from scratch

Config is read from params.yaml in the project root.

Usage:
    python -m pipelines.training_pipeline
"""
import json
import os
import sys

import joblib
import pandas as pd
import psycopg2
import yaml

from src.components.data_ingestion import DataIngestion, read_sparkov_split
from src.components.feature_engineering import FeatureEngineering
from src.components.mlflow_tracking import MLflowTracker
from src.components.model_evaluation import evaluate_model
from src.components.model_training import ModelTrainer, build_catboost_params
from src.utils.exception import CustomException
from src.utils.logger import logging

FEAST_REPO_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "feast", "feature_repo"
))

PG_CONFIG = {
    "host":     os.environ.get("POSTGRES_HOST", "localhost"),
    "port":     int(os.environ.get("POSTGRES_PORT", 5432)),
    "user":     os.environ.get("POSTGRES_USER", "feast"),
    "password": os.environ.get("POSTGRES_PASSWORD", "feast"),
    "dbname":   os.environ.get("POSTGRES_DB", "feast"),
}

# All 34 model features (matches FeatureEngineering.feature_list)
FEATURE_COLS = [
    "hour", "dow", "month", "is_night", "age", "amt_log", "amt_is_round", "distance_km",
    "cc_num_FE", "merchant_FE", "category_FE", "city_FE", "state_FE", "job_FE", "zip_FE",
    "merchant_te", "category_te", "city_te", "state_te", "job_te",
    "amt_per_merchant_mean", "amt_per_merchant_std",
    "amt_per_category_mean", "amt_per_category_std",
    "amt_per_cc_num_mean",   "amt_per_cc_num_std",
    "txn_last_1h",  "txn_last_24h",  "txn_last_168h",
    "amt_sum_last_1h", "amt_sum_last_24h", "amt_sum_last_168h",
]

FEAST_FEATURE_REFS = (
    [f"transaction_features:{c}" for c in [
        "hour", "dow", "month", "is_night", "age", "distance_km",
        "amt", "amt_log", "amt_is_round",
        "category_FE", "city_FE", "state_FE", "job_FE", "zip_FE",
        "category_te", "city_te", "state_te", "job_te",
        "amt_per_category_mean", "amt_per_category_std",
        "is_fraud",
    ]] +
    [f"cc_num_features:{c}" for c in [
        "cc_num_FE", "txn_last_1h", "txn_last_24h", "txn_last_168h",
        "amt_sum_last_1h", "amt_sum_last_24h", "amt_sum_last_168h",
        "amt_per_cc_num_mean", "amt_per_cc_num_std",
    ]] +
    [f"merchant_features:{c}" for c in [
        "merchant_FE", "merchant_te",
        "amt_per_merchant_mean", "amt_per_merchant_std",
    ]]
)


def _load_params() -> dict:
    """Load config from params.yaml in the project root."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    params_path = os.path.join(root, "params.yaml")
    with open(params_path) as f:
        return yaml.safe_load(f)


def load_from_feast() -> tuple:
    """Read pre-computed features from Feast offline store (Postgres).

    1. Gets entity keys (trans_num, cc_num, merchant) + timestamps from
       the `transaction_features` Postgres table
    2. Calls Feast `get_historical_features()` to join all three feature
       views with point-in-time correctness
    3. Splits by timestamp → train (fraudTrain pre-80th), val (post-80th),
       test (fraudTest)

    Returns: (train_df, val_df, test_df)
             Each has all 34 features + is_fraud — no further FE needed.
    """
    from feast import FeatureStore

    logging.info("Loading entity keys from Postgres transaction_features...")
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        entity_df = pd.read_sql("""
            SELECT trans_num, cc_num, merchant, event_timestamp, is_fraud
            FROM fraud_detection.transaction_features
            ORDER BY event_timestamp
        """, conn)
    finally:
        conn.close()
    entity_df["cc_num"] = entity_df["cc_num"].astype("int64")
    entity_df["merchant"] = entity_df["merchant"].astype(str)
    entity_df["trans_num"] = entity_df["trans_num"].astype(str)
    logging.info(f"Entity rows: {len(entity_df):,}")

    logging.info("Querying Feast get_historical_features()...")
    store = FeatureStore(repo_path=FEAST_REPO_PATH)
    df = store.get_historical_features(
        features=FEAST_FEATURE_REFS,
        entity_df=entity_df,
    ).to_df()
    logging.info(f"Feast returned {len(df):,} rows with {len(df.columns)} columns")

    internal_cols = [c for c in df.columns if c.startswith("__") or c in (
        "trans_num", "cc_num", "merchant", "created"
    )]
    df = df.drop(columns=internal_cols, errors="ignore")

    # Time-based split: fraudTest starts 2020-06-21
    split_test = pd.Timestamp("2020-06-21")
    train_full = df[df["event_timestamp"] <  split_test].reset_index(drop=True)
    test_df    = df[df["event_timestamp"] >= split_test].reset_index(drop=True)

    # 80/20 time split of fraudTrain
    cutoff = train_full["event_timestamp"].quantile(0.80)
    val_df   = train_full[train_full.event_timestamp >= cutoff].reset_index(drop=True)
    train_df = train_full[train_full.event_timestamp <  cutoff].reset_index(drop=True)

    for part in [train_df, val_df, test_df]:
        part.drop(columns=["event_timestamp"], inplace=True, errors="ignore")

    logging.info(
        f"train={len(train_df):,} ({train_df['is_fraud'].mean():.4%}) | "
        f"val={len(val_df):,} ({val_df['is_fraud'].mean():.4%}) | "
        f"test={len(test_df):,} ({test_df['is_fraud'].mean():.4%})"
    )
    return train_df, val_df, test_df


def run_training_pipeline():
    cfg = _load_params()

    logging.info("=" * 60)
    logging.info("TRAINING PIPELINE STARTED")
    logging.info(f"  Model     : {cfg['model']['model_name']}")
    logging.info(f"  Threshold : {cfg['data']['tier_thresholds']['tier2_review_queue']}")
    logging.info("=" * 60)

    try:
        # ── Step 1: Data Ingestion ────────────────────────────────────────────
        source = cfg["data"].get("source", "feast")
        logging.info(f"Step 1/4 — Data Ingestion (source={source})")

        if source == "feast":
            train_df, val_df, test_df = load_from_feast()
            feature_engineer = None
            fe_path = "models/feature_engineering.pkl"
            if os.path.exists(fe_path):
                feature_engineer = joblib.load(fe_path)
                logging.info(f"Loaded fitted FeatureEngineering from {fe_path}")
            features = FEATURE_COLS
        else:
            # ── CSV path: run feature engineering from scratch ────────────────
            logging.info("Step 2/4 — Feature Engineering (from scratch)")
            ingestion = DataIngestion()
            train_path, val_path, test_path = ingestion.initiate_data_ingestion()
            train_df = read_sparkov_split(train_path)
            val_df   = read_sparkov_split(val_path)
            test_df  = read_sparkov_split(test_path)
            feature_engineer = FeatureEngineering()
            train_fe = feature_engineer.fit_transform(train_df, compute_velocity=True)
            val_fe   = feature_engineer.transform(val_df, compute_velocity=True)
            test_fe  = feature_engineer.transform(test_df, compute_velocity=True)
            features = feature_engineer.feature_list

            X_train = train_fe[features].values.astype("float32")
            y_train = train_fe["is_fraud"].values
            X_val   = val_fe  [features].values.astype("float32")
            y_val   = val_fe  ["is_fraud"].values
            X_test  = test_fe [features].values.astype("float32")
            y_test  = test_fe ["is_fraud"].values

            logging.info("Step 3/4 — Model Training + Evaluation")
            catboost_params = build_catboost_params(cfg["model"])
            tracker = MLflowTracker(cfg["mlflow"], run_name=cfg["mlflow"]["run_name"])
            with tracker as t:
                trainer = ModelTrainer()
                metadata = trainer.initiate_model_training(
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    feature_engineer=feature_engineer,
                    catboost_params=catboost_params,
                    mlflow_tracker=t,
                )
                model_uri = f"runs:/{t.run.info.run_id}/catboost"
                t.register_model(model_uri, cfg["mlflow"]["registered_name"], stage="Staging")

            from catboost import CatBoostClassifier
            loaded = CatBoostClassifier()
            loaded.load_model(metadata["model_path"])
            eval_metrics = evaluate_model(
                loaded, X_test, y_test,
                tier_thresholds=metadata["tier_thresholds"],
                threshold=cfg["data"]["tier_thresholds"]["tier2_review_queue"],
            )
            _log_summary(metadata, eval_metrics)
            _save_metrics(metadata, eval_metrics)
            return eval_metrics

        # ── Feast path: features already pre-computed ─────────────────────────
        train_fe = train_df[features].astype("float32")
        val_fe   = val_df[features].astype("float32")
        test_fe  = test_df[features].astype("float32")
        train_fe["is_fraud"] = train_df["is_fraud"].values
        val_fe["is_fraud"]   = val_df["is_fraud"].values
        test_fe["is_fraud"]  = test_df["is_fraud"].values

        logging.info(f"Using {len(features)} features")

        X_train = train_fe[features].values.astype("float32")
        y_train = train_fe["is_fraud"].values
        X_val   = val_fe  [features].values.astype("float32")
        y_val   = val_fe  ["is_fraud"].values
        X_test  = test_fe [features].values.astype("float32")
        y_test  = test_fe ["is_fraud"].values

        # ── Step 3: Optuna Tuning (optional) ─────────────────────────────────
        optuna_cfg = cfg.get("optuna", {})
        optuna_enabled = optuna_cfg.get("enabled", False) or \
            os.environ.get("OPTUNA_ENABLED", "").lower() in ("1", "true")

        if optuna_enabled:
            logging.info(f"Step 3/4 — Optuna Tuning ({optuna_cfg['n_trials']} trials)")
            from src.components.optuna_tuning import run_optuna
            optuna_best = run_optuna(
                X_train, y_train, X_val, y_val,
                n_trials=optuna_cfg["n_trials"],
                timeout_minutes=optuna_cfg.get("timeout_minutes", 0),
            )
            catboost_params = build_catboost_params({**cfg["model"], **optuna_best})
            logging.info(f"Optuna best params applied: {optuna_best}")
        else:
            optuna_path = "models/optuna_best.json"
            if os.path.exists(optuna_path):
                with open(optuna_path) as f:
                    saved = json.load(f).get("best_params", {})
                catboost_params = build_catboost_params({**cfg["model"], **saved})
                logging.info(f"Loaded saved Optuna params from {optuna_path}")
            else:
                catboost_params = build_catboost_params(cfg["model"])

        # ── Step 4: Model Training + Evaluation ──────────────────────────────
        logging.info("Step 4/4 — Model Training + Evaluation")
        tracker = MLflowTracker(cfg["mlflow"], run_name=cfg["mlflow"]["run_name"])
        with tracker as t:
            trainer = ModelTrainer()
            metadata = trainer.initiate_model_training(
                X_train, y_train, X_val, y_val, X_test, y_test,
                feature_engineer=feature_engineer,
                catboost_params=catboost_params,
                mlflow_tracker=t,
            )
            model_uri = f"runs:/{t.run.info.run_id}/catboost"
            t.register_model(model_uri, cfg["mlflow"]["registered_name"], stage="Staging")

        from catboost import CatBoostClassifier
        loaded = CatBoostClassifier()
        loaded.load_model(metadata["model_path"])
        eval_metrics = evaluate_model(
            loaded, X_test, y_test,
            tier_thresholds=metadata["tier_thresholds"],
            threshold=cfg["data"]["tier_thresholds"]["tier2_review_queue"],
        )
        _log_summary(metadata, eval_metrics)
        _save_metrics(metadata, eval_metrics)
        return eval_metrics

    except Exception as e:
        logging.exception("Training pipeline failed")
        raise CustomException(e, sys)


def _log_summary(metadata: dict, eval_metrics: dict):
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


def _save_metrics(metadata: dict, eval_metrics: dict):
    os.makedirs("metrics", exist_ok=True)
    metrics = {
        "val_pr_auc":   float(metadata["val_pr_auc"]),
        "val_roc_auc":  float(metadata["val_roc_auc"]),
        "test_pr_auc":  float(metadata["test_pr_auc"]),
        "test_roc_auc": float(metadata["test_roc_auc"]),
        "f1_at_t2":     float(eval_metrics["f1"]),
    }
    with open("metrics/train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    run_training_pipeline()
