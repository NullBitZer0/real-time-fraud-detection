"""Top-level training script for the Sparkov pipeline.

Runs:
  1. Data ingestion (load + time split)
  2. Data validation (schema + row-overlap checks)
  3. Feature engineering (fit on train, apply to all splits)
  4. Model training (CatBoost baseline)
  5. Save artifacts to models/

Usage:
  python -m src.train
"""
import os
import sys
import time
import numpy as np
import pandas as pd

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import logging
from src.utils.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.feature_engineering import FeatureEngineering
from src.components.model_training import ModelTrainer
from src.components.model_evaluation import evaluate_model


def main():
    t_start = time.time()
    logging.info("=" * 70)
    logging.info("Sparkov CatBoost training pipeline")
    logging.info("=" * 70)

    # 1. Data ingestion
    di = DataIngestion()
    train_path, val_path, test_path = di.initiate_data_ingestion()

    # 2. Data validation
    dv = DataValidation()
    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    test_df  = pd.read_csv(test_path)
    dv.validate_columns(train_df)
    dv.validate_split(train_df, val_df, test_df)

    # 3. Feature engineering (fit on train, apply to all)
    logging.info("Feature engineering started")
    fe = FeatureEngineering()
    train_fe = fe.fit_transform(train_df, compute_velocity=True)
    val_fe   = fe.transform(val_df,   compute_velocity=True)
    test_fe  = fe.transform(test_df,  compute_velocity=True)

    features = fe.feature_list
    logging.info(f"Features: {len(features)}")

    X_train = train_fe[features].values
    y_train = train_fe["is_fraud"].values
    X_val   = val_fe  [features].values
    y_val   = val_fe  ["is_fraud"].values
    X_test  = test_fe [features].values
    y_test  = test_fe ["is_fraud"].values

    logging.info(f"X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")

    # 4. Model training
    mt = ModelTrainer()
    metadata = mt.initiate_model_training(
        X_train, y_train, X_val, y_val, X_test, y_test,
        feature_engineer=fe,
    )

    # 5. Print summary
    t_total = time.time() - t_start
    logging.info("=" * 70)
    logging.info("TRAINING COMPLETE")
    logging.info("=" * 70)
    logging.info(f"  val  PR-AUC : {metadata['val_pr_auc']:.4f}")
    logging.info(f"  val  ROC-AUC: {metadata['val_roc_auc']:.4f}")
    logging.info(f"  test PR-AUC : {metadata['test_pr_auc']:.4f}")
    logging.info(f"  test ROC-AUC: {metadata['test_roc_auc']:.4f}")
    logging.info(f"  fit time    : {metadata['fit_time_seconds']:.1f}s")
    logging.info(f"  total time  : {t_total:.1f}s")
    logging.info("")
    logging.info("3-tier analysis (test set):")
    for tier_name, info in metadata["tier_summary"].items():
        if info is None:
            continue
        logging.info(f"  {tier_name}: t={info['threshold']:.4f}  P={info['precision']:.4f}  "
                     f"R={info['recall']:.4f}  F1={info['f1']:.4f}  "
                     f"TP={info['tp']:>5}  FP={info['fp']:>5}  FN={info['fn']:>5}")
    logging.info("")
    logging.info(f"Artifacts saved:")
    logging.info(f"  model:        {metadata['model_path']}")
    logging.info(f"  metadata:     models/metadata.json")
    logging.info(f"  feature_eng:  models/feature_engineering.pkl")


if __name__ == "__main__":
    main()
