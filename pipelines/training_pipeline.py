import joblib
import mlflow
import hydra

import pandas as pd

from omegaconf import DictConfig

from src.components.data_ingestion import DataIngestion
from src.components.preprocessing import DataPreprocessing
from src.components.feature_engineering import FeatureEngineering
from src.components.optuna_tuning import OptunaTuner
from src.components.model_training import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_registry import ModelRegistry
from src.components.model_pusher import ModelPusher

from src.utils.logger import logging


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config"
)
def run_training_pipeline(cfg: DictConfig):
    """
    Full end-to-end training pipeline driven by Hydra config.

    Switch models from CLI without editing any file:
        python -m pipelines.training_pipeline model=xgboost
        python -m pipelines.training_pipeline model=catboost
        python -m pipelines.training_pipeline model=lightgbm

    Override any param from CLI:
        python -m pipelines.training_pipeline data.inference_threshold=0.4
        python -m pipelines.training_pipeline optuna.enabled=true optuna.n_trials=50

    Steps:
        1. Data Ingestion       — splits raw CSV → train.csv / test.csv
        2. Preprocessing        — impute, encode, fit + save scaler/imputer
        3. Feature Engineering  — Amount_log, Hour_sin/cos, V_mean/std/max/min
        4. Optuna Tuning        — optional hyperparameter search
        5. Model Training       — train with params from Hydra config, log to MLflow
        6. Model Evaluation     — evaluate on test set, log metrics to MLflow
        7. Model Registry       — register best run to MLflow Model Registry → Staging
        8. Model Pusher         — pull Staging model and deploy to models/ locally
    """

    logging.info("=" * 60)
    logging.info("TRAINING PIPELINE STARTED")
    logging.info(f"  Model     : {cfg.model.model_name}")
    logging.info(f"  Scaler    : {cfg.data.scaler}")
    logging.info(f"  Threshold : {cfg.data.inference_threshold}")
    logging.info("=" * 60)

    # ── Step 1: Data Ingestion ────────────────────────────────────────────────
    logging.info("Step 1/6 — Data Ingestion")

    ingestion = DataIngestion()
    artifact  = ingestion.initiate_data_ingestion()

    # ── Step 2: Preprocessing ─────────────────────────────────────────────────
    logging.info("Step 2/6 — Preprocessing")

    preprocessor = DataPreprocessing(
        scaler_type=cfg.data.scaler,
        target_col=cfg.data.target_column
    )

    train_df = preprocessor.initiate_preprocessing(
        artifact.train_path,
        is_train=True
    )

    test_df = preprocessor.initiate_preprocessing(
        artifact.test_path,
        is_train=False
    )

    # ── Step 3: Feature Engineering ───────────────────────────────────────────
    logging.info("Step 3/6 — Feature Engineering")

    feature_engineer = FeatureEngineering()

    train_df = feature_engineer.initiate_feature_engineering(train_df)
    test_df  = feature_engineer.initiate_feature_engineering(test_df)

    train_df.to_csv(cfg.data.train_processed, index=False)
    test_df.to_csv(cfg.data.test_processed,   index=False)

    logging.info(
        f"Saved {cfg.data.train_processed} and {cfg.data.test_processed}"
    )

    # ── Step 4: Optuna Tuning (optional) ─────────────────────────────────────
    if cfg.optuna.enabled:
        logging.info("Step 4/6 — Optuna Tuning")

        tuner = OptunaTuner()
        study = tuner.initiate_optuna(
            train_df,
            test_df,
            n_trials=cfg.optuna.n_trials
        )

        logging.info(f"Best Optuna params : {study.best_trial.params}")
        logging.info(f"Best PR-AUC        : {study.best_value:.4f}")

    else:
        logging.info(
            "Step 4/6 — Optuna Tuning SKIPPED "
            "(use optuna.enabled=true from CLI to enable)"
        )

    # ── Step 5: Model Training ────────────────────────────────────────────────
    logging.info("Step 5/6 — Model Training")

    trainer = ModelTrainer()
    metrics = trainer.initiate_model_training(
        train_df,
        test_df,
        cfg           # pass full DictConfig — ModelTrainer uses dot-access
    )

    logging.info(f"Training metrics: {metrics}")

    # ── Step 6: Model Evaluation ──────────────────────────────────────────────
    logging.info("Step 6/6 — Model Evaluation")

    model  = joblib.load(cfg.model_path)

    X_test = test_df.drop(columns=[cfg.data.target_column])
    y_test = test_df[cfg.data.target_column]

    evaluator = ModelEvaluation()

    with mlflow.start_run(run_name="evaluation"):
        eval_metrics = evaluator.evaluate_model(
            model,
            X_test,
            y_test,
            threshold=cfg.data.inference_threshold
        )

    # ── Step 7: Model Registry ────────────────────────────────────────────────
    logging.info("Step 7/8 — Model Registry")

    registry        = ModelRegistry()
    registry_result = registry.register_best_model(
        model_name="FraudDetectionModel",
        experiment_name="fraud_detection_experiment",
        metric="eval_pr_auc",
        stage="Staging"
    )

    logging.info(
        f"Registered v{registry_result['model_version']} → Staging"
    )

    # ── Step 8: Model Pusher ──────────────────────────────────────────────────
    logging.info("Step 8/8 — Model Pusher")

    pusher      = ModelPusher()
    push_result = pusher.push_model(
        model_name="FraudDetectionModel",
        stage="Staging",
        dst_name="model.pkl"
    )

    logging.info(
        f"Model deployed → {push_result['dst_path']}"
    )

    logging.info("=" * 60)
    logging.info("TRAINING PIPELINE COMPLETED")
    logging.info(f"  PR-AUC    : {eval_metrics['pr_auc']:.4f}")
    logging.info(f"  ROC-AUC   : {eval_metrics['roc_auc']:.4f}")
    logging.info(f"  F1        : {eval_metrics['f1']:.4f}")
    logging.info(f"  Threshold : {cfg.data.inference_threshold}")
    logging.info(f"  Registry  : v{registry_result['model_version']} @ Staging")
    logging.info(f"  Deployed  : {push_result['dst_path']}")
    logging.info("=" * 60)

    return eval_metrics


if __name__ == "__main__":
    run_training_pipeline()