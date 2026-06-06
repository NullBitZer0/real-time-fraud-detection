"""Artifact entities for the Sparkov pipeline."""
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class DataIngestionArtifact:
    train_path: str
    val_path:   str
    test_path:  str
    n_train:    int
    n_val:      int
    n_test:     int
    fraud_rate_train: float
    fraud_rate_val:   float
    fraud_rate_test:  float


@dataclass
class ModelTrainingArtifact:
    model_path:        str
    metadata_path:     str
    feature_eng_path:  str
    val_pr_auc:        float
    val_roc_auc:       float
    test_pr_auc:       float
    test_roc_auc:      float
    fit_time_seconds:  float
    tier_summary:      Dict[str, Any]
