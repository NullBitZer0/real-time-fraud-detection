"""Config entities for the Sparkov pipeline."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataIngestionConfig:
    train_path: str  # fraudTrain.csv
    test_path:  str  # fraudTest.csv
    out_train:  str  # artifacts/sparkov_train.csv
    out_val:    str  # artifacts/sparkov_val.csv
    out_test:   str  # artifacts/sparkov_test.csv
    val_quantile: float = 0.80


@dataclass
class DataValidationConfig:
    required_columns: List[str] = field(default_factory=list)


@dataclass
class ModelTrainerConfig:
    model_dir:        str = "models"
    model_path:       str = "models/catboost.cbm"
    metadata_path:    str = "models/metadata.json"
    feature_eng_path: str = "models/feature_engineering.pkl"
    iterations:       int = 500
    depth:            int = 8
    learning_rate:    float = 0.05
    eval_metric:      str = "PRAUC"
    early_stopping:   int = 50
