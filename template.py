import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")

project_name = "fraud_detection"

list_of_files = [
    # GitHub CI/CD
    ".github/workflows/.gitkeep",

    # Hydra Configs
    "configs/config.yaml",
    "configs/model/catboost.yaml",
    "configs/model/lightgbm.yaml",
    "configs/model/xgboost.yaml",

    # Source package
    f"src/__init__.py",
    f"src/{project_name}/__init__.py",

    # Components
    f"src/components/__init__.py",
    f"src/components/data_ingestion.py",
    f"src/components/data_validation.py",
    f"src/components/feature_engineering.py",
    f"src/components/preprocessing.py",
    f"src/components/model_training.py",
    f"src/components/model_evaluation.py",
    f"src/components/optuna_tuning.py",

    # Entity
    f"src/entity/__init__.py",
    f"src/entity/config_entity.py",
    f"src/entity/artifact_entity.py",

    # Utils
    f"src/utils/__init__.py",
    f"src/utils/logger.py",
    f"src/utils/exception.py",

    # MLflow tracking
    f"src/mlflow_tracking.py",

    # Pipelines
    "pipelines/ingest_pipeline.py",
    "pipelines/preprocess_pipeline.py",
    "pipelines/training_pipeline.py",
    "pipelines/optuna_pipeline.py",
    "pipelines/prediction_pipeline.py",

    # Data directories (kept with .gitkeep so git tracks them)
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",

    # Artifacts & model output directories
    "artifacts/.gitkeep",
    "models/.gitkeep",

    # Metrics & logs directories
    "metrics/.gitkeep",
    "logs/.gitkeep",

    # Notebooks
    "notebooks/eda.ipynb",
    "notebooks/experiments.ipynb",

    # API / App
    "api/app.py",

    # DVC
    "dvc.yaml",
    "params.yaml",

    # Project setup
    "setup.py",
    "requirements.txt",
    "README.md",
    "Dockerfile",
    ".gitignore",
    ".dvcignore",
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
