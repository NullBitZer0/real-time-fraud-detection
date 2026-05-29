import yaml
import pandas as pd
from types import SimpleNamespace

from src.components.model_training import ModelTrainer


def load_config(path="params.yaml"):

    with open(path) as f:
        raw = yaml.safe_load(f)

    # Wrap nested dicts so trainer can use config.model.n_estimators etc.
    config = SimpleNamespace(
        model=SimpleNamespace(**raw["model"]),
        data=SimpleNamespace(**raw["data"])
    )

    return config


if __name__ == "__main__":

    config = load_config()

    train_df = pd.read_csv(
        "artifacts/train_processed.csv"
    )

    test_df = pd.read_csv(
        "artifacts/test_processed.csv"
    )

    trainer = ModelTrainer()

    metrics = trainer.initiate_model_training(
        train_df,
        test_df,
        config
    )

    print(metrics)