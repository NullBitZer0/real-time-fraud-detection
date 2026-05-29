import yaml
import pandas as pd

from src.components.model_training import ModelTrainer


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():

    params = load_params()

    train_df = pd.read_csv(params["data"]["train_path"])
    test_df  = pd.read_csv(params["data"]["test_path"])

    trainer = ModelTrainer()

    metrics = trainer.initiate_model_training(
        train_df,
        test_df,
        params
    )

    print(metrics)


if __name__ == "__main__":
    main()