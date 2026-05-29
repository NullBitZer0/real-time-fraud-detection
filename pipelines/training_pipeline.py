import hydra
import pandas as pd

from omegaconf import DictConfig

from src.components.model_training import ModelTrainer


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config"
)
def main(config: DictConfig):

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


if __name__ == "__main__":
    main()