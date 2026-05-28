import pandas as pd
import mlflow

from src.components.optuna_tuning import OptunaTuner


if __name__ == "__main__":

    train_df = pd.read_csv(
        "artifacts/train_processed.csv"
    )

    test_df = pd.read_csv(
        "artifacts/test_processed.csv"
    )

    with mlflow.start_run(
        run_name="optuna_tuning"
    ):

        tuner = OptunaTuner()

        study = tuner.initiate_optuna(
            train_df,
            test_df,
            n_trials=10
        )

        print(
            "Best Parameters:"
        )

        print(
            study.best_trial.params
        )

        print(
            "Best PR AUC:"
        )

        print(
            study.best_value
        )