import yaml
import pandas as pd

from src.components.preprocessing import DataPreprocessing
from src.components.feature_engineering import FeatureEngineering


if __name__ == "__main__":

    # Load scaler type from params.yaml
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    scaler_type = params["data"].get("scaler", "standard")

    preprocessor = DataPreprocessing(scaler_type=scaler_type)
    feature_engineer = FeatureEngineering()

    # ── Train ─────────────────────────────────────────────────────────────────
    # is_train=True → fits scaler/imputer, saves models/scaler.pkl
    train_df = preprocessor.initiate_preprocessing(
        "artifacts/train.csv",
        is_train=True
    )
    train_df = feature_engineer.initiate_feature_engineering(train_df)
    train_df.to_csv("artifacts/train_processed.csv", index=False)

    # ── Test ──────────────────────────────────────────────────────────────────
    # is_train=False → loads saved scaler, only transforms
    test_df = preprocessor.initiate_preprocessing(
        "artifacts/test.csv",
        is_train=False
    )
    test_df = feature_engineer.initiate_feature_engineering(test_df)
    test_df.to_csv("artifacts/test_processed.csv", index=False)

    print("Preprocessing completed — scaler fitted on train only ✓")