import pandas as pd

from src.components.preprocessing import DataPreprocessing
from src.components.feature_engineering import FeatureEngineering


if __name__ == "__main__":

    preprocessor = DataPreprocessing()

    train_df = preprocessor.initiate_preprocessing(
        "artifacts/train.csv"
    )

    test_df = preprocessor.initiate_preprocessing(
        "artifacts/test.csv"
    )

    feature_engineer = FeatureEngineering()

    train_df = (
        feature_engineer
        .initiate_feature_engineering(train_df)
    )

    test_df = (
        feature_engineer
        .initiate_feature_engineering(test_df)
    )

    train_df.to_csv(
        "artifacts/train_processed.csv",
        index=False
    )

    test_df.to_csv(
        "artifacts/test_processed.csv",
        index=False
    )

    print("Preprocessing completed")