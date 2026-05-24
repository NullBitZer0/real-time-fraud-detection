import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.preprocessing import DataPreprocessing
from src.components.feature_engineering import FeatureEngineering
from src.components.model_training import ModelTrainer

from src.logger import logging


if __name__ == "__main__":

    logging.info(
        "Training pipeline started"
    )

    # Ingestion
    ingestion = DataIngestion()

    artifact = ingestion.initiate_data_ingestion()

    # Validation
    validator = DataValidation()

    validator.initiate_validation(
        artifact.train_path
    )

    # Preprocessing
    preprocessor = DataPreprocessing()

    train_df = preprocessor.initiate_preprocessing(
        artifact.train_path
    )

    test_df = preprocessor.initiate_preprocessing(
        artifact.test_path
    )

    # Feature Engineering
    feature_engineer = FeatureEngineering()

    train_df = (
        feature_engineer
        .initiate_feature_engineering(train_df)
    )

    test_df = (
        feature_engineer
        .initiate_feature_engineering(test_df)
    )

    # Model Training
    trainer = ModelTrainer()

    metrics = trainer.initiate_model_training(
        train_df,
        test_df
    )

    print(metrics)

    logging.info(
        "Training pipeline completed"
    )