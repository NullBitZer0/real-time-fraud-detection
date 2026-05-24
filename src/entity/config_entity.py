from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path: str
    train_path: str
    test_path: str


@dataclass
class DataValidationConfig:
    required_columns: list


@dataclass
class ModelTrainerConfig:
    model_path: str