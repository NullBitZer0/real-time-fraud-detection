"""Preprocessing for the Sparkov pipeline.

The CatBoost model works on raw engineered features, so there is no
scaling/imputation/encoding here. The module exists for symmetry with
the legacy pipeline and as a future extension point.
"""

from src.utils.logger import logging


class DataPreprocessing:

    def __init__(self, target_col: str = "is_fraud"):
        self.target_col = target_col

    def initiate_preprocessing(self, df, is_train: bool = True):
        """Identity transform — feature engineering already produced the
        final feature matrix. Kept for API symmetry with the legacy code.
        """
        logging.info(f"Preprocessing: shape={df.shape} (no-op for CatBoost)")
        return df
