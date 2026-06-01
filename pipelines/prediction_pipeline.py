import sys
import yaml
import joblib
import pandas as pd

from src.components.preprocessing import DataPreprocessing
from src.components.feature_engineering import FeatureEngineering

from src.utils.logger import logging
from src.utils.exception import CustomException


# Columns that were scaled during training — must be scaled the same way at inference
SCALE_COLS = [
    "Amount_log", "Hour_sin", "Hour_cos",
]


class PredictionPipeline:
    """
    Inference pipeline for a single transaction or a batch DataFrame.

    Applies the same preprocessing and feature engineering as training,
    then runs the saved LightGBM model.

    Loaded artifacts (from models/):
        model.pkl          — trained LightGBM classifier
        scaler.pkl         — fitted StandardScaler / RobustScaler
        imputer.pkl        — fitted SimpleImputer
        label_encoders.pkl — dict of {col: LabelEncoder}
    """

    def __init__(self, params_path: str = "params.yaml"):

        with open(params_path) as f:
            params = yaml.safe_load(f)

        self.threshold  = params["data"].get("inference_threshold", 0.5)
        self.target_col = params["data"].get("target_column", "Class")
        self.scaler_type = params["data"].get("scaler", "standard")

        # Load saved artifacts produced during training
        self.model          = joblib.load("models/model.pkl")
        self.scaler         = joblib.load("models/scaler.pkl")
        self.imputer        = joblib.load("models/imputer.pkl")
        self.label_encoders = joblib.load("models/label_encoders.pkl")

        # Components — used for transform-only (no fitting)
        self.preprocessor     = DataPreprocessing(
            scaler_type=self.scaler_type,
            target_col=self.target_col
        )
        self.feature_engineer = FeatureEngineering()

        logging.info(
            f"PredictionPipeline ready — threshold={self.threshold}"
        )

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same preprocessing as training — transform only, no fitting.

        Steps:
            1. Drop target column if present (inference on unlabelled data)
            2. Encode categoricals with saved LabelEncoders
            3. Reorder columns to match imputer fit order; fill missing cols as NaN
            4. Impute with saved imputer
            5. Feature engineering (Amount_log, Hour_sin/cos, V stats)
            6. Scale engineered features with saved scaler
        """
        try:

            df = df.copy()

            # 1. Drop target if accidentally included in input
            if self.target_col in df.columns:
                df = df.drop(columns=[self.target_col])

            # 2. Encode categoricals using saved LabelEncoders
            for col, le in self.label_encoders.items():
                if col in df.columns:
                    df[col] = df[col].astype(str)
                    known = set(le.classes_)
                    # Gracefully handle unseen categories → fall back to first class
                    df[col] = df[col].apply(
                        lambda x: x if x in known else le.classes_[0]
                    )
                    df[col] = le.transform(df[col])

            # 3. Align columns to imputer's expected feature order
            expected_cols = list(self.imputer.feature_names_in_)
            for c in expected_cols:
                if c not in df.columns:
                    df[c] = float("nan")
            df = df[expected_cols]

            # 4. Impute using saved imputer (transform only)
            df = pd.DataFrame(
                self.imputer.transform(df),
                columns=expected_cols
            )

            # 5. Feature engineering (same logic as training)
            df = self.feature_engineer.initiate_feature_engineering(df)

            # Drop raw columns — model was trained on V-only + engineered only
            drop_cols = ["Time", "Amount"]
            for c in drop_cols:
                if c in df.columns:
                    df.drop(columns=[c], inplace=True)

            # 6. Scale engineered features using saved scaler (transform only)
            cols_to_scale = [c for c in SCALE_COLS if c in df.columns]
            if cols_to_scale:
                df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict fraud probability and binary label for each row.

        Args:
            df: raw transaction DataFrame (may include or exclude target column)

        Returns:
            Original df with two new columns:
                fraud_probability — float [0.0, 1.0]
                fraud_prediction  — int   (0 = legitimate, 1 = fraud)
        """
        try:

            processed = self.preprocess(df)

            probs = self.model.predict_proba(processed)[:, 1]
            preds = (probs >= self.threshold).astype(int)

            result = df.copy()
            result["fraud_probability"] = probs.round(4)
            result["fraud_prediction"]  = preds

            fraud_count = preds.sum()
            logging.info(
                f"Predicted {len(df)} transactions — "
                f"{fraud_count} flagged as fraud "
                f"({fraud_count / len(df) * 100:.2f}%)"
            )

            return result

        except Exception as e:
            raise CustomException(e, sys)


# ── Run as a quick sanity-check script ───────────────────────────────────────
if __name__ == "__main__":

    pipeline = PredictionPipeline()

    # Load first 1 000 rows of the held-out test set for a quick check
    test_df = pd.read_csv("artifacts/test.csv").head(1000)

    predictions = pipeline.predict(test_df)

    print(predictions[["fraud_probability", "fraud_prediction"]].head(20))
    print(f"\nFlagged as fraud : {predictions['fraud_prediction'].sum()} / {len(predictions)}")
    print(f"Threshold used   : {pipeline.threshold}")
