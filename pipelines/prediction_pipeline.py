import sys
import yaml
import joblib
import numpy as np
import pandas as pd

from src.utils.logger import logging
from src.utils.exception import CustomException


# ── Feature engineering (same as training pipeline) ───────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Amount_log"] = np.log1p(df["Amount"])
    df["Hour"]       = (df["Time"] // 3600) % 24
    df["Hour_sin"]   = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"]   = np.cos(2 * np.pi * df["Hour"] / 24)

    v_cols = [c for c in df.columns if c.startswith("V")]
    df["V_mean"] = df[v_cols].mean(axis=1)
    df["V_std"]  = df[v_cols].std(axis=1)
    df["V_max"]  = df[v_cols].max(axis=1)
    df["V_min"]  = df[v_cols].min(axis=1)

    df.drop(columns=["Time", "Amount", "Hour"], errors="ignore", inplace=True)
    return df


SCALE_COLS = [
    "Amount_log", "Hour_sin", "Hour_cos",
    "V_mean", "V_std", "V_max", "V_min"
]


class PredictionPipeline:

    def __init__(self, params_path: str = "params.yaml"):

        with open(params_path) as f:
            params = yaml.safe_load(f)

        self.threshold = params["data"].get("inference_threshold", 0.5)
        self.target    = params["data"].get("target_column", "Class")

        # Load saved artifacts from training
        self.model          = joblib.load("models/model.pkl")
        self.scaler         = joblib.load("models/scaler.pkl")
        self.imputer        = joblib.load("models/imputer.pkl")
        self.label_encoders = joblib.load("models/label_encoders.pkl")

        logging.info(
            f"PredictionPipeline loaded — threshold={self.threshold}"
        )

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply same preprocessing as training — no fitting."""

        try:

            # Drop target if present (inference on new data)
            if self.target in df.columns:
                df = df.drop(columns=[self.target])

            # Encode categoricals
            for col, le in self.label_encoders.items():
                if col in df.columns:
                    df[col] = df[col].astype(str)
                    # Handle unseen categories gracefully
                    known = set(le.classes_)
                    df[col] = df[col].apply(
                        lambda x: x if x in known else le.classes_[0]
                    )
                    df[col] = le.transform(df[col])

            # Impute
            df = pd.DataFrame(
                self.imputer.transform(df),
                columns=df.columns
            )

            # Feature engineering
            df = add_features(df)

            # Scale
            cols_to_scale = [c for c in SCALE_COLS if c in df.columns]
            if cols_to_scale:
                df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns original df with two new columns:
            fraud_probability : float (0.0 – 1.0)
            fraud_prediction  : int   (0 = normal, 1 = fraud)
        """
        try:

            processed = self.preprocess(df.copy())

            probs = self.model.predict_proba(processed)[:, 1]
            preds = (probs >= self.threshold).astype(int)

            result = df.copy()
            result["fraud_probability"] = probs.round(4)
            result["fraud_prediction"]  = preds

            fraud_count = preds.sum()
            logging.info(
                f"Predicted {len(df)} transactions — "
                f"{fraud_count} flagged as fraud "
                f"({fraud_count/len(df)*100:.2f}%)"
            )

            return result

        except Exception as e:
            raise CustomException(e, sys)


# ── Run as script on test set ─────────────────────────────────────────────────
if __name__ == "__main__":

    pipeline = PredictionPipeline()

    # Load a sample (first 1000 rows of test set for quick check)
    test_df = pd.read_csv("artifacts/test.csv").head(1000)

    predictions = pipeline.predict(test_df)

    print(predictions[["fraud_probability", "fraud_prediction"]].head(20))
    print(f"\nFlagged fraud: {predictions['fraud_prediction'].sum()} / {len(predictions)}")
    print(f"Threshold used: {pipeline.threshold}")
