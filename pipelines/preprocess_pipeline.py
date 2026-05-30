import yaml
import joblib
import os
import pandas as pd

from sklearn.preprocessing import StandardScaler, RobustScaler

from src.components.preprocessing import DataPreprocessing
from src.components.feature_engineering import FeatureEngineering


SCALE_COLS = [
    "Amount_log", "Hour_sin", "Hour_cos",
    "V_mean", "V_std", "V_max", "V_min"
]

if __name__ == "__main__":

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    scaler_type = params["data"].get("scaler", "standard")
    target_col  = params["data"].get("target_column", "Class")

    preprocessor     = DataPreprocessing(scaler_type=scaler_type, target_col=target_col)
    feature_engineer = FeatureEngineering()

    # ── Train: impute → engineer features → fit scaler ───────────────────────
    train_df = preprocessor.initiate_preprocessing(
        "artifacts/train.csv", is_train=True
    )
    train_df = feature_engineer.initiate_feature_engineering(train_df)

    # Fit scaler on engineered features (train only)
    ScalerClass = StandardScaler if scaler_type == "standard" else RobustScaler
    scaler = ScalerClass()
    cols_to_scale = [c for c in SCALE_COLS if c in train_df.columns]
    train_df[cols_to_scale] = scaler.fit_transform(train_df[cols_to_scale])

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    train_df.to_csv("artifacts/train_processed.csv", index=False)

    # ── Test: impute → engineer features → transform only ────────────────────
    test_df = preprocessor.initiate_preprocessing(
        "artifacts/test.csv", is_train=False
    )
    test_df = feature_engineer.initiate_feature_engineering(test_df)

    # Load saved scaler — transform only, no fit
    test_df[cols_to_scale] = scaler.transform(test_df[cols_to_scale])
    test_df.to_csv("artifacts/test_processed.csv", index=False)

    print(f"Preprocessing completed ✓")
    print(f"  Scaler : {scaler_type} (fit on train, transform test)")
    print(f"  Scaled : {cols_to_scale}")
    print(f"  Saved  : models/scaler.pkl")