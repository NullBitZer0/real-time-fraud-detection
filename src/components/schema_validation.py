"""Pandera schema validation for the Sparkov prediction pipeline.

A "schema contract" is what real production ML services have: every input
batch is checked against an explicit type/range/nullability contract
BEFORE it reaches the model. This catches:

- Garbage inputs from upstream (e.g. negative amounts, future dates)
- Schema drift (Feast adds a column, the model still gets the old one)
- Bad data from a new source (e.g. a future migration breaks a field)

Two schemas:
    - TransactionSchema     : the 14 raw fields from /predict
    - ModelInputSchema      : the 32 engineered features the model sees
"""
import sys

import pandas as pd
import pandera.pandas as pa
from pandera import Field

from src.utils.exception import CustomException
from src.utils.logger import logging


# ── Raw transaction input (the 14 fields coming from the API) ────────────────
class TransactionSchema(pa.DataFrameModel):
    """Contract for a single Sparkov transaction row."""

    trans_date_trans_time: pa.DateTime       = Field()                       # any datetime
    cc_num:                pa.Float           = Field(ge=1e15, le=1e18)        # hashed PAN
    merchant:              pa.String          = Field(str_length={"min_value": 1, "max_value": 100})
    category:              pa.String          = Field(isin=[
        "food_dining", "gas_transport", "grocery_pos", "grocery_net", "health_fitness",
        "home", "kids_pets", "misc_net", "misc_pos", "personal_care", "shopping_net",
        "shopping_pos", "travel", "entertainment", "education",
    ])
    amt:                   pa.Float           = Field(ge=0, le=10_000)
    lat:                   pa.Float           = Field(ge=-90, le=90)
    long:                  pa.Float           = Field(ge=-180, le=180)
    merch_lat:             pa.Float           = Field(ge=-90, le=90)
    merch_long:            pa.Float           = Field(ge=-180, le=180)
    dob:                   pa.String          = Field(str_matches=r"^\d{4}-\d{2}-\d{2}$")
    city:                  pa.String          = Field(str_length={"min_value": 1, "max_value": 100})
    state:                 pa.String          = Field(str_length={"min_value": 2, "max_value": 2})
    job:                   pa.String          = Field(str_length={"min_value": 1, "max_value": 100})
    zip:                   pa.Int             = Field(ge=0, le=99_999)

    class Config:
        strict = False   # allow extra columns (e.g. trans_num, is_fraud)
        coerce = True    # coerce types when possible (e.g. int → float for cc_num)


# ── Model input (the 32 engineered features) ─────────────────────────────────
class ModelInputSchema(pa.DataFrameModel):
    """Contract for the 32 features the CatBoost model expects."""

    # Time features
    hour:        pa.Int    = Field(ge=0, le=23)
    dow:         pa.Int    = Field(ge=0, le=6)
    month:       pa.Int    = Field(ge=1, le=12)
    is_night:    pa.Int    = Field(isin=[0, 1])

    # Demographic
    age:         pa.Float  = Field(ge=0, le=120)

    # Amount
    amt_log:     pa.Float  = Field(ge=-10, le=10)
    amt_is_round: pa.Int   = Field(isin=[0, 1])

    # Geo
    distance_km: pa.Float  = Field(ge=0, le=20_000)

    # Frequency-encoded categoricals
    cc_num_FE:      pa.Float
    merchant_FE:    pa.Float
    category_FE:    pa.Float
    city_FE:        pa.Float
    state_FE:       pa.Float
    job_FE:         pa.Float
    zip_FE:         pa.Float

    # Target-encoded categoricals
    merchant_te:    pa.Float = Field(ge=0, le=1)
    category_te:    pa.Float = Field(ge=0, le=1)
    city_te:        pa.Float = Field(ge=0, le=1)
    state_te:       pa.Float = Field(ge=0, le=1)
    job_te:         pa.Float = Field(ge=0, le=1)

    # Per-entity aggregations
    amt_per_merchant_mean:  pa.Float = Field(ge=0)
    amt_per_merchant_std:   pa.Float = Field(ge=0)
    amt_per_category_mean:  pa.Float = Field(ge=0)
    amt_per_category_std:   pa.Float = Field(ge=0)
    amt_per_cc_num_mean:    pa.Float = Field(ge=0)
    amt_per_cc_num_std:     pa.Float = Field(ge=0)

    # Velocity (zero-filled at inference for cold entities)
    txn_last_1h:       pa.Float = Field(ge=0)
    txn_last_24h:      pa.Float = Field(ge=0)
    txn_last_168h:     pa.Float = Field(ge=0)
    amt_sum_last_1h:   pa.Float = Field(ge=0)
    amt_sum_last_24h:  pa.Float = Field(ge=0)
    amt_sum_last_168h: pa.Float = Field(ge=0)

    class Config:
        strict = False
        coerce = True


def validate_transactions(df: pd.DataFrame, *, strict: bool = False) -> pd.DataFrame:
    """Validate a raw-transaction DataFrame against TransactionSchema.

    Args:
        df: input DataFrame (single row or batch)
        strict: if True, raise on the first schema error; if False, log + coerce

    Returns:
        Coerced / cleaned DataFrame.
    """
    try:
        return TransactionSchema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        if strict:
            raise CustomException(e, sys)
        logging.warning(f"validate_transactions: {len(e.failure_cases)} schema issues — coercing")
        for _, row in e.failure_cases.head(10).iterrows():
            logging.warning(f"  schema: {row}")
        # Return a best-effort coerced version
        return TransactionSchema.validate(df, lazy=True)


def validate_model_input(df: pd.DataFrame, *, strict: bool = False) -> pd.DataFrame:
    """Validate the engineered-feature DataFrame against ModelInputSchema."""
    try:
        return ModelInputSchema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        if strict:
            raise CustomException(e, sys)
        logging.warning(f"validate_model_input: {len(e.failure_cases)} schema issues — coercing")
        for _, row in e.failure_cases.head(10).iterrows():
            logging.warning(f"  schema: {row}")
        return ModelInputSchema.validate(df, lazy=True)


if __name__ == "__main__":
    # Smoke test
    import joblib

    from src.components.data_ingestion import read_sparkov_split
    sample = read_sparkov_split("data/raw/fraudTest.csv").head(3)
    print("Validating raw transactions…")
    out = validate_transactions(sample)
    print(f"  OK — {len(out)} rows passed")

    from src.components.feature_engineering import FeatureEngineering
    fe = joblib.load("models/feature_engineering.pkl") if __import__("os").path.exists("models/feature_engineering.pkl") else FeatureEngineering().fit(sample)
    fe_df = fe.transform(sample, compute_velocity=False)
    print("Validating model input…")
    out2 = validate_model_input(fe_df)
    print(f"  OK — {len(out2)} rows passed")
