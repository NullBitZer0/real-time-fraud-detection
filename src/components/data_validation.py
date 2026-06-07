"""Data validation for the Sparkov pipeline.

Checks that the required columns are present in the loaded CSV.
The Sparkov schema is fixed (trans_date_trans_time, cc_num, merchant,
category, amt, lat, long, merch_lat, merch_long, dob, city, state, job,
zip, unix_time, trans_num, is_fraud, first, last, street, gender).
"""

from src.utils.logger import logging

REQUIRED_COLUMNS = [
    "trans_date_trans_time", "cc_num", "merchant", "category", "amt",
    "lat", "long", "merch_lat", "merch_long", "dob", "city", "state",
    "job", "zip", "unix_time", "trans_num", "is_fraud", "first", "last",
    "street", "gender",
]


class DataValidation:

    def validate_columns(self, df):
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise Exception(f"Missing required Sparkov columns: {missing}")
        logging.info(f"Schema OK ({len(df.columns)} columns, {len(REQUIRED_COLUMNS)} required)")

    def validate_split(self, train_df, val_df, test_df):
        """Ensure the three splits have no row overlap."""
        train_set = set(train_df["trans_num"].astype(str))
        val_set   = set(val_df  ["trans_num"].astype(str))
        test_set  = set(test_df ["trans_num"].astype(str))
        overlap_tv = train_set & val_set
        overlap_tt = train_set & test_set
        overlap_vt = val_set   & test_set
        if overlap_tv or overlap_tt or overlap_vt:
            raise Exception(f"Row overlap detected: tv={len(overlap_tv)}, tt={len(overlap_tt)}, vt={len(overlap_vt)}")
        logging.info("No row overlap between train/val/test")
