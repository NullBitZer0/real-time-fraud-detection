"""Evidently-based drift report.

Generates an HTML report comparing the distribution of input features
between `fraudTrain.csv` (reference) and `fraudTest.csv` (current).
The HTML is written to `metrics/drift_report.html` for inclusion in the
README or the MLflow run.

Run:
    python -m src.components.drift_report
"""
import os
import sys
import pandas as pd

from src.utils.logger    import logging
from src.utils.exception import CustomException


def _short_name(x: str) -> str:
    """`fraud_Kirlin and Sons` → `Kirlin and Sons` (to avoid label explosion)."""
    return x.replace("fraud_", "").split(",")[0].strip() if isinstance(x, str) else x


def main(
    reference_path: str = "data/raw/fraudTrain.csv",
    current_path:   str = "data/raw/fraudTest.csv",
    out_html:       str = "metrics/drift_report.html",
    n_sample:       int = 50_000,
):
    """Generate a drift report comparing reference (train) vs current (test)."""
    try:
        os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
        logging.info(f"Drift report: {reference_path} vs {current_path} → {out_html}")

        ref = pd.read_csv(reference_path, parse_dates=["trans_date_trans_time", "dob"]).sample(
            n=min(n_sample, 200_000), random_state=42,
        )
        cur = pd.read_csv(current_path, parse_dates=["trans_date_trans_time", "dob"]).sample(
            n=min(n_sample, 200_000), random_state=42,
        )

        # Compact the dataframes for the report (drop heavy / ID cols)
        cols = [
            "trans_date_trans_time", "amt", "category", "merchant", "cc_num",
            "lat", "long", "merch_lat", "merch_long", "city_pop", "is_fraud",
        ]
        ref_small = ref[cols].copy()
        cur_small = cur[cols].copy()
        # Reduce merchant cardinality
        ref_small["merchant"] = ref_small["merchant"].apply(_short_name)
        cur_small["merchant"] = cur_small["merchant"].apply(_short_name)

        # Generate the report
        from evidently import Dataset, DataDefinition, Report
        from evidently.presets import DataDriftPreset

        dd = DataDefinition(
            numerical_columns=["amt", "lat", "long", "merch_lat", "merch_long", "city_pop"],
            categorical_columns=["category", "merchant"],
            datetime_columns=["trans_date_trans_time"],
        )
        ref_ds = Dataset.from_pandas(ref_small, data_definition=dd)
        cur_ds = Dataset.from_pandas(cur_small, data_definition=dd)

        report = Report([DataDriftPreset()])
        result = report.run(reference_data=ref_ds, current_data=cur_ds)

        # Evidently 0.7+: save as HTML or JSON
        result.save_html(out_html)
        # Also save a JSON snapshot for MLflow logging
        result.save_json(out_html.replace(".html", ".json"))

        # Quick summary in stdout
        try:
            summary = result.dict()
            drift_score = summary.get("metrics", [{}])[0].get("value")
            logging.info(f"Drift report saved → {out_html}  (overall drift score: {drift_score})")
        except Exception:
            logging.info(f"Drift report saved → {out_html}")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
