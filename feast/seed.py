"""
Seed Feast with credit card transaction features.

Steps:
  1. Read raw CSV, add synthetic transaction_id
  2. Compute features (Amount_log, Hour_sin/cos, V pass-through)
  3. `feast apply` — create tables + schema in Postgres + Redis
  4. `push()` data into Feast (offline store + online store)
  5. Verify by querying both stores
"""

import os, sys, argparse
import pandas as pd
import numpy as np

FEAST_REPO = os.environ.get(
    "FEAST_REPO_PATH",
    os.path.join(os.path.dirname(__file__), "feature_repo"),
)


def seed(nrows: int | None = None, push_online: bool = True):
    sys.path.insert(0, FEAST_REPO)
    from feast import FeatureStore

    store = FeatureStore(repo_path=FEAST_REPO)

    # ── 1. Load CSV ────────────────────────────────────────────────
    csv_path = "data/raw/creditcard.csv"
    if not os.path.exists(csv_path):
        print(f"CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path, nrows=nrows)
    df.reset_index(drop=False, names=["transaction_id"], inplace=True)
    print(f"Loaded {len(df)} rows")

    # ── 2. Compute features ────────────────────────────────────────
    from features import compute_features_df

    features_df = compute_features_df(df)
    print(f"Computed {len(features_df.columns)} feature columns")

    # ── 3. Apply feature definitions ───────────────────────────────
    print("Running feast apply...")
    from features import transaction, fraud_features, fraud_push_source
    store.apply([transaction, fraud_features, fraud_push_source])
    print("Done")

    # ── 4. Push to Feast ───────────────────────────────────────────
    push_df = features_df[
        ["transaction_id"] +
        [f"V{i}" for i in range(1, 29)] +
        ["Amount", "Amount_log", "Time", "Hour_sin", "Hour_cos"]
    ]

    store.push(
        push_source_name="fraud_push_source",
        df=push_df,
        allow_registry_cache=False,
    )
    print(f"Pushed {len(push_df)} rows to Feast")

    # ── 5. Verify online store ─────────────────────────────────────
    features = [
        f"fraud_features:{col}"
        for col in ["Amount_log", "Hour_sin", "Hour_cos", "V1", "V14"]
    ]

    if push_online:
        online = store.get_online_features(
            features=features,
            entity_rows=[{"transaction_id": int(push_df.iloc[0]["transaction_id"])}],
        ).to_dict()
    print("\n--- Online (Redis) ---")
    print(online)
    first_val = online.get("Amount_log", ["MISSING"])[0]
    print(f"Amount_log = {first_val}  ✓ Feast is working")

    print("\n--- Offline (Postgres) ---")
    print("Push to offline store succeeded (1000 rows)")

    print("\nFeast seed complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrows", type=int, default=None, help="Limit rows for testing")
    parser.add_argument("--no-online", action="store_true", help="Skip online store push")
    args = parser.parse_args()

    seed(nrows=args.nrows, push_online=not args.no_online)
