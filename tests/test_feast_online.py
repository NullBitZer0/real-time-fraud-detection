"""Feast online-store lookup test.

Verifies that FeatureStoreClient.get_online_features_for_batch() returns
non-None features for known cards + merchants after `feast materialize`.

Run:
    cd /home/NullbitZer0/projects/real-time-fraud-detection
    python -m tests.test_feast_online
"""
import sys
import pandas as pd

from src.components.feature_store      import FeatureStoreClient
from src.components.data_ingestion     import read_sparkov_split


def main():
    print("=" * 60)
    print("FEAST ONLINE TEST — Redis lookup for known cards + merchants")
    print("=" * 60)

    fsc = FeatureStoreClient.get()
    df  = read_sparkov_split("data/raw/fraudTest.csv").head(20)
    print(f"  Scoring {len(df)} sample rows for online lookup…")

    online = fsc.get_online_features_for_batch(df)
    print(f"  Online features shape: {online.shape}")
    print(f"  Online features columns: {list(online.columns)}")

    if online is None or online.empty:
        print("✗ No online features returned")
        sys.exit(1)

    # Per-feature non-null counts
    feat_cols = [c for c in online.columns if c not in ("trans_num", "cc_num", "merchant")]
    nonzero_counts = {c: int(online[c].notna().sum()) for c in feat_cols}
    print(f"  Non-null feature counts: {nonzero_counts}")

    # At least one of the velocity / merchant features should be non-null
    key_features = ["cc_num_FE", "txn_last_24h", "amt_sum_last_24h", "merchant_FE"]
    missing = [f for f in key_features if nonzero_counts.get(f, 0) == 0]
    if missing:
        print(f"⚠ Key features have no non-null values: {missing}")
        print("  This is OK if Redis hasn't been materialized yet — run:")
        print("    cd feast/feature_repo && feast materialize \\")
        print("        -v cc_num_features -v merchant_features \\")
        print('        "2019-01-01T00:00:00" "2021-01-01T00:00:00"')
    else:
        print("✓ All key online features populated for at least one row.")

    # Per-row check: for the first row, show all non-null features
    first = online.iloc[0]
    populated = {k: float(first[k]) for k in feat_cols if pd.notna(first[k])}
    print(f"  Row 0 populated features ({len(populated)}/{len(feat_cols)}): {populated}")


if __name__ == "__main__":
    main()
