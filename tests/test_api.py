"""FastAPI endpoint tests using TestClient (no separate server needed).

Covers the four main endpoints and the Redis decision cache.

Run:
    cd /home/NullbitZer0/projects/real-time-fraud-detection
    python -m tests.test_api
"""
import sys
import time

from fastapi.testclient import TestClient

from api.app import app
from src.components.data_ingestion import read_sparkov_split


def main():
    print("=" * 60)
    print("API TEST — FastAPI TestClient against /predict + /demo/*")
    print("=" * 60)

    results = {"passed": 0, "failed": 0}
    fail_messages = []

    # Use the context-manager form so the lifespan runs (loads the model)
    with TestClient(app) as client:
        # ── /health ────────────────────────────────────────────────────────────
        r = client.get("/health")
        print(f"  /health        → {r.status_code}  {r.json()}")
        if r.status_code == 200 and r.json()["status"] == "ok":
            results["passed"] += 1
        else:
            results["failed"] += 1
            fail_messages.append(f"/health status={r.status_code}")

        if not (r.status_code == 200 and r.json().get("model_loaded")):
            print("⚠ Model not loaded — skipping remaining tests")
            return results

        # ── /predict (single transaction) ──────────────────────────────────────
        df = read_sparkov_split("data/raw/fraudTest.csv").head(1)
        payload = {
            "trans_date_trans_time": str(df["trans_date_trans_time"].iloc[0]),
            "cc_num":    float(df["cc_num"].iloc[0]),
            "merchant":  str (df["merchant"].iloc[0]),
            "category":  str (df["category"].iloc[0]),
            "amt":       float(df["amt"].iloc[0]),
            "lat":       float(df["lat"].iloc[0]),
            "long":      float(df["long"].iloc[0]),
            "merch_lat": float(df["merch_lat"].iloc[0]),
            "merch_long":float(df["merch_long"].iloc[0]),
            "city":      str (df["city"].iloc[0])      if "city"  in df.columns else "",
            "state":     str (df["state"].iloc[0])     if "state" in df.columns else "",
            "job":       str (df["job"].iloc[0])       if "job"   in df.columns else "",
            "dob":       str (df["dob"].iloc[0])       if "dob"   in df.columns else "",
            "zip":       int (df["zip"].iloc[0])       if "zip"   in df.columns else 0,
        }
        r = client.post("/predict", json=payload)
        if r.status_code == 200:
            body = r.json()
            print(f"  /predict       → {r.status_code}  proba={body['fraud_probability']:.4f}  "
                  f"tier={body['tier']}  action={body['action']}  latency={body['latency_ms']:.1f}ms")
            if body["fraud_probability"] < 0 or body["fraud_probability"] > 1:
                results["failed"] += 1
                fail_messages.append("proba out of [0, 1]")
            elif body["tier"] not in (0, 1, 2, 3):
                results["failed"] += 1
                fail_messages.append("tier not in {0,1,2,3}")
            else:
                results["passed"] += 1
        else:
            results["failed"] += 1
            fail_messages.append(f"/predict status={r.status_code} body={r.text[:200]}")

        # ── /demo/decision/{trans_num} (Redis decision cache) ─────────────────
        trans_num = "0f08309c08f6cfcd0a0b8c2b9c40d8a8"  # one from fraudTest
        r = client.get(f"/demo/decision/{trans_num}")
        if r.status_code in (200, 404):
            print(f"  /demo/decision → {r.status_code}  (200=hit, 404=miss or cache off)")
            results["passed"] += 1
        else:
            results["failed"] += 1
            fail_messages.append(f"/demo/decision status={r.status_code}")

        # ── /demo/run-tests (custom mode) ────────────────────────────────────────
        t0 = time.perf_counter()
        r = client.post("/demo/run-tests", json={"mode": "custom", "n_fraud": 50, "n_legit": 50})
        dt = time.perf_counter() - t0
        if r.status_code == 200:
            body = r.json()
            cm = body["confusion_matrix"]
            print(f"  /demo/run-tests (custom) → {r.status_code}  total={body['n_total']}  "
                  f"fraud_rate={body['fraud_rate']:.4f}  macroF1={body['macro_f1']:.4f}  CM={cm}  ({dt:.1f}s)")
            if body["n_total"] != 100:
                results["failed"] += 1
                fail_messages.append(f"n_total != 100 (got {body['n_total']})")
            elif len(body["results"]) != 100:
                results["failed"] += 1
                fail_messages.append(f"results length != 100 (got {len(body['results'])})")
            else:
                results["passed"] += 1
        else:
            results["failed"] += 1
            fail_messages.append(f"/demo/run-tests (custom) status={r.status_code} body={r.text[:200]}")

        # ── /demo/run-tests (real_world mode) ──────────────────────────────────
        t0 = time.perf_counter()
        r = client.post("/demo/run-tests", json={"mode": "real_world", "n_total": 100})
        dt = time.perf_counter() - t0
        if r.status_code == 200:
            body = r.json()
            print(f"  /demo/run-tests (real_world) → {r.status_code}  total={body['n_total']}  "
                  f"fraud={body['n_fraud']}  legit={body['n_legit']}  fraud_rate={body['fraud_rate']:.4f}  ({dt:.1f}s)")
            results["passed"] += 1
        else:
            results["failed"] += 1
            fail_messages.append(f"/demo/run-tests (real_world) status={r.status_code} body={r.text[:200]}")

        # ── /metrics ───────────────────────────────────────────────────────────
        r = client.get("/metrics")
        if r.status_code == 200:
            body = r.json()
            print(f"  /metrics       → {r.status_code}  total={body['total_transactions']}  "
                  f"fraud={body['total_fraud']}  avg_latency={body['avg_latency_ms']:.1f}ms")
            results["passed"] += 1
        else:
            results["failed"] += 1
            fail_messages.append(f"/metrics status={r.status_code}")

        # ── /readyz ────────────────────────────────────────────────────────────
        r = client.get("/readyz")
        if r.status_code == 200:
            body = r.json()
            print(f"  /readyz        → {r.status_code}  ready={body['ready']}  checks={list(body['checks'].keys())}")
            if body.get("ready") is not None:
                results["passed"] += 1
            else:
                results["failed"] += 1
                fail_messages.append("/readyz missing 'ready' field")
        else:
            results["failed"] += 1
            fail_messages.append(f"/readyz status={r.status_code}")

        # ── /metrics/prom ──────────────────────────────────────────────────────
        r = client.get("/metrics/prom")
        if r.status_code == 200 and "fraud_predictions_total" in r.text:
            print(f"  /metrics/prom  → {r.status_code}  bytes={len(r.text)}  has_counter=True")
            results["passed"] += 1
        else:
            results["failed"] += 1
            fail_messages.append(f"/metrics/prom status={r.status_code}")

        # ── /audit/recent ──────────────────────────────────────────────────────
        r = client.get("/audit/recent?n=10")
        if r.status_code == 200:
            body = r.json()
            assert isinstance(body, list), f"expected list, got {type(body)}"
            n = len(body)
            print(f"  /audit/recent  → {r.status_code}  rows={n}  latest={body[0]['trans_num'][:12] if n else '–'}…  tier={body[0]['tier'] if n else '–'}")
            if n == 0:
                # Acceptable if decision_log is empty (we ran /predict + /demo/run-100 above though)
                print("  ⚠ audit log is empty — Postgres may be down")
            results["passed"] += 1
        elif r.status_code == 503:
            # Acceptable: Postgres unavailable in the test environment
            print("  /audit/recent  → 503  (Postgres unavailable — acceptable in CI)")
            results["passed"] += 1
        else:
            results["failed"] += 1
            fail_messages.append(f"/audit/recent status={r.status_code} body={r.text[:200]}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("-" * 60)
    print(f"PASS: {results['passed']}    FAIL: {results['failed']}")
    if fail_messages:
        for msg in fail_messages:
            print(f"  ✗ {msg}")
        sys.exit(1)
    print("✓ All API tests passed.")


if __name__ == "__main__":
    main()
