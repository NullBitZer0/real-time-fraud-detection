import random
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipelines.prediction_pipeline import PredictionPipeline

TEMPLATES = Path(__file__).parent / "templates"
CSV_PATH = ROOT / "data/raw/creditcard.csv"

app = FastAPI(title="Fraud Detection Demo")

legitimate_examples: list[dict] = []
fraud_buckets: dict[str, list[dict]] = {
    "low": [], "medium-low": [], "medium-high": [], "high": [],
}
pipeline: PredictionPipeline | None = None


@app.on_event("startup")
def startup():
    global legitimate_examples, fraud_buckets, pipeline

    pipeline = PredictionPipeline()

    if not CSV_PATH.exists():
        print(f"CSV not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)

    # Legitimate examples — batch predict for speed
    legit = df[df["Class"] == 0].drop(columns=["Class"]).head(500)
    legit_out = pipeline.predict(legit)
    legitimate_examples.clear()
    for i in range(len(legit)):
        row = legit.iloc[i].to_dict()
        row["_prob"] = round(float(legit_out.iloc[i]["fraud_probability"]), 6)
        row["_pred"] = int(legit_out.iloc[i]["fraud_prediction"])
        legitimate_examples.append(row)
    print(f"Loaded {len(legitimate_examples)} legit examples")

    # Fraud examples — batch predict, bucket by confidence
    fraud = df[df["Class"] == 1].drop(columns=["Class"]).head(200)
    fraud_out = pipeline.predict(fraud)
    for bucket in fraud_buckets.values():
        bucket.clear()
    for i in range(len(fraud)):
        row = fraud.iloc[i].to_dict()
        prob = round(float(fraud_out.iloc[i]["fraud_probability"]), 6)
        row["_prob"] = prob
        row["_pred"] = int(fraud_out.iloc[i]["fraud_prediction"])
        if prob < 0.1:
            fraud_buckets["low"].append(row)
        elif prob < 0.7:
            fraud_buckets["medium-low"].append(row)
        elif prob < 0.9:
            fraud_buckets["medium-high"].append(row)
        else:
            fraud_buckets["high"].append(row)
    print(f"Fraud buckets: low={len(fraud_buckets['low'])} "
          f"med-low={len(fraud_buckets['medium-low'])} "
          f"med-high={len(fraud_buckets['medium-high'])} "
          f"high={len(fraud_buckets['high'])}")


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(TEMPLATES / "index.html")


@app.post("/predict")
async def predict(data: dict):
    if pipeline is None:
        return JSONResponse({"error": "Pipeline not initialized"}, status_code=500)

    required = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    for field in required:
        if field not in data:
            return JSONResponse({"error": f"Missing field: {field}"}, status_code=400)

    try:
        row = {col: float(data[col]) for col in required}
        input_df = pd.DataFrame([row])
        result = pipeline.predict(input_df)

        prob = float(result.iloc[0]["fraud_probability"])
        return {
            "fraud": bool(result.iloc[0]["fraud_prediction"]),
            "probability": round(prob, 6),
            "threshold": pipeline.threshold,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/examples")
async def examples(type: str = Query("legit")):
    if pipeline is None:
        return {}

    if type == "legit":
        if not legitimate_examples:
            return {}
        return random.choice(legitimate_examples)

    bucket_map = {"fraud-low": "low", "fraud-medium-low": "medium-low",
                  "fraud-medium-high": "medium-high", "fraud-high": "high"}

    if type == "fraud-random":
        all_fraud = (fraud_buckets["low"] + fraud_buckets["medium-low"]
                     + fraud_buckets["medium-high"] + fraud_buckets["high"])
        return random.choice(all_fraud) if all_fraud else {}

    bucket = bucket_map.get(type, "high")
    rows = fraud_buckets.get(bucket, fraud_buckets["high"])
    if not rows:
        rows = fraud_buckets["high"]

    return random.choice(rows)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return """<!DOCTYPE html><html><body style="font-family:sans-serif;padding:2rem">
<h1>Dashboard</h1>
<p>MLflow tracking and prediction history coming here.</p>
<a href="/">← Back to demo</a>
</body></html>"""
