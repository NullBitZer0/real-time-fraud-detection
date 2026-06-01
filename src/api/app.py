import sys, random
from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipelines.prediction_pipeline import PredictionPipeline

TEMPLATES = Path(__file__).parent / "templates"
CSV_PATH = ROOT / "data/raw/creditcard.csv"

app = FastAPI(title="Fraud Detection Demo")

legitimate_examples: list[dict] = []
fraud_examples: list[dict] = []
pipeline: PredictionPipeline | None = None


@app.on_event("startup")
def startup():
    global legitimate_examples, fraud_examples, pipeline

    pipeline = PredictionPipeline()
    print(f"Pipeline ready — threshold={pipeline.threshold}")

    print(f"CSV_PATH: {CSV_PATH}  exists: {CSV_PATH.exists()}")
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        print(f"CSV loaded: {df.shape}")
        legit = df[df["Class"] == 0].drop(columns=["Class"]).head(500)
        legitimate_examples.extend(legit.to_dict(orient="records"))
        fraud = df[df["Class"] == 1].drop(columns=["Class"]).head(50)
        fraud_examples.extend(fraud.to_dict(orient="records"))
        print(f"Loaded {len(legitimate_examples)} legit + {len(fraud_examples)} fraud examples")


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

        return {
            "fraud": bool(result.iloc[0]["fraud_prediction"]),
            "probability": round(float(result.iloc[0]["fraud_probability"]), 4),
            "threshold": pipeline.threshold,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/examples")
async def examples():
    if not legitimate_examples or not fraud_examples:
        return {"legitimate": {}, "fraud": {}}
    return {
        "legitimate": random.choice(legitimate_examples) if legitimate_examples else {},
        "fraud": random.choice(fraud_examples) if fraud_examples else {},
    }


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return """<!DOCTYPE html><html><body style="font-family:sans-serif;padding:2rem">
<h1>Dashboard</h1>
<p>MLflow tracking and prediction history coming here.</p>
<a href="/">← Back to demo</a>
</body></html>"""
