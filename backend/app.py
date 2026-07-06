"""
app.py — FastAPI backend for URMC LOS & Cost Decision Support
-------------------------------------------------------------
Loads the trained models and serves:
  GET  /api/config    -> UI dropdown options, facilities, default threshold
  GET  /api/metrics   -> cohort stats, model metrics, chart data
  POST /api/predict   -> predicted LOS, predicted cost, high-cost flag, cost drivers
  GET  /              -> the dashboard (static frontend)

Run:  uvicorn app:app --reload --port 8000
"""
import os, json
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
FRONTEND = os.path.join(HERE, "..", "frontend")

# ---- load artifacts once at startup ----
los_model = joblib.load(os.path.join(ART, "los_model.joblib"))
cost_model = joblib.load(os.path.join(ART, "cost_model.joblib"))
encoder = joblib.load(os.path.join(ART, "encoder.joblib"))
config = json.load(open(os.path.join(ART, "config.json")))
metrics = json.load(open(os.path.join(ART, "metrics.json")))

CAT = config["cat_features"]
BASELINE = config["baseline"]

app = FastAPI(title="URMC LOS & Cost Decision Support")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictIn(BaseModel):
    age: str
    admission: str
    severity: str
    drg: str
    payer: str
    facility: Optional[str] = None
    threshold: int = 40000


def _row_to_X(row: dict):
    """Build a single-row feature matrix in the exact CAT order, then encode."""
    df = pd.DataFrame([[row.get(c, BASELINE[c]) for c in CAT]], columns=CAT)
    return encoder.transform(df)


def _predict_cost(row: dict) -> float:
    return float(cost_model.predict(_row_to_X(row))[0])


@app.get("/api/config")
def get_config():
    return {
        "ui_options": config["ui_options"],
        "facilities": config["facilities"],
        "high_cost_default": config["high_cost_default"],
    }


@app.get("/api/metrics")
def get_metrics():
    return metrics


@app.post("/api/predict")
def predict(inp: PredictIn):
    # map UI fields -> full feature row (baseline for everything not set by the UI)
    row = dict(BASELINE)
    row["Age Group"] = inp.age
    row["Type of Admission"] = inp.admission
    row["APR Severity of Illness Description"] = inp.severity
    row["APR DRG Description"] = inp.drg
    row["Payment Typology 1"] = inp.payer
    if inp.facility and inp.facility != "All facilities":
        row["Facility Name"] = inp.facility

    X = _row_to_X(row)
    los = float(los_model.predict(X)[0])
    cost = float(cost_model.predict(X)[0])
    flag = cost >= inp.threshold

    # ---- local cost drivers via ablation (reset each UI feature to baseline) ----
    ui_map = {
        "Severity": "APR Severity of Illness Description",
        "Condition (DRG)": "APR DRG Description",
        "Admission type": "Type of Admission",
        "Age group": "Age Group",
        "Payer": "Payment Typology 1",
    }
    base_cost = _predict_cost(dict(BASELINE))
    drivers = []
    for label, feat in ui_map.items():
        ablated = dict(row)
        ablated[feat] = BASELINE[feat]
        contribution = cost - _predict_cost(ablated)
        drivers.append({"label": label, "value": round(contribution)})
    drivers.sort(key=lambda d: abs(d["value"]), reverse=True)

    return {
        "los": round(los, 1),
        "cost": round(cost),
        "high_cost": flag,
        "base_cost": round(base_cost),
        "drivers": drivers,
    }


# ---- serve frontend ----
if os.path.isdir(FRONTEND):
    @app.get("/")
    def index():
        return FileResponse(os.path.join(FRONTEND, "index.html"))
    app.mount("/", StaticFiles(directory=FRONTEND), name="static")
