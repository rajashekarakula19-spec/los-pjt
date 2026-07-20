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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional
from features import add_derived_to_row

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
COST_TARGET = config.get("cost_target", "raw")

app = FastAPI(title="URMC LOS & Cost Decision Support")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictIn(BaseModel):
    age: str
    admission: str
    severity: str
    drg: str
    payer: str
    diagnosis: Optional[str] = None
    med_surg: Optional[str] = None
    zip_code: Optional[str] = None
    payer2: Optional[str] = None
    hospital_county: Optional[str] = None
    facility: Optional[str] = None
    threshold: int = Field(default=40000, ge=1000, le=500000)


def _row_to_X(row: dict):
    """Build a single-row feature matrix in the exact CAT order, then encode."""
    row = add_derived_to_row(row)
    df = pd.DataFrame([[row.get(c, BASELINE[c]) for c in CAT]], columns=CAT)
    return encoder.transform(df)


def _predict_cost(row: dict) -> float:
    pred = float(cost_model.predict(_row_to_X(row))[0])
    if COST_TARGET == "log1p":
        return max(float(np.expm1(pred)), 0.0)
    return pred


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


@app.get("/api/opportunities")
def get_opportunities(
    facility: Optional[str] = None,
    opportunity_id: Optional[int] = None,
):
    """Return case-mix benchmark results without the legacy chart payload."""
    analysis = metrics.get("opportunity")
    if not analysis:
        raise HTTPException(status_code=503, detail="Opportunity analysis artifact is unavailable")
    facilities = analysis["facilities"]
    opportunities = analysis["opportunities"]
    cases = analysis["case_samples"]
    if facility:
        facilities = [r for r in facilities if r["Facility Name"] == facility]
        opportunities = [r for r in opportunities if r["Facility Name"] == facility]
        cases = [r for r in cases if r["facility"] == facility]
    if opportunity_id is not None:
        if opportunity_id < 0:
            raise HTTPException(status_code=422, detail="Opportunity id must be non-negative")
        if opportunity_id >= len(analysis["opportunities"]):
            raise HTTPException(status_code=404, detail="Opportunity not found")
        opportunities = [analysis["opportunities"][opportunity_id]]
        cases = [r for r in analysis["case_samples"]
                 if r["opportunity_id"] == opportunity_id]
    return {
        "method": analysis["method"],
        "executive": analysis["executive"],
        "validation": analysis["validation"],
        "facilities": facilities,
        "opportunities": opportunities,
        "case_samples": cases,
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": True,
            "opportunity_analysis": "opportunity" in metrics}


@app.post("/api/predict")
def predict(inp: PredictIn):
    # map UI fields -> full feature row (baseline for everything not set by the UI)
    row = dict(BASELINE)
    row["Age Group"] = inp.age
    row["Type of Admission"] = inp.admission
    row["APR Severity of Illness Description"] = inp.severity
    row["APR DRG Description"] = inp.drg
    row["Payment Typology 1"] = inp.payer
    if inp.diagnosis:
        row["CCSR Diagnosis Code"] = inp.diagnosis
    if inp.med_surg:
        row["APR Medical Surgical Description"] = inp.med_surg
    if inp.zip_code:
        row["Zip Code"] = inp.zip_code
    if inp.payer2:
        row["Payment Typology 2"] = inp.payer2
    if inp.hospital_county:
        row["Hospital County"] = inp.hospital_county
    if inp.facility and inp.facility != "All facilities":
        row["Facility Name"] = inp.facility

    X = _row_to_X(row)
    los = max(float(los_model.predict(X)[0]), 0.5)
    cost = _predict_cost(row)
    flag = cost >= inp.threshold

    # ---- local cost drivers via ablation (reset each UI feature to baseline) ----
    ui_map = {
        "Severity": "APR Severity of Illness Description",
        "Condition (DRG)": "APR DRG Description",
        "Admission type": "Type of Admission",
        "Age group": "Age Group",
        "Payer": "Payment Typology 1",
        "Diagnosis": "CCSR Diagnosis Code",
        "Medical/surgical": "APR Medical Surgical Description",
        "ZIP": "Zip Code",
        "Secondary payer": "Payment Typology 2",
        "County": "Hospital County",
        "Facility": "Facility Name",
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
