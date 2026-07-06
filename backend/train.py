"""
train.py — URMC Finance Decision Support: LOS & Cost models
-----------------------------------------------------------
Trains two gradient-boosting regressors on NY SPARCS 2024 (Finger Lakes):
  1. Length-of-stay regressor  (target: Length of Stay, days)
  2. Cost-of-care regressor     (target: Total Costs, USD)

Uses ONLY admission-time features (no leakage from discharge outcomes).
Exports models + encoder + config + metrics to backend/artifacts/.

Run:  python train.py /path/to/SPARCS.csv
"""
import sys, json, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
os.makedirs(ART, exist_ok=True)

CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data", "sparcs.csv")

# ---- admission-time features (known at/near admission; NO discharge leakage) ----
CAT_FEATURES = [
    "Age Group", "Gender", "Race", "Ethnicity", "Type of Admission",
    "APR Severity of Illness Description", "APR Risk of Mortality",
    "APR MDC Description", "APR DRG Description",
    "Payment Typology 1", "Emergency Department Indicator", "Facility Name",
]
# fields the UI estimator lets the user set (subset used for driver panel + dropdowns)
UI_FEATURES = ["Age Group", "Type of Admission",
               "APR Severity of Illness Description", "APR DRG Description",
               "Payment Typology 1"]

HIGH_COST_DEFAULT = 40000  # default $ threshold for the high-cost flag


def load_and_clean(path):
    print(f"Loading {path} ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"  raw rows: {len(df):,}")

    # length of stay: '120+' -> 120, coerce int
    df["LOS"] = pd.to_numeric(
        df["Length of Stay"].astype(str).str.replace("+", "", regex=False),
        errors="coerce")
    # costs already numeric-ish
    df["COST"] = pd.to_numeric(df["Total Costs"].astype(str)
                               .str.replace(",", "", regex=False).str.replace("$", "", regex=False),
                               errors="coerce")

    df = df.dropna(subset=["LOS", "COST"])
    df = df[(df["LOS"] >= 1) & (df["LOS"] <= 120)]
    df = df[(df["COST"] > 0) & (df["COST"] <= df["COST"].quantile(0.999))]  # drop extreme cost outliers

    # group rare DRGs so categorical cardinality stays < 255 (HGB native-cat limit)
    top_drg = df["APR DRG Description"].value_counts().nlargest(180).index
    df["APR DRG Description"] = df["APR DRG Description"].where(
        df["APR DRG Description"].isin(top_drg), other="Other")

    for c in CAT_FEATURES:
        df[c] = df[c].fillna("Unknown").astype(str)

    print(f"  clean rows: {len(df):,}")
    return df


def main():
    df = load_and_clean(CSV)

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X = enc.fit_transform(df[CAT_FEATURES])
    cat_idx = list(range(len(CAT_FEATURES)))  # all features are categorical

    y_los = df["LOS"].values
    y_cost = df["COST"].values

    Xtr, Xte, ylos_tr, ylos_te, ycost_tr, ycost_te = train_test_split(
        X, y_los, y_cost, test_size=0.2, random_state=42)

    common = dict(categorical_features=cat_idx, max_iter=300,
                  learning_rate=0.08, max_depth=8, random_state=42, early_stopping=True)

    print("Training LOS regressor ...")
    los_model = HistGradientBoostingRegressor(**common).fit(Xtr, ylos_tr)
    print("Training cost regressor ...")
    cost_model = HistGradientBoostingRegressor(**common).fit(Xtr, ycost_tr)

    # ---- metrics ----
    los_pred = los_model.predict(Xte)
    cost_pred = cost_model.predict(Xte)

    los_metrics = {
        "mae": round(float(mean_absolute_error(ylos_te, los_pred)), 2),
        "r2": round(float(r2_score(ylos_te, los_pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(ylos_te, los_pred))), 2),
    }
    cost_metrics = {
        "mae": round(float(mean_absolute_error(ycost_te, cost_pred)), 0),
        "r2": round(float(r2_score(ycost_te, cost_pred)), 3),
    }
    # high-cost classification (derived from cost regressor vs actual, at default threshold)
    thr = HIGH_COST_DEFAULT
    true_hi = ycost_te >= thr
    pred_hi = cost_pred >= thr
    tp = int(((pred_hi) & (true_hi)).sum()); fp = int(((pred_hi) & (~true_hi)).sum())
    fn = int(((~pred_hi) & (true_hi)).sum())
    hi_metrics = {
        "threshold": thr,
        "recall": round(tp / (tp + fn) if (tp + fn) else 0, 3),
        "precision": round(tp / (tp + fp) if (tp + fp) else 0, 3),
        "base_rate": round(float(true_hi.mean()), 3),
    }

    # ---- chart data for the frontend ----
    # cost histogram bins
    hist_counts, hist_edges = np.histogram(df["COST"], bins=22, range=(0, df["COST"].quantile(0.98)))
    # scatter sample (predicted vs actual LOS)
    n_sample = min(400, len(ylos_te))
    idx = np.random.RandomState(0).choice(len(ylos_te), n_sample, replace=False)
    scatter = [[round(float(ylos_te[i]), 1), round(float(los_pred[i]), 1)] for i in idx]
    # (actual_cost, predicted_cost) pairs for live threshold recall/precision
    n_cost = min(2000, len(ycost_te))
    cidx = np.random.RandomState(1).choice(len(ycost_te), n_cost, replace=False)
    cost_pairs = [[int(ycost_te[i]), int(cost_pred[i])] for i in cidx]
    # mean LOS by severity
    sev_order = ["Minor", "Moderate", "Major", "Extreme"]
    sev_los = {s: round(float(df.loc[df["APR Severity of Illness Description"] == s, "LOS"].mean()), 1)
               for s in sev_order if (df["APR Severity of Illness Description"] == s).any()}

    cohort = {
        "n": int(len(df)),
        "median_los": round(float(df["LOS"].median()), 1),
        "mean_los": round(float(df["LOS"].mean()), 1),
        "median_cost": int(df["COST"].median()),
        "mean_cost": int(df["COST"].mean()),
        "p95_cost": int(df["COST"].quantile(0.95)),
        "high_cost_rate": round(float((df["COST"] >= thr).mean()), 3),
    }

    # dropdown options for the UI (real values from data)
    def opts(col, limit=None, order=None):
        vc = df[col].value_counts()
        vals = list(vc.index)
        if order:
            vals = [v for v in order if v in vals] + [v for v in vals if v not in order]
        return vals[:limit] if limit else vals

    ui_options = {
        "Age Group": opts("Age Group", order=["0 to 17", "18 to 29", "30 to 49", "50 to 69", "70 or Older"]),
        "Type of Admission": opts("Type of Admission"),
        "APR Severity of Illness Description": ["Minor", "Moderate", "Major", "Extreme"],
        "APR DRG Description": opts("APR DRG Description", limit=25),
        "Payment Typology 1": opts("Payment Typology 1", limit=8),
    }
    facilities = opts("Facility Name", limit=8)

    # grouped cohort stats so KPI filters reflect real data
    def group_stats(col, limit=8):
        out = {}
        vc = df[col].value_counts().nlargest(limit)
        for val in vc.index:
            sub = df[df[col] == val]
            out[str(val)] = {
                "n": int(len(sub)),
                "median_los": round(float(sub["LOS"].median()), 1),
                "median_cost": int(sub["COST"].median()),
                "high_cost_rate": round(float((sub["COST"] >= thr).mean()), 3),
            }
        return out

    cohort_by = {
        "facility": group_stats("Facility Name"),
        "drg": group_stats("APR DRG Description", limit=12),
        "payer": group_stats("Payment Typology 1"),
        "admission": group_stats("Type of Admission"),
    }

    # baseline row (modes) for ablation-based local cost drivers in the API
    baseline = {c: df[c].mode().iloc[0] for c in CAT_FEATURES}

    # ---- save ----
    joblib.dump(los_model, os.path.join(ART, "los_model.joblib"))
    joblib.dump(cost_model, os.path.join(ART, "cost_model.joblib"))
    joblib.dump(enc, os.path.join(ART, "encoder.joblib"))

    config = {
        "cat_features": CAT_FEATURES,
        "ui_features": UI_FEATURES,
        "ui_options": ui_options,
        "facilities": facilities,
        "baseline": baseline,
        "high_cost_default": HIGH_COST_DEFAULT,
    }
    json.dump(config, open(os.path.join(ART, "config.json"), "w"), indent=2)

    metrics = {
        "los": los_metrics, "cost": cost_metrics, "high_cost": hi_metrics,
        "cohort": cohort, "cohort_by": cohort_by, "sev_los": sev_los,
        "scatter": scatter, "cost_pairs": cost_pairs,
        "hist": {"counts": hist_counts.tolist(), "edges": [float(e) for e in hist_edges]},
        "data_source": "NY SPARCS 2024, Finger Lakes region",
    }
    json.dump(metrics, open(os.path.join(ART, "metrics.json"), "w"), indent=2)

    print("\n=== RESULTS ===")
    print("LOS :", los_metrics)
    print("Cost:", cost_metrics)
    print("High-cost:", hi_metrics)
    print("Cohort:", cohort)
    print("Artifacts written to", ART)


if __name__ == "__main__":
    main()
