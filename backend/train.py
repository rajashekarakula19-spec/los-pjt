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
import sys, json, os, math
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
from features import DERIVED_FEATURES, add_derived_features

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
os.makedirs(ART, exist_ok=True)

CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data", "sparcs.csv")

# ---- admission-time features (known at/near admission; NO discharge leakage) ----
RAW_CAT_FEATURES = [
    "Hospital County", "Zip Code",
    "Age Group", "Gender", "Race", "Ethnicity", "Type of Admission",
    "CCSR Diagnosis Code",
    "APR Severity of Illness Description", "APR Risk of Mortality",
    "APR MDC Description", "APR DRG Description", "APR Medical Surgical Description",
    "Payment Typology 1", "Payment Typology 2",
    "Emergency Department Indicator", "Facility Name",
]
CAT_FEATURES = RAW_CAT_FEATURES + DERIVED_FEATURES
# fields the UI estimator lets the user set (subset used for driver panel + dropdowns)
UI_FEATURES = ["Age Group", "Type of Admission",
               "APR Severity of Illness Description", "APR DRG Description",
               "Payment Typology 1", "CCSR Diagnosis Code",
               "APR Medical Surgical Description", "Zip Code"]

# Case-mix features used to establish a neutral expected LOS/cost benchmark.
# Facility and socioeconomic/geographic fields are intentionally excluded: the
# benchmark should expose site variation rather than learn it away, and should
# not normalize differences associated with payer, race, ethnicity, or ZIP.
BENCHMARK_FEATURES = [
    "Age Group", "Gender", "Type of Admission", "CCSR Diagnosis Code",
    "APR Severity of Illness Description", "APR Risk of Mortality",
    "APR MDC Description", "APR DRG Description",
    "APR Medical Surgical Description", "Emergency Department Indicator",
    "DRG Severity Group", "Diagnosis Severity Group", "Age Severity Group",
    "Admission Pathway", "Clinical Service Group", "Population Group",
    "Complexity Score",
]

HIGH_COST_DEFAULT = 40000  # default $ threshold for the high-cost flag

MISSING_MARKERS = {
    "", " ", "nan", "none", "null", "na", "n/a", "not available",
    "not applicable", "unknown", "unk",
}


def clean_category(value):
    """Normalize categorical labels while preserving meaningful SPARCS wording."""
    if pd.isna(value):
        return "Unknown"
    cleaned = " ".join(str(value).strip().split())
    if cleaned.lower() in MISSING_MARKERS:
        return "Unknown"
    return cleaned


def parse_money(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def parse_los(series):
    return pd.to_numeric(
        series.astype(str).str.replace("+", "", regex=False).str.strip(),
        errors="coerce",
    )


def load_and_clean(path):
    print(f"Loading {path} ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"  raw rows: {len(df):,}")

    raw_n = len(df)
    df["LOS"] = parse_los(df["Length of Stay"])
    df["COST"] = parse_money(df["Total Costs"])

    before = len(df)
    df = df.dropna(subset=["LOS", "COST"])
    print(f"  dropped missing LOS/cost: {before - len(df):,}")

    before = len(df)
    df = df[(df["LOS"] >= 1) & (df["LOS"] <= 120)]
    print(f"  dropped invalid LOS: {before - len(df):,}")

    before = len(df)
    df = df[df["COST"] > 0]
    print(f"  dropped non-positive cost: {before - len(df):,}")

    # Keep the central population for training; very small/large costs often add noise.
    cost_low = df["COST"].quantile(0.001)
    cost_high = df["COST"].quantile(0.995)
    before = len(df)
    df = df[(df["COST"] >= cost_low) & (df["COST"] <= cost_high)]
    print(f"  dropped cost outliers outside p0.1-p99.5: {before - len(df):,}")

    for c in RAW_CAT_FEATURES:
        df[c] = df[c].map(clean_category)

    # Remove very tiny categories for fields where they are usually data noise.
    for c in ["Type of Admission", "Payment Typology 1", "Payment Typology 2",
              "Facility Name", "Zip Code", "Hospital County"]:
        counts = df[c].value_counts()
        rare = counts[counts < 25].index
        df[c] = df[c].where(~df[c].isin(rare), "Other")

    # group rare DRGs so categorical cardinality stays < 255 (HGB native-cat limit)
    top_drg = df["APR DRG Description"].value_counts().nlargest(220).index
    df["APR DRG Description"] = df["APR DRG Description"].where(
        df["APR DRG Description"].isin(top_drg), other="Other")
    top_dx = df["CCSR Diagnosis Code"].value_counts().nlargest(220).index
    df["CCSR Diagnosis Code"] = df["CCSR Diagnosis Code"].where(
        df["CCSR Diagnosis Code"].isin(top_dx), other="Other")

    df = add_derived_features(df)
    # HistGradientBoosting supports at most 255 categories per categorical
    # feature. Retain the most common interaction groups and pool the rest.
    for c in ["DRG Severity Group", "Diagnosis Severity Group"]:
        top = df[c].value_counts().nlargest(220).index
        df[c] = df[c].where(df[c].isin(top), "Other")

    print(f"  clean rows: {len(df):,}")
    print(f"  retained rows: {len(df) / raw_n:.1%}")
    return df


def _benchmark_model(cat_count, seed):
    return HistGradientBoostingRegressor(
        categorical_features=list(range(cat_count)), max_iter=180,
        learning_rate=0.08, max_depth=7, min_samples_leaf=35,
        l2_regularization=0.2, random_state=seed, early_stopping=True,
    )


def build_opportunity_analysis(df):
    """Create out-of-fold, case-mix-adjusted facility opportunity metrics."""
    print("Building 3-fold out-of-fold case-mix benchmarks ...")
    bench_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1)
    X = bench_encoder.fit_transform(df[BENCHMARK_FEATURES])
    y_los = df["LOS"].to_numpy(dtype=float)
    y_cost = df["COST"].to_numpy(dtype=float)
    y_cost_log = np.log1p(y_cost)
    expected_los = np.zeros(len(df), dtype=float)
    expected_cost = np.zeros(len(df), dtype=float)

    service_lines = df["Service Line"].astype(str).to_numpy()
    service_line_sizes = {}
    for line_number, service_line in enumerate(sorted(np.unique(service_lines)), start=1):
        line_idx = np.flatnonzero(service_lines == service_line)
        service_line_sizes[service_line] = int(len(line_idx))
        print(f"  service line {line_number}: {service_line} ({len(line_idx):,} records)")
        folds = KFold(n_splits=3, shuffle=True, random_state=2024)
        for fold, (local_train, local_test) in enumerate(folds.split(line_idx), start=1):
            train_idx, test_idx = line_idx[local_train], line_idx[local_test]
            seed = 1000 * line_number + fold
            los_model = _benchmark_model(len(BENCHMARK_FEATURES), seed)
            cost_model = _benchmark_model(len(BENCHMARK_FEATURES), seed + 500)
            los_model.fit(X[train_idx], y_los[train_idx])
            cost_model.fit(X[train_idx], y_cost_log[train_idx])
            expected_los[test_idx] = np.maximum(los_model.predict(X[test_idx]), 0.5)
            expected_cost[test_idx] = np.maximum(
                np.expm1(cost_model.predict(X[test_idx])), 0)

    # Log-cost regression targets the conditional median. Calibrate both OOF
    # series so overall expected totals equal actual totals; group differences
    # therefore represent relative opportunity, not global model bias.
    expected_los *= y_los.sum() / expected_los.sum()
    expected_cost *= y_cost.sum() / expected_cost.sum()
    residual_los = y_los - expected_los
    residual_cost = y_cost - expected_cost

    work = df[["Facility Name", "APR DRG Description", "Type of Admission", "Service Line",
               "Age Group", "APR Severity of Illness Description"]].copy()
    work["actual_los"] = y_los
    work["expected_los"] = expected_los
    work["actual_cost"] = y_cost
    work["expected_cost"] = expected_cost
    work["los_residual"] = residual_los
    work["cost_residual"] = residual_cost

    def summarize(group, dimensions, min_cases):
        rows = []
        grouped = work.groupby(dimensions, observed=True, dropna=False)
        for keys, sub in grouped:
            if len(sub) < min_cases:
                continue
            if not isinstance(keys, tuple):
                keys = (keys,)
            n = len(sub)
            los_diff = float(sub["los_residual"].mean())
            cost_diff = float(sub["cost_residual"].mean())
            los_se = float(sub["los_residual"].std(ddof=1) / np.sqrt(n))
            cost_se = float(sub["cost_residual"].std(ddof=1) / np.sqrt(n))
            los_low, los_high = los_diff - 1.96 * los_se, los_diff + 1.96 * los_se
            cost_low, cost_high = cost_diff - 1.96 * cost_se, cost_diff + 1.96 * cost_se

            # Sensitivity analysis: remove the most extreme 1% on each side.
            # Scale the trimmed mean back to full group volume so it is directly
            # comparable with the raw net gap without claiming literal savings.
            los_q = sub["los_residual"].quantile([0.01, 0.99])
            cost_q = sub["cost_residual"].quantile([0.01, 0.99])
            robust_los = sub["los_residual"].between(los_q.iloc[0], los_q.iloc[1])
            robust_cost = sub["cost_residual"].between(cost_q.iloc[0], cost_q.iloc[1])
            robust_los_diff = float(sub.loc[robust_los, "los_residual"].mean())
            robust_cost_diff = float(sub.loc[robust_cost, "cost_residual"].mean())
            robust_net_cost = robust_cost_diff * n

            positive_cost = sub.loc[sub["cost_residual"] > 0, "cost_residual"].sort_values(ascending=False)
            positive_total = float(positive_cost.sum())
            top_n = max(1, int(np.ceil(n * 0.10)))
            top_share = float(positive_cost.head(top_n).sum() / positive_total) if positive_total else 0.0
            raw_net_cost = float(sub["cost_residual"].sum())
            robust_ratio = (robust_net_cost / raw_net_cost) if raw_net_cost > 0 else 0.0
            if top_share >= 0.65 or robust_ratio < 0.5:
                signal_pattern = "outlier-concentrated"
            elif top_share <= 0.45 and robust_cost_diff > 0:
                signal_pattern = "broad-based"
            else:
                signal_pattern = "mixed"

            confidence_bound_score = max(cost_low, 0) * n
            robust_score = max(robust_net_cost, 0)
            los_z = los_diff / los_se if los_se > 0 else 0.0
            cost_z = cost_diff / cost_se if cost_se > 0 else 0.0
            los_p = 0.5 * math.erfc(los_z / math.sqrt(2))
            cost_p = 0.5 * math.erfc(cost_z / math.sqrt(2))
            combined_p = min(min(los_p, cost_p) * 2, 1.0)
            row = {dimensions[i]: str(keys[i]) for i in range(len(dimensions))}
            row.update({
                "n": int(n),
                "actual_los": round(float(sub["actual_los"].mean()), 2),
                "expected_los": round(float(sub["expected_los"].mean()), 2),
                "los_difference": round(los_diff, 2),
                "los_ci": [round(los_low, 2), round(los_high, 2)],
                "net_bed_days": round(float(sub["los_residual"].sum())),
                "actual_cost": round(float(sub["actual_cost"].mean())),
                "expected_cost": round(float(sub["expected_cost"].mean())),
                "cost_difference": round(cost_diff),
                "cost_ci": [round(cost_low), round(cost_high)],
                "net_cost_difference": round(raw_net_cost),
                "robust_los_difference": round(robust_los_diff, 2),
                "robust_net_bed_days": round(robust_los_diff * n),
                "robust_cost_difference": round(robust_cost_diff),
                "robust_net_cost_difference": round(robust_net_cost),
                "top_10_positive_cost_share": round(top_share, 3),
                "signal_pattern": signal_pattern,
                "p_value": round(combined_p, 8),
                "statistically_higher": bool(los_low > 0 or cost_low > 0),
                "confidence": "high" if n >= 500 else "moderate" if n >= 200 else "directional",
                "opportunity_score": round(min(confidence_bound_score, robust_score)),
            })
            rows.append(row)
        return rows

    def apply_fdr(rows):
        """Benjamini-Hochberg correction across one comparison family."""
        if not rows:
            return rows
        order = sorted(range(len(rows)), key=lambda i: rows[i]["p_value"])
        adjusted = [1.0] * len(rows)
        running = 1.0
        total = len(rows)
        for rank_index in range(total - 1, -1, -1):
            original_index = order[rank_index]
            rank = rank_index + 1
            running = min(running, rows[original_index]["p_value"] * total / rank)
            adjusted[original_index] = min(running, 1.0)
        for row, q_value in zip(rows, adjusted):
            row["q_value"] = round(q_value, 8)
            row["fdr_significant"] = bool(q_value <= 0.05)
        return rows

    facilities = apply_fdr(summarize(work, ["Facility Name"], 100))
    facility_drgs = summarize(work, ["Facility Name", "APR DRG Description"], 100)
    facility_service_lines = summarize(work, ["Facility Name", "Service Line"], 100)
    # Admission type substantially overlaps the service-line definition (for
    # example, Emergency vs Emergency medical). Exclude it from the discovery
    # family to avoid presenting duplicate opportunities as separate findings.
    candidates = apply_fdr(facility_drgs + facility_service_lines)
    opportunities = [r for r in candidates
                     if r["fdr_significant"] and r["statistically_higher"]
                     and r["opportunity_score"] > 0]
    opportunities.sort(key=lambda r: r["opportunity_score"], reverse=True)
    opportunities = opportunities[:40]

    case_samples = []
    for opportunity_id, item in enumerate(opportunities):
        mask = work["Facility Name"].eq(item["Facility Name"])
        if "APR DRG Description" in item:
            mask &= work["APR DRG Description"].eq(item["APR DRG Description"])
        if "Type of Admission" in item:
            mask &= work["Type of Admission"].eq(item["Type of Admission"])
        if "Service Line" in item:
            mask &= work["Service Line"].eq(item["Service Line"])
        sample = work.loc[mask].nlargest(5, "cost_residual")
        for idx, row in sample.iterrows():
            case_samples.append({
                "opportunity_id": opportunity_id,
                "facility": str(row["Facility Name"]),
                "drg": str(row["APR DRG Description"]),
                "admission": str(row["Type of Admission"]),
                "age": str(row["Age Group"]),
                "severity": str(row["APR Severity of Illness Description"]),
                "actual_los": round(float(row["actual_los"]), 1),
                "expected_los": round(float(row["expected_los"]), 1),
                "actual_cost": round(float(row["actual_cost"])),
                "expected_cost": round(float(row["expected_cost"])),
            })

    facilities.sort(key=lambda r: r["net_cost_difference"], reverse=True)
    total_positive_bed_days = round(sum(max(r["net_bed_days"], 0) for r in facilities))
    total_positive_cost = round(sum(max(r["net_cost_difference"], 0) for r in facilities))
    return {
        "method": {
            "name": "Out-of-fold case-mix benchmark",
            "folds": 3,
            "features": BENCHMARK_FEATURES,
            "excluded": ["Facility Name", "Hospital County", "Zip Code", "Race",
                         "Ethnicity", "Payment Typology 1", "Payment Typology 2"],
            "minimum_group_cases": 100,
            "confidence_level": 0.95,
            "service_line_models": service_line_sizes,
            "multiple_comparison_control": "Benjamini-Hochberg FDR at q <= 0.05",
            "robust_trim": "1st-99th percentile residuals retained",
            "concentration_metric": "Share of positive cost difference from top 10% of cases",
            "interpretation": "Differences are investigation signals, not proven waste or savings.",
        },
        "executive": {
            "discharges": int(len(work)),
            "actual_bed_days": round(float(y_los.sum())),
            "expected_bed_days": round(float(expected_los.sum())),
            "actual_cost": round(float(y_cost.sum())),
            "expected_cost": round(float(expected_cost.sum())),
            "positive_facility_bed_day_gap": total_positive_bed_days,
            "positive_facility_cost_gap": total_positive_cost,
            "facilities_review": sum(r["fdr_significant"] and r["statistically_higher"] for r in facilities),
            "opportunities_review": len(opportunities),
        },
        "validation": {
            "los_mae": round(float(mean_absolute_error(y_los, expected_los)), 2),
            "los_r2": round(float(r2_score(y_los, expected_los)), 3),
            "cost_mae": round(float(mean_absolute_error(y_cost, expected_cost))),
            "cost_r2": round(float(r2_score(y_cost, expected_cost)), 3),
            "aggregate_calibration": "Overall expected totals calibrated to actual totals",
        },
        "facilities": facilities,
        "opportunities": opportunities,
        "case_samples": case_samples,
    }


def main():
    df = load_and_clean(CSV)
    opportunity = build_opportunity_analysis(df)

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X = enc.fit_transform(df[CAT_FEATURES])
    cat_idx = list(range(len(CAT_FEATURES)))  # all features are categorical

    y_los = df["LOS"].values
    y_cost = df["COST"].values
    y_cost_log = np.log1p(y_cost)

    Xtr, Xte, ylos_tr, ylos_te, ycost_tr, ycost_te, ycost_log_tr, ycost_log_te = train_test_split(
        X, y_los, y_cost, y_cost_log, test_size=0.2, random_state=42)

    common = dict(categorical_features=cat_idx, max_iter=300,
                  learning_rate=0.08, max_depth=8, random_state=42, early_stopping=True)

    print("Training LOS regressor ...")
    los_model = HistGradientBoostingRegressor(**common).fit(Xtr, ylos_tr)
    print("Training cost regressor on log(cost) ...")
    cost_model = HistGradientBoostingRegressor(**common).fit(Xtr, ycost_log_tr)

    # ---- metrics ----
    los_pred = los_model.predict(Xte)
    cost_pred = np.expm1(cost_model.predict(Xte))
    cost_pred = np.maximum(cost_pred, 0)

    los_metrics = {
        "mae": round(float(mean_absolute_error(ylos_te, los_pred)), 2),
        "r2": round(float(r2_score(ylos_te, los_pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(ylos_te, los_pred))), 2),
    }
    cost_metrics = {
        "mae": round(float(mean_absolute_error(ycost_te, cost_pred)), 0),
        "r2": round(float(r2_score(ycost_te, cost_pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(ycost_te, cost_pred))), 0),
        "median_ae": round(float(np.median(np.abs(ycost_te - cost_pred))), 0),
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
        "Payment Typology 2": opts("Payment Typology 2", limit=8),
        "CCSR Diagnosis Code": opts("CCSR Diagnosis Code", limit=25),
        "APR Medical Surgical Description": opts("APR Medical Surgical Description"),
        "Zip Code": opts("Zip Code", limit=25),
        "Hospital County": opts("Hospital County", limit=8),
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
        "cost_target": "log1p",
        "target_label": "Total Costs",
    }
    json.dump(config, open(os.path.join(ART, "config.json"), "w"), indent=2)

    metrics = {
        "los": los_metrics, "cost": cost_metrics, "high_cost": hi_metrics,
        "cohort": cohort, "cohort_by": cohort_by, "sev_los": sev_los,
        "scatter": scatter, "cost_pairs": cost_pairs,
        "hist": {"counts": hist_counts.tolist(), "edges": [float(e) for e in hist_edges]},
        "opportunity": opportunity,
        "data_source": "NY SPARCS 2024, Finger Lakes region",
        "target_label": "Total Costs",
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
