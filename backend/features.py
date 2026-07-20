"""Shared, outcome-free feature engineering for training and inference."""

import pandas as pd


DERIVED_FEATURES = [
    "DRG Severity Group",
    "Diagnosis Severity Group",
    "Age Severity Group",
    "Admission Pathway",
    "Clinical Service Group",
    "Population Group",
    "Complexity Score",
    "Has Secondary Payer",
    "Service Line",
]

LEVEL = {"Minor": 1, "Moderate": 2, "Major": 3, "Extreme": 4}


def _join(left, right):
    return left.astype(str) + " | " + right.astype(str)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with features that do not use LOS, cost, or disposition."""
    out = df.copy()
    severity = out["APR Severity of Illness Description"].fillna("Unknown")
    mortality = out["APR Risk of Mortality"].fillna("Unknown")
    out["DRG Severity Group"] = _join(out["APR DRG Description"], severity)
    out["Diagnosis Severity Group"] = _join(out["CCSR Diagnosis Code"], severity)
    out["Age Severity Group"] = _join(out["Age Group"], severity)
    out["Admission Pathway"] = _join(
        out["Type of Admission"], out["Emergency Department Indicator"])
    out["Clinical Service Group"] = _join(
        out["APR MDC Description"], out["APR Medical Surgical Description"])

    age = out["Age Group"].astype(str)
    admission = out["Type of Admission"].astype(str)
    out["Population Group"] = "Adult"
    out.loc[age.eq("0-17") | age.eq("0 to 17"), "Population Group"] = "Pediatric"
    out.loc[age.str.contains("70", na=False), "Population Group"] = "Older adult"
    out.loc[admission.eq("Newborn"), "Population Group"] = "Newborn"

    severity_score = severity.map(LEVEL)
    mortality_score = mortality.map(LEVEL)
    score = severity_score + mortality_score
    out["Complexity Score"] = score.map(
        lambda value: str(int(value)) if pd.notna(value) else "Unknown")
    payer2 = out["Payment Typology 2"].fillna("Unknown").astype(str)
    out["Has Secondary Payer"] = payer2.ne("Unknown").map({True: "Yes", False: "No"})

    # Mutually exclusive, interpretable cohorts supported by the observed
    # clustering structure. These use no LOS, cost, procedure, or disposition.
    mdc = out["APR MDC Description"].astype(str)
    med_surg = out["APR Medical Surgical Description"].astype(str)
    ed = out["Emergency Department Indicator"].astype(str)
    out["Service Line"] = "Other medical"
    out.loc[med_surg.eq("Surgical"), "Service Line"] = "Other surgical"
    out.loc[
        med_surg.eq("Medical")
        & (admission.eq("Emergency") | ed.eq("Y")),
        "Service Line",
    ] = "Emergency medical"
    out.loc[
        med_surg.eq("Surgical") & admission.eq("Elective"),
        "Service Line",
    ] = "Elective surgical"
    out.loc[age.eq("0-17") | age.eq("0 to 17"), "Service Line"] = "Pediatric"
    out.loc[
        mdc.str.contains("PREGNANCY|CHILDBIRTH|PUERPERIUM", case=False, na=False),
        "Service Line",
    ] = "Maternity"
    out.loc[
        admission.eq("Newborn") | mdc.str.contains("NEWBORN|NEONATE", case=False, na=False),
        "Service Line",
    ] = "Newborn"
    return out


def add_derived_to_row(row: dict) -> dict:
    frame = add_derived_features(pd.DataFrame([row]))
    return frame.iloc[0].to_dict()
