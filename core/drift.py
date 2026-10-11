"""Drift monitoring with the Population Stability Index (PSI).

The question: "is the data the model sees today still like the data it was trained on?"

PSI compares two distributions of one variable, bin by bin:
    PSI = sum over bins of (share_now - share_then) * ln(share_now / share_then)
Rule of thumb used almost everywhere in credit/telecom risk:
    < 0.10   stable
    0.10-0.25  watch (something moved, look at it)
    > 0.25   alert (investigate; consider retraining)

Bins come from the REFERENCE (training) data only. Missing values get their own
bin, so "suddenly many more gaps" is also detected. For categories, a value never
seen in training goes into an 'other' bin.

We also compute PSI on the model's output score: if the scores shift, the model
is seeing a different population or reacting to something new, even if no single
input moved much.

Limit: PSI sees input/score drift, not whether the model is still RIGHT. To know
that you need the real outcomes (labels), which arrive 60 days later in this use
case. The monitoring plan in the architecture note covers both.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WATCH, ALERT = 0.10, 0.25
EPS = 1e-4  # avoids log(0) and division by zero for empty bins


def status_for(psi: float) -> str:
    return "stable" if psi < WATCH else "watch" if psi < ALERT else "alert"


def _psi_from_counts(ref: np.ndarray, new: np.ndarray) -> float:
    p = np.clip(ref / ref.sum(), EPS, None)
    q = np.clip(new / new.sum(), EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce").to_numpy(dtype=float)
    cur = pd.to_numeric(current, errors="coerce").to_numpy(dtype=float)
    ref_ok = ref[~np.isnan(ref)]
    if len(ref_ok) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref_ok, np.linspace(0, 1, bins + 1)[1:-1]))  # interior cut points

    def counts(values: np.ndarray) -> np.ndarray:
        ok = values[~np.isnan(values)]
        idx = np.searchsorted(edges, ok, side="right")  # 0 .. len(edges)
        c = np.bincount(idx, minlength=len(edges) + 1).astype(float)
        return np.append(c, float(np.isnan(values).sum()))  # last bin = missing

    return _psi_from_counts(counts(ref), counts(cur))


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    ref = reference.astype(object).where(reference.notna(), "missing").astype(str)
    cur = current.astype(object).where(current.notna(), "missing").astype(str)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    cats = sorted(set(ref))
    ref_c = np.array([(ref == c).sum() for c in cats] + [0], dtype=float)
    cur_c = np.array([(cur == c).sum() for c in cats] + [(~cur.isin(cats)).sum()], dtype=float)  # last = unseen
    return _psi_from_counts(ref_c, cur_c)


def drift_report(bundle: dict, current_X: pd.DataFrame, current_scores: np.ndarray) -> pd.DataFrame:
    """One row per feature plus one for the model score, sorted worst first."""
    rows = []
    for col in bundle["numeric"]:
        rows.append({"variable": col, "kind": "numeric", "psi": psi_numeric(bundle["reference_X"][col], current_X[col])})
    for col in bundle["categorical"]:
        rows.append({"variable": col, "kind": "categorical", "psi": psi_categorical(bundle["reference_X"][col], current_X[col])})
    rows.append({"variable": "MODEL SCORE", "kind": "score",
                 "psi": psi_numeric(pd.Series(bundle["reference_scores"]), pd.Series(current_scores))})
    out = pd.DataFrame(rows)
    out["status"] = out["psi"].map(lambda v: "n/a" if pd.isna(v) else status_for(v))
    return out.sort_values("psi", ascending=False, kind="stable").reset_index(drop=True)
