"""Tests for the generic ML pipeline and the drift module.

No database needed: they build a small synthetic table with a KNOWN signal, so we
can check the pipeline finds it (and does not invent one where there is none).

Run from the project root:  python -m pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from core.drift import psi_categorical, psi_numeric, status_for
from core.ml_pipeline import MlError, compute_metrics, top_fraction_mask, train_from_frame
from core.use_case import load_use_case


@pytest.fixture(scope="module")
def uc():
    return load_use_case("churn")


def make_frame(uc, n=1500, seed=0, signal=True):
    """Churn-shaped table. If signal=True, low activity + failed payments raise churn probability."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({uc.entity.key: np.arange(1, n + 1)})
    for col in uc.features.numeric:
        df[col] = rng.normal(50, 15, n)
    df["avg_active_days_3m"] = rng.uniform(0, 30, n)
    df["failed_payments_90d"] = rng.poisson(0.3, n).astype(float)
    for col in uc.features.categorical:
        df[col] = rng.choice(["a", "b", "c"], n)
    df.loc[rng.random(n) < 0.15, "avg_satisfaction"] = np.nan  # gaps, as in the real data
    df.loc[rng.random(n) < 0.05, "region"] = None
    if signal:
        logit = -2.2 - 0.25 * (df["avg_active_days_3m"] - 15) + 1.3 * df["failed_payments_90d"]
        p = 1 / (1 + np.exp(-logit))
    else:
        p = np.full(n, 0.1)
    df[uc.target.name] = (rng.random(n) < p).astype(int)
    return df


def test_baseline_is_chance_and_model_beats_it(uc):
    r = train_from_frame(make_frame(uc), uc)
    comp = r["comparison"].set_index("model")
    assert comp.loc[uc.models.baseline, "test_roc_auc"] == pytest.approx(0.5, abs=1e-9)
    best = r["best_name"]
    assert comp.loc[best, "test_roc_auc"] > 0.70
    assert r["card"]["beats_baseline_in_cv"] is True


def test_finds_the_planted_drivers(uc):
    r = train_from_frame(make_frame(uc), uc)
    top2 = list(r["importance"]["feature"].head(2))
    assert set(top2) == {"avg_active_days_3m", "failed_payments_90d"}
    d = r["importance"].set_index("feature")["direction"]
    assert d["avg_active_days_3m"] == "higher value -> lower score"
    assert d["failed_payments_90d"] == "higher value -> higher score"


def test_no_signal_means_no_skill(uc):
    """If labels are pure noise the pipeline must NOT report a good model."""
    r = train_from_frame(make_frame(uc, signal=False), uc)
    assert r["comparison"].set_index("model").loc[r["best_name"], "test_roc_auc"] < 0.62


def test_same_seed_same_result(uc):
    a = train_from_frame(make_frame(uc), uc)
    b = train_from_frame(make_frame(uc), uc)
    pd.testing.assert_frame_equal(a["comparison"], b["comparison"])
    pd.testing.assert_frame_equal(a["scored"], b["scored"])
    assert a["card"]["data_sha256"] == b["card"]["data_sha256"]


def test_out_of_fold_scores_cover_everyone_once(uc):
    df = make_frame(uc)
    r = train_from_frame(df, uc)
    s = r["scored"]
    assert len(s) == len(df) and s[uc.entity.key].is_unique
    assert s["risk_decile"].between(1, 10).all()
    assert s["flagged"].sum() == int(np.ceil(uc.evaluation.flag_top_fraction * len(df)))


def test_refuses_too_few_positives(uc):
    df = make_frame(uc, n=300)
    df[uc.target.name] = 0
    df.loc[:3, uc.target.name] = 1
    with pytest.raises(MlError):
        train_from_frame(df, uc)


def test_missing_column_is_a_clear_error(uc):
    with pytest.raises(MlError):
        train_from_frame(make_frame(uc).drop(columns=["tenure_days"]), uc)


def test_top_fraction_ties_are_random_not_positional():
    """A constant score must not let row order decide who is 'flagged'."""
    scores = np.zeros(1000)
    a = top_fraction_mask(scores, 0.1, seed=1)
    assert a.sum() == 100
    assert not a[:100].all()  # not simply the first 100 rows


def test_lift_math():
    y = np.array([1] * 10 + [0] * 90)
    perfect = compute_metrics("classification", y, y.astype(float), flag_fraction=0.1, seed=0)
    assert perfect["precision"] == 1.0 and perfect["recall"] == 1.0 and perfect["lift"] == pytest.approx(10.0)


def test_regression_config_runs_with_no_code_change(uc):
    """The same pipeline handles a regression use case: only the config differs."""
    rng = np.random.default_rng(3)
    n = 800
    df = pd.DataFrame({"id": np.arange(n), "x1": rng.normal(size=n), "x2": rng.normal(size=n),
                       "grp": rng.choice(["u", "v"], n)})
    df["y"] = 10 + 3 * df["x1"] - 2 * (df["grp"] == "v") + rng.normal(0, 0.5, n)
    reg = uc.model_copy(deep=True, update={
        "problem_type": "regression",
        "entity": uc.entity.model_copy(update={"key": "id"}),
        "target": uc.target.model_copy(update={"name": "y"}),
        "features": uc.features.model_copy(update={"numeric": ["x1", "x2"], "categorical": ["grp"]}),
        "models": uc.models.model_copy(update={"baseline": "dummy_regressor",
                                               "candidates": ["linear_regression", "gradient_boosting_regressor"]}),
        "metrics": uc.metrics.model_copy(update={"primary": "r2", "report": ["r2", "mae"]}),
    })
    r = train_from_frame(df, reg)
    comp = r["comparison"].set_index("model")
    assert comp.loc["dummy_regressor", "test_r2"] < 0.05
    assert comp.loc[r["best_name"], "test_r2"] > 0.9
    assert r["importance"]["feature"].iloc[0] == "x1"


# ---------------- drift ----------------
def test_psi_same_distribution_is_stable():
    rng = np.random.default_rng(0)
    a, b = pd.Series(rng.normal(0, 1, 5000)), pd.Series(rng.normal(0, 1, 5000))
    assert psi_numeric(a, b) < 0.05


def test_psi_detects_a_shift():
    rng = np.random.default_rng(0)
    a, b = pd.Series(rng.normal(0, 1, 5000)), pd.Series(rng.normal(1.0, 1, 5000))
    assert status_for(psi_numeric(a, b)) == "alert"


def test_psi_detects_new_missing_values():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 5000))
    b = a.copy()
    b[:1500] = np.nan  # 30% of the feed suddenly empty
    assert psi_numeric(a, b) > 0.25


def test_psi_categorical_new_category_and_mix_change():
    a = pd.Series(["x"] * 500 + ["y"] * 500)
    assert psi_categorical(a, pd.Series(["x"] * 500 + ["y"] * 500)) < 0.01
    assert psi_categorical(a, pd.Series(["x"] * 900 + ["y"] * 100)) > 0.25
    assert psi_categorical(a, pd.Series(["x"] * 400 + ["y"] * 400 + ["z"] * 200)) > 0.1
