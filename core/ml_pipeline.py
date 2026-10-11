"""The generic ML pipeline.

Nothing in here knows about churn. It reads everything from the use-case config:
which columns are numeric / categorical, which column is the target, which
models to compare, which metric decides, how big the test set is, the seed.
Swap the config + the SQL and the same code trains a different model.

What it does (in this order)
  1. Reads the training table, keeps only the columns the config names.
  2. Splits once into train / test (stratified for classification, seeded).
  3. Builds ONE sklearn Pipeline per model: preprocessing + model. Because the
     preprocessing lives inside the Pipeline, it is fitted on training rows only
     (no test-set information leaks into medians / scaling / categories).
  4. Compares the baseline and every candidate:
        - hyper-parameters are tuned with cross-validation INSIDE the train split
        - the model is picked by that cross-validation score, never by the test set
        - the test set is touched once per model, only to report
  5. Explains the chosen model with permutation importance (+ direction).
  6. Scores every entity with out-of-fold predictions (nobody is scored by a
     model that saw their own label).
  7. Saves the model, a model card, and the tables the report/dashboard use.

Outputs go to outputs/ml/<use_case>/ and are identical on every run
(same data + same seed + same library versions).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core.config import PROJECT_ROOT
from core.use_case import UseCase

logger = logging.getLogger(__name__)

CV_FOLDS = 5
MIN_PER_CLASS = 2 * CV_FOLDS  # need at least this many examples of each class to cross-validate


class MlError(Exception):
    """The data or config cannot be trained on (not enough rows, one class only, ...)."""


# --------------------------------------------------------------------------------------
# Model registry. Names are exactly the ones use_case.ALLOWED_MODELS accepts.
# Each entry: (how to build the estimator from a seed, the small grid searched with CV).
# Grids are deliberately small: this is a prototype, and a small grid is easy to explain.
# --------------------------------------------------------------------------------------
REGISTRY: dict[str, tuple] = {
    # classification
    "dummy_classifier": (lambda s: DummyClassifier(strategy="prior"), {}),
    "logistic_regression": (lambda s: LogisticRegression(max_iter=2000, random_state=s), {"model__C": [0.1, 1.0, 10.0]}),
    "random_forest": (
        lambda s: RandomForestClassifier(n_estimators=300, random_state=s, n_jobs=1),
        {"model__max_depth": [4, 8], "model__min_samples_leaf": [5, 20]},
    ),
    "gradient_boosting": (
        lambda s: GradientBoostingClassifier(learning_rate=0.05, subsample=0.8, random_state=s),
        {"model__n_estimators": [100, 200], "model__max_depth": [2, 3]},
    ),
    # regression
    "dummy_regressor": (lambda s: DummyRegressor(strategy="mean"), {}),
    "linear_regression": (lambda s: LinearRegression(), {}),
    "random_forest_regressor": (
        lambda s: RandomForestRegressor(n_estimators=300, random_state=s, n_jobs=1),
        {"model__max_depth": [4, 8], "model__min_samples_leaf": [5, 20]},
    ),
    "gradient_boosting_regressor": (
        lambda s: GradientBoostingRegressor(learning_rate=0.05, subsample=0.8, random_state=s),
        {"model__n_estimators": [100, 200], "model__max_depth": [2, 3]},
    ),
}

# metric name in the use-case config -> sklearn scorer name (used for CV selection + permutation importance)
SKLEARN_SCORER = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "accuracy": "accuracy",
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
    "mape": "neg_mean_absolute_percentage_error",
}
LOWER_IS_BETTER = {"mae", "rmse", "mape"}  # only matters for how we print / compare (sklearn negates them)


# --------------------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------------------
def prepare_frame(df: pd.DataFrame, uc: UseCase) -> pd.DataFrame:
    """Keep the key, the features and the target; give every column a plain, predictable type."""
    needed = [uc.entity.key, *uc.feature_columns, uc.target.name]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise MlError(f"Training table is missing columns the config expects: {missing}")
    out = df[needed].copy()
    for col in uc.features.numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    for col in uc.features.categorical:
        s = out[col].astype(object)
        out[col] = s.where(s.notna(), np.nan)  # one kind of "missing" (NaN) for the imputer
    out[uc.target.name] = pd.to_numeric(out[uc.target.name])
    return out.reset_index(drop=True)


def build_preprocessor(uc: UseCase) -> ColumnTransformer:
    """Numbers: fill gaps with the median (and remember which were missing), then scale.
    Categories: gaps become their own category 'missing', then one-hot encode.
    Unseen categories at prediction time are ignored instead of crashing."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    parts = []
    if uc.features.numeric:
        parts.append(("num", numeric, list(uc.features.numeric)))
    if uc.features.categorical:
        parts.append(("cat", categorical, list(uc.features.categorical)))
    return ColumnTransformer(parts, remainder="drop")


def build_pipeline(uc: UseCase, model_name: str) -> tuple[Pipeline, dict]:
    make, grid = REGISTRY[model_name]
    return Pipeline([("prep", build_preprocessor(uc)), ("model", make(uc.evaluation.random_seed))]), grid


def _scores(pipe: Pipeline, X: pd.DataFrame, problem_type: str) -> np.ndarray:
    return pipe.predict_proba(X)[:, 1] if problem_type == "classification" else pipe.predict(X)


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def top_fraction_mask(scores: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """True for the riskiest ceil(fraction * n) entities.
    Ties are broken randomly (seeded) so a model with no skill is not accidentally rewarded
    or punished by the order the rows happen to be in."""
    n = len(scores)
    k = max(1, math.ceil(fraction * n))
    tiebreak = np.random.default_rng(seed).random(n)
    order = np.lexsort((tiebreak, -scores))  # primary: score descending, then random
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def compute_metrics(problem_type: str, y_true, scores, *, flag_fraction: float, seed: int) -> dict:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    if problem_type == "regression":
        nonzero = np.all(y_true != 0)
        return {
            "mae": float(mean_absolute_error(y_true, scores)),
            "rmse": float(math.sqrt(mean_squared_error(y_true, scores))),
            "r2": float(r2_score(y_true, scores)),
            "mape": float(mean_absolute_percentage_error(y_true, scores)) if nonzero else float("nan"),
        }
    flagged = top_fraction_mask(scores, flag_fraction, seed)
    positives = int(y_true.sum())
    tp = int((flagged & (y_true == 1)).sum())
    k = int(flagged.sum())
    precision = tp / k if k else 0.0
    recall = tp / positives if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    base_rate = positives / len(y_true)
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        # the four below are measured at the operating point: "flag the riskiest flag_fraction"
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": float((flagged.astype(int) == y_true).mean()),
        "lift": precision / base_rate if base_rate else float("nan"),
        "flagged": k,
    }


# --------------------------------------------------------------------------------------
# Explanations
# --------------------------------------------------------------------------------------
def _direction(X: pd.DataFrame, scores: np.ndarray, col: str, numeric: bool) -> str:
    s = pd.Series(scores, index=X.index)
    if numeric:
        x = X[col]
        ok = x.notna()
        if ok.sum() < 30 or x[ok].nunique() < 3:
            return "not enough variation to tell"
        rho = x[ok].corr(s[ok], method="spearman")
        if rho > 0.05:
            return "higher value -> higher score"
        if rho < -0.05:
            return "higher value -> lower score"
        return "no clear direction"
    grp = s.groupby(X[col].fillna("missing")).agg(["mean", "count"])
    grp = grp[grp["count"] >= 20]
    if len(grp) < 2:
        return "not enough variation to tell"
    hi, lo = grp["mean"].idxmax(), grp["mean"].idxmin()
    return f"highest score: {hi}; lowest: {lo}"


def explain(pipe: Pipeline, X_test: pd.DataFrame, y_test, uc: UseCase, primary: str) -> pd.DataFrame:
    """Permutation importance on the TEST set: shuffle one feature, see how much the score drops.
    A big drop = the model leans on that feature. Works for any model type."""
    result = permutation_importance(
        pipe, X_test, y_test, scoring=SKLEARN_SCORER[primary], n_repeats=20,
        random_state=uc.evaluation.random_seed, n_jobs=1,
    )
    scores = _scores(pipe, X_test, uc.problem_type)
    rows = []
    for i, col in enumerate(X_test.columns):
        numeric = col in uc.features.numeric
        mean, std = float(result.importances_mean[i]), float(result.importances_std[i])
        rows.append({
            "feature": col,
            "kind": "numeric" if numeric else "categorical",
            "importance": mean,
            "std": std,
            "clearly_matters": bool(mean - 2 * std > 0),  # drop is bigger than 2 std of the shuffling noise
            "direction": _direction(X_test, scores, col, numeric),
        })
    return pd.DataFrame(rows).sort_values("importance", ascending=False, kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def _splitter(problem_type: str, seed: int):
    cls = StratifiedKFold if problem_type == "classification" else KFold
    return cls(n_splits=CV_FOLDS, shuffle=True, random_state=seed)


def _data_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()


def _round(obj, nd=6):
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, nd) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if math.isnan(obj) else round(float(obj), nd)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def train_from_frame(raw: pd.DataFrame, uc: UseCase, out_dir: Path | None = None) -> dict:
    """Train, compare, explain, score, save. Returns everything the notebook / report needs."""
    seed = uc.evaluation.random_seed
    primary = uc.metrics.primary
    df = prepare_frame(raw, uc)
    key, target = uc.entity.key, uc.target.name
    X, y = df[uc.feature_columns], df[target]

    if len(df) < 100:
        raise MlError(f"Only {len(df)} rows: too few to train and test honestly (need at least 100).")
    if uc.problem_type == "classification":
        counts = y.value_counts()
        if len(counts) < 2 or counts.min() < MIN_PER_CLASS:
            raise MlError(f"Target needs at least {MIN_PER_CLASS} examples of each class; got {counts.to_dict()}")

    # ---- 1. one split, made once, seeded ----
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=uc.evaluation.test_size, random_state=seed,
        stratify=y if uc.problem_type == "classification" else None,
    )
    cv = _splitter(uc.problem_type, seed)
    flag = uc.evaluation.flag_top_fraction

    # ---- 2. fit + compare every model ----
    names = [uc.models.baseline, *uc.models.candidates]
    fitted: dict[str, GridSearchCV] = {}
    rows = []
    for name in names:
        pipe, grid = build_pipeline(uc, name)
        search = GridSearchCV(pipe, grid or {}, scoring=SKLEARN_SCORER[primary], cv=cv, refit=True, n_jobs=1)
        search.fit(X_tr, y_tr)
        fitted[name] = search
        test_scores = _scores(search.best_estimator_, X_te, uc.problem_type)
        m = compute_metrics(uc.problem_type, y_te, test_scores, flag_fraction=flag, seed=seed)
        rows.append({
            "model": name,
            "role": "baseline" if name == uc.models.baseline else "candidate",
            "cv_score": float(search.best_score_),  # sklearn sign convention (errors are negative)
            "best_params": json.dumps({k.replace("model__", ""): v for k, v in search.best_params_.items()}, sort_keys=True),
            **{f"test_{k}": v for k, v in m.items()},
        })
        logger.info("%s: cv %s=%.4f", name, primary, search.best_score_)
    comparison = pd.DataFrame(rows)

    # ---- 3. choose by CROSS-VALIDATION on train only (the test set plays no part) ----
    cand = comparison[comparison["role"] == "candidate"]
    if cand.empty:
        raise MlError("Config lists no candidate models.")
    best_name = cand.sort_values("cv_score", ascending=False, kind="stable").iloc[0]["model"]
    comparison["selected"] = comparison["model"] == best_name
    best = fitted[best_name].best_estimator_
    base_cv = float(comparison.loc[comparison["role"] == "baseline", "cv_score"].iloc[0])
    best_cv = float(comparison.loc[comparison["selected"], "cv_score"].iloc[0])
    beats_baseline = best_cv > base_cv

    # ---- 4. explain the chosen model ----
    importance = explain(best, X_te, y_te, uc, primary)

    # ---- 5. out-of-fold scores for EVERY entity ----
    method = "predict_proba" if uc.problem_type == "classification" else "predict"
    oof = cross_val_predict(clone(best), X, y, cv=cv, method=method, n_jobs=1)
    oof = oof[:, 1] if uc.problem_type == "classification" else oof
    scored = pd.DataFrame({key: df[key], "score": oof, "actual": y.to_numpy()})
    if uc.problem_type == "classification":
        scored["risk_rank"] = scored["score"].rank(method="first", ascending=False).astype(int)
        scored["risk_decile"] = (10 - ((scored["risk_rank"] - 1) * 10 // len(scored))).astype(int)  # 10 = riskiest
        scored["flagged"] = top_fraction_mask(scored["score"].to_numpy(), flag, seed).astype(int)
    scored = scored.sort_values(key).reset_index(drop=True)

    # ---- 6. model card ----
    final_test = comparison.loc[comparison["selected"]].iloc[0]
    card = {
        "use_case": uc.id,
        "problem_type": uc.problem_type,
        "target": target,
        "primary_metric": primary,
        "seed": seed,
        "rows": {"total": len(df), "train": len(X_tr), "test": len(X_te)},
        "positive_rate": {"train": float(y_tr.mean()), "test": float(y_te.mean())} if uc.problem_type == "classification" else None,
        "features": {"numeric": list(uc.features.numeric), "categorical": list(uc.features.categorical)},
        "selected_model": best_name,
        "selected_params": json.loads(final_test["best_params"]),
        "baseline_model": uc.models.baseline,
        "cv_score": {"baseline": base_cv, "selected": best_cv},
        "beats_baseline_in_cv": bool(beats_baseline),
        "operating_point": {"flag_top_fraction": flag} if uc.problem_type == "classification" else None,
        "test_metrics": {k[5:]: final_test[k] for k in comparison.columns if k.startswith("test_")},
        "data_sha256": _data_hash(df),
        "parameters": uc.parameters,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "notes": [
            "Final model is trained on the train split only; the test split was never used to choose anything.",
            "Split is random (stratified), not by time. See README limitations.",
            "scores.csv uses out-of-fold predictions: each entity is scored by a model that did not see it.",
        ],
    }

    result = {
        "comparison": comparison, "importance": importance, "scored": scored, "card": _round(card),
        "best_name": best_name, "pipeline": best,
        "y_test": y_te.to_numpy(), "test_scores": _scores(best, X_te, uc.problem_type),
        "all_test_scores": {n: _scores(fitted[n].best_estimator_, X_te, uc.problem_type) for n in names},
        "X_train": X_tr, "X_test": X_te, "frame": df,
    }

    if out_dir is not None:
        save_outputs(result, uc, out_dir)
    return result


def save_outputs(result: dict, uc: UseCase, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    comp = result["comparison"].copy()
    num = comp.select_dtypes("number").columns
    comp[num] = comp[num].round(6)
    comp.to_csv(out_dir / "model_comparison.csv", index=False)
    result["importance"].round({"importance": 6, "std": 6}).to_csv(out_dir / "feature_importance.csv", index=False)
    result["scored"].round({"score": 6}).to_csv(out_dir / "scores.csv", index=False)
    (out_dir / "model_card.json").write_text(json.dumps(result["card"], indent=2, sort_keys=True), encoding="utf-8")
    # The saved bundle carries the training features + their scores: drift monitoring compares against them.
    reference = result["X_train"].reset_index(drop=True)
    bundle = {
        "pipeline": result["pipeline"],
        "use_case": uc.id,
        "features": uc.feature_columns,
        "numeric": list(uc.features.numeric),
        "categorical": list(uc.features.categorical),
        "problem_type": uc.problem_type,
        "reference_X": reference,
        "reference_scores": _scores(result["pipeline"], reference, uc.problem_type),
    }
    joblib.dump(bundle, out_dir / "model.joblib")


def default_out_dir(uc: UseCase) -> Path:
    return PROJECT_ROOT / "outputs" / "ml" / uc.id


def train(schema: str, uc: UseCase, out_dir: Path | None = None) -> dict:
    """Read the workspace's training_set table and train on it."""
    from core.sql_runner import read_training_set  # local import: keeps this module usable without a database

    return train_from_frame(read_training_set(schema), uc, out_dir if out_dir is not None else default_out_dir(uc))
