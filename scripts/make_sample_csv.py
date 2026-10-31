"""Write a small SAMPLE file for trying the 'Use my own data' path (it is not real data).

A made-up micro-lender: one row per loan, with the kind of mess real exports have (blank and "N/A" values,
"Yes" / "yes " spellings, a few repeated loan ids, a typo in a number, one absurd income). The outcome
(`defaulted`) depends on income, loan size, late payments and guarantor, plus noise.

    python -m scripts.make_sample_csv                 # writes samples/microloans_sample.csv
    python -m scripts.make_sample_csv --rows 3000 --seed 7
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make(rows: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = np.round(rng.lognormal(7.4, 0.5, rows), 0)
    amount = np.round(income * rng.uniform(0.5, 4.0, rows) / 10) * 10
    term = rng.choice([3, 6, 9, 12, 18], rows)
    prior = rng.poisson(1.5, rows)
    late = np.minimum(rng.poisson(0.6 + 0.3 * prior * rng.random(rows), rows), 8)
    guarantor = rng.random(rows) < 0.45
    employment = rng.choice(["Salaried", "Self-employed", "Trader", "Student"], rows, p=[0.3, 0.3, 0.3, 0.1])
    branch = rng.choice(["Madina", "Kasoa", "Tema", "Kumasi", "Takoradi"], rows)
    applied = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 600, rows), unit="D")
    logit = (-4.1 + 1.1 * (amount / income) + 0.45 * late - 0.7 * guarantor - 0.00015 * income
             + 0.5 * (employment == "Trader") + 0.04 * term + rng.normal(0, 0.6, rows))
    defaulted = rng.random(rows) < 1 / (1 + np.exp(-logit))

    df = pd.DataFrame({
        "loan_id": [f"LN{100000 + i}" for i in range(rows)],
        "applied_on": applied.strftime("%Y-%m-%d"),
        "branch": branch,
        "employment_type": employment,
        "monthly_income_ghs": income.astype(int).astype(object),
        "loan_amount_ghs": amount.astype(int).astype(object),
        "term_months": term,
        "prior_loans": prior,
        "late_payments_prior": late,
        "has_guarantor": np.where(guarantor, "Yes", "No"),
        "defaulted": np.where(defaulted, "yes", "no"),
    })
    # --- realistic mess, injected on purpose
    def pick(frac):
        return rng.choice(rows, int(rows * frac), replace=False)
    df.loc[pick(0.06), "monthly_income_ghs"] = ""
    df.loc[pick(0.02), "monthly_income_ghs"] = "N/A"
    df.loc[pick(0.003), "monthly_income_ghs"] = "abc"
    df.loc[pick(0.003), "monthly_income_ghs"] = 9_999_999
    df.loc[pick(0.04), "employment_type"] = ""
    df.loc[pick(0.05), "has_guarantor"] = "yes "
    df.loc[pick(0.03), "has_guarantor"] = "YES"
    df.loc[pick(0.04), "branch"] = df["branch"].str.lower()
    df.loc[pick(0.02), "applied_on"] = ""
    dup = df.sample(int(rows * 0.01), random_state=seed)
    return pd.concat([df, dup], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("samples/microloans_sample.csv"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame = make(args.rows, args.seed)
    frame.to_csv(args.out, index=False)
    print(f"Wrote {len(frame):,} rows to {args.out}  (default rate {(frame['defaulted'] == 'yes').mean():.1%})")
