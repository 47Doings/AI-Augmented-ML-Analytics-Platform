# Monitoring plan

A model that was good on the day it was trained is not necessarily good today. Customers change, prices change, a
source system starts sending something new. Monitoring answers four questions, in this order, because a problem in an
earlier layer makes the later ones meaningless.

| Layer | Question | What is measured | Where it exists today |
|---|---|---|---|
| 1. Data | Is the input data healthy? | The 35 data-quality checks (duplicates, orphans, missing values, invalid dates, future payments); row counts per table; freshness (latest payment/usage date) | `sql/common/data_quality.sql`, shown in results and the report |
| 2. Features | Do today's inputs look like the training inputs? | PSI per feature, including a "missing" bin, so a column that suddenly goes empty is caught | `core/drift.py`, pipeline "Drift check" step |
| 3. Score | Is the model's output distribution stable? | PSI of the score itself (the earliest warning, even when no single feature moves) | same, row "MODEL SCORE" |
| 4. Performance | Is it still right? | Once real outcomes arrive: ROC AUC / average precision, precision and lift in the top 10% (classification); MAE / RMSE (regression), compared with the model card | **Proposed.** Needs labels, which arrive after the label window (60 days for churn, 30 for revenue) |
| 5. The pipeline and the AI text | Did the run work, and was the AI text trustworthy? | Run status and duration per step; share of runs where the AI answer was accepted vs rejected vs fallback; numbers checked per summary | `platform.runs`, `insights.json` (`mode`, `attempts`, `validation`) |

## Thresholds and what happens

| Signal | Level | Action |
|---|---|---|
| Any high-severity data-quality check worse than last run by more than 2x, or a table with no new rows for 2 days | Block | Do not score or retrain. Alert the data owner. Show the last good results with a "stale" banner. |
| Feature PSI below 0.10 | Stable | Nothing. |
| Feature PSI 0.10 to 0.25 | Watch | Record it; review at the weekly check. One feature on watch is normal. |
| Feature PSI above 0.25, or 3 or more features on watch, or score PSI above 0.10 | Alert | Investigate the cause first (new plan, price change, source bug). If the change is real, retrain on recent data. |
| Performance (when labels exist) drops by more than 0.05 ROC AUC, or top-decile lift falls below 2x, from the model card values | Alert | Retrain; compare the challenger with the current model on the same recent data before replacing it. |
| AI answer rejected on both attempts in more than 20% of runs in a week | Review | Look at the rejected errors in `insights.json`; adjust the prompt or the allowed wording. Template keeps reports flowing meanwhile. |

The thresholds for PSI are the common conventions (0.10 and 0.25), not tuned values. They should be revisited once
there is real history, because PSI is noisy on small samples (this prototype scores a few hundred customers).

## Cadence

- **Every run (nightly in production):** data quality gate, feature and score PSI, run status. Automatic.
- **Weekly:** someone looks at the trend charts; confirms the watch items.
- **When labels mature (about 60 days later for churn):** score the old predictions against what happened, and record
  the real AUC and lift next to the model-card values.
- **Retraining:** on an alert, or on a fixed schedule (monthly or quarterly), always as champion vs challenger with
  the same seed and recorded library versions, so the comparison is fair and reproducible.

## Why PSI

PSI compares how the values were spread across bins at training time against now. It needs no labels, so it works
immediately, and it is easy to explain. It does not say the model is *wrong*, only that the world it is seeing has
moved; that is why performance monitoring (layer 4) is still needed once outcomes are known.

## How this runs in production

See the architecture note: a scheduler runs the pipeline; each run writes its metrics (PSI per feature, DQ counts,
durations, AI mode) to a monitoring table; Grafana charts them and alerts on the thresholds above.
