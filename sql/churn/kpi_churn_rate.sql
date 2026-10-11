-- KPI: churn rate = share of customers active at the cutoff who cancel inside the label window.
-- Reads training_set (build it first). Must return one column called value.

-- name: kpi_churn_rate
SELECT avg(churned::numeric) AS value
FROM training_set;
