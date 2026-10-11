-- KPI: share of payment attempts that failed in the last complete month before the cutoff.
-- Parameter: %(cutoff_date)s. Must return one column called value.

-- name: kpi_payment_failure_rate
SELECT coalesce(count(*) FILTER (WHERE status = 'failed')::numeric / NULLIF(count(*), 0), 0) AS value
FROM clean_payments
WHERE paid_at >= (date_trunc('month', %(cutoff_date)s::date + 1) - interval '1 month')::timestamp
  AND paid_at <  date_trunc('month', %(cutoff_date)s::date + 1)::timestamp;
