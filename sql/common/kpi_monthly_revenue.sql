-- KPI: monthly recurring revenue = successful payments in the last complete month before the cutoff.
-- Parameter: %(cutoff_date)s. Must return one column called value.

-- name: kpi_monthly_revenue
SELECT coalesce(sum(amount_ghs_filled), 0) AS value
FROM clean_payments
WHERE status = 'success'
  AND paid_at >= (date_trunc('month', %(cutoff_date)s::date + 1) - interval '1 month')::timestamp
  AND paid_at <  date_trunc('month', %(cutoff_date)s::date + 1)::timestamp;
