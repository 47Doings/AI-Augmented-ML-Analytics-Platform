-- =====================================================================
-- training_table.sql (revenue)  -  one row per active customer:
-- features known at the cutoff + the money they pay in the next 30 days
--
-- Parameters (from use_cases/revenue.yaml):
--   %(cutoff_date)s         the "today" of the forecast
--   %(label_window_days)s   how many days ahead we add up payments
--
-- TARGET
--   revenue_next_30d = sum of SUCCESSFUL payments (amount_ghs_filled) made in the
--   window AFTER the cutoff. 0 if the customer paid nothing (failed, late, or left).
--
-- THE RULE THAT PREVENTS LEAKAGE (same as the churn use case)
--   Features use only data on or before the cutoff. Only the target looks past it.
--   scripts/verify_sql.py proves it by deleting all post-cutoff data and checking
--   no feature changes.
--
-- WHO IS IN THE TABLE: the same population as the churn use case
--   (real customers, still subscribed on the cutoff day, cancellation date known).
--
-- Techniques: CTEs, window functions (ROW_NUMBER for the latest payment),
-- conditional aggregation (FILTER), joins, date arithmetic. No percent sign
-- may appear in this file (the driver treats it as a parameter marker).
-- =====================================================================

-- name: training_table
WITH params AS (
    SELECT %(cutoff_date)s::date                                              AS cutoff,
           %(label_window_days)s::int                                         AS window_days,
           (date_trunc('month', %(cutoff_date)s::date + 1) - interval '1 month')::date AS last_full_month
),

sub_at_cutoff AS (
    SELECT s.*,
           ROW_NUMBER() OVER (PARTITION BY s.customer_id
                              ORDER BY s.start_date DESC, s.subscription_id DESC) AS rn
    FROM clean_subscriptions s
    CROSS JOIN params p
    WHERE s.start_date <= p.cutoff
),

population AS (
    SELECT c.customer_id, c.signup_date, c.region, c.acquisition_channel, c.customer_type,
           sc.plan_id
    FROM clean_customers c
    JOIN sub_at_cutoff sc       ON sc.customer_id = c.customer_id AND sc.rn = 1
    JOIN clean_subscriptions l  ON l.customer_id  = c.customer_id AND l.is_latest
    CROSS JOIN params p
    WHERE NOT c.is_duplicate
      AND c.signup_date <= p.cutoff
      AND (sc.end_date IS NULL OR sc.end_date > p.cutoff)
      AND NOT (l.status = 'cancelled' AND l.end_date IS NULL)
),

-- NOTE: AS MATERIALIZED makes PostgreSQL compute each CTE once. Without it the planner
-- (which under-estimates the size of these CTEs) re-runs the payment CTEs once per customer.
-- the clean payments view is expensive (it removes duplicates with a window function), so
-- compute it ONCE and let the three payment CTEs below share the result
pay AS MATERIALIZED (
    SELECT * FROM clean_payments
),

-- billing history: the 90 days up to and including the cutoff day
pay_hist AS MATERIALIZED (
    SELECT pay.customer_id,
           count(*) FILTER (WHERE pay.status = 'success')                       AS successful_payments_90d,
           count(*) FILTER (WHERE pay.status = 'failed')                        AS failed_payments_90d,
           count(*) FILTER (WHERE pay.status = 'reversed')                      AS reversed_payments_90d,
           avg(pay.amount_ghs_filled) FILTER (WHERE pay.status = 'success')     AS avg_billed_ghs_90d
    FROM pay
    CROSS JOIN params p
    WHERE pay.paid_at >= (p.cutoff - 90)::timestamp
      AND pay.paid_at <  (p.cutoff + 1)::timestamp
    GROUP BY pay.customer_id
),

-- the most recent successful payment up to the cutoff (window function)
last_pay AS MATERIALIZED (
    SELECT customer_id, last_paid_at, last_payment_method
    FROM (
        SELECT pay.customer_id,
               pay.paid_at AS last_paid_at,
               pay.method  AS last_payment_method,
               ROW_NUMBER() OVER (PARTITION BY pay.customer_id
                                  ORDER BY pay.paid_at DESC, pay.payment_id DESC) AS rn
        FROM pay
        CROSS JOIN params p
        WHERE pay.status = 'success'
          AND pay.paid_at < (p.cutoff + 1)::timestamp
    ) x
    WHERE rn = 1
),

usage AS MATERIALIZED (
    SELECT u.customer_id,
           avg(u.data_used_gb) AS avg_data_used_gb_3m,
           avg(u.active_days)  AS avg_active_days_3m
    FROM clean_usage u
    CROSS JOIN params p
    WHERE u.usage_month <= p.last_full_month
      AND u.usage_month >= (p.last_full_month - interval '2 months')::date
    GROUP BY u.customer_id
),

tickets AS MATERIALIZED (
    SELECT t.customer_id,
           count(*)                                                                   AS tickets_90d,
           avg(t.satisfaction_score) FILTER (WHERE t.resolved_at < (p.cutoff + 1)::timestamp) AS avg_satisfaction
    FROM clean_tickets t
    CROSS JOIN params p
    WHERE t.opened_at >= (p.cutoff - 90)::timestamp
      AND t.opened_at <  (p.cutoff + 1)::timestamp
    GROUP BY t.customer_id
),

-- THE TARGET: the only place that looks after the cutoff
future AS MATERIALIZED (
    SELECT pay.customer_id,
           sum(pay.amount_ghs_filled) AS revenue_next_30d
    FROM pay
    CROSS JOIN params p
    WHERE pay.status = 'success'
      AND pay.paid_at >= (p.cutoff + 1)::timestamp
      AND pay.paid_at <  (p.cutoff + 1 + p.window_days)::timestamp
    GROUP BY pay.customer_id
)

SELECT
    pop.customer_id,
    -- numeric features
    (p.cutoff - pop.signup_date)                                  AS tenure_days,
    pl.monthly_price_ghs,
    pl.contract_months,
    COALESCE(ph.successful_payments_90d, 0)                       AS successful_payments_90d,
    COALESCE(ph.failed_payments_90d, 0)                           AS failed_payments_90d,
    COALESCE(ph.reversed_payments_90d, 0)                         AS reversed_payments_90d,
    round(ph.avg_billed_ghs_90d, 2)                               AS avg_billed_ghs_90d,
    (p.cutoff - lp.last_paid_at::date)                            AS days_since_last_payment,
    round(u.avg_data_used_gb_3m, 2)                               AS avg_data_used_gb_3m,
    round(u.avg_active_days_3m, 1)                                AS avg_active_days_3m,
    COALESCE(tk.tickets_90d, 0)                                   AS tickets_90d,
    round(tk.avg_satisfaction, 2)                                 AS avg_satisfaction,
    -- categorical features
    pl.plan_name,
    pop.region,
    pop.acquisition_channel,
    pop.customer_type,
    COALESCE(lp.last_payment_method, 'none')                      AS last_payment_method,
    -- target
    COALESCE(f.revenue_next_30d, 0)::float8                       AS revenue_next_30d
FROM population pop
CROSS JOIN params p
JOIN plans pl             ON pl.plan_id = pop.plan_id
LEFT JOIN pay_hist ph     ON ph.customer_id = pop.customer_id
LEFT JOIN last_pay lp     ON lp.customer_id = pop.customer_id
LEFT JOIN usage u         ON u.customer_id  = pop.customer_id
LEFT JOIN tickets tk      ON tk.customer_id = pop.customer_id
LEFT JOIN future f        ON f.customer_id  = pop.customer_id
ORDER BY pop.customer_id;
