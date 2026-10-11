-- =====================================================================
-- training_table.sql  -  one row per customer: features + label (churn)
--
-- Parameters (from use_cases/churn.yaml):
--   %(cutoff_date)s         the "today" of the prediction
--   %(label_window_days)s   how far ahead we look for a cancellation
--
-- THE RULE THAT PREVENTS LEAKAGE
--   Features use only data known ON OR BEFORE the cutoff.
--   The label looks at the label_window_days AFTER the cutoff.
--   If a feature touched anything after the cutoff, the model would be
--   "peeking at the answer". scripts/verify_sql.py proves this does not happen
--   by deleting all post-cutoff data and checking the features do not change.
--
-- WHO IS IN THE TABLE
--   * real customers (is_duplicate = FALSE) who signed up on or before the cutoff
--   * who were still customers on the cutoff day
--   * whose cancellation date, if any, is known (otherwise the label cannot be trusted)
--
-- Techniques: CTEs, window functions (ROW_NUMBER), conditional aggregation
-- (FILTER), joins, date arithmetic. No percent sign may appear in this file
-- (the driver treats it as a parameter marker).
-- =====================================================================

-- name: training_table
WITH params AS (
    SELECT %(cutoff_date)s::date                                              AS cutoff,
           %(label_window_days)s::int                                         AS window_days,
           -- the last month that is COMPLETELY over on the cutoff day
           (date_trunc('month', %(cutoff_date)s::date + 1) - interval '1 month')::date AS last_full_month
),

-- the subscription each customer was on at the cutoff (window function)
sub_at_cutoff AS (
    SELECT s.*,
           ROW_NUMBER() OVER (PARTITION BY s.customer_id
                              ORDER BY s.start_date DESC, s.subscription_id DESC) AS rn
    FROM clean_subscriptions s
    CROSS JOIN params p
    WHERE s.start_date <= p.cutoff
),

-- who we are predicting for
population AS (
    SELECT c.customer_id, c.signup_date, c.region, c.acquisition_channel, c.customer_type,
           sc.plan_id,
           l.status   AS latest_status,
           l.end_date AS latest_end_date
    FROM clean_customers c
    JOIN sub_at_cutoff sc       ON sc.customer_id = c.customer_id AND sc.rn = 1
    JOIN clean_subscriptions l  ON l.customer_id  = c.customer_id AND l.is_latest
    CROSS JOIN params p
    WHERE NOT c.is_duplicate
      AND c.signup_date <= p.cutoff
      AND (sc.end_date IS NULL OR sc.end_date > p.cutoff)                -- still a customer at the cutoff
      AND NOT (l.status = 'cancelled' AND l.end_date IS NULL)            -- cancellation date unknown: exclude
),

-- usage: the 3 complete months before the cutoff vs the 3 before those
usage AS (
    SELECT u.customer_id,
           avg(u.data_used_gb) FILTER (WHERE u.usage_month >= (p.last_full_month - interval '2 months')::date)  AS avg_data_used_gb_3m,
           avg(u.active_days)  FILTER (WHERE u.usage_month >= (p.last_full_month - interval '2 months')::date)  AS avg_active_days_3m,
           avg(u.data_used_gb) FILTER (WHERE u.usage_month <  (p.last_full_month - interval '2 months')::date)  AS avg_data_used_gb_prev3m,
           -- how many months the earlier window actually has (short-tenure customers have 0-1)
           count(u.data_used_gb) FILTER (WHERE u.usage_month <  (p.last_full_month - interval '2 months')::date) AS prev3m_months
    FROM clean_usage u
    CROSS JOIN params p
    WHERE u.usage_month <= p.last_full_month
      AND u.usage_month >= (p.last_full_month - interval '5 months')::date
    GROUP BY u.customer_id
),

-- payments in the 90 days up to and including the cutoff day
payments90 AS (
    SELECT pay.customer_id,
           count(*) FILTER (WHERE pay.status = 'failed') AS failed_payments_90d
    FROM clean_payments pay
    CROSS JOIN params p
    WHERE pay.paid_at >= (p.cutoff - 90)::timestamp
      AND pay.paid_at <  (p.cutoff + 1)::timestamp
    GROUP BY pay.customer_id
),

-- tickets: counts over 90 days; satisfaction over 180 days (it is sparse).
-- A ticket only counts as resolved / rated if that happened by the cutoff.
tickets AS (
    SELECT t.customer_id,
           count(*) FILTER (WHERE t.opened_at >= (p.cutoff - 90)::timestamp)                              AS tickets_90d,
           count(*) FILTER (WHERE t.opened_at >= (p.cutoff - 90)::timestamp AND t.category = 'network')   AS network_tickets_90d,
           avg(t.resolution_hours) FILTER (WHERE t.opened_at >= (p.cutoff - 90)::timestamp
                                             AND t.resolved_at < (p.cutoff + 1)::timestamp)               AS avg_resolution_hours_90d,
           avg(t.satisfaction_score) FILTER (WHERE t.resolved_at < (p.cutoff + 1)::timestamp)             AS avg_satisfaction
    FROM clean_tickets t
    CROSS JOIN params p
    WHERE t.opened_at >= (p.cutoff - 180)::timestamp
      AND t.opened_at <  (p.cutoff + 1)::timestamp
    GROUP BY t.customer_id
)

SELECT
    pop.customer_id,
    -- numeric features
    (p.cutoff - pop.signup_date)                                                      AS tenure_days,
    pl.monthly_price_ghs,
    pl.contract_months,
    round(u.avg_data_used_gb_3m, 2)                                                   AS avg_data_used_gb_3m,
    -- only meaningful when the earlier window has at least 2 months of data; otherwise NULL
    CASE WHEN u.prev3m_months >= 2
         THEN round(100.0 * (u.avg_data_used_gb_3m - u.avg_data_used_gb_prev3m)
                          / NULLIF(u.avg_data_used_gb_prev3m, 0), 1)
    END                                                                               AS usage_change_pct,
    round(u.avg_active_days_3m, 1)                                                    AS avg_active_days_3m,
    COALESCE(pm.failed_payments_90d, 0)                                               AS failed_payments_90d,
    COALESCE(tk.tickets_90d, 0)                                                       AS tickets_90d,
    COALESCE(tk.network_tickets_90d, 0)                                               AS network_tickets_90d,
    round(tk.avg_resolution_hours_90d, 1)                                             AS avg_resolution_hours_90d,
    round(tk.avg_satisfaction, 2)                                                     AS avg_satisfaction,
    -- categorical features
    pl.plan_name,
    pop.region,
    pop.acquisition_channel,
    pop.customer_type,
    -- label: cancelled inside the window AFTER the cutoff
    CASE WHEN pop.latest_status = 'cancelled'
          AND pop.latest_end_date >  p.cutoff
          AND pop.latest_end_date <= p.cutoff + p.window_days
         THEN 1 ELSE 0 END                                                            AS churned
FROM population pop
CROSS JOIN params p
JOIN plans pl             ON pl.plan_id = pop.plan_id
LEFT JOIN usage u         ON u.customer_id  = pop.customer_id
LEFT JOIN payments90 pm   ON pm.customer_id = pop.customer_id
LEFT JOIN tickets tk      ON tk.customer_id = pop.customer_id
ORDER BY pop.customer_id;
