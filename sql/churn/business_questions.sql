-- =====================================================================
-- business_questions.sql  -  the questions the retention team would ask
--
-- Every query starts from the CLEAN views. Queries 5 and 6 read
-- training_set, which is built from training_table.sql (run that first).
-- Parameter: %(data_end_date)s. No percent sign appears in parametrised
-- queries (the driver reads it as a parameter marker).
-- =====================================================================

-- name: monthly_churn_trend
-- Question: is churn getting better or worse month by month?
-- Techniques: CTEs, generate_series, conditional aggregation, window functions (moving average, LAG)
WITH lifecycle AS (
    SELECT c.signup_date,
           CASE WHEN s.status = 'cancelled' THEN s.end_date END AS churn_date
    FROM clean_customers c
    JOIN clean_subscriptions s ON s.customer_id = c.customer_id AND s.is_latest
    WHERE NOT c.is_duplicate
      AND NOT (s.status = 'cancelled' AND s.end_date IS NULL)
),
months AS (
    SELECT generate_series((SELECT date_trunc('month', min(signup_date)) FROM lifecycle),
                           %(data_end_date)s::date, interval '1 month')::date AS month_start
),
monthly AS (
    SELECT m.month_start,
           count(*) FILTER (WHERE l.signup_date < m.month_start
                              AND (l.churn_date IS NULL OR l.churn_date >= m.month_start))          AS customers_at_start,
           count(*) FILTER (WHERE l.churn_date >= m.month_start
                              AND l.churn_date <  (m.month_start + interval '1 month'))             AS churned,
           count(*) FILTER (WHERE l.signup_date >= m.month_start
                              AND l.signup_date <  (m.month_start + interval '1 month'))            AS new_customers
    FROM months m
    CROSS JOIN lifecycle l
    GROUP BY m.month_start
)
SELECT month_start, new_customers, customers_at_start, churned,
       round(100.0 * churned / NULLIF(customers_at_start, 0), 2)                                    AS churn_rate_pct,
       round(100.0 * AVG(churned::numeric / NULLIF(customers_at_start, 0))
                 OVER (ORDER BY month_start ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2)           AS churn_rate_3m_avg_pct,
       churned - LAG(churned) OVER (ORDER BY month_start)                                           AS change_vs_prev_month
FROM monthly
ORDER BY month_start;

-- name: churn_by_plan
-- Question: which plans lose the most customers?
-- Techniques: CTE, join, conditional aggregation
WITH base AS (
    SELECT s.plan_id, s.status
    FROM clean_subscriptions s
    JOIN clean_customers c ON c.customer_id = s.customer_id
    WHERE s.is_latest
      AND NOT c.is_duplicate
      AND NOT (s.status = 'cancelled' AND s.end_date IS NULL)
)
SELECT p.plan_name, p.monthly_price_ghs, p.contract_months,
       count(*)                                                              AS customers,
       count(*) FILTER (WHERE b.status = 'cancelled')                        AS cancelled,
       round(100.0 * count(*) FILTER (WHERE b.status = 'cancelled') / count(*), 1) AS churn_pct
FROM base b
JOIN plans p ON p.plan_id = b.plan_id
GROUP BY p.plan_name, p.monthly_price_ghs, p.contract_months
ORDER BY churn_pct DESC;

-- name: churn_by_channel_region
-- Question: which channel + region combinations churn most? (top 2 per channel)
-- Techniques: CTEs, conditional aggregation, window function RANK, subquery in FROM
WITH base AS (
    SELECT c.acquisition_channel, c.region, (s.status = 'cancelled') AS churned
    FROM clean_customers c
    JOIN clean_subscriptions s ON s.customer_id = c.customer_id AND s.is_latest
    WHERE NOT c.is_duplicate
      AND NOT (s.status = 'cancelled' AND s.end_date IS NULL)
),
segments AS (
    SELECT acquisition_channel, region,
           count(*)                                       AS customers,
           count(*) FILTER (WHERE churned)                AS cancelled,
           round(100.0 * count(*) FILTER (WHERE churned) / count(*), 1) AS churn_pct
    FROM base
    GROUP BY acquisition_channel, region
    HAVING count(*) >= 40
)
SELECT *
FROM (SELECT segments.*,
             RANK() OVER (PARTITION BY acquisition_channel ORDER BY churn_pct DESC) AS rank_in_channel
      FROM segments) ranked
WHERE rank_in_channel <= 2
ORDER BY acquisition_channel, rank_in_channel;

-- name: tenure_at_churn
-- Question: how long do customers stay before they leave?
-- Techniques: CTEs, CASE bucketing, window functions over aggregates (share and running total)
WITH churned AS (
    SELECT (s.end_date - c.signup_date) AS tenure_days
    FROM clean_customers c
    JOIN clean_subscriptions s ON s.customer_id = c.customer_id AND s.is_latest
    WHERE NOT c.is_duplicate AND s.status = 'cancelled' AND s.end_date IS NOT NULL
),
bucketed AS (
    SELECT CASE WHEN tenure_days < 90  THEN '1: under 3 months'
                WHEN tenure_days < 180 THEN '2: 3-6 months'
                WHEN tenure_days < 365 THEN '3: 6-12 months'
                ELSE                        '4: 12+ months' END AS tenure_bucket
    FROM churned
)
SELECT tenure_bucket,
       count(*)                                                              AS churned_customers,
       round(100.0 * count(*) / sum(count(*)) OVER (), 1)                    AS pct_of_all_churn,
       round(100.0 * sum(count(*)) OVER (ORDER BY tenure_bucket) / sum(count(*)) OVER (), 1) AS cumulative_pct
FROM bucketed
GROUP BY tenure_bucket
ORDER BY tenure_bucket;

-- name: signals_by_outcome
-- Question: what do customers who are about to leave look like, compared with those who stay?
-- Reads training_set (features measured BEFORE the cutoff, outcome AFTER it).
-- Techniques: conditional label, aggregation
SELECT CASE churned WHEN 1 THEN 'cancels within the window' ELSE 'stays' END AS outcome,
       count(*)                                          AS customers,
       round(avg(failed_payments_90d), 2)                AS avg_failed_payments_90d,
       round(avg(tickets_90d), 2)                        AS avg_tickets_90d,
       round(avg(network_tickets_90d), 2)                AS avg_network_tickets_90d,
       round(avg(usage_change_pct), 1)                   AS avg_usage_change_pct,
       round(avg(avg_data_used_gb_3m), 1)                AS avg_data_gb_3m,
       round(avg(tenure_days))                           AS avg_tenure_days
FROM training_set
GROUP BY churned
ORDER BY churned DESC;

-- name: high_risk_segments
-- Question: which plan + tenure groups churn clearly more than average? (and by how much)
-- Techniques: subqueries (scalar subquery in HAVING and in SELECT), CASE bucketing
SELECT plan_name,
       CASE WHEN tenure_days < 180 THEN 'under 6 months'
            WHEN tenure_days < 365 THEN '6-12 months'
            ELSE '12+ months' END                                            AS tenure_band,
       count(*)                                                              AS customers,
       sum(churned)                                                          AS will_churn,
       round(100.0 * avg(churned), 1)                                        AS churn_pct,
       round(avg(churned) / (SELECT avg(churned) FROM training_set), 2)      AS times_the_average
FROM training_set
GROUP BY plan_name, tenure_band
HAVING count(*) >= 30
   AND avg(churned) > (SELECT avg(churned) FROM training_set)
ORDER BY churn_pct DESC;

-- name: monthly_revenue_trend
-- Question: how much money comes in each month, and how often do payments fail?
-- Techniques: CTE, conditional aggregation, window functions (LAG growth, running total)
WITH monthly AS (
    SELECT date_trunc('month', paid_at)::date                                AS month_start,
           sum(amount_ghs_filled) FILTER (WHERE status = 'success')          AS revenue_ghs,
           count(*) FILTER (WHERE status = 'success')                        AS successful_payments,
           count(*) FILTER (WHERE status = 'failed')                         AS failed_payments
    FROM clean_payments
    WHERE paid_at < (%(data_end_date)s::date + 1)::timestamp
    GROUP BY 1
)
SELECT month_start,
       round(revenue_ghs, 2)                                                 AS revenue_ghs,
       successful_payments,
       round(100.0 * failed_payments / NULLIF(successful_payments + failed_payments, 0), 2) AS failure_rate_pct,
       round(100.0 * (revenue_ghs - LAG(revenue_ghs) OVER (ORDER BY month_start))
                   / NULLIF(LAG(revenue_ghs) OVER (ORDER BY month_start), 0), 1)            AS growth_pct,
       round(sum(revenue_ghs) OVER (ORDER BY month_start), 2)                AS cumulative_revenue_ghs
FROM monthly
ORDER BY month_start;
