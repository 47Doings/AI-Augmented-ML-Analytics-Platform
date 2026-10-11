-- =====================================================================
-- business_questions.sql (revenue)  -  what the finance team would ask
--
-- Every query starts from the CLEAN views. Query 6 reads training_set
-- (build it first with: python -m scripts.run_sql training --use-case revenue).
-- Parameters: %(cutoff_date)s, %(data_end_date)s. No percent sign appears in
-- parametrised queries (the driver reads it as a parameter marker).
-- Revenue = successful payments, using amount_ghs_filled.
-- =====================================================================

-- name: monthly_revenue_trend
-- Question: is revenue growing, and how stable is it?
-- Techniques: CTE, aggregation with FILTER, window functions (LAG, moving average)
WITH monthly AS (
    SELECT date_trunc('month', paid_at)::date                         AS month_start,
           sum(amount_ghs_filled) FILTER (WHERE status = 'success')   AS revenue_ghs,
           count(*) FILTER (WHERE status = 'success')                 AS successful_payments,
           count(*) FILTER (WHERE status = 'failed')                  AS failed_payments
    FROM clean_payments
    WHERE paid_at <  date_trunc('month', %(cutoff_date)s::date + 1)::timestamp
    GROUP BY 1
)
SELECT month_start,
       round(revenue_ghs, 0)                                                              AS revenue_ghs,
       successful_payments,
       failed_payments,
       round(100.0 * (revenue_ghs - LAG(revenue_ghs) OVER (ORDER BY month_start))
                   / NULLIF(LAG(revenue_ghs) OVER (ORDER BY month_start), 0), 1)          AS mom_growth_pct,
       round(avg(revenue_ghs) OVER (ORDER BY month_start ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 0) AS moving_avg_3m
FROM monthly
ORDER BY month_start;

-- name: revenue_by_plan
-- Question: which plans bring the money in the last complete month?
-- Techniques: joins, aggregation, window function (share of total)
SELECT pl.plan_name,
       count(DISTINCT p.customer_id)                                 AS paying_customers,
       round(sum(p.amount_ghs_filled), 0)                            AS revenue_ghs,
       round(100.0 * sum(p.amount_ghs_filled) / sum(sum(p.amount_ghs_filled)) OVER (), 1) AS share_of_revenue_pct,
       round(avg(p.amount_ghs_filled), 2)                            AS avg_payment_ghs
FROM clean_payments p
JOIN subscriptions s ON s.subscription_id = p.subscription_id
JOIN plans pl        ON pl.plan_id = s.plan_id
WHERE p.status = 'success'
  AND p.paid_at >= (date_trunc('month', %(cutoff_date)s::date + 1) - interval '1 month')::timestamp
  AND p.paid_at <  date_trunc('month', %(cutoff_date)s::date + 1)::timestamp
GROUP BY pl.plan_name
ORDER BY revenue_ghs DESC;

-- name: failure_rate_by_method
-- Question: which payment method fails most (money we should be able to collect)?
-- Techniques: conditional aggregation, ratio, filtering to the last 6 complete months
SELECT method,
       count(*)                                                           AS attempts,
       count(*) FILTER (WHERE status = 'failed')                          AS failed,
       round(100.0 * count(*) FILTER (WHERE status = 'failed') / count(*), 1) AS failure_rate_pct,
       round(sum(amount_ghs_filled) FILTER (WHERE status = 'failed'), 0)  AS failed_value_ghs
FROM clean_payments
WHERE paid_at >= (date_trunc('month', %(cutoff_date)s::date + 1) - interval '6 months')::timestamp
  AND paid_at <  date_trunc('month', %(cutoff_date)s::date + 1)::timestamp
GROUP BY method
ORDER BY failure_rate_pct DESC;

-- name: top_region_channel_by_revenue
-- Question: where does revenue come from (region x acquisition channel), ranked inside each region?
-- Techniques: joins to clean customers (duplicates excluded), RANK() window function
WITH per_customer AS MATERIALIZED (       -- aggregate the payments first, then join the (small) customer table
    SELECT p.customer_id, sum(p.amount_ghs_filled) AS revenue_ghs
    FROM clean_payments p
    WHERE p.status = 'success'
      AND p.paid_at >= (date_trunc('month', %(cutoff_date)s::date + 1) - interval '12 months')::timestamp
      AND p.paid_at <  date_trunc('month', %(cutoff_date)s::date + 1)::timestamp
    GROUP BY p.customer_id
),
rev AS (
    SELECT c.region, c.acquisition_channel,
           count(*)             AS customers,
           sum(pc.revenue_ghs)  AS revenue_ghs
    FROM per_customer pc
    JOIN clean_customers c ON c.customer_id = pc.customer_id AND NOT c.is_duplicate
    GROUP BY c.region, c.acquisition_channel
)
SELECT region, acquisition_channel, customers, round(revenue_ghs, 0) AS revenue_ghs,
       round(revenue_ghs / customers, 0)                                   AS revenue_per_customer_ghs,
       RANK() OVER (PARTITION BY region ORDER BY revenue_ghs DESC)         AS rank_in_region
FROM rev
ORDER BY region, rank_in_region;

-- name: revenue_concentration
-- Question: how much of the money comes from the top 10 percent of customers?
-- Techniques: CTE, NTILE and cumulative SUM window functions
WITH paid AS MATERIALIZED (
    SELECT p.customer_id, sum(p.amount_ghs_filled) AS revenue_ghs
    FROM clean_payments p
    WHERE p.status = 'success'
      AND p.paid_at >= (date_trunc('month', %(cutoff_date)s::date + 1) - interval '12 months')::timestamp
      AND p.paid_at <  date_trunc('month', %(cutoff_date)s::date + 1)::timestamp
    GROUP BY p.customer_id
),
per_customer AS (                       -- real customers only (duplicate accounts excluded)
    SELECT pd.customer_id, pd.revenue_ghs
    FROM paid pd
    JOIN clean_customers c ON c.customer_id = pd.customer_id AND NOT c.is_duplicate
),
bucketed AS (
    SELECT *, NTILE(10) OVER (ORDER BY revenue_ghs DESC, customer_id) AS decile
    FROM per_customer
)
SELECT decile,
       count(*)                                                                    AS customers,
       round(sum(revenue_ghs), 0)                                                  AS revenue_ghs,
       round(100.0 * sum(revenue_ghs) / sum(sum(revenue_ghs)) OVER (), 1)          AS share_pct,
       round(100.0 * sum(sum(revenue_ghs)) OVER (ORDER BY decile)
                   / sum(sum(revenue_ghs)) OVER (), 1)                             AS cumulative_share_pct
FROM bucketed
GROUP BY decile
ORDER BY decile;

-- name: forecast_vs_list_price
-- Question: if every active customer paid their plan price, what would the next 30 days bring,
--           and how much is missing? (reads training_set; the gap is the money at risk)
-- Techniques: aggregation over the training table, ratio, total row via UNION ALL
WITH by_plan AS (
    SELECT plan_name,
           count(*)                         AS customers,
           sum(monthly_price_ghs)           AS expected_ghs,
           sum(revenue_next_30d)::numeric   AS actual_ghs
    FROM training_set
    GROUP BY plan_name
)
SELECT plan_name, customers, expected_ghs, actual_ghs, shortfall_ghs, shortfall_pct
FROM (
    SELECT plan_name, customers, round(expected_ghs, 0) AS expected_ghs, round(actual_ghs, 0) AS actual_ghs,
           round(expected_ghs - actual_ghs, 0)                                     AS shortfall_ghs,
           round(100.0 * (expected_ghs - actual_ghs) / NULLIF(expected_ghs, 0), 1) AS shortfall_pct,
           0 AS sort_group, expected_ghs AS sort_value
    FROM by_plan
    UNION ALL
    SELECT 'ALL PLANS', sum(customers), round(sum(expected_ghs), 0), round(sum(actual_ghs), 0),
           round(sum(expected_ghs) - sum(actual_ghs), 0),
           round(100.0 * (sum(expected_ghs) - sum(actual_ghs)) / NULLIF(sum(expected_ghs), 0), 1),
           1, 0
    FROM by_plan
) t
ORDER BY sort_group, sort_value DESC;
