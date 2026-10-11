-- =====================================================================
-- data_quality.sql  -  data-quality checks on the RAW tables
--
-- One result table: one row per check, saying how many rows are affected,
-- what kind of problem it is, how serious it is, and what the platform does
-- about it. It lives in sql/common/ because it describes the DATASET, not a
-- particular use case; any use case can point to it.
--
-- Parameter: %(data_end_date)s  (the day the data was exported)
--
-- Techniques used here:
--   * conditional aggregation  - count(*) FILTER (WHERE ...), many checks, one scan per table
--   * CTEs                     - one block of counts per table
--   * window functions         - ROW_NUMBER() to find duplicates, LAG() to find missing months
--   * subqueries               - NOT EXISTS to find accounts without a subscription
--   * unpivoting               - CROSS JOIN LATERAL (VALUES ...) turns one wide row into many check rows
--
-- Note: no percent sign appears in a query that takes parameters, because
-- the driver treats it as a placeholder marker.
-- =====================================================================

-- name: data_quality
WITH
cust AS (
    SELECT
        count(*) AS total,
        count(*) FILTER (WHERE region IS NULL)                                              AS region_missing,
        count(*) FILTER (WHERE region IS NOT NULL AND region NOT IN
            ('Greater Accra','Ashanti','Western','Central','Eastern','Volta','Northern','Bono')) AS region_inconsistent_spelling,
        count(*) FILTER (WHERE email IS NULL)                                               AS email_missing,
        count(*) FILTER (WHERE phone IS NULL)                                               AS phone_missing,
        count(*) FILTER (WHERE phone IS NOT NULL AND phone !~ '^0[0-9]{9}$')                AS phone_format_inconsistent,
        count(*) FILTER (WHERE full_name IS NULL)                                           AS name_missing,
        count(*) FILTER (WHERE full_name IS NOT NULL
                           AND (full_name = upper(full_name) OR full_name = lower(full_name))) AS name_case_inconsistent,
        count(*) FILTER (WHERE customer_type NOT IN ('individual','business'))              AS customer_type_inconsistent_case,
        count(*) FILTER (WHERE acquisition_channel IS NULL)                                 AS acquisition_channel_missing,
        count(*) FILTER (WHERE lower(customer_type) = 'individual' AND birth_year IS NULL)  AS birth_year_missing_individuals,
        count(*) FILTER (WHERE birth_year IS NOT NULL AND birth_year NOT BETWEEN 1930 AND 2008) AS birth_year_implausible,
        -- subquery: accounts that never got a subscription
        (SELECT count(*) FROM customers c
          WHERE NOT EXISTS (SELECT 1 FROM subscriptions s WHERE s.customer_id = c.customer_id)) AS accounts_without_subscription,
        -- window function: later accounts that share a phone number with an earlier one
        (SELECT count(*) FROM (
            SELECT ROW_NUMBER() OVER (PARTITION BY right(regexp_replace(phone, '\D', '', 'g'), 9)
                                      ORDER BY signup_date, customer_id) AS rn
            FROM customers WHERE phone IS NOT NULL) x
          WHERE rn > 1)                                                                     AS duplicate_accounts_by_phone
    FROM customers
),
sub AS (
    SELECT
        count(*) AS total,
        count(*) FILTER (WHERE lower(status) = 'cancelled' AND end_date IS NULL)            AS cancelled_but_end_date_missing,
        count(*) FILTER (WHERE end_date < start_date)                                       AS end_date_before_start_date,
        count(*) FILTER (WHERE status <> lower(status))                                     AS status_inconsistent_case,
        count(*) FILTER (WHERE lower(status) = 'cancelled' AND cancel_reason IS NULL)       AS cancel_reason_missing
    FROM subscriptions
),
pay AS (
    SELECT
        count(*) AS total,
        count(*) FILTER (WHERE amount_ghs IS NULL)                                          AS amount_missing,
        count(*) FILTER (WHERE amount_ghs < 0)                                              AS amount_negative,
        count(*) FILTER (WHERE method NOT IN ('momo','card','bank','cash'))                 AS method_inconsistent,
        count(*) FILTER (WHERE status NOT IN ('success','failed','reversed'))               AS status_inconsistent_case,
        count(*) FILTER (WHERE subscription_id IS NULL)                                     AS subscription_id_missing,
        count(*) FILTER (WHERE paid_at >= %(data_end_date)s::date + 1)                      AS paid_at_in_future,
        (SELECT count(*) FROM (
            SELECT ROW_NUMBER() OVER (PARTITION BY customer_id, subscription_id, paid_at, amount_ghs, method, status
                                      ORDER BY payment_id) AS rn
            FROM payments) x
          WHERE rn > 1)                                                                     AS exact_duplicate_rows
    FROM payments
),
use AS (
    SELECT
        count(*) AS total,
        count(*) FILTER (WHERE data_used_gb IS NULL)                                        AS data_used_missing,
        count(*) FILTER (WHERE data_used_gb < 0)                                            AS data_used_negative,
        count(*) FILTER (WHERE active_days NOT BETWEEN 0 AND 31)                            AS active_days_over_31,
        count(*) FILTER (WHERE usage_month <> date_trunc('month', usage_month)::date)       AS month_not_first_of_month,
        (SELECT count(*) FROM (
            SELECT ROW_NUMBER() OVER (PARTITION BY customer_id, date_trunc('month', usage_month)
                                      ORDER BY usage_id) AS rn
            FROM usage_monthly) x
          WHERE rn > 1)                                                                     AS conflicting_duplicate_month_rows,
        -- window function: months skipped between a customer's consecutive usage rows
        (SELECT coalesce(sum(gap - 1), 0) FROM (
            SELECT (extract(year FROM m) * 12 + extract(month FROM m))
                 - (extract(year FROM prev_m) * 12 + extract(month FROM prev_m)) AS gap
            FROM (SELECT date_trunc('month', usage_month)::date AS m,
                         LAG(date_trunc('month', usage_month)::date)
                             OVER (PARTITION BY customer_id ORDER BY date_trunc('month', usage_month)) AS prev_m
                  FROM usage_monthly) a
            WHERE prev_m IS NOT NULL AND m <> prev_m) b
          WHERE gap > 1)                                                                    AS missing_months_between_rows
    FROM usage_monthly
),
tix AS (
    SELECT
        count(*) AS total,
        count(*) FILTER (WHERE resolved_at < opened_at)                                     AS resolved_before_opened,
        count(*) FILTER (WHERE satisfaction_score IS NOT NULL
                           AND satisfaction_score NOT BETWEEN 1 AND 5)                      AS satisfaction_out_of_range,
        count(*) FILTER (WHERE category NOT IN ('billing','network','speed','sim/device','other')) AS category_inconsistent,
        count(*) FILTER (WHERE priority NOT IN ('low','medium','high'))                     AS priority_inconsistent_case,
        (SELECT count(*) FROM (
            SELECT ROW_NUMBER() OVER (PARTITION BY customer_id, opened_at, resolved_at, category, priority, satisfaction_score
                                      ORDER BY ticket_id) AS rn
            FROM support_tickets) x
          WHERE rn > 1)                                                                     AS exact_duplicate_rows
    FROM support_tickets
)
SELECT table_name, check_id, dimension, severity, issues, total,
       round(100.0 * issues / NULLIF(total, 0), 2) AS pct_affected,
       handling
FROM (
    SELECT 'customers' AS table_name, s.total, v.*
    FROM cust s CROSS JOIN LATERAL (VALUES
        ('customers.accounts_without_subscription',   'integrity',    'high',   s.accounts_without_subscription,  'Excluded: no subscription means nothing to predict'),
        ('customers.duplicate_accounts_by_phone',     'uniqueness',   'high',   s.duplicate_accounts_by_phone,    'Flagged is_duplicate in clean_customers; excluded from analysis'),
        ('customers.region_missing',                  'completeness', 'medium', s.region_missing,                 'Set to Unknown'),
        ('customers.region_inconsistent_spelling',    'consistency',  'medium', s.region_inconsistent_spelling,   'Mapped to 8 canonical regions'),
        ('customers.email_missing',                   'completeness', 'low',    s.email_missing,                  'Left missing (not used by the model)'),
        ('customers.phone_missing',                   'completeness', 'medium', s.phone_missing,                  'Left missing; duplicates with no phone cannot be matched'),
        ('customers.phone_format_inconsistent',       'consistency',  'low',    s.phone_format_inconsistent,      'Reduced to last 9 digits (phone_key)'),
        ('customers.name_missing',                    'completeness', 'low',    s.name_missing,                   'Left missing'),
        ('customers.name_case_inconsistent',          'consistency',  'low',    s.name_case_inconsistent,         'Title-cased'),
        ('customers.customer_type_inconsistent_case', 'consistency',  'medium', s.customer_type_inconsistent_case,'Lower-cased'),
        ('customers.acquisition_channel_missing',     'completeness', 'medium', s.acquisition_channel_missing,    'Set to unknown'),
        ('customers.birth_year_missing_individuals',  'completeness', 'low',    s.birth_year_missing_individuals, 'Left missing (not used by the model)'),
        ('customers.birth_year_implausible',          'validity',     'low',    s.birth_year_implausible,         'Set to NULL')
    ) AS v(check_id, dimension, severity, issues, handling)
    UNION ALL
    SELECT 'subscriptions', s.total, v.*
    FROM sub s CROSS JOIN LATERAL (VALUES
        ('subscriptions.cancelled_but_end_date_missing', 'completeness', 'high',   s.cancelled_but_end_date_missing, 'Excluded from training: cancellation date unknown, label untrustworthy'),
        ('subscriptions.end_date_before_start_date',     'validity',     'high',   s.end_date_before_start_date,     'End date set to unknown (NULL)'),
        ('subscriptions.status_inconsistent_case',       'consistency',  'medium', s.status_inconsistent_case,       'Lower-cased'),
        ('subscriptions.cancel_reason_missing',          'completeness', 'low',    s.cancel_reason_missing,          'Left missing (reported as unknown)')
    ) AS v(check_id, dimension, severity, issues, handling)
    UNION ALL
    SELECT 'payments', s.total, v.*
    FROM pay s CROSS JOIN LATERAL (VALUES
        ('payments.exact_duplicate_rows',       'uniqueness',   'high',   s.exact_duplicate_rows,     'Removed in clean_payments'),
        ('payments.amount_missing',             'completeness', 'medium', s.amount_missing,           'Filled with the plan price'),
        ('payments.amount_negative',            'validity',     'medium', s.amount_negative,          'Treated as invalid, filled with the plan price'),
        ('payments.method_inconsistent',        'consistency',  'low',    s.method_inconsistent,      'Normalised to momo/card/bank/cash'),
        ('payments.status_inconsistent_case',   'consistency',  'medium', s.status_inconsistent_case, 'Lower-cased'),
        ('payments.subscription_id_missing',    'integrity',    'low',    s.subscription_id_missing,  'Kept; amount cannot be filled from a plan'),
        ('payments.paid_at_in_future',          'validity',     'high',   s.paid_at_in_future,        'Ignored: every analysis stops at the cutoff or export date')
    ) AS v(check_id, dimension, severity, issues, handling)
    UNION ALL
    SELECT 'usage_monthly', s.total, v.*
    FROM use s CROSS JOIN LATERAL (VALUES
        ('usage_monthly.conflicting_duplicate_month_rows', 'uniqueness',   'high',   s.conflicting_duplicate_month_rows, 'Collapsed to one row per customer-month (keeps the original)'),
        ('usage_monthly.data_used_missing',                'completeness', 'medium', s.data_used_missing,                'Left missing; averages skip it'),
        ('usage_monthly.data_used_negative',               'validity',     'medium', s.data_used_negative,               'Set to NULL'),
        ('usage_monthly.active_days_over_31',              'validity',     'low',    s.active_days_over_31,              'Set to NULL'),
        ('usage_monthly.month_not_first_of_month',         'consistency',  'low',    s.month_not_first_of_month,         'Normalised to the 1st'),
        ('usage_monthly.missing_months_between_rows',      'completeness', 'medium', s.missing_months_between_rows,      'Left as gaps; averages use the months that exist')
    ) AS v(check_id, dimension, severity, issues, handling)
    UNION ALL
    SELECT 'support_tickets', s.total, v.*
    FROM tix s CROSS JOIN LATERAL (VALUES
        ('support_tickets.exact_duplicate_rows',       'uniqueness',   'medium', s.exact_duplicate_rows,       'Removed in clean_tickets'),
        ('support_tickets.resolved_before_opened',     'validity',     'medium', s.resolved_before_opened,     'Resolution time set to unknown'),
        ('support_tickets.satisfaction_out_of_range',  'validity',     'low',    s.satisfaction_out_of_range,  'Set to NULL'),
        ('support_tickets.category_inconsistent',      'consistency',  'low',    s.category_inconsistent,      'Normalised to 5 categories'),
        ('support_tickets.priority_inconsistent_case', 'consistency',  'low',    s.priority_inconsistent_case, 'Lower-cased')
    ) AS v(check_id, dimension, severity, issues, handling)
) checks
ORDER BY CASE severity WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END, issues DESC, check_id;
