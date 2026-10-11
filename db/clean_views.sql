-- =====================================================================
-- clean_views.sql  -  the CLEAN layer (Step 4)
--
-- The tables from schema.sql are the RAW layer: they hold the mess exactly as
-- a real system would. These views sit on top and apply the cleaning rules
-- ONCE, in one place, so every later query (business questions, training
-- table, KPIs) starts from clean data and nobody repeats the rules.
--
--     raw tables  ->  clean_* views  ->  training_set  ->  ML
--
-- Run inside a workspace schema (like schema.sql). Safe to run repeatedly:
-- CREATE OR REPLACE VIEW. Views bind to the tables in whatever schema is on
-- the search_path when they are created, so each workspace gets its own.
--
-- The raw data is never modified. A rule can change without reloading data.
-- =====================================================================

-- ---------------------------------------------------------------------
-- clean_customers
--   * region spellings collapsed to 8 canonical names ('Unknown' if missing)
--   * customer_type / channel lower-cased; missing channel -> 'unknown'
--   * names tidied (trim + Title Case); email lower-cased
--   * implausible birth years -> NULL
--   * phone_key = last 9 digits, so 0241234567, +233241234567 and
--     "024 123 4567" all become the same key
--   * is_duplicate = TRUE for every account after the first one that
--     shares a phone_key (window function)
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW clean_customers AS
WITH norm AS (
    SELECT c.*,
           lower(regexp_replace(btrim(c.region), '\s+', ' ', 'g'))      AS region_key,
           right(regexp_replace(c.phone, '\D', '', 'g'), 9)             AS phone_digits
    FROM customers c
)
SELECT
    customer_id,
    initcap(NULLIF(btrim(full_name), ''))                                AS full_name,
    lower(NULLIF(btrim(email), ''))                                      AS email,
    NULLIF(phone_digits, '')                                             AS phone_key,
    CASE
        WHEN region_key IN ('greater accra', 'gt. accra', 'accra')       THEN 'Greater Accra'
        WHEN region_key IN ('ashanti', 'ashanti region', 'kumasi')       THEN 'Ashanti'
        WHEN region_key IN ('western', 'western region')                 THEN 'Western'
        WHEN region_key IN ('central', 'central region')                 THEN 'Central'
        WHEN region_key IN ('eastern', 'eastern region')                 THEN 'Eastern'
        WHEN region_key IN ('volta', 'volta region')                     THEN 'Volta'
        WHEN region_key IN ('northern', 'northern region')               THEN 'Northern'
        WHEN region_key IN ('bono', 'brong ahafo')                       THEN 'Bono'
        WHEN region_key IS NULL OR region_key = ''                       THEN 'Unknown'
        ELSE initcap(region_key)
    END                                                                  AS region,
    CASE WHEN lower(btrim(customer_type)) IN ('individual', 'business')
         THEN lower(btrim(customer_type)) ELSE 'unknown' END             AS customer_type,
    COALESCE(lower(NULLIF(btrim(acquisition_channel), '')), 'unknown')   AS acquisition_channel,
    signup_date,
    CASE WHEN birth_year BETWEEN 1930 AND 2008 THEN birth_year END       AS birth_year,
    CASE
        WHEN NULLIF(phone_digits, '') IS NULL THEN FALSE   -- no phone: cannot match, do not guess
        ELSE ROW_NUMBER() OVER (PARTITION BY NULLIF(phone_digits, '')
                                ORDER BY signup_date, customer_id) > 1
    END                                                                  AS is_duplicate
FROM norm;

-- ---------------------------------------------------------------------
-- clean_subscriptions
--   * status lower-cased
--   * an end_date earlier than start_date is impossible -> treated as unknown (NULL)
--   * is_latest = this is the customer's newest subscription (window function).
--     A plan change closes the old row as 'cancelled' / 'plan_change', so only
--     the LATEST row says whether the customer has really left.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW clean_subscriptions AS
SELECT
    s.subscription_id,
    s.customer_id,
    s.plan_id,
    s.start_date,
    CASE WHEN s.end_date >= s.start_date THEN s.end_date END             AS end_date,
    lower(btrim(s.status))                                               AS status,
    (s.end_date < s.start_date)                                          AS end_date_invalid,
    s.cancel_reason,
    ROW_NUMBER() OVER (PARTITION BY s.customer_id
                       ORDER BY s.start_date DESC, s.subscription_id DESC) = 1   AS is_latest
FROM subscriptions s;

-- ---------------------------------------------------------------------
-- clean_payments
--   * exact duplicate payments removed (keeps the lowest payment_id)
--   * method / status normalised
--   * a zero or negative amount is invalid -> NULL in amount_ghs
--   * amount_ghs_filled = amount, or the plan's monthly price when the amount
--     is missing/invalid (billing is a fixed price per plan)
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW clean_payments AS
WITH ranked AS (
    SELECT p.*,
           ROW_NUMBER() OVER (
               PARTITION BY p.customer_id, p.subscription_id, p.paid_at, p.amount_ghs,
                            lower(btrim(p.method)), lower(btrim(p.status))
               ORDER BY p.payment_id) AS rn
    FROM payments p
)
SELECT
    r.payment_id,
    r.customer_id,
    r.subscription_id,
    r.paid_at,
    CASE WHEN r.amount_ghs > 0 THEN r.amount_ghs END                     AS amount_ghs,
    COALESCE(CASE WHEN r.amount_ghs > 0 THEN r.amount_ghs END,
             pl.monthly_price_ghs)                                       AS amount_ghs_filled,
    CASE lower(btrim(r.method))
        WHEN 'momo'          THEN 'momo'
        WHEN 'mobile money'  THEN 'momo'
        WHEN 'card'          THEN 'card'
        WHEN 'bank'          THEN 'bank'
        WHEN 'bank transfer' THEN 'bank'
        WHEN 'cash'          THEN 'cash'
        ELSE 'other'
    END                                                                  AS method,
    lower(btrim(r.status))                                               AS status
FROM ranked r
LEFT JOIN subscriptions s ON s.subscription_id = r.subscription_id
LEFT JOIN plans pl        ON pl.plan_id = s.plan_id
WHERE r.rn = 1;

-- ---------------------------------------------------------------------
-- clean_usage
--   * month normalised to the 1st; duplicate months collapsed (keeps the
--     original row = lowest usage_id)
--   * negative GB -> NULL; active_days outside 0-31 -> NULL
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW clean_usage AS
WITH ranked AS (
    SELECT u.*,
           date_trunc('month', u.usage_month)::date AS month_start,
           ROW_NUMBER() OVER (PARTITION BY u.customer_id, date_trunc('month', u.usage_month)
                              ORDER BY u.usage_id) AS rn
    FROM usage_monthly u
)
SELECT
    usage_id,
    customer_id,
    month_start                                                          AS usage_month,
    CASE WHEN data_used_gb >= 0 THEN data_used_gb END                    AS data_used_gb,
    CASE WHEN active_days BETWEEN 0 AND 31 THEN active_days END          AS active_days
FROM ranked
WHERE rn = 1;

-- ---------------------------------------------------------------------
-- clean_tickets
--   * exact duplicate tickets removed
--   * category / priority normalised
--   * resolved_at earlier than opened_at is impossible -> NULL (treated as unresolved)
--   * satisfaction outside 1-5 -> NULL
--   * resolution_hours derived from the cleaned timestamps
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW clean_tickets AS
WITH ranked AS (
    SELECT t.*,
           ROW_NUMBER() OVER (
               PARTITION BY t.customer_id, t.opened_at, t.resolved_at, t.category,
                            t.priority, t.satisfaction_score
               ORDER BY t.ticket_id) AS rn
    FROM support_tickets t
),
fixed AS (
    SELECT r.ticket_id, r.customer_id, r.opened_at,
           CASE WHEN r.resolved_at >= r.opened_at THEN r.resolved_at END AS resolved_at,
           CASE
               WHEN lower(r.category) LIKE '%bill%'                              THEN 'billing'
               WHEN lower(r.category) LIKE '%network%'                           THEN 'network'
               WHEN lower(r.category) LIKE '%speed%' OR lower(r.category) LIKE '%slow%' THEN 'speed'
               WHEN lower(r.category) LIKE '%sim%'                               THEN 'sim/device'
               ELSE 'other'
           END                                                                   AS category,
           lower(btrim(r.priority))                                              AS priority,
           CASE WHEN r.satisfaction_score BETWEEN 1 AND 5 THEN r.satisfaction_score END AS satisfaction_score
    FROM ranked r
    WHERE r.rn = 1
)
SELECT f.*,
       EXTRACT(EPOCH FROM (f.resolved_at - f.opened_at)) / 3600.0        AS resolution_hours
FROM fixed f;

-- ---------------------------------------------------------------------
-- Descriptions stored in the database. The AI layer (Step 8) can read these.
-- ---------------------------------------------------------------------
COMMENT ON VIEW clean_customers     IS 'Cleaned customers. is_duplicate = a later account sharing a phone number with an earlier one. Exclude duplicates from analysis.';
COMMENT ON VIEW clean_subscriptions IS 'Cleaned subscriptions. is_latest marks each customer''s newest row; only that row says whether the customer has left. cancel_reason plan_change is NOT churn.';
COMMENT ON VIEW clean_payments      IS 'Cleaned payments, exact duplicates removed. Revenue = status success, using amount_ghs_filled.';
COMMENT ON VIEW clean_usage         IS 'Cleaned monthly usage, one row per customer per month.';
COMMENT ON VIEW clean_tickets       IS 'Cleaned support tickets with resolution_hours; unresolved tickets have NULL resolved_at.';
