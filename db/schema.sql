-- =====================================================================
-- schema.sql  —  business data template (Step 1)
--
-- Demo business: a fictional Accra-based mobile data / home broadband
-- subscription company. Starting use case: customer churn.
--
-- HOW THIS FILE IS USED
--   Every workspace gets its own Postgres schema (e.g. ws_demo).
--   This file is a TEMPLATE that is run inside that schema:
--
--       CREATE SCHEMA ws_demo;
--       SET search_path TO ws_demo;
--       \i schema.sql
--
--   That is why no table below has a schema prefix or a workspace_id.
--
-- DESIGN NOTE: this is the RAW layer. Constraints are deliberately
-- loose (text instead of enums, no UNIQUE on email/phone, nullable
-- descriptive columns) because the synthetic data must contain
-- realistic mess: missing values, duplicates, inconsistent spellings.
-- The SQL layer (Step 4) finds and handles that mess. Primary keys and
-- foreign keys are still enforced so the relationships are real.
-- =====================================================================

-- ---------------------------------------------------------------------
-- plans: the products the company sells (small lookup table)
-- ---------------------------------------------------------------------
CREATE TABLE plans (
    plan_id            integer       PRIMARY KEY,
    plan_name          text          NOT NULL,
    monthly_price_ghs  numeric(10,2) NOT NULL,
    data_cap_gb        integer,                      -- NULL = unlimited
    contract_months    integer       NOT NULL DEFAULT 0   -- 0 = no contract
);

-- ---------------------------------------------------------------------
-- customers: one row per person/business who signed up
-- (duplicates with different customer_ids will be injected on purpose)
-- ---------------------------------------------------------------------
CREATE TABLE customers (
    customer_id          integer PRIMARY KEY,
    full_name            text,
    email                text,
    phone                text,
    region               text,        -- messy on purpose: 'Greater Accra' / 'greater accra' / 'Accra'
    customer_type        text,        -- 'individual' or 'business'
    acquisition_channel  text,        -- 'online', 'agent', 'referral', 'walk-in'
    signup_date          date NOT NULL,
    birth_year           integer      -- often missing
);

-- ---------------------------------------------------------------------
-- subscriptions: a customer's plan history.
-- One customer can have several rows (upgrade/downgrade/cancel).
-- Churn is derived from here: a customer whose last subscription has
-- status 'cancelled' and an end_date has churned.
-- ---------------------------------------------------------------------
CREATE TABLE subscriptions (
    subscription_id  integer PRIMARY KEY,
    customer_id      integer NOT NULL REFERENCES customers(customer_id),
    plan_id          integer NOT NULL REFERENCES plans(plan_id),
    start_date       date    NOT NULL,
    end_date         date,            -- NULL = still running
    status           text    NOT NULL, -- 'active', 'cancelled', 'suspended'
    cancel_reason    text             -- only filled for cancelled rows (and not always)
);

-- ---------------------------------------------------------------------
-- payments: money received (or failed attempts)
-- ---------------------------------------------------------------------
CREATE TABLE payments (
    payment_id       bigint        PRIMARY KEY,
    customer_id      integer       NOT NULL REFERENCES customers(customer_id),
    subscription_id  integer       REFERENCES subscriptions(subscription_id),
    paid_at          timestamp     NOT NULL,
    amount_ghs       numeric(10,2),
    method           text,         -- 'momo', 'card', 'bank', 'cash'
    status           text          -- 'success', 'failed', 'reversed'
);

-- ---------------------------------------------------------------------
-- usage_monthly: how much each customer actually used the service
-- No UNIQUE (customer_id, usage_month) so duplicate rows can exist.
-- ---------------------------------------------------------------------
CREATE TABLE usage_monthly (
    usage_id       bigint        PRIMARY KEY,
    customer_id    integer       NOT NULL REFERENCES customers(customer_id),
    usage_month    date          NOT NULL,   -- first day of the month
    data_used_gb   numeric(8,2),
    active_days    integer
);

-- ---------------------------------------------------------------------
-- support_tickets: complaints and requests
-- ---------------------------------------------------------------------
CREATE TABLE support_tickets (
    ticket_id           bigint    PRIMARY KEY,
    customer_id         integer   NOT NULL REFERENCES customers(customer_id),
    opened_at           timestamp NOT NULL,
    resolved_at         timestamp,            -- NULL = still open (or never logged)
    category            text,                 -- 'billing', 'network', 'speed', 'sim/device', 'other'
    priority            text,                 -- 'low', 'medium', 'high'
    satisfaction_score  integer               -- 1-5, mostly missing
);

-- ---------------------------------------------------------------------
-- Indexes on the foreign keys we will join and group on constantly
-- ---------------------------------------------------------------------
CREATE INDEX idx_subscriptions_customer ON subscriptions (customer_id);
CREATE INDEX idx_payments_customer      ON payments (customer_id);
CREATE INDEX idx_payments_paid_at       ON payments (paid_at);
CREATE INDEX idx_usage_customer_month   ON usage_monthly (customer_id, usage_month);
CREATE INDEX idx_tickets_customer       ON support_tickets (customer_id);

-- ---------------------------------------------------------------------
-- Column documentation stored IN the database.
-- Later (Step 8) the AI layer can read these so it knows what columns
-- mean instead of guessing from names.
-- ---------------------------------------------------------------------
COMMENT ON TABLE  customers          IS 'One row per customer account. May contain duplicate people under different ids.';
COMMENT ON TABLE  subscriptions      IS 'Plan history per customer. Latest row per customer shows current state; cancelled + end_date means churned.';
COMMENT ON TABLE  payments           IS 'Payment attempts. Only status = success counts as revenue.';
COMMENT ON TABLE  usage_monthly      IS 'Monthly service usage per customer.';
COMMENT ON TABLE  support_tickets    IS 'Customer support interactions.';
COMMENT ON COLUMN subscriptions.end_date          IS 'NULL means the subscription is still running.';
COMMENT ON COLUMN payments.amount_ghs             IS 'Amount in Ghana cedis (GHS).';
COMMENT ON COLUMN support_tickets.satisfaction_score IS '1 (worst) to 5 (best); frequently missing.';
