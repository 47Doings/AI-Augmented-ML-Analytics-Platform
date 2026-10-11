"""Synthetic data generator for the demo business: an Accra mobile data / home
broadband subscription company.

HOW IT WORKS (read this before the interview)
---------------------------------------------
1. SIMULATE A CLEAN WORLD. For each customer we decide who they are, which plan
   they buy, and whether/when they cancel. Cancellation is driven by customer
   traits (channel, plan, contract, region, tenure) plus randomness.
2. GENERATE BEHAVIOUR THAT FOLLOWS THE DECISION. Customers who are about to
   cancel tend to show "distress" in the months before: usage drops, payments
   fail, support tickets rise. About 30% of churners are "silent" (no warning
   signs) and some happy customers have scary-looking patches (false alarms),
   so the signal is real but imperfect. That is what a model can learn from.
3. INJECT REAL-WORLD MESS. Duplicates, missing values, inconsistent spellings,
   impossible dates and amounts. Every injected problem is counted and written
   to manifest.json, so the SQL data-quality checks can be verified against an
   answer key.

Everything is driven by ONE random seed. Same seed + same library versions =
byte-identical CSV files. Nothing here reads the clock.
"""
from __future__ import annotations

import calendar
import hashlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Fixed world constants
# ---------------------------------------------------------------------------
DATA_START = date(2023, 7, 1)  # earliest possible signup
LAST_SIGNUP = date(2026, 9, 15)  # latest possible signup
DATA_END = date(2026, 9, 30)  # the data was "exported" on this day

# (plan_id, name, monthly_price_ghs, data_cap_gb (None = unlimited), contract_months)
PLANS = [
    (1, "Starter 5GB", 35.00, 5, 0),
    (2, "Basic 15GB", 70.00, 15, 0),
    (3, "Standard 40GB", 120.00, 40, 6),
    (4, "Home Fibre", 250.00, None, 12),
    (5, "Business Pro", 400.00, None, 12),
]
PLAN_P = {
    "individual": [0.30, 0.30, 0.22, 0.18, 0.00],
    "business": [0.00, 0.10, 0.20, 0.30, 0.40],
}
ALLOWED_PLANS = {"individual": [0, 1, 2, 3], "business": [1, 2, 3, 4]}

CHANNELS = ["online", "agent", "referral", "walk-in"]
CHANNEL_P = [0.35, 0.30, 0.15, 0.20]
CHANNEL_EFFECT = {"online": 0.35, "agent": 0.0, "referral": -0.45, "walk-in": -0.10}

REGIONS = ["Greater Accra", "Ashanti", "Western", "Central", "Eastern", "Volta", "Northern", "Bono"]
REGION_P = [0.45, 0.22, 0.08, 0.07, 0.07, 0.04, 0.04, 0.03]
REGION_EFFECT = {"Volta": 0.25, "Northern": 0.30, "Bono": 0.20}  # weaker network coverage

METHODS = ["momo", "card", "bank", "cash"]
METHOD_P = {"individual": [0.62, 0.12, 0.04, 0.22], "business": [0.30, 0.20, 0.40, 0.10]}

CANCEL_REASONS = ["price", "network_quality", "competitor_offer", "moved", "service_not_needed", "billing_dispute", "other"]
CANCEL_REASON_P = [0.28, 0.24, 0.14, 0.10, 0.12, 0.06, 0.06]

TICKET_CATEGORIES = ["billing", "network", "speed", "sim/device", "other"]
TICKET_P_NORMAL = np.array([0.25, 0.25, 0.15, 0.15, 0.20])
TICKET_P_DISTRESS = np.array([0.20, 0.38, 0.25, 0.07, 0.10])

# Monthly cancel probability for an "average" customer = sigmoid(BASE_LOGIT).
BASE_LOGIT = -2.95

FIRST_NAMES = [
    "Kwame", "Kofi", "Yaw", "Kwabena", "Kwesi", "Kojo", "Yoofi", "Nana", "Ama", "Akosua", "Abena", "Adwoa",
    "Efua", "Esi", "Afia", "Akua", "Yaa", "Ewurama", "Mohammed", "Abdul", "Ibrahim", "Fatima", "Mariam",
    "Samuel", "Daniel", "Grace", "Joyce", "Emmanuel", "Isaac", "Prince", "Michael", "Comfort", "Gifty",
    "Linda", "Priscilla", "Eric", "Richard", "Benjamin", "Vivian", "Josephine",
]
LAST_NAMES = [
    "Mensah", "Owusu", "Boateng", "Asante", "Appiah", "Agyemang", "Tetteh", "Quaye", "Addo", "Ofori", "Osei",
    "Adjei", "Annan", "Amoah", "Ansah", "Acheampong", "Darko", "Armah", "Sarpong", "Baidoo", "Danso", "Yeboah",
    "Bonsu", "Lamptey", "Ankrah", "Quartey", "Dadzie", "Amankwah", "Fosu", "Gyasi", "Tagoe", "Wiredu",
    "Abdulai", "Issah", "Yakubu", "Tandoh", "Kuffour", "Nartey", "Ayittey", "Ocloo",
]
BUSINESS_SUFFIXES = ["Ventures", "Enterprise", "Trading", "Logistics", "Consult", "Services"]
PHONE_PREFIXES = ["24", "54", "55", "59", "20", "50", "26", "56", "27", "57"]
EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com"]

REGION_ALIASES = {
    "Greater Accra": ["greater accra", "GREATER ACCRA", "Accra", "Gt. Accra", "Greater  Accra"],
    "Ashanti": ["ashanti", "ASHANTI", "Ashanti Region", "Kumasi"],
    "Western": ["western", "WESTERN", "Western Region"],
    "Central": ["central", "CENTRAL", "Central Region"],
    "Eastern": ["eastern", "EASTERN", "Eastern Region"],
    "Volta": ["volta", "VOLTA", "Volta Region"],
    "Northern": ["northern", "NORTHERN", "Northern Region"],
    "Bono": ["bono", "BONO", "Brong Ahafo"],
}


@dataclass(frozen=True)
class GenConfig:
    n_customers: int = 5000
    seed: int = 42
    # Used ONLY to compute the answer key (ground-truth labels) in the manifest.
    # The pipeline itself reads these from the use case config, not from here.
    cutoff_date: date = date(2026, 6, 30)
    label_window_days: int = 60


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def _next_month(d: date) -> date:
    return date(d.year + (d.month == 12), d.month % 12 + 1, 1)


def _dim(d: date) -> int:
    return calendar.monthrange(d.year, d.month)[1]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-x)))


def _pick(rng: np.random.Generator, seq, p=None):
    """rng.choice over a list that returns a plain Python object."""
    return seq[int(rng.choice(len(seq), p=p))]


def _plan_idx_on(history: list[tuple[date, int]], d: date) -> int:
    idx = history[0][1]
    for start, plan_idx in history:
        if start <= d:
            idx = plan_idx
    return idx


# ---------------------------------------------------------------------------
# Step 1 + 2: simulate the clean world
# ---------------------------------------------------------------------------
def _simulate_clean(cfg: GenConfig, rng: np.random.Generator):
    customers, subs, pays, usage, tickets, truth = [], [], [], [], [], []
    used_phones: set[str] = set()
    sub_id = pay_id = use_id = tix_id = 0
    span_days = (LAST_SIGNUP - DATA_START).days

    for cid in range(1, cfg.n_customers + 1):
        # ---- who is this customer? ----
        ctype = "business" if rng.random() < 0.12 else "individual"
        channel = _pick(rng, CHANNELS, CHANNEL_P)
        region = _pick(rng, REGIONS, REGION_P)
        signup = DATA_START + timedelta(days=int(span_days * rng.beta(1.3, 1.0)))
        plan_idx = int(rng.choice(len(PLANS), p=PLAN_P[ctype]))

        first, last = _pick(rng, FIRST_NAMES), _pick(rng, LAST_NAMES)
        if ctype == "business":
            name = f"{last} {_pick(rng, BUSINESS_SUFFIXES)}"
            email = f"info@{last.lower()}{int(rng.integers(1, 99))}.com.gh"
            birth_year = None
        else:
            name = f"{first} {last}"
            email = f"{first.lower()}.{last.lower()}{int(rng.integers(1, 999))}@{_pick(rng, EMAIL_DOMAINS)}"
            birth_year = int(rng.integers(1958, 2006))
        while True:
            phone = "0" + _pick(rng, PHONE_PREFIXES) + f"{int(rng.integers(0, 10_000_000)):07d}"
            if phone not in used_phones:
                used_phones.add(phone)
                break
        customers.append((cid, name, email, phone, region, ctype, channel, signup, birth_year))

        # ---- does this customer cancel, and when? (static traits + tenure only) ----
        contract = PLANS[plan_idx][4]
        risk = (
            BASE_LOGIT
            + CHANNEL_EFFECT[channel]
            + REGION_EFFECT.get(region, 0.0)
            + (-0.5 if ctype == "business" else 0.0)
            + (-0.5 if contract > 0 else 0.0)
            + (0.3 if plan_idx == 0 else 0.0)
            + float(rng.normal(0, 0.5))
        )
        churn_date: date | None = None
        m = _next_month(_month_start(signup))
        tenure = 1
        while m <= DATA_END:
            tenure_effect = 0.6 if tenure <= 3 else (-0.2 if tenure >= 12 else 0.0)
            if rng.random() < _sigmoid(risk + tenure_effect):
                lo = max(m, signup + timedelta(days=20))
                hi = min(date(m.year, m.month, _dim(m)), DATA_END)
                if lo > hi:
                    lo = hi
                churn_date = lo + timedelta(days=int(rng.integers(0, (hi - lo).days + 1)))
                break
            m = _next_month(m)
            tenure += 1
        end_eff = churn_date or DATA_END

        months: list[date] = []
        m = _month_start(signup)
        while m <= end_eff:
            months.append(m)
            m = _next_month(m)

        # ---- "distress" per month: 0 = normal, 1 = about to leave ----
        distress = np.zeros(len(months))
        silent = False
        if churn_date is not None:
            silent = bool(rng.random() < 0.30)  # leaves without warning signs
            if not silent:
                lead = int(rng.integers(2, 5))  # warning signs start 2-4 months before
                for j in range(lead):
                    k = len(months) - 1 - j
                    if k >= 0:
                        distress[k] = max(distress[k], 1.0 - 0.7 * j / lead)
        elif len(months) >= 4 and rng.random() < 0.12:  # false alarm on a customer who stays
            start = int(rng.integers(0, len(months) - 2))
            length = int(rng.integers(2, 4))
            distress[start : start + length] = float(rng.uniform(0.3, 0.7))

        # ---- plan history (a few customers switch plan once) ----
        history = [(signup, plan_idx)]
        if (end_eff - signup).days >= 150 and rng.random() < 0.15:
            change_date = signup + timedelta(days=int(rng.integers(90, (end_eff - signup).days - 30)))
            allowed = ALLOWED_PLANS[ctype]
            up, down = plan_idx + 1, plan_idx - 1
            if rng.random() < 0.55 and up in allowed:
                new_idx = up
            elif down in allowed:
                new_idx = down
            else:
                new_idx = up
            history.append((change_date, new_idx))

        # ---- subscriptions ----
        if len(history) == 2:
            sub_id += 1
            subs.append((sub_id, cid, history[0][1] + 1, signup, history[1][0], "cancelled", "plan_change"))
            final_start, final_plan = history[1]
        else:
            final_start, final_plan = signup, plan_idx
        sub_id += 1
        final_sub_id = sub_id
        if churn_date is not None:
            if (not silent) and rng.random() < 0.35:
                reason = "network_quality"
            else:
                reason = _pick(rng, CANCEL_REASONS, CANCEL_REASON_P)
            subs.append((sub_id, cid, final_plan + 1, final_start, churn_date, "cancelled", reason))
        else:
            subs.append((sub_id, cid, final_plan + 1, final_start, None, "active", None))
        first_sub_id = final_sub_id - 1 if len(history) == 2 else final_sub_id
        change_date_or_none = history[1][0] if len(history) == 2 else None

        truth.append((cid, signup, churn_date))

        # ---- behaviour, month by month ----
        frac_use = float(rng.uniform(0.2, 0.9))  # how much of the plan this customer uses
        base_days = int(rng.integers(12, 28))
        pref_method = _pick(rng, METHODS, METHOD_P[ctype])
        ticket_base = 0.13 if ctype == "business" else 0.09

        for k, m in enumerate(months):
            d = float(distress[k])
            dim = _dim(m)
            lo_day = max(m, signup)
            hi_day = min(date(m.year, m.month, dim), end_eff)
            if lo_day > hi_day:
                continue
            active_span = (hi_day - lo_day).days + 1
            frac = active_span / dim

            # usage
            cur_plan = PLANS[_plan_idx_on(history, hi_day)]
            cap = cur_plan[3]
            base_gb = cap * frac_use if cap is not None else 40.0 + 120.0 * frac_use
            gb = base_gb * float(rng.lognormal(0, 0.18)) * (1 - 0.55 * d) * frac
            gb = min(gb, (cap or 1e9) * 1.25)
            days_active = int(np.clip(round(base_days * (1 - 0.4 * d) * frac + float(rng.normal(0, 1.5))), 1, dim))
            use_id += 1
            usage.append((use_id, cid, m, round(gb, 2), days_active))

            # payment (billed on the signup day-of-month, capped at 28)
            pay_day = date(m.year, m.month, min(signup.day, 28))
            if signup <= pay_day <= end_eff:
                price = PLANS[_plan_idx_on(history, pay_day)][2]
                method = pref_method if rng.random() < 0.85 else _pick(rng, METHODS, METHOD_P[ctype])
                when = datetime(pay_day.year, pay_day.month, pay_day.day, int(rng.integers(6, 22)), int(rng.integers(0, 60)))
                u = rng.random()
                sid = first_sub_id if (change_date_or_none and pay_day < change_date_or_none) else final_sub_id
                if u < 0.005:
                    status = "reversed"
                elif u < 0.005 + 0.04 + 0.30 * d:
                    status = "failed"
                else:
                    status = "success"
                pay_id += 1
                pays.append((pay_id, cid, sid, when, price, method, status))
                if status == "failed" and rng.random() < 0.6:
                    retry = when + timedelta(days=int(rng.integers(1, 6)))
                    if retry.date() <= min(end_eff, DATA_END):
                        pay_id += 1
                        pays.append((pay_id, cid, sid, retry, price, method, "success"))

            # support tickets
            n_tix = int(rng.poisson(ticket_base * (1 + 5 * d) * frac))
            for _ in range(n_tix):
                day = lo_day + timedelta(days=int(rng.integers(0, active_span)))
                opened = datetime(day.year, day.month, day.day, int(rng.integers(7, 21)), int(rng.integers(0, 60)))
                probs = (1 - d) * TICKET_P_NORMAL + d * TICKET_P_DISTRESS
                category = TICKET_CATEGORIES[int(rng.choice(5, p=probs / probs.sum()))]
                priority = _pick(rng, ["low", "medium", "high"], [0.45 - 0.2 * d, 0.40, 0.15 + 0.2 * d])
                hours = float(rng.lognormal(np.log(30), 0.8)) * (1 + 0.8 * d)
                resolved = None if rng.random() < 0.05 else opened + timedelta(hours=hours)
                if resolved is not None and resolved.date() > DATA_END:
                    resolved = None
                score = None
                if resolved is not None and rng.random() < 0.30:
                    score = int(np.clip(round(4.2 - 0.02 * hours + float(rng.normal(0, 0.8))), 1, 5))
                tix_id += 1
                tickets.append((tix_id, cid, opened, resolved, category, priority, score))

    return customers, subs, pays, usage, tickets, truth


# ---------------------------------------------------------------------------
# Step 3: inject real-world mess (each problem is counted for the answer key)
# ---------------------------------------------------------------------------
def _idx(rng: np.random.Generator, n: int, frac: float) -> np.ndarray:
    """Exactly round(frac * n) distinct row positions."""
    k = min(n, int(round(frac * n)))
    return np.sort(rng.choice(n, size=k, replace=False))


def _pick_among(rng: np.random.Generator, column: pd.Series, frac: float, n: int) -> np.ndarray:
    """round(frac * n) distinct positions, chosen only among rows where `column` has a value."""
    candidates = np.flatnonzero(column.notna().to_numpy())
    k = min(len(candidates), int(round(frac * n)))
    return np.sort(rng.choice(candidates, size=k, replace=False))


def _name_variant(name: str, rng: np.random.Generator) -> str:
    kind = int(rng.integers(0, 3))
    if kind == 0:
        return name.upper()
    if kind == 1:
        return name.lower()
    if len(name) > 4:  # typo: swap two neighbouring letters
        i = int(rng.integers(1, len(name) - 2))
        return name[:i] + name[i + 1] + name[i] + name[i + 2 :]
    return name


def _region_variant(region: str, rng: np.random.Generator) -> str:
    options = REGION_ALIASES.get(region, []) + [region.lower(), region.upper(), region + " "]
    return _pick(rng, options)


def _inject(tables: dict[str, pd.DataFrame], rng: np.random.Generator):
    issues: dict[str, int] = {}
    dup_pairs: list[tuple[int, int]] = []

    # ------------------------------ customers ------------------------------
    cu = tables["customers"]
    n0 = len(cu)
    # 1. duplicate people registered twice under a new id
    dup_src = _idx(rng, n0, 0.02)
    new_rows = []
    next_id = int(cu["customer_id"].max())
    for i in dup_src:
        row = cu.iloc[i].copy()
        next_id += 1
        dup_pairs.append((int(row["customer_id"]), next_id))
        row["customer_id"] = next_id
        row["full_name"] = _name_variant(str(row["full_name"]), rng)
        if rng.random() < 0.5:
            row["email"] = None
        row["signup_date"] = row["signup_date"] + timedelta(days=int(rng.integers(0, 6)))
        new_rows.append(row)
    cu = pd.concat([cu, pd.DataFrame(new_rows)], ignore_index=True)
    issues["customers.duplicate_person_rows"] = len(dup_src)
    n = len(cu)

    # Missing values are chosen among rows that still have a value, so the counts are exact
    # (an already-missing value is never "injected" a second time).
    nulls = _pick_among(rng, cu["region"], 0.03, n)
    cu.loc[nulls, "region"] = None
    issues["customers.region_missing"] = len(nulls)
    sel = _pick_among(rng, cu["region"], 0.08, n)
    for i in sel:
        cu.at[i, "region"] = _region_variant(str(cu.at[i, "region"]), rng)
    issues["customers.region_inconsistent_spelling"] = len(sel)

    nulls = _pick_among(rng, cu["email"], 0.12, n)
    cu.loc[nulls, "email"] = None
    issues["customers.email_missing"] = len(nulls)
    nulls = _pick_among(rng, cu["phone"], 0.04, n)
    cu.loc[nulls, "phone"] = None
    issues["customers.phone_missing"] = len(nulls)

    fmt_cnt = 0
    for i in _idx(rng, n, 0.25):
        p = cu.at[i, "phone"]
        if isinstance(p, str) and p.startswith("0"):
            cu.at[i, "phone"] = _pick(rng, ["+233" + p[1:], "233" + p[1:], p[:3] + " " + p[3:6] + " " + p[6:]])
            fmt_cnt += 1
    issues["customers.phone_format_inconsistent"] = fmt_cnt

    by = cu["birth_year"].astype(object)
    individuals = [i for i in range(n) if cu.at[i, "customer_type"] == "individual" and by.iloc[i] is not None and not pd.isna(by.iloc[i])]
    sel = rng.choice(individuals, size=int(round(0.30 * len(individuals))), replace=False)
    cu.loc[sel, "birth_year"] = None
    issues["customers.birth_year_missing_individuals"] = len(sel)
    remaining = [i for i in individuals if pd.notna(cu.at[i, "birth_year"])]
    sel = rng.choice(remaining, size=int(round(0.006 * n)), replace=False)
    for i in sel:
        cu.at[i, "birth_year"] = int(_pick(rng, [1890, 1900, 2015, 2030, 0]))
    issues["customers.birth_year_implausible"] = len(sel)

    nulls = _pick_among(rng, cu["full_name"], 0.01, n)
    cu.loc[nulls, "full_name"] = None
    issues["customers.name_missing"] = len(nulls)
    sel = _pick_among(rng, cu["full_name"], 0.06, n)
    for i in sel:
        cu.at[i, "full_name"] = cu.at[i, "full_name"].upper() if rng.random() < 0.5 else cu.at[i, "full_name"].lower()
    issues["customers.name_case_inconsistent"] = len(sel)
    sel = _idx(rng, n, 0.04)
    for i in sel:
        v = str(cu.at[i, "customer_type"])
        cu.at[i, "customer_type"] = v.capitalize() if rng.random() < 0.5 else v.upper()
    issues["customers.customer_type_inconsistent_case"] = len(sel)
    nulls = _pick_among(rng, cu["acquisition_channel"], 0.02, n)
    cu.loc[nulls, "acquisition_channel"] = None
    issues["customers.acquisition_channel_missing"] = len(nulls)
    cu["birth_year"] = cu["birth_year"].astype("Int64")
    tables["customers"] = cu

    # ------------------------------ subscriptions ------------------------------
    su = tables["subscriptions"]
    final_cancelled = su.index[(su["status"] == "cancelled") & (su["cancel_reason"] != "plan_change")].to_numpy()
    sel = rng.choice(final_cancelled, size=int(round(0.006 * len(final_cancelled))), replace=False)
    su.loc[sel, "end_date"] = None
    issues["subscriptions.cancelled_but_end_date_missing"] = len(sel)
    sel = rng.choice(final_cancelled, size=int(round(0.25 * len(final_cancelled))), replace=False)
    su.loc[sel, "cancel_reason"] = None
    issues["subscriptions.cancel_reason_missing"] = len(sel)
    with_end = su.index[su["end_date"].notna()].to_numpy()
    sel = rng.choice(with_end, size=int(round(0.004 * len(with_end))), replace=False)
    for i in sel:
        su.at[i, "end_date"] = su.at[i, "start_date"] - timedelta(days=int(rng.integers(1, 30)))
    issues["subscriptions.end_date_before_start_date"] = len(sel)
    sel = _idx(rng, len(su), 0.005)
    for i in sel:
        v = su.at[i, "status"]
        su.at[i, "status"] = v.capitalize() if rng.random() < 0.5 else v.upper()
    issues["subscriptions.status_inconsistent_case"] = len(sel)
    active = su.index[su["status"] == "active"].to_numpy()
    sel = rng.choice(active, size=int(round(0.015 * len(active))), replace=False)
    su.loc[sel, "status"] = "suspended"
    issues["subscriptions.active_marked_suspended"] = len(sel)
    tables["subscriptions"] = su

    # ------------------------------ payments ------------------------------
    pa = tables["payments"]
    n = len(pa)
    sel = _idx(rng, n, 0.005)
    pa.loc[sel, "amount_ghs"] = None
    issues["payments.amount_missing"] = len(sel)
    pos = pa.index[pa["amount_ghs"].notna()].to_numpy()
    sel = rng.choice(pos, size=int(round(0.002 * n)), replace=False)
    pa.loc[sel, "amount_ghs"] = -pa.loc[sel, "amount_ghs"]
    issues["payments.amount_negative"] = len(sel)
    sel = _idx(rng, n, 0.06)
    for i in sel:
        pa.at[i, "method"] = _pick(rng, {"momo": ["MoMo", "MOMO", "mobile money"], "card": ["Card", "CARD"], "bank": ["Bank", "bank transfer"], "cash": ["Cash", "CASH"]}[pa.at[i, "method"]])
    issues["payments.method_inconsistent"] = len(sel)
    sel = _idx(rng, n, 0.01)
    for i in sel:
        v = pa.at[i, "status"]
        pa.at[i, "status"] = v.upper() if rng.random() < 0.5 else v.capitalize()
    issues["payments.status_inconsistent_case"] = len(sel)
    sel = _idx(rng, n, 0.02)
    pa["subscription_id"] = pa["subscription_id"].astype("Int64")
    pa.loc[sel, "subscription_id"] = pd.NA
    issues["payments.subscription_id_missing"] = len(sel)
    sel = _idx(rng, n, 0.001)
    for i in sel:  # a timestamp after the export date: impossible, so it must be a data error
        pa.at[i, "paid_at"] = pd.Timestamp(DATA_END) + pd.Timedelta(days=int(rng.integers(1, 120)), hours=int(rng.integers(6, 22)))
    issues["payments.paid_at_in_future"] = len(sel)
    sel = _idx(rng, n, 0.01)
    dup = pa.iloc[sel].copy()
    dup["payment_id"] = np.arange(int(pa["payment_id"].max()) + 1, int(pa["payment_id"].max()) + 1 + len(dup))
    pa = pd.concat([pa, dup], ignore_index=True)
    issues["payments.exact_duplicate_rows"] = len(sel)
    tables["payments"] = pa

    # ------------------------------ usage_monthly ------------------------------
    us = tables["usage_monthly"]
    n = len(us)
    drop = _idx(rng, n, 0.03)
    us = us.drop(index=drop).reset_index(drop=True)
    issues["usage_monthly.months_missing"] = len(drop)
    n = len(us)
    sel = _idx(rng, n, 0.015)
    us.loc[sel, "data_used_gb"] = None
    issues["usage_monthly.data_used_missing"] = len(sel)
    pos = us.index[us["data_used_gb"].notna()].to_numpy()
    sel = rng.choice(pos, size=int(round(0.002 * n)), replace=False)
    us.loc[sel, "data_used_gb"] = -us.loc[sel, "data_used_gb"]
    issues["usage_monthly.data_used_negative"] = len(sel)
    us["active_days"] = us["active_days"].astype("Int64")
    sel = _idx(rng, n, 0.003)
    for i in sel:
        us.at[i, "active_days"] = int(_pick(rng, [45, 60, 99]))
    issues["usage_monthly.active_days_over_31"] = len(sel)
    sel = _idx(rng, n, 0.003)
    for i in sel:
        m = us.at[i, "usage_month"]
        us.at[i, "usage_month"] = date(m.year, m.month, 15)
    issues["usage_monthly.month_not_first_of_month"] = len(sel)
    sel = _idx(rng, n, 0.01)
    dup = us.iloc[sel].copy()
    dup["usage_id"] = np.arange(int(us["usage_id"].max()) + 1, int(us["usage_id"].max()) + 1 + len(dup))
    dup["data_used_gb"] = (dup["data_used_gb"] * rng.uniform(0.9, 1.1, size=len(dup))).round(2)
    us = pd.concat([us, dup], ignore_index=True)
    issues["usage_monthly.conflicting_duplicate_month_rows"] = len(sel)
    tables["usage_monthly"] = us

    # ------------------------------ support_tickets ------------------------------
    tk = tables["support_tickets"]
    n = len(tk)
    resolved = tk.index[tk["resolved_at"].notna()].to_numpy()
    sel = rng.choice(resolved, size=int(round(0.006 * n)), replace=False)
    for i in sel:
        tk.at[i, "resolved_at"] = tk.at[i, "opened_at"] - pd.Timedelta(hours=int(rng.integers(1, 72)))
    issues["support_tickets.resolved_before_opened"] = len(sel)
    tk["satisfaction_score"] = tk["satisfaction_score"].astype("Int64")
    scored = tk.index[tk["satisfaction_score"].notna()].to_numpy()
    sel = rng.choice(scored, size=max(1, int(round(0.02 * len(scored)))), replace=False)
    for i in sel:
        tk.at[i, "satisfaction_score"] = int(_pick(rng, [0, 6, 7, 10]))
    issues["support_tickets.satisfaction_out_of_range"] = len(sel)
    cat_alias = {"billing": ["Billing", "BILLING"], "network": ["Network", "network issue", "NETWORK"], "speed": ["Speed", "slow internet"], "sim/device": ["SIM / Device", "sim device"], "other": ["Other"]}
    sel = _idx(rng, n, 0.05)
    for i in sel:
        tk.at[i, "category"] = _pick(rng, cat_alias[tk.at[i, "category"]])
    issues["support_tickets.category_inconsistent"] = len(sel)
    sel = _idx(rng, n, 0.04)
    for i in sel:
        tk.at[i, "priority"] = str(tk.at[i, "priority"]).upper() if rng.random() < 0.5 else str(tk.at[i, "priority"]).capitalize()
    issues["support_tickets.priority_inconsistent_case"] = len(sel)
    sel = _idx(rng, n, 0.008)
    dup = tk.iloc[sel].copy()
    dup["ticket_id"] = np.arange(int(tk["ticket_id"].max()) + 1, int(tk["ticket_id"].max()) + 1 + len(dup))
    tk = pd.concat([tk, dup], ignore_index=True)
    issues["support_tickets.exact_duplicate_rows"] = len(sel)
    tables["support_tickets"] = tk

    return issues, dup_pairs


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def generate(cfg: GenConfig):
    """Return (tables, manifest, answer_key). Pure function of cfg: no clock, no files."""
    rng = np.random.default_rng(cfg.seed)
    customers, subs, pays, usage, tickets, truth = _simulate_clean(cfg, rng)

    tables = {
        "plans": pd.DataFrame(PLANS, columns=["plan_id", "plan_name", "monthly_price_ghs", "data_cap_gb", "contract_months"]),
        "customers": pd.DataFrame(customers, columns=["customer_id", "full_name", "email", "phone", "region", "customer_type", "acquisition_channel", "signup_date", "birth_year"]),
        "subscriptions": pd.DataFrame(subs, columns=["subscription_id", "customer_id", "plan_id", "start_date", "end_date", "status", "cancel_reason"]),
        "payments": pd.DataFrame(pays, columns=["payment_id", "customer_id", "subscription_id", "paid_at", "amount_ghs", "method", "status"]),
        "usage_monthly": pd.DataFrame(usage, columns=["usage_id", "customer_id", "usage_month", "data_used_gb", "active_days"]),
        "support_tickets": pd.DataFrame(tickets, columns=["ticket_id", "customer_id", "opened_at", "resolved_at", "category", "priority", "satisfaction_score"]),
    }
    tables["customers"] = tables["customers"].astype({"full_name": object, "email": object, "phone": object, "birth_year": object})
    tables["plans"]["data_cap_gb"] = tables["plans"]["data_cap_gb"].astype("Int64")
    tables["payments"]["paid_at"] = pd.to_datetime(tables["payments"]["paid_at"])
    tables["support_tickets"]["opened_at"] = pd.to_datetime(tables["support_tickets"]["opened_at"])
    tables["support_tickets"]["resolved_at"] = pd.to_datetime(tables["support_tickets"]["resolved_at"])
    tables["subscriptions"] = tables["subscriptions"].astype({"end_date": object, "cancel_reason": object})
    tables["payments"]["subscription_id"] = tables["payments"]["subscription_id"].astype("Int64")

    # Ground truth from the CLEAN simulation, before any mess is injected.
    cutoff, window = cfg.cutoff_date, timedelta(days=cfg.label_window_days)
    key_rows = []
    for cid, signup, churn in truth:
        existed = signup <= cutoff
        active = existed and (churn is None or churn > cutoff)
        label = 1 if (active and churn is not None and churn <= cutoff + window) else 0
        key_rows.append((cid, int(existed), int(active), label, churn))
    answer_key = pd.DataFrame(key_rows, columns=["customer_id", "existed_at_cutoff", "active_at_cutoff", "churned_in_window", "true_churn_date"])

    issues, dup_pairs = _inject(tables, rng)
    for name in tables:  # deterministic order
        tables[name] = tables[name].reset_index(drop=True)

    manifest = {
        "generator": "datagen.telecom",
        "seed": cfg.seed,
        "n_customers_before_duplicates": cfg.n_customers,
        "data_window": {"first_signup": DATA_START.isoformat(), "last_signup": LAST_SIGNUP.isoformat(), "data_end": DATA_END.isoformat()},
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "row_counts": {name: int(len(df)) for name, df in tables.items()},
        "injected_issues": issues,
        "ground_truth": {
            "cutoff_date": cutoff.isoformat(),
            "label_window_days": cfg.label_window_days,
            "customers_existing_at_cutoff": int(answer_key["existed_at_cutoff"].sum()),
            "active_at_cutoff": int(answer_key["active_at_cutoff"].sum()),
            "churned_in_window": int(answer_key["churned_in_window"].sum()),
            "churn_rate_in_window": round(float(answer_key["churned_in_window"].sum() / max(1, answer_key["active_at_cutoff"].sum())), 4),
            "note": "Computed from the clean simulation BEFORE mess was injected. Use it to check the SQL label logic.",
        },
    }
    answer_key["churned_in_window"] = answer_key["churned_in_window"].astype(int)
    return tables, manifest, answer_key, pd.DataFrame(dup_pairs, columns=["original_customer_id", "duplicate_customer_id"])


def write_outputs(tables, manifest, answer_key, dup_pairs, out_dir: Path) -> dict[str, str]:
    """Write CSVs (+ answer key + manifest). Returns {file: sha256}."""
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    key_dir = out_dir / "answer_key"
    key_dir.mkdir(exist_ok=True)
    hashes: dict[str, str] = {}
    kw = dict(index=False, lineterminator="\n", date_format="%Y-%m-%d %H:%M:%S")
    for name, df in tables.items():
        path = out_dir / f"{name}.csv"
        df.to_csv(path, **kw)
        hashes[f"{name}.csv"] = hashlib.sha256(path.read_bytes()).hexdigest()
    answer_key.to_csv(key_dir / "labels_at_cutoff.csv", **kw)
    dup_pairs.to_csv(key_dir / "duplicate_customer_pairs.csv", **kw)
    manifest = {**manifest, "file_sha256": hashes}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return hashes
