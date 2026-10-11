"""Tests for the synthetic data generator.

Run from the project root:  python -m pytest -q
Uses a small dataset so it finishes in a few seconds.
"""
import hashlib
from datetime import date

import pandas as pd
import pytest

from datagen.telecom import DATA_END, GenConfig, generate


def _hash(tables):
    h = hashlib.sha256()
    for name in sorted(tables):
        h.update(tables[name].to_csv(index=False, lineterminator="\n").encode())
    return h.hexdigest()


@pytest.fixture(scope="module")
def small():
    return generate(GenConfig(n_customers=600, seed=7))


def test_same_seed_gives_identical_data():
    a, *_ = generate(GenConfig(n_customers=200, seed=1))
    b, *_ = generate(GenConfig(n_customers=200, seed=1))
    assert _hash(a) == _hash(b)


def test_different_seed_gives_different_data():
    a, *_ = generate(GenConfig(n_customers=200, seed=1))
    b, *_ = generate(GenConfig(n_customers=200, seed=2))
    assert _hash(a) != _hash(b)


def test_foreign_keys_are_valid(small):
    t, *_ = small
    cust = set(t["customers"]["customer_id"])
    subs = set(t["subscriptions"]["subscription_id"])
    plans = set(t["plans"]["plan_id"])
    assert set(t["subscriptions"]["customer_id"]) <= cust
    assert set(t["subscriptions"]["plan_id"]) <= plans
    for name in ("payments", "usage_monthly", "support_tickets"):
        assert set(t[name]["customer_id"]) <= cust, name
    assert set(t["payments"]["subscription_id"].dropna()) <= subs


def test_primary_keys_are_unique(small):
    t, *_ = small
    for name, key in [("plans", "plan_id"), ("customers", "customer_id"), ("subscriptions", "subscription_id"),
                      ("payments", "payment_id"), ("usage_monthly", "usage_id"), ("support_tickets", "ticket_id")]:
        assert t[name][key].is_unique, name


def test_no_activity_after_data_end_except_injected_future_payments(small):
    t, manifest, *_ = small
    future = int((t["payments"]["paid_at"].dt.date > DATA_END).sum())
    injected = manifest["injected_issues"]["payments.paid_at_in_future"]
    # every injected one is in the future; exact duplicates (made afterwards) may copy a few of them
    assert injected > 0 and injected <= future <= injected * 2
    assert (t["support_tickets"]["opened_at"].dt.date <= DATA_END).all()


def test_mess_is_actually_present(small):
    t, manifest, *_ = small
    issues = manifest["injected_issues"]
    assert issues["customers.duplicate_person_rows"] > 0
    assert t["customers"]["email"].isna().sum() > 0
    assert (t["payments"]["amount_ghs"] < 0).sum() > 0
    assert t["usage_monthly"].duplicated(["customer_id", "usage_month"]).sum() > 0
    assert t["customers"]["region"].nunique() > 8  # spelling variants exist


def test_duplicates_share_a_phone_with_their_original(small):
    t, _, _, dup_pairs = small
    cu = t["customers"].set_index("customer_id")
    same = 0
    for orig, dup in zip(dup_pairs["original_customer_id"], dup_pairs["duplicate_customer_id"]):
        if pd.notna(cu.at[orig, "phone"]) and cu.at[orig, "phone"] == cu.at[dup, "phone"]:
            same += 1
    assert same > 0


def test_answer_key_is_consistent(small):
    _, manifest, key, _ = small
    gt = manifest["ground_truth"]
    assert gt["active_at_cutoff"] == key["active_at_cutoff"].sum()
    assert gt["churned_in_window"] == key["churned_in_window"].sum()
    assert (key["churned_in_window"] <= key["active_at_cutoff"]).all()
    assert 0.02 < gt["churn_rate_in_window"] < 0.30
