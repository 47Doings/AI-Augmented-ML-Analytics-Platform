"""Tests for the business report. No database needed (uses tests/fixtures).

Run:  python -m pytest -q tests/test_report.py
"""
import copy
import json
from pathlib import Path

import pytest

from core.ai.insights import generate_insights
from core.report import build_report_html
from core.use_case import load_use_case

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(params=["churn", "revenue"])
def case(request):
    uc = load_use_case(request.param)
    res = json.loads((FIX / f"results_{request.param}.json").read_text(encoding="utf-8"))
    return uc, res, generate_insights(res, uc, None)


def test_report_has_every_section_and_the_computed_kpis(case):
    uc, res, ins = case
    html = build_report_html(res, ins, "Acme")
    for must in ("Executive summary", "Key findings", "Recommendations", "How good is the model?",
                 "Business questions answered with SQL", "Data quality", "Drift monitoring", "Appendix"):
        assert must in html, must
    for k in res["kpis"]:
        assert k["shown"] in html  # the headline numbers are exactly the computed ones
    assert res["model"]["selected"] in html and "for Acme" in html
    assert "Template summary" in html  # the summary is labelled with how it was produced
    assert "<script" not in html.lower() and "ws_" not in html


def test_every_summary_sentence_in_the_report_is_the_checked_one(case):
    uc, res, ins = case
    html = build_report_html(res, ins)
    from html import escape
    for key in ("executive_summary", "recommendations"):
        for b in ins["document"][key]:
            assert escape(b["text"]) in html


def test_hostile_text_is_shown_as_text_not_markup(case):
    uc, res, ins = case
    bad = copy.deepcopy(res)
    bad["use_case"]["name"] = "<script>alert(1)</script>"
    first = next(iter(bad["business_questions"].values()))
    first["rows"][0][0] = "<img src=x onerror=alert(1)>"
    ins2 = copy.deepcopy(ins)
    ins2["document"]["key_findings"][0]["text"] = "<b onclick=alert(1)>hi</b> a long enough sentence"
    html = build_report_html(bad, ins2, "<i>Biz</i>")
    assert "<script>alert" not in html and "<img src=x" not in html and "<b onclick" not in html and "<i>Biz" not in html
    assert "&lt;script&gt;" in html


def test_report_still_builds_without_insights_or_optional_parts(case):
    uc, res, _ = case
    thin = copy.deepcopy(res)
    thin["drift"], thin["data_quality"], thin["business_questions"] = None, None, {}
    html = build_report_html(thin, None)
    assert "No written summary" in html and "Drift monitoring" not in html
