"""The business report: one self-contained HTML page (print it to PDF from the browser).

It reads ONLY results.json and insights.json, never the database or raw data, so every
number on the page is a number the pipeline computed and saved. Nothing in here
mentions churn or revenue: titles, KPIs and tables come from the results themselves.

Everything that came from data or from the AI is passed through html.escape, so a
hostile plan name or a hostile AI sentence is shown as text and never runs as markup.
Charts are inline SVG (no JavaScript, no images to fetch), so the file works offline and
prints cleanly.
"""
from __future__ import annotations

from datetime import date
from html import escape as _e

SECTION_TITLES = [
    ("executive_summary", "Executive summary"),
    ("key_findings", "Key findings"),
    ("recommendations", "Recommendations"),
    ("caveats", "Caveats"),
]
MODE_LABEL = {
    "ai": "Written by AI; every number checked against computed facts",
    "fallback": "Template summary built from computed facts (no AI provider answered)",
    "none": "No summary available",
}

CSS = """
:root{--ink:#1c2430;--muted:#5d6877;--line:#d9dee5;--accent:#1f5fbf;--bg:#f4f6f9;--good:#15803d;--warn:#b45309;--bad:#b91c1c}
*{box-sizing:border-box}
body{font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:var(--ink);background:var(--bg);margin:0}
main{max-width:900px;margin:0 auto;padding:28px 24px 60px;background:#fff}
h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:30px 0 8px;padding-bottom:4px;border-bottom:2px solid var(--accent)}
h3{font-size:14px;margin:18px 0 6px}
.sub{color:var(--muted);margin:0 0 4px}.note{color:var(--muted);font-size:12px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:16px 0}
.kpi{border:1px solid var(--line);border-radius:8px;padding:12px 14px}.kpi b{display:block;font-size:24px}.kpi span{color:var(--muted);font-size:12px}
.badge{display:inline-block;font-size:11px;font-weight:600;border:1px solid var(--line);border-radius:999px;padding:2px 9px}
.badge.ai{color:var(--good)}.badge.fallback{color:var(--warn)}.badge.none{color:var(--bad)}
ul.items{padding-left:18px;margin:6px 0}ul.items li{margin:5px 0}
.type{font-size:11px;border:1px solid var(--line);border-radius:999px;padding:0 7px;color:var(--muted)}
.cites{color:var(--muted);font-size:11px}
table{border-collapse:collapse;width:100%;font-size:12px;margin:6px 0}
th,td{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left}th{background:var(--bg)}td.n{text-align:right;font-variant-numeric:tabular-nums}
tr.sel td{background:#eaf2ff;font-weight:600}
.pill{font-size:11px;padding:0 7px;border-radius:999px;border:1px solid var(--line)}
.pill.stable{color:var(--good)}.pill.watch{color:var(--warn)}.pill.alert{color:var(--bad)}
svg text{font:11px system-ui,sans-serif;fill:var(--ink)}svg .m{fill:var(--muted)}
.foot{margin-top:34px;font-size:11px;color:var(--muted);border-top:1px solid var(--line);padding-top:10px}
@media print{body{background:#fff}main{padding:0;max-width:none}h2,h3{break-after:avoid}table,svg,.kpi{break-inside:avoid}}
"""


# ------------------------------------------------------------------ small helpers
def _fmt(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.3f}".rstrip("0").rstrip(".")
    return _e(str(v))


def _table(t: dict | None, limit: int = 8, selected_by: str | None = None, pill: str | None = None, latest: bool = False) -> str:
    if not t or not t.get("columns"):
        return ""
    cols, rows = t["columns"], t["rows"]
    shown = rows[-limit:] if latest else rows[:limit]
    head = "".join(f"<th>{_e(c.replace('_', ' '))}</th>" for c in cols)
    body = []
    for r in shown:
        cls = ' class="sel"' if selected_by and r[cols.index(selected_by)] is True else ""
        cells = []
        for c, v in zip(cols, r):
            if pill == c and v:
                cells.append(f'<td><span class="pill {_e(str(v))}">{_e(str(v))}</span></td>')
            else:
                cells.append(f'<td class="{"n" if isinstance(v, (int, float)) and not isinstance(v, bool) else ""}">{_fmt(v)}</td>')
        body.append(f"<tr{cls}>{''.join(cells)}</tr>")
    more = f'<p class="note">Showing {len(shown)} of {t.get("total_rows", len(rows))} rows.</p>' if t.get("total_rows", len(rows)) > len(shown) else ""
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>{more}"


def _bars(labels: list, values: list, fmt=lambda v: f"{v:.3f}", weak: list | None = None, width: int = 760) -> str:
    """Horizontal bar chart as inline SVG. Bars from 0; labels left, values right."""
    if not labels:
        return ""
    vals = [v or 0 for v in values]
    top = max([abs(v) for v in vals] + [1e-9])
    row_h, left, right = 22, 190, 70
    plot = width - left - right
    out = [f'<svg viewBox="0 0 {width} {row_h * len(labels) + 4}" width="100%" role="img">']
    for i, (lab, v) in enumerate(zip(labels, vals)):
        y = i * row_h + 2
        w = max(0.0, v / top * plot)
        opacity = "0.4" if weak and weak[i] else "1"
        out.append(f'<text x="0" y="{y + 14}">{_e(str(lab))[:30]}</text>'
                   f'<rect x="{left}" y="{y + 3}" width="{w:.1f}" height="12" rx="2" fill="#1f5fbf" opacity="{opacity}"/>'
                   f'<text class="m" x="{left + plot + 8}" y="{y + 14}">{_e(fmt(v))}</text>')
    out.append("</svg>")
    return "".join(out)


def _col(t: dict, name: str) -> list:
    i = t["columns"].index(name)
    return [r[i] for r in t["rows"]]


def _items(items: list[dict]) -> str:
    lis = []
    for b in items:
        tag = f'<span class="type">{_e(b["type"].replace("_", " "))}</span> ' if b.get("type") else ""
        lis.append(f'<li>{tag}{_e(b["text"])} <span class="cites">[{_e(", ".join(b["facts"]))}]</span></li>')
    return f'<ul class="items">{"".join(lis)}</ul>'


def _source_label(res: dict) -> str:
    return "file uploaded by the business" if res["use_case"].get("data_source") == "upload" else "synthetic demo data"


# ------------------------------------------------------------------ sections
def _insights_html(ins: dict | None) -> str:
    if not ins or not ins.get("document"):
        return '<h2>Summary</h2><p class="note">No written summary is available for this run.</p>'
    mode = ins.get("mode", "none")
    out = [f'<p><span class="badge {_e(mode)}">{_e(MODE_LABEL.get(mode, mode))}</span></p>']
    for key, title in SECTION_TITLES:
        items = ins["document"].get(key) or []
        if items:
            out.append(f"<h2>{_e(title)}</h2>{_items(items)}")
    v = ins.get("validation") or {}
    out.append(f'<p class="note">Checks passed: {v.get("numbers_checked", 0)} numbers matched to facts, '
               f'{v.get("facts_cited", 0)} facts cited. Bracketed ids point to the facts in the appendix.</p>')
    return "".join(out)


def _model_html(res: dict) -> str:
    m, card = res["model"], res["model"]["card"]
    tm = card["test_metrics"]
    if res["use_case"]["problem_type"] == "classification":
        line = (f"Flagging the riskiest {card['operating_point']['flag_top_fraction'] * 100:.0f}% of entities on held-out data: "
                f"precision {tm['precision'] * 100:.1f}%, recall {tm['recall'] * 100:.1f}%, lift {tm['lift']:.1f}x the average rate.")
    else:
        line = f"Error on held-out data: MAE {tm['mae']:.2f}, RMSE {tm['rmse']:.2f}, R² {tm['r2']:.3f}."
    imp = m["importance"]
    top = list(range(min(8, len(imp["rows"]))))
    out = [
        "<h2>How good is the model?</h2>",
        '<p class="note">A baseline that ignores the inputs is compared with real models. The model was chosen by cross-validation on the training split; '
        "test columns use rows it never saw.</p>",
        _table(m["comparison"], limit=6, selected_by="selected"),
        f"<p>Selected: <b>{_e(m['selected'])}</b>. {_e(line)}</p>",
        "<h3>What the model leans on</h3>",
        '<p class="note">Permutation importance on the test set; faded bars are within noise. This shows association in the data, not cause.</p>',
        _bars([_col(imp, "feature")[i] for i in top], [_col(imp, "importance")[i] for i in top],
              fmt=lambda v: f"{v:.3f}", weak=[not _col(imp, "clearly_matters")[i] for i in top]),
    ]
    if m.get("deciles"):
        d = m["deciles"]
        out += ["<h3>Do the risk scores rank real outcomes? (decile 10 = highest scores)</h3>",
                _bars([f"Decile {x}" for x in _col(d, "risk_decile")], _col(d, "rate"), fmt=lambda v: f"{v * 100:.1f}%")]
    return "".join(out)


def _questions_html(res: dict) -> str:
    qs = res.get("business_questions") or {}
    if not qs:
        return ""
    out = ["<h2>Business questions answered with SQL</h2>"]
    for name, t in qs.items():
        latest = bool(t["columns"]) and "month" in t["columns"][0].lower()
        out.append(f"<h3>{_e(name.replace('_', ' ').capitalize())}</h3>{_table(t, limit=8, latest=latest)}")
    return "".join(out)


def _quality_html(res: dict) -> str:
    dq = res.get("data_quality")
    drift = res.get("drift")
    out = []
    if dq:
        out += ["<h2>Data quality</h2>", '<p class="note">Checks run in SQL on the raw tables; worst first.</p>', _table(dq, limit=8)]
    if drift:
        out += ["<h2>Drift monitoring</h2>",
                f'<p class="note">Same SQL at a later date ({_e(drift["later_cutoff"])}, {drift["entities"]:,} entities) compared with the training data. '
                "PSI below 0.10 stable, 0.10 to 0.25 watch, above 0.25 alert.</p>",
                _table(drift, limit=10, pill="status")]
    return "".join(out)


def _appendix_html(res: dict, ins: dict | None) -> str:
    card = res["model"]["card"]
    out = ["<h2>Appendix: where the numbers come from</h2>"]
    if ins and ins.get("facts"):
        out.append('<p class="note">Every sentence in the summary cites these computed facts.</p><table><tbody>' +
                   "".join(f"<tr><td><b>{_e(f['id'])}</b></td><td>{_e(f['text'])}</td></tr>" for f in ins["facts"]) + "</tbody></table>")
    v = card.get("versions", {})
    out.append(f'<p class="note">Rows: {card["rows"]["train"]:,} train, {card["rows"]["test"]:,} test. Seed {card["seed"]}. '
               f'Libraries: {_e(", ".join(f"{k} {x}" for k, x in v.items()))}. Data source: {_e(_source_label(res))}.</p>')
    if ins:
        out.append(f'<p class="note">Summary provider: {_e(str(ins.get("provider")))}; generated {_e(str(ins.get("generated_at", "")))}.</p>')
    return "".join(out)


# ------------------------------------------------------------------ the page
def build_report_html(results: dict, insights: dict | None = None, business_name: str | None = None) -> str:
    uc = results["use_case"]
    kpis = "".join(f'<div class="kpi"><b>{_e(k["shown"])}</b><span>{_e(k["label"])}</span></div>' for k in results["kpis"])
    who = f" for {_e(business_name)}" if business_name else ""
    body = [
        f"<h1>{_e(uc['name'])}: business report</h1>",
        f'<p class="sub">{_e(uc["business_question"])}</p>',
        f'<p class="note">Prepared{who} on {date.today().isoformat()}. Data: {_e(_source_label(results))}.'
        + (f' Data window ends {_e(str(uc["parameters"]["data_end_date"]))}.' if uc["parameters"].get("data_end_date") else "") + "</p>",
        f'<div class="kpis">{kpis}</div>',
        _insights_html(insights),
        _model_html(results),
        _questions_html(results),
        _quality_html(results),
        _appendix_html(results, insights),
        '<p class="foot">Generated by the AI-augmented ML analytics platform from results.json and insights.json only. '
        "Figures describe patterns in past data and do not prove cause.</p>",
    ]
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{_e(uc['name'])} report</title><style>{CSS}</style></head><body><main>{''.join(body)}</main></body></html>")
