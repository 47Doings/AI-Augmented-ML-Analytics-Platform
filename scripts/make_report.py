"""Write the business report for a finished run, as HTML and (if Playwright is installed) PDF.

Run from the project root, pointing at a results folder, for example:

    python -m scripts.make_report outputs/workspaces/<schema>/churn

It reads only results.json and insights.json in that folder. Output goes next to them:
report.html, and report.pdf when a headless browser is available (pip install playwright;
playwright install chromium). Without it, open report.html and use Print > Save as PDF.
"""
import argparse
import json
import sys
from pathlib import Path

from core.report import build_report_html


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", type=Path, help="folder containing results.json (and insights.json)")
    args = p.parse_args()
    res_file, ins_file = args.folder / "results.json", args.folder / "insights.json"
    if not res_file.is_file():
        print(f"No results.json in {args.folder}. Run the analysis first.")
        return 1
    ins = json.loads(ins_file.read_text(encoding="utf-8")) if ins_file.is_file() else None
    html_path = args.folder / "report.html"
    html_path.write_text(build_report_html(json.loads(res_file.read_text(encoding="utf-8")), ins), encoding="utf-8")
    print(f"Wrote {html_path}")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright not installed: open the HTML file and use Print > Save as PDF.")
        return 0
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.goto(html_path.resolve().as_uri())
        pdf = args.folder / "report.pdf"
        page.pdf(path=str(pdf), format="A4", print_background=True, margin={"top": "14mm", "bottom": "14mm", "left": "12mm", "right": "12mm"})
        browser.close()
    print(f"Wrote {pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
