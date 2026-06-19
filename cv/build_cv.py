#!/usr/bin/env python3
"""
Build François-Pierre Paty's CV: cv.json -> HTML (Jinja2) -> PDF (WeasyPrint).

Usage:
    python3 cv/build_cv.py

Outputs:
    assets/cv/cv.pdf      the CV (linked from the website sidebar)
    cv/cv.preview.html    rendered HTML, handy for quick browser preview

Edit cv/cv.json to change content, then re-run. No LaTeX required.
"""

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "cv.json"
TEMPLATE = "template.html.j2"
PREVIEW = HERE / "cv.preview.html"
PDF_OUT = ROOT / "assets" / "cv" / "cv.pdf"


def main() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))

    env = Environment(
        loader=FileSystemLoader(str(HERE)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    html = env.get_template(TEMPLATE).render(**data)

    PREVIEW.write_text(html, encoding="utf-8")

    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    # base_url = HERE so the relative <link href="cv.css"> resolves
    HTML(string=html, base_url=str(HERE)).write_pdf(str(PDF_OUT))

    print(f"Wrote {PDF_OUT.relative_to(ROOT)}")
    print(f"Wrote {PREVIEW.relative_to(ROOT)} (preview)")


if __name__ == "__main__":
    main()
