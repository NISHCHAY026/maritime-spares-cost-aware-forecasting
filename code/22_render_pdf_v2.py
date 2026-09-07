"""
Render paper_draft_v2.md to PDF, with v2 figures embedded
in an Appendix at the end.
"""

from __future__ import annotations

import re
from pathlib import Path

from markdown_pdf import MarkdownPdf, Section

import config as C


PAPER_DIR = C.PROJECT_DIR / "docs" / "paper"
SRC = PAPER_DIR / "paper_draft_v2.md"
DST = PAPER_DIR / "paper_draft_v2.pdf"
FIG_DIR = PAPER_DIR / "figures_clean"


CSS = """
@page { size: A4; margin: 18mm 18mm 18mm 18mm; }
body { font-family: 'Calibri','Helvetica','Arial',sans-serif;
       font-size: 11pt; line-height: 1.45; color: #222; }
h1 { font-family: 'Cambria','Georgia',serif; font-size: 18pt;
     margin-top: 14pt; margin-bottom: 8pt; page-break-after: avoid; }
h2 { font-family: 'Cambria','Georgia',serif; font-size: 15pt;
     margin-top: 14pt; margin-bottom: 6pt; page-break-after: avoid; }
h3 { font-family: 'Cambria','Georgia',serif; font-size: 13pt;
     margin-top: 10pt; margin-bottom: 4pt; page-break-after: avoid; }
h4 { font-family: 'Cambria','Georgia',serif; font-size: 11.5pt; }
p  { margin: 0 0 6pt 0; text-align: justify; }
ul, ol { margin: 0 0 6pt 18pt; }
li { margin-bottom: 2pt; }
code { font-family: 'Consolas','Courier New',monospace; font-size: 10pt;
       background: #f4f4f4; padding: 0 3px; }
pre  { font-family: 'Consolas','Courier New',monospace; font-size: 9.5pt;
       background: #f4f4f4; padding: 6pt; border: 1px solid #e2e2e2;
       border-radius: 3px; line-height: 1.3; overflow-x: auto; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 10pt; width: 100%; }
th, td { border: 1px solid #c8c8c8; padding: 4pt 6pt; vertical-align: top; }
th { background: #ececec; font-weight: 600; text-align: left; }
tr:nth-child(2n) td { background: #fafafa; }
img { max-width: 100%; height: auto; display: block; margin: 8pt auto; }
hr { border: 0; border-top: 1px solid #c8c8c8; margin: 12pt 0; }
"""


def insert_figure_block(md_text: str) -> str:
    fig_files = sorted(p for p in FIG_DIR.glob("fig*.png"))
    if not fig_files:
        return md_text
    lines = ["", "---", "",
             "## Embedded figures (printable copy)", ""]
    for fp in fig_files:
        title = fp.stem.replace("_", " ").title()
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"![{title}]({fp.resolve().as_posix()})")
        lines.append("")
    return md_text + "\n".join(lines)


def main():
    md = SRC.read_text(encoding="utf-8")
    md = insert_figure_block(md)
    pdf = MarkdownPdf(toc_level=2)
    pdf.add_section(Section(md, toc=True), user_css=CSS)
    pdf.meta["title"] = ("Policy Formula Choice Dominates Forecaster Choice "
                          "for Inventory Cost in Intermittent Maritime Spare Parts")
    pdf.meta["author"] = "[Author TBD]"
    pdf.meta["subject"] = ("Cost-aware intermittent-demand benchmark on the focal"
                            "operating segment of a multi-brand cruise operator")
    pdf.meta["keywords"] = ("intermittent demand; spare-parts forecasting; "
                              "inventory; cost-aware evaluation; foundation models; "
                              "policy formula; maritime")
    pdf.save(str(DST))
    print(f"Wrote {DST}  ({DST.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
