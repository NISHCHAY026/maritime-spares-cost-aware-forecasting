"""
Render the Markdown paper draft to a printable PDF.

Output:
  docs/paper/paper_draft.pdf

Layout choices (geared for printing on A4/Letter):
  * 11 pt body, 18/15/13 pt headings, serif headings + sans body for legibility
  * 18 mm margins on all sides
  * Tables get a thin border + alternate row tint
  * Images centered with caption-like spacing
"""

from __future__ import annotations

import re
from pathlib import Path

from markdown_pdf import MarkdownPdf, Section

import config as C


PAPER_DIR = C.PROJECT_DIR / "docs" / "paper"
SRC = PAPER_DIR / "paper_draft.md"
DST = PAPER_DIR / "paper_draft.pdf"


CSS = """
@page { size: A4; margin: 18mm 18mm 18mm 18mm; }
body { font-family: 'Calibri', 'Helvetica', 'Arial', sans-serif;
       font-size: 11pt; line-height: 1.45; color: #222; }
h1 { font-family: 'Cambria', 'Georgia', serif; font-size: 18pt;
     margin-top: 14pt; margin-bottom: 8pt; page-break-after: avoid; }
h2 { font-family: 'Cambria', 'Georgia', serif; font-size: 15pt;
     margin-top: 14pt; margin-bottom: 6pt; page-break-after: avoid; }
h3 { font-family: 'Cambria', 'Georgia', serif; font-size: 13pt;
     margin-top: 10pt; margin-bottom: 4pt; page-break-after: avoid; }
h4 { font-family: 'Cambria', 'Georgia', serif; font-size: 11.5pt;
     margin-top: 8pt; margin-bottom: 3pt; page-break-after: avoid; }
p  { margin: 0 0 6pt 0; text-align: justify; }
ul, ol { margin: 0 0 6pt 18pt; }
li { margin-bottom: 2pt; }
blockquote { border-left: 3px solid #ccc; padding: 0 8pt; color: #555; }
code { font-family: 'Consolas', 'Courier New', monospace; font-size: 10pt;
       background: #f4f4f4; padding: 0 3px; }
pre  { font-family: 'Consolas', 'Courier New', monospace; font-size: 9.5pt;
       background: #f4f4f4; padding: 6pt; border: 1px solid #e2e2e2;
       border-radius: 3px; line-height: 1.3; overflow-x: auto; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 10pt; width: 100%; }
th, td { border: 1px solid #c8c8c8; padding: 4pt 6pt; vertical-align: top; }
th { background: #ececec; font-weight: 600; text-align: left; }
tr:nth-child(2n) td { background: #fafafa; }
img { max-width: 100%; height: auto; display: block; margin: 8pt auto; }
hr { border: 0; border-top: 1px solid #c8c8c8; margin: 12pt 0; }
.caption { font-size: 9.5pt; color: #555; text-align: center; margin-top: 2pt; }
"""


def fix_image_paths(md_text: str) -> str:
    """
    The markdown references figures via relative paths like
    `figures/fig2_acc_vs_cost.png` (only inside Markdown image syntax).
    The PDF renderer needs absolute paths because we feed it the markdown
    string directly. We DO NOT touch parquet / .py file paths mentioned
    in prose since those aren't images — only `![...](path)` constructs.
    """
    fig_dir = PAPER_DIR / "figures"
    def _resolver(match: re.Match) -> str:
        alt, target = match.group(1), match.group(2)
        if target.startswith(("http://", "https://", "/")) or Path(target).is_absolute():
            return match.group(0)
        candidate = (PAPER_DIR / target).resolve()
        if not candidate.exists():
            candidate = (fig_dir / Path(target).name).resolve()
        return f"![{alt}]({candidate.as_posix()})"
    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _resolver, md_text)


def insert_figure_block(md_text: str) -> str:
    """
    The current draft references figures by name in captions but doesn't
    embed the PNGs. For the printable PDF we want the actual images. We
    add an "Embedded figures" section at the end with each figure inline.
    """
    fig_dir = PAPER_DIR / "figures"
    fig_files = sorted(p for p in fig_dir.glob("fig*.png"))
    if not fig_files:
        return md_text
    lines = ["", "---", "",
             "## Embedded figures (printable copy)", "",
             "These are the rendered figures for the captions in the "
             "Figures and Tables section above. The originals at 300 DPI "
             "live in `docs/paper/figures/`.", ""]
    for fp in fig_files:
        name = fp.stem.replace("_", " ").title()
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"![{name}]({fp.resolve().as_posix()})")
        lines.append("")
    return md_text + "\n".join(lines)


def main():
    md = SRC.read_text(encoding="utf-8")
    md = fix_image_paths(md)
    md = insert_figure_block(md)

    pdf = MarkdownPdf(toc_level=2)
    pdf.add_section(Section(md, toc=True), user_css=CSS)
    pdf.meta["title"] = ("When Forecast Accuracy Misleads Inventory Decisions: "
                          "A Cost-Aware Benchmark on Lumpy Spare Parts")
    pdf.meta["author"] = "[Author TBD]"
    pdf.meta["subject"] = ("Cost-aware intermittent-demand forecasting "
                            "benchmark on a multi-brand maritime fleet")
    pdf.meta["keywords"] = ("intermittent demand; spare-parts forecasting; "
                              "inventory; cost-aware evaluation; benchmarking")
    pdf.save(str(DST))

    print(f"Wrote {DST}  ({DST.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
