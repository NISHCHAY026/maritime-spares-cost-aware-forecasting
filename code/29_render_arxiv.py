"""
Render docs/paper/arxiv_v1.md (clean-window arXiv preprint) to PDF.

The figures are referenced in-text as "Figure A".."Figure F"; they are
appended as a captioned figure appendix at the end (markdown-pdf has no
float placement). Figure files live in docs/paper/figures_clean/.
"""

from __future__ import annotations

from markdown_pdf import MarkdownPdf, Section

import config as C


PAPER_DIR = C.PROJECT_DIR / "docs" / "paper"
SRC = PAPER_DIR / "arxiv_v1.md"
DST = PAPER_DIR / "arxiv_v1.pdf"
FIG_DIR = PAPER_DIR / "figures_clean"

# letter -> (filename stem, caption)
FIG_CAPTIONS = {
    "A": ("figA_censoring",
          "Demand profile across panel quarters. The original test window "
          "(Q21-28) is right-censored; the clean window is Q17-20."),
    "B": ("figB_tau",
          "Mean per-SKU Kendall's tau (MAE-rank vs cost-rank), split by "
          "demand activity. The zero-demand value is mechanical and excluded."),
    "C": ("figC_acc_cost",
          "Accuracy (test MAE) vs simulated cost, clean window. The most "
          "accurate model (Chronos) is cheapest; the second (MA) is dearest."),
    "D": ("figD_calibration",
          "Predictive-quantile calibration, clean window. Q99 of LightGBM "
          "and Chronos covers only ~94% of realised demand."),
    "E": ("figE_fillmatched",
          "Fill-matched native-quantile vs normal-approximation cost. The "
          "native-quantile advantage appears only for calibrated ZIP."),
    "F": ("figF_deployed",
          "Deployed vs model-driven policy cost on the overlap SKUs. The "
          "best model beats the deployed policy in direction."),
}


CSS = """
@page { size: A4; margin: 18mm 18mm 18mm 18mm; }
body { font-family: 'Calibri','Helvetica','Arial',sans-serif;
       font-size: 11pt; line-height: 1.45; color: #222; }
h1 { font-family: 'Cambria','Georgia',serif; font-size: 17pt;
     margin-top: 14pt; margin-bottom: 8pt; page-break-after: avoid; }
h2 { font-family: 'Cambria','Georgia',serif; font-size: 14pt;
     margin-top: 14pt; margin-bottom: 6pt; page-break-after: avoid; }
h3 { font-family: 'Cambria','Georgia',serif; font-size: 12.5pt;
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
.cap { font-size: 9.5pt; color: #555; text-align: center; margin: 2pt 0 12pt 0; }
hr { border: 0; border-top: 1px solid #c8c8c8; margin: 12pt 0; }
"""


def figure_appendix() -> str:
    lines = ["", "---", "", "## Figures", ""]
    for letter, (stem, caption) in FIG_CAPTIONS.items():
        fp = FIG_DIR / f"{stem}.png"
        if not fp.exists():
            print(f"  WARNING missing figure: {fp}")
            continue
        lines.append(f"![Figure {letter}]({fp.resolve().as_posix()})")
        lines.append("")
        lines.append(f"<p class='cap'><b>Figure {letter}.</b> {caption}</p>")
        lines.append("")
    return "\n".join(lines)


def main():
    md = SRC.read_text(encoding="utf-8")
    md = md + "\n" + figure_appendix()
    pdf = MarkdownPdf(toc_level=2)
    pdf.add_section(Section(md, toc=True), user_css=CSS)
    pdf.meta["title"] = ("Accuracy, Calibration, and Cost in Intermittent-Demand "
                         "Forecasting: Evaluation Pitfalls from a Maritime "
                         "Spare-Parts Study")
    pdf.meta["author"] = "Nishchay Patel"
    pdf.meta["subject"] = ("Cost-aware intermittent-demand benchmark on the "
                           "operating segment of a multi-brand cruise group "
                           "(clean, uncensored window)")
    pdf.meta["keywords"] = ("intermittent demand; spare-parts inventory; "
                            "cost-aware forecast evaluation; calibration; "
                            "right-censoring; maritime operations")
    pdf.save(str(DST))
    print(f"Wrote {DST}  ({DST.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
