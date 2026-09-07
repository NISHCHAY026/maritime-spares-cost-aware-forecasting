# Overleaf package: *When Does Forecast Accuracy Predict Inventory Cost?*

Everything needed to compile the paper. No external files, no bibliography
database, no custom class.

## Contents

```
main.tex                     the paper (self-contained, article class)
figures_clean/               the six figures the paper includes
  figA_censoring.png         Figure 1  nonzero-demand share by quarter
  figB_tau.png               Figure 3  per-SKU Kendall tau distribution
  figC_acc_cost.png          Figure 2  accuracy against simulated cost
  figD_calibration.png       Figure 4  predictive-quantile coverage
  figE_fillmatched.png       Figure 5  fill-matched native-quantile policy
  figF_deployed.png          Figure 6  deployed policy comparison
```

## Compiling

Upload the whole folder to Overleaf (New Project, Upload Project, or drag the
zip in). Overleaf detects `main.tex` automatically.

Compiler: **pdfLaTeX**. Two passes, which Overleaf runs by default. There is no
`.bib` file, so no BibTeX or Biber step is needed; the reference list is a
hand-formatted `list` environment at the end of `main.tex`.

Locally:

```bash
pdflatex main.tex && pdflatex main.tex
```

## Verified before packaging

Compiled here with MiKTeX-pdfTeX 4.23 (pdfLaTeX, two passes):

- exit 0, **0 errors**
- **0 overfull boxes, 0 underfull boxes**
- **0 undefined references or citations**
- 19 pages, all 6 figures embedded, 6 numbered tables, 6 numbered figures

## Preamble

Only stock CTAN packages, all present in Overleaf's default TeX Live:
`inputenc`, `fontenc`, `lmodern`, `geometry`, `amsmath`, `amssymb`, `booktabs`,
`graphicx`, `caption`, `microtype`, `parskip`, `hyperref`.

`\graphicspath{{figures_clean/}}` is set in the preamble, so the figure
directory must keep its name and stay next to `main.tex`.

## Notes for editing

- `main.tex` is byte-identical to `docs/paper/arxiv_v1.tex` in the project
  repository. Edit one and copy it over rather than letting the two drift.
- The source is pure ASCII. Keep it that way: `inputenc` with `utf8` will
  accept more, but arXiv is happier with ASCII and the project's claims
  verifier assumes it.
- Numbers in the text are checked against the analysis ledgers by
  `code/33_verify_claims.py` (46 claims). Change a number in the paper and that
  script should be re-run.
