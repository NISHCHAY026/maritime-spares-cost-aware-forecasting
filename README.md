# When Does Forecast Accuracy Predict Inventory Cost?

Code, anonymised data and the full manuscript for a study of cost-aware
intermittent-demand forecasting on a maritime spare-parts fleet.

Eleven forecasters are scored on 15,348 SKUs of real operational data, both on
point accuracy and on the cost they produce inside an (s, S) inventory
simulation, against the operator's own deployed policy as a baseline.

## The finding

Forecast accuracy predicts inventory cost only where holding cost dominates.
The fleet-level rank correlation between mean absolute error and simulated cost
is **+0.591** where holding cost dominates and **-0.345** where stockout cost
dominates. Because 96.8 % of these parts are stockout-dominated, the pooled
correlation is **+0.018**, indistinguishable from zero. That reproduces, on
independent operational data and under a different policy class, the boundary
Theodorou, Spiliotis and Assimakopoulos (2025) reported on M5 retail data.

The sign change is not evidence that accuracy predicts cost. MAE is minimised at
the median of realised demand, which is zero for 77.3 % of these SKUs, and on
such a series MAE is provably monotone in the forecast. Ranking forecasters by
MAE is therefore ranking them by how much they forecast: the per-SKU rank
correlation between forecast level and MAE has median +1.000. The crossover
follows analytically from the newsvendor critical fractile q\* = Cu/(Cu+Co),
which reaches 0.5 at exactly the observed $1,600 unit price.

The obvious remedy fails. Scoring at each SKU's critical fractile inherits the
same confound with the sign reversed. The confound is escapable only under a
policy that consumes the whole predictive distribution, and then only when
scored at the fractile the decision actually reads.

On public M5 data the confound does not transfer: the analytic condition holds
for 10.1 % of M5 series against 77.3 % here. Its severity is a measurable
property of the evaluation window rather than of intermittency.

## Layout

```
paper/       manuscript: LaTeX source, Markdown, compiled PDF, figures
overleaf/    self-contained package for Overleaf (main.tex + figures)
code/        the full pipeline, numbered in run order
data/        anonymised dataset (see data/README.md)
results/     result ledgers in JSON and text, one per analysis
ANONYMIZATION.md   what was anonymised, how, and what is withheld
```

## Reproducing

```bash
pip install -r code/requirements.txt
```

Point the pipeline at this repository and run the analyses. Each script writes a
ledger into `results/`.

```bash
export SPARES_PROJECT_DIR=$(pwd)
python code/26_clean_full.py        # the main comparison
python code/32_cost_regime.py       # the regime split
python code/39_inference.py         # paired bootstrap over SKUs
python code/41_design_weights.py    # population estimates
python code/33_verify_claims.py     # checks the paper against the ledgers
```

`33_verify_claims.py` is the one to run after any change. It cross-checks 46
numbered claims in the manuscript against the result ledgers and exits non-zero
if any has drifted. It exists because an earlier draft contained numbers no
artifact supported.

Two scripts need inputs that are not part of this release. `01_anonymize.py`
needs the raw source files and the secret salt, neither of which is published.
`38_m5_replication.py` needs the M5 competition tables; set `M5_DIR` to a
directory holding them.

## What this data is, and is not

The dataset is a stratified sample and a reduced population panel, both
anonymised with salted keyed hashing. It is published with the operating
company's approval. Read `ANONYMIZATION.md` before drawing conclusions from it,
and `data/README.md` for the column meanings and the sampling design.

Two cautions that matter for anyone reusing it:

- The sample is **equal-allocation stratified**, so a raw mean is a sample
  quantity, not a fleet quantity. Design weights run to 96:1 and are supplied in
  `data/design_weights.csv`. Kish's effective sample size under them is 3,127,
  not 15,348.
- Simulated cost is not realised cost. The simulator is not validated against
  observed stockouts, because the source data contain none. Section 7 of the
  paper lists this and fourteen other limitations.

## Citing

If you use the data or the code, please cite the paper. See `CITATION.cff`.

## Licence

Code is MIT (see `LICENSE`). The dataset is **not** covered by that licence: it
is operational data published with the operating company's approval for research
use. See `data/TERMS.md`.
