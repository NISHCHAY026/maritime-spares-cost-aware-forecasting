# Dataset

Anonymised maritime spare-parts demand, forecasting inputs and deployed-policy
parameters. Published with the operating company's approval. Read
`../ANONYMIZATION.md` for the anonymisation method and the disclosure
inventory, and `TERMS.md` before redistributing.

## Files

| File | Rows | What it is |
|---|---|---|
| `analysis_sample.parquet` | 15,348 | the SKUs the paper analyses |
| `population_panel.parquet` | 219,783 | the panel the sample was drawn from |
| `deployed_policy.parquet` | 13,168 | the operator's deployed (s, S) parameters |
| `design_weights.csv` | 19 | stratum population and sample counts, and w_h |
| `results_per_sku.parquet` | 15,348 | per-SKU MAE and simulated cost, all 11 models |
| `results_per_sku_equalinfo.parquet` | 15,348 | the same under equalised information sets |
| `stockclass_levels.csv` | 93 | the opaque stock-class levels |
| `forecasts_chronos_clean.parquet` | 15,348 | Chronos-T5-small forecasts, 16-quarter context (Q1-16), 1,000 seeded samples |
| `forecasts_chronos_ctx12.parquet` | 15,348 | the same on the 12-quarter context (Q1-12), for the specification grid |

The two Chronos files are model outputs, included because regenerating them
takes several GPU-hours; every other model in the paper refits from
`analysis_sample.parquet` in minutes. With them present, `46_grid_inference.py`
runs all nine grid arms with eleven models.

## Columns

Keys. `STOCK_ITEM_NUMBER` is a salted keyed BLAKE2b surrogate for the part type,
consistent across every file. `FORECAST_ID` identifies the forecast run. Neither
can be inverted without the secret salt, which is never published.

Demand. `ACTUALS_QTR01` to `ACTUALS_QTR28` are quarterly consumption. The paper
trains on quarters 1 to 16 and tests on 17 to 20; quarters 21 to 28 are
right-censored by reporting lag and are used only in the censoring experiment.

Attributes. `SEGMENTATION_GRP` is the Syntetos-Boylan demand class (Smooth,
Erratic, Intermittent, Lumpy, Unknown). `ABC_GRP` is the ABC value class.
`STRATEGY_GRP`, `VELOCITY_MODE`, `CRITICALITY_MODE`, `STATUS_MODE` and
`IS_SLOW_MOVER` are operational categoricals. `STOCKCLASS_MODE` is an equipment
class, relabelled to opaque `SC_00` to `SC_92`: the underlying three-letter codes
are the operator's internal taxonomy, and the relabelling preserves sort order so
the pipeline's categorical encoding, and therefore every published result, is
unchanged.

Economics. `UNIT_PRICE_USD` drives the holding-versus-stockout regime split, at a
crossover of $1,600 under the paper's cost configuration. `LEAD_TIME_MEAN` is in
days. `USAGE_PER_YEAR_MEAN` is the operator's own usage estimate.

Deployed baselines. `SBA_ALPHA` and `SES_ALPHA` are the smoothing constants from
the operator's deployed configuration, so the classical methods here are
evaluated as deployed rather than tuned. `DEPLOYED_NEW_MIN_MEAN` and
`DEPLOYED_NEW_MAX_MEAN` are the deployed reorder point and order-up-to level.

## The sampling design matters

`analysis_sample.parquet` is an **equal-allocation stratified sample** of
`population_panel.parquet`, stratified on `SEGMENTATION_GRP` x `ABC_GRP` with a
per-cell cap of 1,579. Sampling fractions run from 1.0 % to 100 %, so the design
weight w_h = N_h/n_h spans 1.00 to 96.12 and the Intermittent x C cell alone is
69 % of the population against 10 % of the sample.

A raw mean over the sample is therefore a statement about the sample, not the
fleet. Use `design_weights.csv` for population estimates. Kish's effective sample
size under those weights is 3,127. Section 5.9 of the paper reports both.

## What is not here

The raw source files, the secret salt and the surrogate-to-raw mappings are never
published. Also dropped from these tables, as having no role in the analysis: the
operator's per-quarter forecast, error and bias series (644 columns), internal
system insert timestamps, and purchase dates.
