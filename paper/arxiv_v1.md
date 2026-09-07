# When Does Forecast Accuracy Predict Inventory Cost? Evidence from a Maritime Spare-Parts Fleet

**Author.** Nishchay Patel

**Abstract.**
Whether forecast accuracy predicts inventory cost is usually answered by
correlating the two across methods. On 15,348 maritime spare-parts
series under an (s, S) policy, with the operator's deployed policy as baseline,
that statistic fails twice. It is confounded: MAE is minimised at the median
of demand, zero for 77.3 % of these series, so ranking methods by MAE
ranks them by forecast level, six copies of one forecaster differing only by
a scale factor reproduce the full regime pattern, correlating at +1.000 where
holding cost dominates. The regime boundary follows from the newsvendor
fractile at a derivable unit price, here \$1,600. The statistic is also fragile,
changing sign across nine defensible evaluation designs while per-SKU
statistics barely move. Scoring the critical
fractile inherits the confound; scoring the fractile the replenishment decision
reads escapes it. On M5 the zero-median condition holds for
10.1 % of series, helping reconcile the conflicting evidence.

**Keywords.** intermittent demand; spare-parts inventory; cost-aware forecast
evaluation; cost regime; foundation models; maritime operations.

---

## 1. Introduction

Forecasting models are judged by accuracy metrics such as MAE, MASE and RMSSE.
In spare-parts management the model is not the end of the line: it feeds an
(s, S) inventory policy, and what that policy costs is holding cost plus
whatever stockouts cost. Whether the two criteria agree has been asked for two
decades (Syntetos & Boylan 2006; Teunter et al. 2010; Prak & Teunter 2019;
Kolassa 2016), and the answers have not been consistent.

Recent work sharpens the question from two directions. Petropoulos, Wang &
Disney (2019) compared inventory performance across the M3 methods, and
Theodorou, Spiliotis & Assimakopoulos (2025) found on the M5 data that the
accuracy-to-cost link is conditional on the cost structure: accuracy is more
relevant when holding cost is comparable to or larger than lost-sales cost,
and where lost-sales cost dominates the preferable method may not be the most
accurate one. Separately, it is known that point-error metrics misbehave on
mostly-zero series: MAE rewards a flat zero forecast, which motivated MASE
(Hyndman & Koehler 2006), and Wallström & Segerstedt (2010) and Kolassa (2016,
2020) argue the problem runs deeper than any one metric.

This paper connects the two observations and pushes the conclusion further
than either. We work on the spare-parts operation of a multi-brand cruise
group, three brands and 40 vessels, around 220k SKUs with quarterly history,
with the (s, S) parameters the operator actually runs as a baseline. On this
data, under policies that read the forecast through its mean, the regime
pattern is not a fact about forecast quality at all. It is
what the metric-level confound looks like after it passes through a cost
function; its boundary is derivable from the cost parameters alone; it
reproduces in full on six copies of one forecaster that differ only by a scale
factor; and the fleet-level statistic that exhibits it flips sign under
evaluation-design choices that studies in this literature rarely report.

**Contributions.**

* **MAE is confounded with forecast level, by identity rather than by
  tendency.** For a flat forecast, MAE is minimised at the median of realised
  demand. That median is zero for 77.3 % of these SKUs, and on such a SKU MAE
  is strictly increasing in the forecast, so ranking forecasters by MAE ranks
  them by level. Per SKU the level-MAE rank correlation has median +1.000,
  with 78.6 % of SKUs at exactly +1, and six scaled copies of one forecaster,
  which contain identical information about demand, reproduce the full regime
  pattern at +1.000 where holding cost dominates.
* **The regime boundary has a single-period derivation.** The cost-minimising
  forecast is the quantile at the newsvendor critical fractile
  q* = Cu/(Cu+Co), which crosses the 0.5 fractile MAE targets at a derivable
  unit price, \$1,600 here, and the sign of the accuracy-cost correlation
  differs across that price as predicted.
  Where the zero-median condition binds, the regime pattern is therefore the
  confound's fingerprint: real, predictable from the cost parameters, and not
  evidence that better forecasts cost less. On M5 the condition binds for one
  series in ten, so the boundary Theodorou et al. (2025) report cannot be
  wholly this identity at work.
* **The fleet-level statistic is fragile as an instrument.** The correlation
  between method-mean accuracy and method-mean cost rests on eleven points.
  Across nine defensible evaluation designs it runs from -0.103 to +1.000 in
  the holding-dominated group, and dropping one model moves it from +0.591 to
  +0.455. Its SKU-bootstrap interval holds the design fixed and so understates
  the uncertainty that matters. The per-SKU τ, computed over 15,348 SKUs,
  stays between +0.333 and +0.435 in every design.
* **The obvious remedy fails, and the working one is narrower than expected.**
  Scoring at each SKU's own q* is confounded with level just as badly in the
  opposite direction (per-SKU pinball-level rank correlation median -1.000).
  Under a policy that consumes the whole predictive distribution the confound
  is escapable, but only by scoring at the fractile the decision reads:
  pinball at the service level predicts cost better than MAE (median +0.800
  against +0.632) while a generic five-quantile CRPS does worse (+0.400). This
  comparison rests on the four models that emit quantiles.
* **The severity is measurable in advance.** On the public M5 data the
  level-MAE statistic has median +0.194 because the analytic condition holds
  for only 10.1 % of M5 series against 77.3 % here; conditioning on it, the
  mechanism reappears (+0.614 against +0.157). The confound is a property of
  how zero-dense the evaluation window is, not of intermittent demand as such,
  and competition-data and spare-parts studies sit on opposite sides of it.
* **Secondary results.** The within-SKU and fleet-level statistics answer
  different questions and disagree (+0.387 against +0.018). A zero-shot
  Chronos-T5 model is the most accurate and the cheapest point estimate of
  eleven forecasters, though separable from almost none of them. Ten of eleven
  model-driven policies beat the operator's deployed parameterisation, with a
  bootstrap interval on the gap of [-60.2 %, -41.0 %]. And deliberately
  scoring against right-censored quarters does not reorder the forecasters.

Section 2 places the work; Section 3 covers data; Section 4 methods; Section 5
results; Sections 6 to 8 discuss, bound and conclude. A reproducibility note
records the analysis history, including several corrections.

---

## 2. Related work

**Intermittent-demand forecasting.** Croston (1972) smoothed demand size and
inter-demand interval separately. Syntetos & Boylan (2005) corrected its
inversion bias, giving SBA; Teunter, Syntetos & Babai (2011) smoothed the
demand probability instead (TSB); Nikolopoulos et al. (2011) introduced
temporal aggregation (ADIDA/IMAPA); Kourentzes (2014) added parameter tuning.
MAPE is undefined on mostly-zero series and plain MAE rewards a flat zero
forecast, so Hyndman & Koehler (2006) standardised the scaled error (MASE).

**Accuracy versus inventory performance.** Syntetos & Boylan (2006) showed
empirically that accuracy and stock-control rankings can diverge; Teunter et
al. (2010) and Prak & Teunter (2019) developed the gap analytically; Teunter &
Duncan (2009) showed the accuracy ranking of intermittent-demand methods
depends on the error measure chosen, and Syntetos, Nikolopoulos & Boylan
(2010) proposed judging forecasts by accuracy-implication metrics rather than
accuracy itself. On the
metric side, Wallström & Segerstedt (2010) found that error measures for
intermittent series disagree with each other and with stock-control outcomes,
Kolassa (2016) argued point-error metrics are the wrong target for count data,
and Kolassa (2020) showed the "best" point forecast is a property of the error
measure chosen. Petropoulos, Wang & Disney (2019) measured inventory
performance across M3 methods, and Theodorou, Spiliotis & Assimakopoulos
(2025) found on M5 that the accuracy-cost relationship depends on the balance
between holding and lost-sales cost. This paper argues those two lines are
reporting the same phenomenon: the regime dependence is what the metric-level
confound produces once it passes through a cost function, its boundary is
derivable from the cost parameters, and the fleet-level statistic that
exhibits it is fragile to evaluation design.

**Probabilistic forecasting and calibration.** Gneiting et al. (2007)
formalised calibration and sharpness. We use predictive quantiles to set
safety stock and report the coverage they achieve.

**Global and foundation-model forecasting.** Cross-learning across series
(Salinas et al. 2020; Bandara et al. 2020; Makridakis et al. 2022) set up the
zero-shot foundation models, of which Chronos (Ansari et al. 2024) is one.
Vandeput (2020) cautions that machine learners over-predict when most
observations are zero. Head-to-head evaluations of a foundation model against
Croston-family methods on intermittent spares, scored by inventory cost rather
than accuracy alone, appear to be scarce.

**Inventory policy.** The (s, S) machinery is textbook (Silver, Pyke & Thomas
2017; Snyder & Shen 2019).

---

## 3. Data and deployed pipeline

### 3.1 Source, scope and sampling

The operator runs four maintenance-management instances, three legacy and one
migrated. We extract spare-part consumption, the parts master, the
forecast-analysis table and the deployed (s, S) parameter table, restricted to
the cruise group's operating segment. Anonymisation is keyed and salted: every
identifying field (brand, vessel, SKU, part type, vendor, department) becomes a
namespaced surrogate, consistent across tables. Free-text fields are dropped.
Dates, quantities, prices, lead times, the demand-pattern segmentation, ABC
class and criticality are preserved.

The analysis runs on a **stratified sample of 15,348 SKUs** drawn from the
219,783-SKU forecast panel by equal allocation across demand-pattern and ABC
cells, with a per-cell cap of 1,579. Equal allocation buys precision in the
small, high-value cells that a proportional sample would barely reach, and it
is the right design for a paired model comparison. It is the wrong design for
reading a raw mean as a fleet quantity: sampling fractions run from 1.0 % to
100 %, so the design weight w_h = N_h/n_h spans 1.00 to 96.12, and the single
Intermittent x C cell is 69 % of the panel against 10 % of the sample.

We therefore report both. Model comparisons, which are paired within SKU, use
the sample unweighted. Every quantity stated as a property of the fleet is also
reported as a Horvitz-Thompson estimate carrying w_h, with a bootstrap that
resamples within strata (Section 5.10). Kish's effective sample size under those
weights is 3,127, not 15,348, and we quote it wherever a weighted interval is
given.

### 3.2 Right-censoring and the evaluation window

Each SKU carries 28 quarterly observations. The fraction with nonzero demand
holds at about 5 to 7 % through panel quarter 20, then falls to roughly 1 % by
quarters 25 to 28 (Figure A). This is a reporting lag rather than a demand
collapse: the consumption log is stable through the most recent complete
calendar quarter and only then drops, because recent transactions have not all
been entered. The last panel quarters are incomplete.

We therefore evaluate on **train = panel quarters 1 to 16, test = quarters 17
to 20**, four quarters at roughly 5 % nonzero, matching the training level. Of
the 15,348 SKUs, 5,929 (38.6 %) see positive demand in that window. Section 5.9
reports what happens when the test window is deliberately placed in the
incomplete region instead.

---

## 4. Methods

### 4.1 Forecasters

Eleven models in three tiers. **Classical (seven):** the operator's deployed
SBA, SES and two-quarter moving average, plus Croston without bias correction,
TSB, ADIDA and IMAPA. **Distributional (two):** an intercept-only
Zero-Inflated Poisson (ZIP) and a Hurdle Negative-Binomial (HNB), fit by
method-of-moments and turned into predictive quantiles by sampling 1,000 draws.
**Global and foundation (two):** a global LightGBM (Ke et al. 2017) with
separate occurrence, size and quantile heads, and Chronos-T5-small run zero-shot on the 16-quarter
context with 1,000 seeded sample paths.

Two implementation points matter for reproducibility. LightGBM's quantile
heads are fit on positive-target rows, so they estimate the size quantile
conditional on occurrence; the predictive quantile of demand is the hurdle
mixture of that with the occurrence probability, not the conditional quantile
gated by an indicator. And LightGBM is non-deterministic across runs when
multithreaded unless its seeds, `deterministic`, `force_row_wise` and thread
count are all pinned. Both points are elaborated in the reproducibility note,
because getting either wrong changes the conclusions.

### 4.2 (s, S) cost simulator

For each SKU and forecaster we build an (s, S) policy and replay the
test-window demand: quarterly review, integer-quarter lead times
(⌈days/91.3125⌉, minimum one), and holding cost charged on end-of-quarter
positive on-hand. Default configuration: holding 25 %/yr of unit value,
ordering \$50 per order, stockout \$100 per unit short. Section 5.8 sweeps all
three.

Four modelling choices shape these results and are stated explicitly, because
each one is consequential on a four-quarter window.

* **Unmet demand is lost, not backordered.** A stockout is charged once per
  unit short and the shortfall does not carry into the next quarter.
* **Inventory starts saturated** at on-hand = S. On this window that is not a
  neutral choice: 81.8 % of SKUs never place an order in the four test
  quarters, so for most of the sample the simulation measures the cost of
  holding an initial position rather than the cost of replenishing. Starting
  from zero instead raises mean cost by about 5 % but drops mean fill from
  0.965 to 0.830, so the start condition matters far more for service than for
  cost. The specification grid of Section 5.4 re-runs the whole comparison from
  four opening positions and three windows, so this choice is tested rather
  than assumed.
* **Fill rate is 1.0 by definition when a SKU has no realised demand**, which
  is 61.4 % of the sample. Reported fleet fill rates are therefore mostly
  definitional; the demand-active mean fill is closer to 0.91. We use fill only
  to match service between the two policy formulas, never as a headline result.
* **Cost is dominated by holding.** Under the default configuration the split
  is 90.5 % holding, 9.0 % stockout, 0.5 % ordering. Since holding cost is
  close to monotone in forecast level, the cost ranking is substantially a
  ranking of forecast bias, and Section 5.3 should be read with that in mind.

The service level enters only through z = Φ⁻¹(SL) in the safety-stock term, so
it is applied as a cycle-service (Type-1) quantity even though the criticality
tiers were specified as availability targets.

**Normal-approximation policy.** The reorder point is
s = D̄·LT + z·√(LT·σ_D² + D̄²·σ_LT²), with EOQ from the standard formula and
S = s + EOQ. σ_D is held model-agnostic, one training-window standard
deviation shared by every forecaster, so the only thing that varies across
forecasters is the point forecast D̄. This isolates point-forecast quality, at
the cost of not testing forecaster-specific uncertainty (Section 7).

**Native-quantile policy.** For the four models that emit quantiles we set s
from the model's own predictive quantile of lead-time demand at the SKU's
criticality-tiered service level, then rescale by a bisection-searched
multiplier κ until fleet fill matches that model's normal-approximation
policy, and compare cost at equal service.

### 4.3 Metrics and the cost regime

Point accuracy is MAE, with RMSSE as a genuinely independent check. We do not
report MASE as a robustness check on the rank statistic: MASE is a strictly
positive per-SKU rescaling of MAE, so the per-SKU rank correlation is identical
to the MAE one by construction and corroborates nothing.

We report the accuracy-cost relationship at two levels, because they answer
different questions. The **within-SKU** statistic is the mean per-SKU Kendall's
τ between MAE-rank and cost-rank across the eleven forecasters, with SKU-level
bootstrap CIs. The **fleet-level** statistic is the Spearman correlation
between each forecaster's mean MAE and its mean cost, across the eleven
forecasters. ZIP and HNB share a point forecast (Section 5.1), so the eleven
forecasters contribute ten distinct points to this correlation; we keep both
and let the tie stand. The second statistic is what a practitioner implicitly
uses when choosing one model for the whole fleet.

**Cost regime.** Under the default configuration, per-unit-quarter holding cost
is (0.25/4)·price and the stockout penalty is a flat \$100, so the two are equal
at a unit price of \$1,600. SKUs below that are stockout-dominated; above it,
holding-dominated. This gives a direct test of the Theodorou et al. boundary.

---

## 5. Results

All results are on the uncensored window: train quarters 1 to 16, test 17 to 20.

### 5.1 Accuracy

Mean test MAE, best to worst: Chronos 6.525, MA 7.598, SES 7.889, IMAPA 8.201,
ADIDA 8.311, HNB 8.595, ZIP 8.595, TSB 8.633, LGBM 8.775, SBA 8.948, Croston
9.107 (Figure C, x-axis). Chronos wins zero-shot with no fine-tuning. SBA and
Croston come last, which fits: a bias-corrected estimator is not built to
minimise raw error when most periods are zero.

ZIP and HNB have algebraically identical point forecasts here (both reduce to
the training sample mean), so the study contains ten distinct point
forecasters, not eleven. We keep both because their predictive distributions
differ, which matters in Section 5.7.

### 5.2 Cost

Mean cost per SKU, cheapest to dearest: Chronos \$2,097, HNB \$2,554, ZIP
\$2,555, TSB \$2,564, SBA \$2,581, Croston \$2,585, IMAPA \$2,607, SES \$2,657,
ADIDA \$2,691, MA \$2,879, LGBM \$2,936, at fill rates from 0.956 to 0.980
(Figure C, y-axis).

Chronos is both the most accurate and the cheapest. The second-most-accurate
model, the moving average, is the second most expensive. That pair is the whole
problem this paper is about, and Section 5.3 explains why it happens.

The ordering is almost entirely unresolvable, and we quantify that rather than
noting it in passing. Simulated cost is extremely right-skewed: 3.2 % of SKUs
carry 82.3 % of it. Those are not an independently identified tail; they are
the holding-dominated group of Section 5.4, since the same unit-price threshold
defines both. A paired bootstrap over SKUs (B = 20,000) gives Chronos a
mean of \$2,097 with a 95 % interval of [\$545, \$5,089]. On the paired
difference against the cheapest model, **only one of ten models is
distinguishable**: LightGBM, at +\$839 [+212, +1,834]. Chronos against
Hurdle-NB is +\$457 [-232, +1,611]. So Chronos is the cheapest *point estimate*
but is not statistically separable from nine of the other ten, and no claim in
this paper rests on the cost ordering of adjacent models.

### 5.3 MAE is confounded with forecast level

**Within-SKU.** Mean per-SKU Kendall's τ between MAE-rank and cost-rank:

| SKU subset | n (τ defined) | τ | 95 % CI |
|---|---|---|---|
| Demand-active | 5,341 | +0.387 | [+0.369, +0.405] |
| Zero-demand | 9,419 | +0.999 | mechanical, excluded |
| Pooled | 14,760 | +0.778 | inflated by zero-demand |

On a SKU with no realised demand both MAE and holding cost increase with
forecast magnitude, so τ is positive by construction; we rely on the
demand-active value. Under RMSSE the same statistic falls to +0.175
[+0.154, +0.192], so even within-SKU the answer depends on the error metric.

**Fleet-level.** Across the eleven forecasters, Spearman ρ between mean MAE and
mean cost is **+0.018 (p = 0.958)**: no relationship at all. The within-SKU and
fleet-level statistics are both correct and they disagree, because they ask
different questions. Averaging ranks within SKUs measures whether a better
forecast for a given part tends to cost less for that part. Correlating model
means measures whether a more accurate model is a cheaper model for the fleet,
which is the question a practitioner selecting one forecaster is actually
asking.

**The mechanism.** The two statistics above disagree because neither is
measuring forecast quality alone. Start analytically. For a flat forecast f > 0 evaluated on a window
d_1, ..., d_H, mean absolute error has derivative

    dMAE/df = ( #{t : f > d_t} - #{t : f < d_t} ) / H,

so if more than half of the d_t are zero, MAE is strictly increasing in f for
every f > 0. On such a SKU, ranking forecasters by MAE is *identically* ranking
them by forecast level, inverted. This is not a tendency that might wash out in
a larger sample; it is a property of the metric on zero-dense data.

The condition binds here. The median of the test window is zero for **77.3 %**
of SKUs (61.4 % are entirely zero), and even among demand-active SKUs 41.3 %
still have a zero median. Measured per SKU across the eleven forecasters, the
rank correlation between forecast level and MAE has **median +1.000**, with
78.6 % of SKUs at exactly +1 (n = 14,760). The same statistic against simulated
cost has median +1.000. These are computed over 15,348 SKUs rather than over
the eleven model means, so they do not rest on an eleven-point correlation.

The fleet-level view is the aggregate of that:

| subset | ρ(MAE, cost) | ρ(MAE, level) | partial ρ(MAE, cost \| level) | ρ(MAE, holding) | ρ(MAE, stockout) |
|---|---|---|---|---|---|
| All | +0.018 | +0.982 | -0.187 | +0.018 | -0.509 |
| Stockout-dominated | -0.345 | +0.964 | +0.548 | +0.827 | -0.509 |
| Holding-dominated | +0.591 | **+1.000** | **+0.000** | +0.591 | +0.091 |

In the holding-dominated group, where the raw correlation is strongest, MAE and
forecast level are perfectly rank-correlated and the partial correlation is
exactly zero: accuracy and level cannot be separated there at all. The
mechanism is visible in the last two columns. Forecast level raises holding
cost (ρ = +0.827, p = 0.002) and lowers stockout cost (ρ = -0.509). Which of
those two components dominates the total is exactly what the cost regime
decides, and that, not forecast quality, is what sets the sign of the
accuracy-cost relationship.

The practical consequence is stronger than "accuracy is a weak proxy". Because
the confound is structural rather than incidental, MAE cannot serve as a
model-selection criterion for these SKUs in *either* regime: where it appears
to work it is standing in for level, and where it appears to fail it is also
standing in for level, with the sign flipped by the cost structure.

**The confound, demonstrated with no information differences at all.** Take
the deployed SBA forecast and multiply it by k ∈ {0.25, 0.5, 0.75, 1, 1.5, 2}.
The six variants contain identical information about demand; they differ only
in level. On SKUs with a positive base forecast and a zero test median, MAE is
strictly increasing in k on **100.0 %** of SKUs, which is the derivative above
verified exactly.

| k | mean MAE | mean cost, stockout-dom. | mean cost, holding-dom. |
|---|---|---|---|
| 0.25 | 6.903 | \$563 | \$53,117 |
| 0.50 | 7.383 | \$498 | \$57,667 |
| 1.00 | 8.916 | \$470 | \$66,745 |
| 2.00 | 14.162 | \$498 | \$84,991 |

The fleet-level statistic reproduces the entire regime pattern on this family.
ρ(mean MAE, mean cost) across the six variants is **+1.000** where holding
cost dominates, and +1.000 pooled, because the holding-dominated group carries
82.3 % of the dollars. In the stockout-dominated group mean cost is U-shaped
in k, minimised near k = 1, so the correlation there is -0.486 and means
nothing: a monotone metric against a non-monotone cost. A statistic that
awards a perfect accuracy-to-cost relationship to six copies of the same
forecast is not measuring forecast quality. We suggest this scale-family
placebo as a standard check for accuracy-versus-cost studies: whatever
relationship the pipeline awards to scaled copies is its level effect, and
only signal beyond that is evidence about accuracy.

### 5.4 The regime boundary: derivable, and fragile as an instrument

**The empirical split.** Splitting at the \$1,600 crossover:

| regime | n SKUs | fleet ρ(MAE, cost) | within-SKU τ | most accurate | cheapest |
|---|---|---|---|---|---|
| Stockout-dominated (price < \$1,600) | 14,861 | -0.345 (p = 0.298) | +0.385 | Chronos | LGBM |
| Holding-dominated (price ≥ \$1,600) | 487 | **+0.591 (p = 0.056)** | +0.491 | Chronos | Chronos |

A paired bootstrap over SKUs puts intervals on these. The holding-dominated
correlation is +0.591 with a 95 % interval of **[+0.49, +1.00]**, which excludes
zero, so it survives inference that resamples SKUs rather than relying on an
eleven-point asymptotic test (whose p was 0.056). The stockout-dominated
correlation is -0.345 with an interval of **[-0.76, +0.44]**, which includes
zero: there is no detectable relationship there, and we do not claim a negative
one. Pooled across all SKUs the interval is [-0.29, +0.75], also including zero.

We report these intervals to two decimals deliberately. At B = 2,000 the
holding-dominated lower bound was not stable in the third decimal, ranging from
+0.409 to +0.545 across five seeds; at B = 20,000, which is what we now run, the
same sweep spans +0.464 to +0.500. The percentile endpoint of a rank correlation
over eleven model means is the fragile statistic here, so alongside it we report
the bootstrap share of replicates at or below zero, which is stable: 0.0014 in
the holding-dominated group, 0.727 in the stockout-dominated group and 0.447
pooled.

**The crossover is derivable.** In a newsvendor trade-off with underage cost Cu
and overage cost Co, the cost-minimising order quantity is the demand quantile
at the critical fractile q* = Cu/(Cu+Co). Here Cu is the \$100 stockout penalty
and Co is (0.25/4) x price per quarter, so

    q*(price) = 100 / (100 + 0.0625 x price),

which equals 0.5 at exactly **price = \$1,600**: the crossover used in the
table above, derived rather than observed. The derivation is single-period and
myopic. With a saturated start most units are held for several quarters, which
pushes the true break-even price below \$1,600, so we use \$1,600 as the
definition of the split rather than as an estimate of the optimum. MAE implicitly targets the 0.5
fractile for every SKU regardless of price, so it is aligned with the cost
structure at exactly one price and misaligned on both sides, which is why the
correlation reverses sign instead of merely weakening. The misalignment is
severe: the median q* here is **0.981**, and 96.8 % of SKUs require q* > 0.5.

Under this design, the split reproduces the boundary Theodorou et al. (2025)
report on M5 retail data: accuracy appears to predict cost where holding cost
dominates and not where stockouts dominate, and because 96.8 % of these parts
are stockout-dominated the pooled correlation is nil. A study that reports one
pooled number would conclude accuracy does not matter; a study of high-value
slow movers would conclude it does. But Section 5.3 says what this pattern is
made of, and the derivation above says where its boundary must sit. The regime
split is the level confound passed through a cost function, and reproducing it
here, in a different industry, under a different policy class, against a
deployed baseline and in a far more intermittent regime, is evidence about the
metric and the cost parameters rather than about forecast quality.

**How much does the instrument itself move?** Everything above is computed
over eleven method means under one evaluation design, and eleven points invite
the question of what happens under another. We re-ran the full comparison
under nine designs: four opening inventory positions (saturated at S, the
published choice; uniform in [s, S]; exactly at s; empty) crossed with three
fully uncensored windows. The arms share the ten forecasters that can be fit
on every window, since Chronos needs the 16-quarter context; dropping it alone
moves the holding-dominated correlation from +0.591 to +0.455.

| start | window (train → test) | SKUs ordering | stockout share | ρ pooled | ρ stockout-dom. | ρ holding-dom. | τ |
|---|---|---|---|---|---|---|---|
| S | 1-16 → 17-20 | 18.2 % | 9.1 % | -0.309 | -0.285 | +0.455 | +0.405 |
| mid | 1-16 → 17-20 | 26.2 % | 10.4 % | -0.309 | -0.358 | +0.455 | +0.337 |
| s | 1-16 → 17-20 | 36.3 % | 11.9 % | -0.309 | -0.236 | +0.455 | +0.333 |
| zero | 1-16 → 17-20 | 72.3 % | 36.2 % | -0.455 | -0.273 | +0.418 | +0.435 |
| S | 1-12 → 13-16 | 18.9 % | 34.0 % | +0.685 | -0.212 | -0.103 | +0.411 |
| mid | 1-12 → 13-16 | 27.8 % | 36.0 % | +0.600 | -0.418 | -0.103 | +0.347 |
| S | 1-12 → 13-20 | 28.6 % | 24.6 % | +0.345 | +0.091 | +1.000 | +0.403 |
| mid | 1-12 → 13-20 | 36.2 % | 25.7 % | +0.273 | +0.055 | +1.000 | +0.345 |
| zero | 1-12 → 13-20 | 66.2 % | 40.6 % | +0.127 | -0.055 | +0.927 | +0.396 |

Two things are true at once. The opening position, which is the obvious worry
(under a saturated start only 18.2 % of SKUs ever place an order, and cost is
90.5 % holding), barely matters: starting every SKU empty raises the ordering
share to 72.3 %, lifts stockout cost to 36.2 % of the total, and moves the
holding-dominated correlation from +0.455 to +0.418. The regime result is not
an artifact of an unexercised policy. The window is another matter. The same
statistic is -0.103 on train 1-12 test 13-16 and +1.000 when the test window
extends to 13-20. The first comparison moves the test period along with the
training length, so we attribute it to neither alone; the second holds
training fixed and still spans most of the range a correlation can take. All
three windows are uncensored, and any of them could have been the published
design.

The quantity that does not care is the per-SKU τ, computed over 15,348 SKUs
rather than ten means: it sits between +0.333 and +0.435 in every arm. The
SKU-bootstrap interval of [+0.49, +1.00] quoted above is therefore the smaller
of two uncertainties. It captures sampling variation with the design held
fixed, and the design variation is wider than the interval.

### 5.5 What to use instead

**The obvious remedy does not work.** If MAE targets the wrong fractile, score
each SKU at its own q* instead. We tested that and it fails. Pinball loss at q*
is confounded with level just as badly in the opposite direction: the per-SKU
rank correlation between that loss and forecast level has median **-1.000**,
and fleet-level ρ(pinball, cost) is **-0.491** against MAE's +0.018. It helps
only where holding dominates (+0.809, p = 0.003). Swapping the metric swaps the
direction of the confound rather than removing it.

The reason is structural. Under the model-agnostic σ, the policy sees the
forecast only through the scalar D̄, so simulated cost is a deterministic
function of level and *no* point-error metric can add information about cost.
There is no second channel to exploit.

**A distributional policy does open one, but only just.** Under the
native-quantile policy the reorder point is s = LT · Q_α, so the policy reads a
quantile rather than a mean. That is a genuine second channel: the per-SKU rank
correlation between level and Q_α has median +0.738, with |ρ| > 0.99 on only
14.4 % of SKUs, and cost tracks Q_α (median ρ = +1.000) more tightly than it
tracks level (+0.816). Scoring against native-quantile cost:

| metric | per-SKU ρ with cost, median |
|---|---|
| MAE | +0.632 |
| pinball at the service level | **+0.800** |
| CRPS over the five fitted quantiles | +0.400 |

The ordering is the useful result, and it is not "distributional beats point".
A generic distributional score is *worse* than MAE, because CRPS averages over
the whole quantile grid including the median region the decision never reads.
What wins is the loss evaluated at the specific fractile the policy consumes.
This is Gneiting's (2011) consistency principle observed inside a deployed
policy chain: a scoring function should be consistent for the functional the
decision reads, and the failure of an unweighted CRPS here is the case for
quantile-weighted scoring made by Gneiting & Ranjan (2011).
Only four models emit quantiles, so the fleet-level version of this comparison
rests on four points and we do not interpret it.

### 5.6 Does the confound generalise? A test on public M5 data

If the confound were a property of intermittent demand, it would reappear on
the M5 competition data. It does not. Across 30,490 M5 series and eight
classical forecasters, the per-series rank correlation between forecast level
and MAE has median **+0.194**, with 1.5 % of series at exactly +1, against
median +1.000 and 78.6 % here. By demand class the M5 medians are ordered as
expected (Smooth +0.157, Erratic +0.229, Intermittent +0.253, Lumpy +0.301) but
none is close to the maritime result.

The analytic condition explains the gap. Median demand is zero over the
out-of-sample window for **10.1 %** of M5 series against **77.3 %** here. So the difference is how
zero-dense the evaluation window is, not the horizon length and not the
Syntetos-Boylan class label. Conditioning on the analytic condition, the
mechanism does reappear on M5: the median rank correlation is **+0.614** on the
3,082 series where median demand is zero, against **+0.157** on the 27,384
where it is not.

The claim is therefore conditional and its severity is measurable in advance
from the evaluation window alone. That also reconciles a disagreement rather
than adding to it: competition-data studies and spare-parts studies sit on
opposite sides of this condition, which is one reason they reach different
conclusions about whether accuracy predicts inventory cost.

### 5.7 Predictive-quantile policies

Fill-matched cost change of the native-quantile policy against the
normal-approximation policy, with the coverage its 99th percentile achieves:

| Model | κ to match fill | Q99 coverage | cost Δ at matched fill |
|---|---|---|---|
| ZIP | 0.88 | 0.963 | -14.8 % |
| HNB | 0.91 | 0.973 | +1.5 % |
| LGBM | 0.51 | 0.995 | -64.4 % |
| Chronos | 3.83 | 0.961 | +4.2 % |

All four reach the target fill. The two models whose upper tails are best
calibrated at the nominal 99 % level, LightGBM (0.995) and HNB (0.973), sit at
opposite ends of the cost outcome, so calibration at a single level does not by
itself determine whether a predictive-quantile policy pays. The ZIP saving is
stable across sampling seeds (mean -15.2 %, range -15.5 % to -14.8 % over five
seeds). A global multiplicative conformal correction moves every one of these
by less than half a percentage point, so it neither rescues nor damages any
model here.

We make no general claim from this table. The honest summary is that a
predictive-quantile policy helped two of four models on this data and was
approximately neutral for the other two, and that we cannot separate the effect
of tail calibration from the effect of the fill-matching procedure itself.

### 5.8 Deployed policy comparison

On the 1,579 SKUs where the deployed parameter set can be matched, ten of the
eleven model-driven policies beat the deployed policy on cost. The best, HNB,
costs \$567 against \$1,151, a 51 % reduction. LightGBM is the exception at
\$1,543, or 34 % more expensive than deployed, because its point forecast
over-stocks and this overlap is richer in demand-active SKUs than the sample as
a whole.

The direction is robust and now has an interval: a paired bootstrap over the
overlap SKUs puts the gap at -50.7 % with a 95 % interval of
**[-60.2 %, -41.0 %]**, comfortably excluding zero. The cheapest model in each
cell beats deployed in 80 of 80 cost-configuration cells and in all four service-level scenarios, by
49 % to 53 %. Which model is cheapest is configuration-dependent (IMAPA wins 31
cells, Chronos 23, HNB 13, ZIP 9, ADIDA and SBA 2 each), so there is no single
best forecaster to crown.

We do not report the magnitude with confidence. The data contain no realised
stockouts, so the simulator cannot be validated against ground truth. Against
the recorded inventory snapshot it tracks in shape but not in level: simulated
deployed on-hand correlates with recorded stock at +0.663 in logs but sits at
about half its level (median ratio 0.50), and the simulated deployed fill of
0.864 falls short of the operator's stated 0.95 to 0.99 goal. The simulator
under-stocks relative to reality. Model-versus-model comparisons share the
simulator and are unaffected; the absolute deployed figures are directional.
The overlap is also a biased subsample, over-weighted toward lumpy and
high-ABC parts. We do not compare fill on the overlap, because model fill was
computed sample-wide rather than on this subset.

### 5.9 A negative result: censoring does not reorder forecasters

Because the recent quarters are incomplete, an obvious worry is that a test
window placed there would flatter models that under-forecast. We tested it.
Holding the training window (quarters 1 to 16) and the horizon (four quarters)
fixed, and moving only the test window into the incomplete region (quarters 25
to 28), the demand-active fraction falls from 38.6 % to 11.4 %, confirming the
censoring is real and severe. The cost ranking does not move: the ordering of
all ten models is identical across the two windows, the moving average is
ninth of ten in both, and the within-SKU τ is unchanged (+0.405 clean against
+0.426 censored, overlapping intervals).

A ranking change does appear if one compares the clean window against the
original naive split (train quarters 1 to 20, test 21 to 28), where the moving
average becomes the cheapest model. But that comparison moves the training
window and doubles the horizon as well as shifting the test period, so it
cannot be attributed to censoring. We report this because we initially believed
the opposite, and because the controlled version of the experiment is cheap and
we suspect other cost-aware studies have not run it.

### 5.10 Two design checks: information sets and design weights

Two things about the setup would not survive a referee. Three of the eleven
forecasters saw less history than the rest, and the sample is not the fleet
whose means the paper reports. We fixed both and report what moved.

**Equal information sets.** Croston, SBA and TSB update their smoothed state
from the previous observation, so reading the forecast at the last training
column returns a forecast whose state has consumed only fifteen of the sixteen
training quarters. The final quarter reaches it only through the initialisation
constants, whose weight decays as (1 - alpha)^T. Every other forecaster here
conditions on the full window: SES takes an explicit extra step, the moving
average reads the last two quarters directly, ADIDA and IMAPA end with a
smoothing step on the most recent aggregated block, ZIP and Hurdle-NB fit
moments over all sixteen columns, and LightGBM and Chronos take lags and
context ending at the last quarter. Three of eleven models were being scored on
a shorter history than their competitors.

Running one further iteration of the same recurrence removes the asymmetry. The
three affected models improve, by 0.36 %, 0.37 % and 2.12 % of MAE and 0.30 % to
0.56 % of cost, and the other eight are bit-identical, which is the check that
the change touched nothing else. No conclusion moves. Chronos remains both the
most accurate and the cheapest; the holding-dominated correlation is unchanged
at +0.591; the within-SKU τ is +0.387 either way; the deployed gap moves from
-50.7 % to -50.9 %. Two labels do change: TSB rises from fourth to second
cheapest, and it replaces Hurdle-NB as the cheapest model on the deployed
overlap. Since Section 5.2 already establishes that adjacent models are not
separable, neither label carries an inferential claim.

**Design weights.** Sampling fractions run from 1.0 % to 100 %, so the
Horvitz-Thompson estimate carrying w_h = N_h/n_h answers a different question
from the raw mean: what the fleet looks like, rather than what the sample does.
The two differ substantially.

| quantity | sample | population |
|---|---|---|
| Mean cost per SKU (Chronos) | \$2,097 | \$434 |
| Demand-active SKU share | 38.6 % | 14.8 % |
| Holding-dominated SKU share | 3.2 % | 6.0 % |
| Cost share in holding-dominated | 82.3 % | 62.1 % |
| Mean within-SKU τ (demand-active) | +0.387 | +0.436 |
| ρ(MAE, cost), holding-dominated | +0.591 | **+0.991** |
| ρ(MAE, cost), stockout-dominated | -0.345 | **-0.882** |
| ρ(MAE, cost), pooled | +0.018 | +0.109 |
| Deployed gap | -50.7 % | -52.1 % |

Every dollar figure in this paper is therefore a sample figure, and the fleet is
far more intermittent than the sample: weighted, only 14.8 % of SKUs see demand
in the test window. Chronos remains the cheapest model under weighting and the
deployed gap is unchanged in direction and size.

The regime contrast sharpens rather than dissolving, which is convenient enough
that it deserves scrutiny. A weighted correlation can be one heavy stratum
wearing a large weight, so we decomposed it. The holding-dominated result is
robust: it is positive within every major stratum (+0.845, +0.864, +0.809,
+0.864 for the four largest by weight); removing any of the four largest
strata leaves it above +0.97, and the full leave-one-stratum-out minimum is
+0.69, at a stratum carrying 0.5 % of the weight. The stockout-dominated
result is not robust in this way. It falls from -0.882 to
-0.182 when the Intermittent x C cell is removed, and the within-stratum values
have no consistent sign, running from -0.709 to +0.936. That cell is 68 % of
the fleet by weight, so -0.882 is a true statement about this population and a
statement about one stratum at the same time.

So the contrast survives both checks and the magnitude of its negative half
does not. The correlation is positive where holding cost dominates, in the
sample and in the population alike; Section 5.3 says what it is made of. Where stockouts dominate it does not, and how negative
the relationship looks depends on the stratum. Kish's effective sample size
under these weights is 3,127 overall and
162 in the holding-dominated group, so the weighted intervals are wide by
construction, and we report them alongside the unweighted ones rather than in
place of them. These are the two checks the regime contrast survives; the
window sensitivity of Section 5.4 is the one it does not.

---

## 6. Discussion

The results cohere around one point: the fleet-level correlation between
method-mean accuracy and method-mean cost, which is how this literature
usually asks whether accuracy matters, does not measure what it appears to
measure on intermittent data, and the failure has structure.

It is confounded wherever the zero-median condition binds. MAE-rank is
level-rank there, cost decomposes into a component level raises and a
component level lowers, and the cost regime picks the sign. And it is fragile
even setting the confound aside: eleven points under one design, moving across
most of the admissible range under defensible design changes. Neither failure
is visible from inside a single study, which may be part of why two decades of
asking the question have not produced a consistent answer.

What should an accuracy-versus-cost study report instead? Four things, all
cheap.

First, the zero-median share of its evaluation window. That number determines
whether a median-targeting metric can separate forecast quality from forecast
level at all: at 77.3 % it cannot, at M5's 10.1 % it largely can. It is
computable before any model is fit.

Second, a scale-family placebo. Score k-scaled copies of one forecaster with
the study's own pipeline. Whatever accuracy-cost relationship the pipeline
awards them is its level effect, and only signal beyond that is evidence about
accuracy. Ours awards them +1.000 where holding cost dominates.

Third, per-SKU paired statistics alongside the method-mean correlation, and
the correlation's movement across at least one alternative window. A sampling
interval on a fixed design is not the uncertainty that matters here.

Fourth, the critical fractile. q* = Cu/(Cu+Co) locates the regime boundary
analytically, at \$1,600 here with a median q* of 0.981, and where a
distributional forecast is available it should be scored at the fractile the
policy reads, not with a generic proper score, which we find performs worse than MAE
in a four-model comparison.

None of this says accuracy is worthless. The within-SKU τ of +0.387 among
demand-active SKUs is stable across all nine evaluation designs and consistent
with better forecasts costing less for a given part. What fails is the
instrument that aggregates this into "the accurate model is the cheap model".
And the deployed-policy comparison shows where the money is: the safety-stock
parameterisation separates the deployed policy from the model-driven
alternatives by far more than the forecasters separate from each other.

---

## 7. Limitations

1. **Four-quarter test horizon**, forced by the data vintage. Short for
   intermittent demand.
2. **One operator, one segment.** External validity is unestablished; this is a
   replication on one new domain, not a general law.
3. **The design weights are large and the effective sample is small.** Section
   5.10 reports Horvitz-Thompson estimates alongside the unweighted ones, so
   fleet claims are no longer read off a raw sample mean. But w_h reaches 96,
   Kish's effective n is 3,127, and in the holding-dominated group it is 162.
   Weighted intervals are correspondingly wide, and the weighted
   stockout-dominated correlation is carried by a single stratum.
4. **Cost concentration, and what it costs us in power.** 3.2 % of SKUs carry
   82.3 % of simulated cost, so sample-mean cost is effectively a statement
   about a small high-value subset. Under a paired bootstrap only one of ten
   models is distinguishable from the cheapest, so the cost ranking should be
   read as descriptive. A design that trims or stratifies the tail would have
   more power, at the cost of no longer describing the fleet's actual spend.
5. **Simulator not validated against realised stockouts**, because the data
   contain none. It under-stocks by roughly 2× against the recorded snapshot.
6. **Lost-sales rather than backorder dynamics**, which suits consumable
   spares but not repairable or critical items where demand genuinely queues.
7. **The window is the exposure, not the opening position.** With on-hand
   initialised at S, 81.8 % of SKUs never reorder within the four test
   quarters. The specification grid of Section 5.4 tests this directly:
   opening positions from saturated to empty move the holding-dominated
   correlation by less than 0.04, while the choice of evaluation window moves
   it across most of its range. What remains unresolved is which window is the
   right one, and eleven method means cannot settle that.
8. **Cost is 90.5 % holding**, so the cost ranking is close to a bias ranking
   and does not isolate the value of a forecast's shape.
9. **Assumed cost parameters**, swept but not operator-supplied. The regime
   crossover at \$1,600 is a consequence of the assumed \$100 stockout penalty;
   a different penalty moves the boundary, though not the qualitative result.
10. **Model-agnostic σ_D**, which isolates point-forecast quality but does not
    test forecaster-specific uncertainty.
11. **Untuned baselines.** ZIP and HNB are intercept-only and share a point
    forecast; LightGBM is recursive multi-step; Chronos is zero-shot.
12. **The parameters are inherited, not fitted.** The information-set
    asymmetry that used to sit here is fixed in Section 5.9. What remains is
    that the smoothing constants for SBA, Croston, TSB, SES and the aggregate
    methods come from the operator's deployed configuration rather than being
    optimised on the training window, so these methods are evaluated as
    deployed rather than at their best.
13. **Coarse granularity**: quarterly review, integer-quarter lead times, and a
    cycle-service z applied to criticality tiers specified as availability
    targets.
14. **The distributional comparison rests on four models.** Only ZIP, Hurdle-NB,
    LightGBM and Chronos emit quantiles, so the fleet-level version of Section
    5.5's metric comparison has four points and is not interpreted; the per-SKU
    rank correlations over four models can also take only a few discrete values.
15. **The M5 comparison is a metric test, not a cost test.** We reuse
    per-series accuracy and bias for eight classical methods computed in a
    companion study on the public M5 data, and recover forecast level from
    bias; we do not run an inventory simulation on M5, so
    Section 5.6 establishes the confound's prevalence there but not its cost
    consequences.

---

## 8. Conclusion

On 15,348 SKUs of real maritime spare-parts data, the statistic this
literature uses to ask whether forecast accuracy predicts inventory cost turns
out, under policies that read the forecast through its mean, to measure
forecast level. MAE is minimised at the median of realised
demand, which is zero for 77.3 % of these SKUs, so MAE-rank is level-rank by
identity, with per-SKU rank correlation median +1.000; six scaled copies of
one forecaster, containing identical information about demand, reproduce the
full regime pattern at +1.000 where holding cost dominates. The boundary
between regimes is the newsvendor critical fractile crossing the median, at a
derivable unit price of \$1,600, with the correlation's sign differing across
that price as predicted.
The instrument is fragile besides: across nine defensible evaluation designs
the holding-dominated correlation runs from -0.103 to +1.000, while the
per-SKU τ stays between +0.333 and +0.435. Scoring at q* inherits the confound
with the sign reversed; what works is scoring the fractile the replenishment
decision reads, under a policy that consumes the predictive distribution. On
public M5 data the zero-median condition holds for 10.1 % of series against
77.3 % here, which helps reconcile why competition-data and spare-parts studies
disagree about whether accuracy matters. Three secondary results: ten of
eleven model-driven policies beat the deployed parameterisation, direction
bounded away from zero ([-60.2 %, -41.0 %]); a zero-shot foundation model is
the most accurate and cheapest point estimate but separable from almost none
of its competitors; and censoring of the most recent quarters, though real and
severe, does not reorder forecasters. The design checks of Section 5.10,
equalised information sets and design weights, leave these conclusions
standing. What they do not leave standing is any temptation to read an
eleven-point correlation as a stable fact about the world.

---

## Reproducibility and analysis note

The full pipeline (anonymisation, sampling, all eleven forecasters, the (s, S)
simulator and every robustness check) is released as code, and the anonymised
data are released with it. Results were produced under a claims-verification
script that cross-checks every number in this paper against the analysis
ledgers; it is part of the release.

This paper went through an adversarial review that changed its conclusions, and
the corrections are worth recording because each one is a plausible failure
mode for a cost-aware forecasting study.

1. **A predictive-quantile composition bug.** LightGBM's quantiles were
   composed as the conditional-size quantile multiplied by an indicator that
   the occurrence probability exceeded 0.5. That collapsed the entire
   predictive distribution to a point mass at zero for two thirds of the sample
   and left the rest on an un-remapped conditional quantile. Corrected to the
   hurdle mixture, LightGBM's Q99 coverage moved from 0.941 to 0.995 and its
   fill-matched cost outcome from +204 % to -64.4 %. An earlier draft reported
   the buggy figure as evidence that machine-learning models have tails too
   thin for inventory use.
2. **Monte Carlo sample size masquerading as miscalibration.** Chronos was
   sampled 30 times while ZIP and HNB were sampled 1,000 times, which biases
   the empirical 0.99 quantile downward. At 1,000 seeded draws its Q99 coverage
   moved from 0.939 to 0.961 and it reached target fill at κ = 3.83 rather than
   failing at the search bound.
3. **Non-deterministic training, in two layers.** LightGBM produced MAE of
   8.526, 8.555 and 8.987 on three runs at the same seed. The first layer was
   the model: its stochastic parameters were unseeded and LightGBM is
   non-deterministic when multithreaded, so we pinned the seeds,
   `deterministic`, `force_row_wise` and the thread count. That was not
   sufficient. The second layer was the data: Polars `unique()` does not
   guarantee row order, and LightGBM's bagging samples rows by position, so a
   reshuffled frame still produced a different model. Only after sorting the
   analysis frame to a canonical order did two separate processes agree
   bit-for-bit. Every number here is from the sorted pipeline; the regime
   correlation in particular moved from +0.673 (p = 0.023) to +0.591
   (p = 0.056), which is why we do not lean on it.
4. **An unsupported headline.** An earlier draft claimed censoring flipped the
   cheapest forecaster to the most expensive. No artifact supported it, and the
   controlled experiment in Section 5.9 shows it is false.
5. **An over-strong dominance claim.** An earlier draft stated the deployed
   policy was beaten by every model-driven policy. LightGBM does not beat it.
6. **A vacuous robustness check.** An earlier draft offered MASE as
   corroboration of the MAE-based rank statistic. MASE is a positive per-SKU
   rescaling of MAE, so the statistic is identical by construction.
7. **A statistic with too little power.** The earliest version measured the
   accuracy-cost relationship as a Spearman correlation over eleven
   forecasters, whose bootstrap interval spanned most of [-1, 1].
8. **An unfair comparison.** SBA, Croston and TSB were scored on fifteen
   training quarters against sixteen for the other eight models, because their
   state update lags the observation by one period. Section 5.10 equalises it.
   All three improve and nothing else moves.
9. **A bootstrap interval quoted more precisely than it was estimated.** At
   B = 2,000 the holding-dominated lower bound ranged from +0.409 to +0.545
   across seeds, and an earlier draft quoted +0.527 from one of them. B is now
   20,000, the interval is quoted to two decimals, and the stable statistic,
   the bootstrap share of replicates at or below zero (0.0014), is reported
   with it.
10. **Fleet claims read off an equal-allocation sample.** Design weights reach
    96 : 1 here, and the unweighted mean cost is roughly six times the
    population estimate. Section 5.10 reports both.
11. **Sampling-only uncertainty presented as the uncertainty.** An earlier
    draft gave the holding-dominated correlation a bootstrap interval of
    [+0.49, +1.00] and treated that as the finding's uncertainty. The interval
    resamples SKUs with the evaluation design held fixed. Varying the design
    instead (Section 5.4) moves the same statistic from -0.103 to +1.000, so
    the interval is the smaller of the two uncertainties.

One earlier discrepancy is now resolved. A previous draft reported the
Hurdle-NB fill-matched delta as +2.1 % in one script and -0.5 % in another, on
the same window and configuration, and flagged the inconsistency rather than
choosing. The non-determinism fixes of item 3 were the cause: after the
canonical-ordering re-run the two scripts agree at +1.5 %, the value in
Section 5.7's table.

---

## References

Ansari, A. F., et al. (2024). *Chronos: Learning the Language of Time
Series.* Transactions on Machine Learning Research. arXiv:2403.07815.

Bandara, K., Bergmeir, C., & Smyl, S. (2020). Forecasting across time
series databases using recurrent neural networks on groups of similar
series: a clustering approach. *Expert Systems with Applications*, 140,
112896.

Croston, J. D. (1972). Forecasting and stock control for intermittent
demands. *Operational Research Quarterly*, 23(3), 289-303.

Gneiting, T. (2011). Making and evaluating point forecasts. *Journal of the
American Statistical Association*, 106(494), 746-762.

Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007). Probabilistic
forecasts, calibration and sharpness. *JRSS B*, 69(2), 243-268.

Gneiting, T., & Ranjan, R. (2011). Comparing density forecasts using
threshold- and quantile-weighted scoring rules. *Journal of Business &
Economic Statistics*, 29(3), 411-422.

Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of
forecast accuracy. *International Journal of Forecasting*, 22(4), 679-688.

Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting
decision tree. *NeurIPS* 30.

Kolassa, S. (2016). Evaluating predictive count data distributions in
retail sales forecasting. *International Journal of Forecasting*, 32(3),
788-803.

Kolassa, S. (2020). Why the "best" point forecast depends on the error or
accuracy measure. *International Journal of Forecasting*, 36(1), 208-211.

Kourentzes, N. (2014). On intermittent demand model optimisation and
selection. *International Journal of Production Economics*, 156, 180-190.

Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). The M5
accuracy competition: results, findings, and conclusions. *International
Journal of Forecasting*, 38(4), 1346-1364.

Nikolopoulos, K., Syntetos, A. A., Boylan, J. E., Petropoulos, F., &
Assimakopoulos, V. (2011). An aggregate-disaggregate intermittent demand
approach (ADIDA) to forecasting: an empirical proposition and analysis.
*JORS*, 62(3), 544-554.

Petropoulos, F., Wang, X., & Disney, S. M. (2019). The inventory performance
of forecasting methods: evidence from the M3 competition data.
*International Journal of Forecasting*, 35(1), 251-265.

Prak, D., & Teunter, R. (2019). A general method for addressing
forecasting uncertainty in inventory models. *International Journal of
Forecasting*, 35(1), 224-238.

Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020).
DeepAR: probabilistic forecasting with autoregressive recurrent networks.
*International Journal of Forecasting*, 36(3), 1181-1191.

Silver, E. A., Pyke, D. F., & Thomas, D. J. (2017). *Inventory and
Production Management in Supply Chains* (4th ed.). CRC Press.

Snyder, L. V., & Shen, Z.-J. M. (2019). *Fundamentals of Supply Chain
Theory* (2nd ed.). Wiley.

Syntetos, A. A., & Boylan, J. E. (2005). The accuracy of intermittent
demand estimates. *International Journal of Forecasting*, 21(2), 303-314.

Syntetos, A. A., & Boylan, J. E. (2006). On the stock control performance
of intermittent demand estimators. *International Journal of Production
Economics*, 103(1), 36-47.

Syntetos, A. A., Nikolopoulos, K., & Boylan, J. E. (2010). Judging the judges
through accuracy-implication metrics: The case of inventory forecasting.
*International Journal of Forecasting*, 26(1), 134-143.

Teunter, R. H., Babai, M. Z., & Syntetos, A. A. (2010). ABC
classification: service levels and inventory costs. *Production and
Operations Management*, 19(3), 343-352.

Teunter, R. H., & Duncan, L. (2009). Forecasting intermittent demand: a
comparative study. *Journal of the Operational Research Society*, 60(3),
321-329.

Teunter, R. H., Syntetos, A. A., & Babai, M. Z. (2011). Intermittent
demand: linking forecasting to inventory obsolescence. *European Journal
of Operational Research*, 214(3), 606-615.

Theodorou, E., Spiliotis, E., & Assimakopoulos, V. (2025). Forecast accuracy
and inventory performance: insights on their relationship from the M5
competition data. *European Journal of Operational Research*, 322(2),
414-426. doi:10.1016/j.ejor.2024.12.033.

Vandeput, N. (2020). *Inventory Optimization: Models and Simulations.*
De Gruyter.

Wallström, P., & Segerstedt, A. (2010). Evaluation of forecasting error
measurements and techniques for intermittent demand. *International Journal
of Production Economics*, 128(2), 625-636.
