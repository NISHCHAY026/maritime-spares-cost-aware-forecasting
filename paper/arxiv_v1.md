# When Does Forecast Accuracy Predict Inventory Cost? Evidence from a Maritime Spare-Parts Fleet

**Author.** Nishchay Patel

**Abstract.**
Forecasters are usually chosen on point accuracy, but accuracy is not what an
inventory operation pays for. Theodorou, Spiliotis and Assimakopoulos (2025)
showed on M5 retail data that the link between accuracy and inventory cost is
conditional on the cost structure: accuracy matters when holding cost is
comparable to or larger than the cost of a stockout, and much less when
stockouts dominate. We test that boundary out of domain, on 15,348 SKUs of
real maritime spare-parts data from a multi-brand cruise group, under an
(s, S) policy rather than order-up-to, with the operator's own deployed policy
as a baseline. The boundary is visible: the fleet-level rank correlation
between mean absolute error and simulated cost is +0.591 where holding cost
dominates, with a bootstrap interval of [+0.49, +1.00] that excludes zero
(B = 20,000; the bootstrap puts 0.14 % of replicates at or below zero), while
where stockout cost dominates the point estimate is -0.345 with an interval of
[-0.76, +0.44] that does not. Because 96.8 % of these parts sit in the
stockout-dominated regime the pooled correlation is indistinguishable from zero
(+0.018). Carrying the sampling design weights, which reach 96 : 1 because the
sample is equal-allocation, sharpens the contrast rather than dissolving it
(+0.991 against -0.882), but only the positive side survives dropping the one
stratum that is 68 % of the fleet. The sign change is not evidence that
accuracy predicts cost. MAE is minimised at the median of realised demand,
which is zero for 77.3 % of these SKUs, and on such a series MAE is provably
monotone in the forecast, so ranking forecasters by MAE is ranking them by how
much they forecast: the per-SKU rank correlation between level and MAE has
median +1.000. The regime boundary itself follows analytically. With underage
cost Cu and overage cost Co the cost-minimising forecast is the quantile at the
critical fractile Cu/(Cu+Co), which equals the 0.5 fractile MAE targets at
exactly one unit price, here \$1,600, and that is precisely the observed
crossover. We test the obvious remedy and it fails: scoring at each SKU's
critical fractile is confounded with level just as badly in the opposite
direction. The confound is escapable only under a policy that consumes the
whole predictive distribution, and then only by scoring at the fractile the
decision actually reads, since a generic distributional score performs worse
than MAE. Finally we test generality on the public M5 data and it does not
transfer: the analytic condition holds for 10.1 % of M5 series against 77.3 %
here, and the confound is correspondingly mild. The severity is therefore a
measurable property of the evaluation window rather than of intermittency, and
that reconciles why competition-data and spare-parts studies disagree. Two secondary results: a
zero-shot Chronos-T5 model is the most accurate and the cheapest forecaster in
the set, and ten of the eleven model-driven policies beat the deployed
parameterisation on cost, though our simulator is not validated against
realised stockouts so we report that direction without its magnitude. We also
report a negative result. Scoring against right-censored quarters, which is a
real feature of this data, does not reorder the forecasters at all. The
anonymised data and the full pipeline are released.

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

Recent work explains part of that inconsistency. Petropoulos, Wang & Disney
(2019) compared inventory performance across the M3 methods, and Theodorou,
Spiliotis & Assimakopoulos (2025) went further on the M5 data: they find the
accuracy-to-cost link is weak and conditional, and specifically that accuracy
is more relevant when holding cost is similar to or larger than lost-sales
cost. Where lost-sales cost dominates, they note, the preferable method may
not be the most accurate one, and they single out intermittent products.

That is a testable prediction, and it has not been tested outside retail
competition data. We test it on the spare-parts operation of a multi-brand
cruise group: three brands, 40 vessels, around 220k SKUs with quarterly
forecast history. Our setting differs from theirs in three ways that matter.
The demand is far more intermittent than M5 retail. The policy is (s, S)
rather than order-up-to. And the baseline is not another model but the (s, S)
parameters the operator actually runs.

**Contributions.**

* **MAE is confounded with forecast level, by identity rather than by
  tendency.** For a flat forecast, MAE is minimised at the median of realised
  demand. That median is zero for 77.3 % of these SKUs, and on such a SKU MAE
  is strictly increasing in the forecast, so MAE-rank is level-rank inverted.
  Per SKU across the eleven forecasters the level-MAE rank correlation has
  median +1.000, with 78.6 % of SKUs at exactly +1.
* **The regime boundary has an exact derivation.** The cost-minimising forecast
  is the quantile at the newsvendor critical fractile q* = Cu/(Cu+Co). Here
  q* = 0.5 exactly at a unit price of \$1,600, which is the empirically
  observed crossover. MAE implicitly targets 0.5 for every SKU regardless of
  price, so it is aligned with the cost structure at one price and points the
  wrong way on both sides of it. That is why the correlation reverses sign
  rather than merely weakening.
* **The obvious remedy fails, and the working one is narrower than expected.**
  Scoring at each SKU's own q* is confounded with level just as badly in the
  opposite direction (per-SKU pinball-level rank correlation median -1.000).
  Under a policy that consumes the whole predictive distribution the confound
  is escapable, but only by scoring at the fractile the decision reads: pinball
  at the service level predicts cost better than MAE (median +0.800 against
  +0.632) while a generic five-quantile CRPS does worse (+0.400).
* **The severity is measurable in advance, and does not generalise.** On the
  public M5 data the same statistic has median +0.194 with 1.5 % of series at
  +1, because the analytic condition holds for only 10.1 % of M5 series against
  77.3 % here. Conditioning on it, the mechanism does reappear (median +0.614
  where the condition is met against +0.157 where it is not). So the confound
  is a property of how zero-dense the evaluation window is, not of intermittent
  demand as such, and competition-data and spare-parts studies sit on opposite
  sides of it.
* **The level of aggregation also changes the answer.** The within-SKU rank
  correlation is +0.387 while the fleet-level correlation across forecasters,
  which is what a practitioner choosing one model implicitly uses, is +0.018.
  These are different estimands rather than contradictory results.
* **A zero-shot foundation model is simultaneously the most accurate and the
  cheapest forecaster** in a field of eleven that includes the operator's
  deployed methods and the standard Croston family.
* **A negative result on censoring.** The most recent quarters of this feed
  are incomplete. Holding the training window and the horizon fixed and moving
  only the test window into the censored region leaves the cost ranking
  unchanged. Censoring is real in the data and consequential for reported
  demand, but it does not reorder forecasters.

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
al. (2010) and Prak & Teunter (2019) developed the gap analytically; Kolassa
(2016) argued point-error metrics are the wrong target for count data.
Petropoulos, Wang & Disney (2019) measured inventory performance across M3
methods. Theodorou, Spiliotis & Assimakopoulos (2025) supply the result this
paper builds on: using the M5 data, quantile forecasts and an order-up-to
policy, they find the accuracy-cost relationship depends on the balance
between holding and lost-sales cost. Our contribution is to test that
dependence on real operational data, in a much more intermittent regime, under
a different policy class, against a deployed baseline.

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
219,789-SKU forecast panel by equal allocation across demand-pattern and ABC
cells, with a per-cell cap of 1,579. Equal allocation buys precision in the
small, high-value cells that a proportional sample would barely reach, and it
is the right design for a paired model comparison. It is the wrong design for
reading a raw mean as a fleet quantity: sampling fractions run from 1.0 % to
100 %, so the design weight w_h = N_h/n_h spans 1.00 to 96.12, and the single
Intermittent x C cell is 69 % of the panel against 10 % of the sample.

We therefore report both. Model comparisons, which are paired within SKU, use
the sample unweighted. Every quantity stated as a property of the fleet is also
reported as a Horvitz-Thompson estimate carrying w_h, with a bootstrap that
resamples within strata (Section 5.9). Kish's effective sample size under those
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
the 15,348 SKUs, 5,929 (38.6 %) see positive demand in that window. Section 5.6
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
ordering \$50 per order, stockout \$100 per unit short. Section 5.5 sweeps all
three.

Four modelling choices shape these results and are stated explicitly, because
each one is consequential on a four-quarter window.

* **Unmet demand is lost, not backordered.** A stockout is charged once per
  unit short and the shortfall does not carry into the next quarter.
* **Inventory starts saturated** at on-hand = S. On this window that is not a
  neutral choice: 83.0 % of SKUs never place an order in the four test
  quarters, so for most of the sample the simulation measures the cost of
  holding an initial position rather than the cost of replenishing. Starting
  from zero instead raises mean cost by about 5 % but drops mean fill from
  0.965 to 0.828, so the start condition matters far more for service than for
  cost.
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
forecasters. The second is what a practitioner implicitly uses when choosing
one model for the whole fleet.

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
differ, which matters in Section 5.4.

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
the holding-dominated group of Section 5.3, since the same unit-price threshold
defines both. A paired bootstrap over SKUs (B = 20,000) gives Chronos a
mean of \$2,097 with a 95 % interval of [\$543, \$5,050]. On the paired
difference against the cheapest model, **only one of ten models is
distinguishable**: LightGBM, at +\$839 [+219, +1,806]. Chronos against
Hurdle-NB is +\$457 [-231, +1,605]. So Chronos is the cheapest *point estimate*
but is not statistically separable from nine of the other ten, and no claim in
this paper rests on the cost ordering of adjacent models.

### 5.3 The accuracy-cost relationship depends on regime and on aggregation

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

**By cost regime.** Splitting at the \$1,600 crossover:

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

This is the Theodorou et al. prediction, reproduced on independent operational
data under a different policy class. Where holding cost dominates, accuracy
predicts cost and the most accurate model is the cheapest. Where stockout cost
dominates, the correlation is negative and the most accurate model is not the
cheapest. Because 96.8 % of these parts are stockout-dominated, the pooled
fleet correlation is nil, and any study of intermittent spares that reports a
single pooled number will conclude accuracy does not matter, while a study of
high-value slow movers will conclude it does.

**Why the sign changes: MAE is confounded with forecast level.** The regime
pattern above is not evidence that better forecasts cost less.

Start analytically. For a flat forecast f > 0 evaluated on a window
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

### 5.7 Why the boundary sits where it does, and what to use instead

**The crossover is derivable.** In a newsvendor trade-off with underage cost Cu
and overage cost Co, the cost-minimising order quantity is the demand quantile
at the critical fractile q* = Cu/(Cu+Co). Here Cu is the \$100 stockout penalty
and Co is (0.25/4) x price per quarter, so

    q*(price) = 100 / (100 + 0.0625 x price),

which equals 0.5 at exactly **price = \$1,600**. That is the regime crossover of
Section 5.3, derived rather than observed. MAE implicitly targets the 0.5
fractile for every SKU regardless of price, so it is aligned with the cost
structure at exactly one price and misaligned on both sides, which is why the
correlation reverses sign instead of merely weakening. The misalignment is
severe: the median q* here is **0.981**, and 96.8 % of SKUs require q* > 0.5.

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
Only four models emit quantiles, so the fleet-level version of this comparison
rests on four points and we do not interpret it.

### 5.8 Does the confound generalise? A test on public M5 data

If the confound were a property of intermittent demand, it would reappear on
the M5 competition data. It does not. Across 30,490 M5 series and eight
classical forecasters, the per-series rank correlation between forecast level
and MAE has median **+0.194**, with 1.5 % of series at exactly +1, against
median +1.000 and 78.6 % here. By demand class the M5 medians are ordered as
expected (Smooth +0.157, Erratic +0.229, Intermittent +0.253, Lumpy +0.301) but
none is close to the maritime result.

The analytic condition explains the gap. Median demand is zero over the
out-of-sample window for **10.1 %** of M5 series against **77.3 %** here; at a
matched four-period horizon M5 is still only 5.2 %. So the difference is how
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

### 5.4 Predictive-quantile policies

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

### 5.5 Deployed policy comparison

On the 1,579 SKUs where the deployed parameter set can be matched, ten of the
eleven model-driven policies beat the deployed policy on cost. The best, HNB,
costs \$567 against \$1,151, a 51 % reduction. LightGBM is the exception at
\$1,543, or 34 % more expensive than deployed, because its point forecast
over-stocks and this overlap is richer in demand-active SKUs than the sample as
a whole.

The direction is robust and now has an interval: a paired bootstrap over the
overlap SKUs puts the gap at -50.7 % with a 95 % interval of
**[-60.4 %, -41.0 %]**, comfortably excluding zero. The cheapest model in each
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

### 5.6 A negative result: censoring does not reorder forecasters

Because the recent quarters are incomplete, an obvious worry is that a test
window placed there would flatter models that under-forecast. We tested it.
Holding the training window (quarters 1 to 16) and the horizon (four quarters)
fixed, and moving only the test window into the incomplete region (quarters 25
to 28), the demand-active fraction falls from 38.6 % to 11.4 %, confirming the
censoring is real and severe. The cost ranking does not move: the ordering of
all ten models is identical across the two windows, the moving average is
ninth of ten in both, and the within-SKU τ is unchanged (+0.404 clean against
+0.419 censored, overlapping intervals).

A ranking change does appear if one compares the clean window against the
original naive split (train quarters 1 to 20, test 21 to 28), where the moving
average becomes the cheapest model. But that comparison moves the training
window and doubles the horizon as well as shifting the test period, so it
cannot be attributed to censoring. We report this because we initially believed
the opposite, and because the controlled version of the experiment is cheap and
we suspect other cost-aware studies have not run it.

### 5.9 Two design checks: information sets and design weights

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
+0.864 for the four largest by weight) and leave-one-stratum-out never takes it
below +0.973. The stockout-dominated result is not. It falls from -0.882 to
-0.182 when the Intermittent x C cell is removed, and the within-stratum values
have no consistent sign, running from -0.709 to +0.936. That cell is 68 % of
the fleet by weight, so -0.882 is a true statement about this population and a
statement about one stratum at the same time.

So the contrast survives both checks and the magnitude of its negative half
does not. Accuracy predicts cost where holding cost dominates, in the sample and
in the population alike. Where stockouts dominate it does not, and how negative
the relationship looks depends on the stratum. Kish's effective sample size
under these weights is 3,127 overall and
162 in the holding-dominated group, so the weighted intervals are wide by
construction, and we report them alongside the unweighted ones rather than in
place of them.

---

## 6. Discussion

The results cohere around one point: "does forecast accuracy predict inventory
cost" is not a well-posed question until two things are fixed, the cost regime
and the level of aggregation.

Fix the regime and the answer is clean. Where holding cost dominates, accuracy
predicts cost strongly and the most accurate model is the cheapest one. Where
stockout cost dominates, accuracy carries no fleet-level signal and can point
the wrong way. That is Theodorou et al.'s finding, and it survives transfer to
a different industry, a much more intermittent demand process, a different
policy class and a deployed baseline. The transfer is the contribution; the
sign reversal is the sharpening.

Fix the aggregation level and a second apparent contradiction dissolves. A
positive within-SKU τ and a null fleet-level ρ on the same data are not in
tension. They are different estimands, and the fleet-level one is closer to the
decision a planner actually makes.

For practice this yields two cheap diagnostics, both computable before any
model is fitted.

The first is on the evaluation window. Compute the share of SKUs whose test
window has a zero median. That share is what determines whether MAE can
separate forecast quality from forecast level at all. At 77.3 % it cannot; at
the 10.1 % we measure on M5 it largely can. This is a property of the data and
the horizon, not of the models, and it is knowable in advance.

The second is on the cost structure. Compute q* = Cu/(Cu+Co) per SKU. Where it
sits near 0.5, MAE is aligned with the decision. Where it does not, and here
the median is 0.981, the metric is targeting the wrong fractile and its
relationship with cost will take whatever sign the cost structure imposes. If a
distributional forecast is available, score it at q* under a policy that
actually consumes the distribution, and not with a generic proper scoring rule,
which we find performs worse than MAE.

Both diagnostics point the same way as our third result: the safety-stock
parameterisation separates the deployed policy from every reasonable
model-driven alternative by far more than the forecasters separate from each
other.

---

## 7. Limitations

1. **Four-quarter test horizon**, forced by the data vintage. Short for
   intermittent demand.
2. **One operator, one segment.** External validity is unestablished; this is a
   replication on one new domain, not a general law.
3. **The design weights are large and the effective sample is small.** Section
   5.9 reports Horvitz-Thompson estimates alongside the unweighted ones, so
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
7. **Saturated start and a short window interact.** With on-hand initialised
   at S, 83.0 % of SKUs never reorder within the four test quarters, so the
   comparison largely measures holding cost on an initial position. A longer
   window, or a zero start, would exercise the replenishment logic more.
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
    5.7's metric comparison has four points and is not interpreted; the per-SKU
    rank correlations over four models can also take only a few discrete values.
15. **The M5 comparison is a metric test, not a cost test.** We reuse published
    per-series accuracy and bias from the M5 classical benchmark and recover
    forecast level from bias; we do not run an inventory simulation on M5, so
    Section 5.8 establishes the confound's prevalence there but not its cost
    consequences.

---

## 8. Conclusion

On 15,348 SKUs of real maritime spare-parts data, the relationship between
forecast accuracy and simulated inventory cost is conditional in the way
Theodorou, Spiliotis and Assimakopoulos (2025) predicted from M5 retail data.
Where holding cost dominates the fleet-level rank correlation is +0.591; where
stockout cost dominates it is -0.345; pooled across a population that is 96.8 %
stockout-dominated it is +0.018. But the pattern is a confound rather than a
signal. MAE is minimised at the median of realised demand, which is zero for
77.3 % of these SKUs, so MAE-rank is level-rank by identity, with per-SKU
correlation median +1.000. The crossover follows analytically: the critical
fractile q* = Cu/(Cu+Co) equals the 0.5 fractile MAE targets at exactly
\$1,600. Scoring at q* instead does not help, because it inherits the confound
with the sign reversed; only a policy that consumes the whole predictive
distribution opens a usable channel, and then only when scored at the fractile
the decision reads. On public M5 data the confound is mild (median +0.194)
because the analytic condition holds for 10.1 % of series rather than 77.3 %,
so its severity is a measurable property of the evaluation window rather than
of intermittency. The answer
also depends on aggregation: the same data give a moderate positive within-SKU
τ of +0.387 and a null fleet-level correlation. A zero-shot foundation model is
both the most accurate and the cheapest forecaster in the set, and ten of
eleven model-driven policies beat the operator's deployed parameterisation in
direction. Censoring of the most recent quarters, though real and severe in
this feed, does not reorder the forecasters. The regime contrast survives both
design checks. Equalising the information sets across forecasters leaves it at
+0.591, and carrying the sampling design weights raises it to +0.991, positive
within every major stratum. Its negative half travels less well. That one
collapses from -0.882 to -0.182 once the stratum holding 68 % of the fleet is
removed, so what we claim is the contrast and not the magnitude on the
stockout-dominated side.

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
   controlled experiment in Section 5.6 shows it is false.
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
   state update lags the observation by one period. Section 5.9 equalises it.
   All three improve and nothing else moves.
9. **A bootstrap interval quoted more precisely than it was estimated.** At
   B = 2,000 the holding-dominated lower bound ranged from +0.409 to +0.545
   across seeds, and an earlier draft quoted +0.527 from one of them. B is now
   20,000, the interval is quoted to two decimals, and the stable statistic,
   the bootstrap share of replicates at or below zero (0.0014), is reported
   with it.
10. **Fleet claims read off an equal-allocation sample.** Design weights reach
    96 : 1 here, and the unweighted mean cost is roughly six times the
    population estimate. Section 5.9 reports both.

One discrepancy remains unresolved: the Hurdle-NB fill-matched delta is +2.1 %
in the main analysis script and -0.5 % in the robustness script, on the same
window and nominal configuration. We report the main-script value and flag the
inconsistency rather than choosing silently.

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

Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007). Probabilistic
forecasts, calibration and sharpness. *JRSS B*, 69(2), 243-268.

Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of
forecast accuracy. *International Journal of Forecasting*, 22(4), 679-688.

Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting
decision tree. *NeurIPS* 30.

Kolassa, S. (2016). Evaluating predictive count data distributions in
retail sales forecasting. *International Journal of Forecasting*, 32(3),
788-803.

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

Teunter, R. H., Babai, M. Z., & Syntetos, A. A. (2010). ABC
classification: service levels and inventory costs. *Production and
Operations Management*, 19(3), 343-352.

Teunter, R. H., Syntetos, A. A., & Babai, M. Z. (2011). Intermittent
demand: linking forecasting to inventory obsolescence. *European Journal
of Operational Research*, 214(3), 606-615.

Theodorou, E., Spiliotis, E., & Assimakopoulos, V. (2025). Forecast accuracy
and inventory performance: insights on their relationship from the M5
competition data. *European Journal of Operational Research*, 322(2),
414-426. doi:10.1016/j.ejor.2024.12.033.

Vandeput, N. (2020). *Inventory Optimization: Models and Simulations.*
De Gruyter.
