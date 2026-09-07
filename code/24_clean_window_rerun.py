"""
Audit fix #1: re-run the three findings on a CLEAN, uncensored window.

Motivation
----------
The panel's nominal test window (QTR25-28) shows a 5-6x demand collapse
vs training (frac-nonzero ~1% vs ~6%). The consumption log confirms this
is right-censoring: calendar consumption is stable through 2026Q1
(~320-410k txn/qtr) then cliffs to 71k (2026Q2, partial) and ~300 txn
(2026Q3-Q4, data not yet landed). Evaluating forecasts / simulating cost
against those quarters scores models against demand that has not been
recorded.

Clean re-run
------------
Both train and test are drawn from the stable pre-spike, pre-censoring
regime QTR01-21:
    train = QTR01-16   (16 quarters, stable)
    test  = QTR17-20   (4 quarters, stable; means ~1.0-1.4, ~5% nonzero)
This avoids the QTR21 spike and the QTR22-28 censored tail. The test
window is shorter than the original 8 quarters because there are not 8
uncensored post-training quarters available — an unavoidable consequence
of the data vintage, and itself part of the finding.

Models: 10 (7 classical + ZIP + HNB + LGBM). Chronos deferred (15 min);
add later if the 10-model picture warrants.

Compares clean-window results to the contaminated-window results for all
three findings.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import polars as pl

import config as C
from baselines import sba as sba_fn, ses as ses_fn, ma2 as ma_fn
from models_classical import croston, tsb, adida, imapa
from models_distributional import (fit_zip, sample_zip, fit_hurdle_nb,
                                    sample_hurdle_nb, quantiles_from_samples)
from models_ml import LGBForecaster
from simulator import (CostConfig, normal_policy, simulate,
                       lead_days_to_qtrs, service_level_for)
from unified_stats import per_sku_kendall_tau, bootstrap_mean_ci

SAMPLE_DIR = C.SAMPLE_DIR
ROB_DIR = SAMPLE_DIR / "robustness"
ACTUAL_COLS = [f"ACTUALS_QTR{i:02d}" for i in range(1, 29)]

TRAIN_LO, TRAIN_HI = 0, 16     # python slice [0:16]  => QTR01-16
TEST_LO,  TEST_HI  = 16, 20    # [16:20]              => QTR17-20
QUANTS = (50, 80, 90, 95, 99)


def constant_h(point, h):
    return np.repeat(point[:, None], h, axis=1)


def main():
    t0 = time.time()
    cfg = CostConfig()

    sample = (pl.scan_parquet(SAMPLE_DIR / "sample_skus.parquet")
              .select(["STOCK_ITEM_NUMBER", "FORECAST_ID",
                       "SEGMENTATION_GRP", "ABC_GRP", "STRATEGY_GRP",
                       "VELOCITY_MODE", "CRITICALITY_MODE", "STOCKCLASS_MODE",
                       "UNIT_PRICE_USD", "LEAD_TIME_MEAN", "USAGE_PER_YEAR_MEAN",
                       "IS_SLOW_MOVER", "SBA_ALPHA", "SES_ALPHA"])
              .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect())
    panel = (pl.scan_parquet(SAMPLE_DIR / "panel.parquet")
             .select(["STOCK_ITEM_NUMBER", "FORECAST_ID"] + ACTUAL_COLS)
             .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect())
    df = sample.join(panel, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner")
    N = df.height
    print(f"SEGMENT panel: {N:,}   train=QTR01-16  test=QTR17-20")

    actuals = df.select(ACTUAL_COLS).to_numpy().astype(np.float64)
    train = actuals[:, TRAIN_LO:TRAIN_HI]
    test  = actuals[:, TEST_LO:TEST_HI]
    H = test.shape[1]
    test_dmd = test.sum(axis=1)
    pos = test_dmd > 0
    print(f"  test demand-active SKUs: {pos.sum():,} ({100*pos.mean():.1f}%)  "
          f"(contaminated window was 34.0%)")

    sba_alpha = df["SBA_ALPHA"].cast(pl.Float64).fill_null(0.1).to_numpy()
    ses_alpha = df["SES_ALPHA"].cast(pl.Float64).fill_null(0.5).to_numpy()
    unit_price = df["UNIT_PRICE_USD"].fill_null(1.0).cast(pl.Float64).to_numpy()
    unit_price = np.where(unit_price > 0, unit_price, 1.0)
    lead_days = df["LEAD_TIME_MEAN"].fill_null(60.0).cast(pl.Float64).to_numpy()
    lead_days = np.where(lead_days > 0, lead_days, 60.0)
    lead_qtrs = lead_days_to_qtrs(lead_days)
    lead_std_qtrs = (lead_days * 0.3) / 91.3125
    sigma = train.std(axis=1, ddof=0)
    crit = df["CRITICALITY_MODE"].fill_null("Normal").to_list()
    sl = np.array([service_level_for(c, cfg) for c in crit])

    # ---- fit forecasters on the clean train window -> first-test-qtr point ----
    def enc(col):
        s = df[col].fill_null("__NA__").cast(pl.Utf8)
        cats = sorted(set(s.to_list())); mp = {c: i for i, c in enumerate(cats)}
        return np.array([mp[v] for v in s.to_list()], dtype=np.float64)

    forecasts = {}   # name -> dict(point=(N,), mae=(N,), quantiles=optional)

    forecasts["SBA"]     = {"point": sba_fn(train, sba_alpha).forecast[:, -1]}
    ses_tr = ses_fn(train, ses_alpha)
    forecasts["SES"]     = {"point": ses_alpha*train[:, -1] + (1-ses_alpha)*ses_tr[:, -1]}
    forecasts["MA"]      = {"point": (train[:, -1] + train[:, -2]) / 2.0}
    forecasts["CROSTON"] = {"point": croston(train, sba_alpha, sba=False).forecast[:, -1]}
    forecasts["TSB"]     = {"point": tsb(train, sba_alpha).forecast[:, -1]}
    forecasts["ADIDA"]   = {"point": adida(train, ses_alpha, k=None)["forecast_per_period"]}
    forecasts["IMAPA"]   = {"point": imapa(train, ses_alpha, k_max=None)["forecast_per_period"]}

    zp = fit_zip(train)
    zq = quantiles_from_samples(sample_zip(zp["pi"], zp["lam"], 1000, 42),
                                qs=[q/100 for q in QUANTS])
    forecasts["ZIP"] = {"point": zp["mean"],
                        "quantiles": {q: zq[q/100] for q in QUANTS}}
    hp = fit_hurdle_nb(train)
    hq = quantiles_from_samples(sample_hurdle_nb(hp["p"], hp["mu"], hp["alpha"], 1000, 42),
                                qs=[q/100 for q in QUANTS])
    forecasts["HURDLE_NB"] = {"point": hp["mean"],
                              "quantiles": {q: hq[q/100] for q in QUANTS}}

    static = {k: enc(c) for k, c in [
        ("f_seg", "SEGMENTATION_GRP"), ("f_abc", "ABC_GRP"),
        ("f_strategy", "STRATEGY_GRP"), ("f_velocity", "VELOCITY_MODE"),
        ("f_criticality", "CRITICALITY_MODE"), ("f_stockclass", "STOCKCLASS_MODE")]}
    static["f_price"] = unit_price
    static["f_leadtime"] = lead_days
    static["f_usage"] = df["USAGE_PER_YEAR_MEAN"].fill_null(0.0).cast(pl.Float64).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lg = LGBForecaster()
        lg.fit(train, static, train_quantiles=[q/100 for q in QUANTS], seed=42)
        lpred = lg.predict_recursive(train, static, h=H)
    forecasts["LGBM"] = {"point": lpred["mean"][:, 0],
                         "quantiles": {q: lpred["quantiles"].get(q/100, lpred["mean"][:, 0])
                                       for q in QUANTS}}

    # ---- MAE on clean test + normal-policy cost ----
    models = list(forecasts)
    mae_arr  = np.zeros((N, len(models)))
    cost_arr = np.zeros((N, len(models)))
    fill_by_model = {}
    for j, name in enumerate(models):
        pt = np.maximum(forecasts[name]["point"], 0.0)
        fc_h = constant_h(pt, H)
        mae_arr[:, j] = np.abs(fc_h - test).mean(axis=1)
        pol = normal_policy(pt, sigma, lead_qtrs.astype(np.float64),
                            lead_std_qtrs, unit_price, sl, cfg)
        sim = simulate(test, pol["s"], pol["S"], lead_qtrs, unit_price, cfg=cfg)
        cost_arr[:, j] = sim["total_cost"]
        fill_by_model[name] = sim["fill_rate"].mean()

    # ---- Finding 1: per-SKU tau, split ----
    taus = per_sku_kendall_tau(mae_arr, cost_arr)
    ci_all = bootstrap_mean_ci(taus)
    ci_pos = bootstrap_mean_ci(taus[pos])
    ci_zero = bootstrap_mean_ci(taus[~pos])

    # ---- Finding 2: calibration on clean test ----
    calib = {}
    for name in ("ZIP", "HURDLE_NB", "LGBM"):
        cov = {}
        for q in QUANTS:
            qv = np.maximum(forecasts[name]["quantiles"][q], 0.0)[:, None]
            cov[q] = float((test <= qv).mean())
        calib[name] = cov

    # ---- Finding 3: deployed comparison on clean test ----
    pp = (pl.scan_parquet(SAMPLE_DIR / "policy_panel.parquet")
          .select(["STOCK_ITEM_NUMBER", "FORECAST_ID",
                   "DEPLOYED_NEW_MIN_MEAN", "DEPLOYED_NEW_MAX_MEAN"])
          .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect())
    ov = df.select(["STOCK_ITEM_NUMBER", "FORECAST_ID"]).with_row_index("idx").join(
        pp, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner")
    idx = ov["idx"].to_numpy()
    s_dep = ov["DEPLOYED_NEW_MIN_MEAN"].fill_null(0.0).cast(pl.Float64).to_numpy()
    S_dep = np.maximum(ov["DEPLOYED_NEW_MAX_MEAN"].fill_null(0.0).cast(pl.Float64).to_numpy(),
                       s_dep + 1.0)
    sim_dep = simulate(test[idx], s_dep, S_dep, lead_qtrs[idx], unit_price[idx], cfg=cfg)
    dep_cost = sim_dep["total_cost"].mean(); dep_fill = sim_dep["fill_rate"].mean()
    # best model on the overlap
    ov_model_cost = {}
    for j, name in enumerate(models):
        ov_model_cost[name] = float(cost_arr[idx, j].mean())
    best_model = min(ov_model_cost, key=ov_model_cost.get)
    best_cost = ov_model_cost[best_model]

    # ---------------- report ----------------
    L = []
    L.append("CLEAN-WINDOW RE-RUN (train QTR01-16, test QTR17-20)  vs  contaminated (train Q1-20, test Q21-28)")
    L.append("=" * 90)
    L.append(f"N={N:,}   test demand-active: {100*pos.mean():.1f}% (contaminated 34.0%)")
    L.append("")
    L.append("FINDING 1 — per-SKU Kendall tau (MAE-rank vs cost-rank):")
    L.append(f"  demand-active : clean tau={ci_pos['mean']:+.3f} CI[{ci_pos['lo']:+.3f},{ci_pos['hi']:+.3f}] N={ci_pos['n_valid']:,}"
             f"   |  contaminated +0.49 [+0.47,+0.50]")
    L.append(f"  zero-demand   : clean tau={ci_zero['mean']:+.3f} CI[{ci_zero['lo']:+.3f},{ci_zero['hi']:+.3f}] N={ci_zero['n_valid']:,}"
             f"   |  contaminated +0.86")
    L.append(f"  pooled        : clean tau={ci_all['mean']:+.3f} CI[{ci_all['lo']:+.3f},{ci_all['hi']:+.3f}]"
             f"          |  contaminated +0.73")
    L.append("")
    L.append("FINDING 2 — predictive-quantile calibration (coverage; target shown):")
    L.append(f"  {'MODEL':<10} {'Q50→.50':>9} {'Q90→.90':>9} {'Q95→.95':>9} {'Q99→.99':>9}")
    for name, cov in calib.items():
        L.append(f"  {name:<10} {cov[50]:>9.3f} {cov[90]:>9.3f} {cov[95]:>9.3f} {cov[99]:>9.3f}")
    L.append("  (contaminated Q99 coverage was 0.965-0.988; target 0.99)")
    L.append("")
    L.append(f"FINDING 3 — deployed vs best model on overlap (n={len(idx):,}):")
    L.append(f"  deployed   mean cost ${dep_cost:>9,.0f}   fill {dep_fill:.3f}")
    L.append(f"  best model ({best_model}) ${best_cost:>9,.0f}   "
             f"gap {100*(best_cost-dep_cost)/dep_cost:+.1f}%   (contaminated gap -37%, deployed fill 0.85)")
    L.append("")
    L.append("Per-model clean-window MAE / cost / fill:")
    order = sorted(models, key=lambda m: cost_arr[:, models.index(m)].mean())
    L.append(f"  {'MODEL':<10} {'meanMAE':>9} {'meanCost':>10} {'fill':>7}")
    for name in order:
        j = models.index(name)
        L.append(f"  {name:<10} {mae_arr[:,j].mean():>9.3f} {cost_arr[:,j].mean():>10,.0f} {fill_by_model[name]:>7.3f}")

    report = "\n".join(L)
    (ROB_DIR / "clean_window_rerun.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\nDone in {time.time()-t0:.1f}s   wrote {ROB_DIR/'clean_window_rerun.txt'}")


if __name__ == "__main__":
    main()
