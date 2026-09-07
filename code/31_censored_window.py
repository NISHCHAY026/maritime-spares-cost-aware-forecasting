"""
The censored-window comparison, actually run.

The paper's first methodological contribution claims that scoring against
right-censored quarters flips the cheapest forecaster (the two-quarter MA)
into the most expensive. No file in the repo supported that: every
"contaminated" figure in 24_clean_window_rerun.py is a hard-coded string
literal, and the only file that does carry censored-window costs
(output/sample/sim_summary.txt) ranks MA 9th of 11, not 1st.

This runs the same models over three windows so the claim can be tested:

  A  clean            train QTR01-16, test QTR17-20   (H=4)
  B  censored         train QTR01-16, test QTR25-28   (H=4)
  C  original naive   train QTR01-20, test QTR21-28   (H=8)

A vs B is the experiment the contribution needs: identical training data and
identical horizon length, with only the test window moved into the censored
region. C is what the first analysis actually did, and differs from A in both
the training window AND the horizon, so it cannot isolate censoring.

Chronos is excluded: its forecasts exist only for the clean window and
re-running it per window costs hours. All three windows therefore use the same
ten models, which also fixes the panel's like-for-like complaint (the paper
compared a 10-model censored tau against an 11-model clean tau).

Writes output/sample/segment/robustness/censored_window.{txt,json}
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import polars as pl

import config as C
from baselines import sba as sba_fn, ses as ses_fn
from models_classical import croston, tsb, adida, imapa
from models_distributional import fit_zip, fit_hurdle_nb
from models_ml import LGBForecaster
from simulator import (CostConfig, normal_policy, simulate,
                       lead_days_to_qtrs, service_level_for)
from unified_stats import per_sku_kendall_tau, bootstrap_mean_ci

SAMPLE_DIR = C.SAMPLE_DIR
ROB_DIR = SAMPLE_DIR / "robustness"
ACTUAL_COLS = [f"ACTUALS_QTR{i:02d}" for i in range(1, 29)]

WINDOWS = [
    ("A_clean",          16, 16, 20),
    ("B_censored",       16, 24, 28),
    ("C_naive_original", 20, 20, 28),
]


def load():
    sample = (pl.scan_parquet(SAMPLE_DIR / "sample_skus.parquet").select(
        ["STOCK_ITEM_NUMBER", "FORECAST_ID", "SEGMENTATION_GRP", "ABC_GRP",
         "STRATEGY_GRP", "VELOCITY_MODE", "CRITICALITY_MODE", "STOCKCLASS_MODE",
         "UNIT_PRICE_USD", "LEAD_TIME_MEAN", "USAGE_PER_YEAR_MEAN",
         "SBA_ALPHA", "SES_ALPHA"]).unique(
        subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect())
    panel = (pl.scan_parquet(SAMPLE_DIR / "panel.parquet").select(
        ["STOCK_ITEM_NUMBER", "FORECAST_ID"] + ACTUAL_COLS).unique(
        subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect())
    return sample.join(panel, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner").sort(["STOCK_ITEM_NUMBER", "FORECAST_ID"])


def run_window(df, actuals, train_hi, test_lo, test_hi, cfg):
    train = actuals[:, :train_hi]
    test = actuals[:, test_lo:test_hi]
    N, H = df.height, test.shape[1]

    sba_a = df["SBA_ALPHA"].cast(pl.Float64).fill_null(0.1).to_numpy()
    ses_a = df["SES_ALPHA"].cast(pl.Float64).fill_null(0.5).to_numpy()
    price = df["UNIT_PRICE_USD"].fill_null(1.0).cast(pl.Float64).to_numpy()
    price = np.where(price > 0, price, 1.0)
    ld = df["LEAD_TIME_MEAN"].fill_null(60.0).cast(pl.Float64).to_numpy()
    ld = np.where(ld > 0, ld, 60.0)
    lq = lead_days_to_qtrs(ld)
    lsq = (ld * 0.3) / 91.3125
    sigma = train.std(axis=1, ddof=0)
    sl = np.array([service_level_for(c, cfg)
                   for c in df["CRITICALITY_MODE"].fill_null("Normal").to_list()])

    def enc(col):
        s = df[col].fill_null("__NA__").cast(pl.Utf8)
        mp = {c: i for i, c in enumerate(sorted(set(s.to_list())))}
        return np.array([mp[v] for v in s.to_list()], np.float64)

    F = {}
    F["SBA"] = sba_fn(train, sba_a).forecast[:, -1]
    st = ses_fn(train, ses_a)
    F["SES"] = ses_a * train[:, -1] + (1 - ses_a) * st[:, -1]
    F["MA"] = (train[:, -1] + train[:, -2]) / 2.0
    F["CROSTON"] = croston(train, sba_a, sba=False).forecast[:, -1]
    F["TSB"] = tsb(train, sba_a).forecast[:, -1]
    F["ADIDA"] = adida(train, ses_a, k=None)["forecast_per_period"]
    F["IMAPA"] = imapa(train, ses_a, k_max=None)["forecast_per_period"]
    F["ZIP"] = fit_zip(train)["mean"]
    F["HURDLE_NB"] = fit_hurdle_nb(train)["mean"]

    static = {k: enc(c) for k, c in [
        ("f_seg", "SEGMENTATION_GRP"), ("f_abc", "ABC_GRP"),
        ("f_strategy", "STRATEGY_GRP"), ("f_velocity", "VELOCITY_MODE"),
        ("f_criticality", "CRITICALITY_MODE"), ("f_stockclass", "STOCKCLASS_MODE")]}
    static["f_price"] = price
    static["f_leadtime"] = ld
    static["f_usage"] = df["USAGE_PER_YEAR_MEAN"].fill_null(0.0).cast(pl.Float64).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lg = LGBForecaster(lags=(1, 2, 3, 4), rolls=(2, 4))
        lg.fit(train, static, train_quantiles=[0.5, 0.8, 0.9, 0.95, 0.99], seed=42)
        F["LGBM"] = lg.predict_recursive(train, static, h=H)["mean"][:, 0]

    models = list(F)
    mae = np.zeros((N, len(models)))
    cost = np.zeros((N, len(models)))
    fill = {}
    for j, m in enumerate(models):
        pt = np.maximum(F[m], 0.0)
        mae[:, j] = np.abs(np.repeat(pt[:, None], H, 1) - test).mean(1)
        pol = normal_policy(pt, sigma, lq.astype(float), lsq, price, sl, cfg)
        sim = simulate(test, pol["s"], pol["S"], lq, price, cfg=cfg)
        cost[:, j] = sim["total_cost"]
        fill[m] = float(sim["fill_rate"].mean())

    pos = test.sum(axis=1) > 0
    taus = per_sku_kendall_tau(mae, cost)
    ci_pos = bootstrap_mean_ci(taus[pos])

    return {
        "models": models,
        "mae": {m: float(mae[:, j].mean()) for j, m in enumerate(models)},
        "cost": {m: float(cost[:, j].mean()) for j, m in enumerate(models)},
        "fill": fill,
        "demand_active_frac": float(pos.mean()),
        "H": int(H),
        "tau_pos": ci_pos,
    }


def main():
    t0 = time.time()
    cfg = CostConfig()
    df = load()
    actuals = df.select(ACTUAL_COLS).to_numpy().astype(np.float64)
    print(f"N = {df.height:,}")

    res = {}
    for label, tr_hi, te_lo, te_hi in WINDOWS:
        print(f"running {label}: train Q1-{tr_hi}, test Q{te_lo+1}-{te_hi} ...", flush=True)
        res[label] = run_window(df, actuals, tr_hi, te_lo, te_hi, cfg)
        res[label]["spec"] = f"train Q1-{tr_hi}, test Q{te_lo+1}-{te_hi}"

    L = ["CENSORED-WINDOW COMPARISON (10 models, like-for-like)", "=" * 78, ""]
    for label, r in res.items():
        L.append(f"{label}  [{r['spec']}]  H={r['H']}  "
                 f"demand-active {100*r['demand_active_frac']:.1f}%  "
                 f"demand-active tau {r['tau_pos']['mean']:+.3f} "
                 f"[{r['tau_pos']['lo']:+.3f},{r['tau_pos']['hi']:+.3f}]")
        order = sorted(r["cost"], key=r["cost"].get)
        L.append(f"  {'rank':<5}{'MODEL':<12}{'meanMAE':>9}{'meanCost':>12}{'fill':>8}")
        for i, m in enumerate(order, 1):
            L.append(f"  {i:<5}{m:<12}{r['mae'][m]:>9.3f}{r['cost'][m]:>12,.0f}{r['fill'][m]:>8.3f}")
        L.append("")

    L.append("MA cost rank by window (the contribution's claim):")
    for label, r in res.items():
        order = sorted(r["cost"], key=r["cost"].get)
        L.append(f"  {label:<18} MA rank {order.index('MA')+1} of {len(order)}"
                 f"   (cheapest = {order[0]}, dearest = {order[-1]})")

    rep = "\n".join(L)
    (ROB_DIR / "censored_window.txt").write_text(rep, encoding="utf-8")
    (ROB_DIR / "censored_window.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n" + rep)
    print(f"\nDone {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
