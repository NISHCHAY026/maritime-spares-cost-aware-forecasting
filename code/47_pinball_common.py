"""
The metric comparison of Section 5.5, done properly: common SKU set, paired
differences, bootstrap intervals.

The referee panel's charge: the flagship ordering (pinball at the service
level +0.800 > MAE +0.632 > CRPS +0.400) compares medians of per-SKU rank
correlations computed on NON-IDENTICAL SKU subsets (a correlation is undefined
wherever a metric is constant across the four models), carries no uncertainty,
and the corresponding means nearly coincide. This script restricts all three
metrics to the SKUs where every one of them is defined, reports means alongside
medians, and puts a SKU-bootstrap interval on the PAIRED per-SKU differences
(pinball minus MAE, CRPS minus MAE), which is the statistic the claim actually
needs.

The forecasting and policy machinery replicates 37_distributional.py exactly
(same fits, same seeds, same native-quantile cost).

Writes output/sample/segment/robustness/pinball_common.{txt,json}
"""
from __future__ import annotations

import importlib.util
import json
import warnings

import numpy as np
import polars as pl

import config as C
from models_distributional import (fit_zip, sample_zip, fit_hurdle_nb,
                                   sample_hurdle_nb, quantiles_from_samples)
from models_ml import LGBForecaster
from simulator import (CostConfig, simulate, lead_days_to_qtrs,
                       service_level_for)
from policy_native_quantile import native_quantile_policy

S = C.SAMPLE_DIR
ROB = S / "robustness"
AC = [f"ACTUALS_QTR{i:02d}" for i in range(1, 29)]
QUANTS = (50, 80, 90, 95, 99)
B = 10_000
SEED = 42

_spec = importlib.util.spec_from_file_location(
    "dist37", str(C.PROJECT_DIR / "code" / "37_distributional.py"))
dist37 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dist37)


def main():
    cfg = CostConfig()
    samp = (pl.scan_parquet(S / "sample_skus.parquet").select(
        ["STOCK_ITEM_NUMBER", "FORECAST_ID", "SEGMENTATION_GRP", "ABC_GRP",
         "STRATEGY_GRP", "VELOCITY_MODE", "CRITICALITY_MODE", "STOCKCLASS_MODE",
         "UNIT_PRICE_USD", "LEAD_TIME_MEAN", "USAGE_PER_YEAR_MEAN"])
        .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect())
    pan = (pl.scan_parquet(S / "panel.parquet")
           .select(["STOCK_ITEM_NUMBER", "FORECAST_ID"] + AC)
           .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect())
    df = samp.join(pan, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"],
                   how="inner").sort(["STOCK_ITEM_NUMBER", "FORECAST_ID"])

    act = df.select(AC).to_numpy().astype(float)
    train, test = act[:, :16], act[:, 16:20]
    N, H = df.height, test.shape[1]

    price = df["UNIT_PRICE_USD"].fill_null(1.0).cast(pl.Float64).to_numpy()
    price = np.where(price > 0, price, 1.0)
    ld = df["LEAD_TIME_MEAN"].fill_null(60.0).cast(pl.Float64).to_numpy()
    ld = np.where(ld > 0, ld, 60.0)
    lq = lead_days_to_qtrs(ld)
    sl = np.array([service_level_for(c, cfg)
                   for c in df["CRITICALITY_MODE"].fill_null("Normal").to_list()])

    def enc(col):
        s = df[col].fill_null("__NA__").cast(pl.Utf8)
        mp = {c: i for i, c in enumerate(sorted(set(s.to_list())))}
        return np.array([mp[v] for v in s.to_list()], np.float64)

    Q, P = {}, {}
    zp = fit_zip(train)
    zq = quantiles_from_samples(sample_zip(zp["pi"], zp["lam"], 1000, 42),
                                qs=[q / 100 for q in QUANTS])
    Q["ZIP"] = {q / 100: zq[q / 100] for q in QUANTS}; P["ZIP"] = zp["mean"]
    hp = fit_hurdle_nb(train)
    hq = quantiles_from_samples(
        sample_hurdle_nb(hp["p"], hp["mu"], hp["alpha"], 1000, 42),
        qs=[q / 100 for q in QUANTS])
    Q["HURDLE_NB"] = {q / 100: hq[q / 100] for q in QUANTS}; P["HURDLE_NB"] = hp["mean"]
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
        lg.fit(train, static, train_quantiles=[q / 100 for q in QUANTS], seed=42)
        lp = lg.predict_recursive(train, static, h=H)
    Q["LGBM"] = {q / 100: lp["quantiles"][q / 100] for q in QUANTS}
    P["LGBM"] = lp["mean"][:, 0]
    cp = S / "forecasts_chronos_clean.parquet"
    cd = df.select(["STOCK_ITEM_NUMBER", "FORECAST_ID"]).join(
        pl.read_parquet(cp), on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="left")
    Q["CHRONOS"] = {q / 100: cd[f"Q_{q:02d}"].fill_null(0.0).to_numpy() for q in QUANTS}
    P["CHRONOS"] = cd["POINT_Q17"].fill_null(0.0).to_numpy()

    models = list(Q)
    Mn = len(models)
    levels = [q / 100 for q in QUANTS]
    mae = np.zeros((N, Mn)); pin = np.zeros((N, Mn)); crp = np.zeros((N, Mn))
    ncost = np.zeros((N, Mn))
    for j, m in enumerate(models):
        pt = np.maximum(P[m], 0.0)
        mae[:, j] = np.abs(np.repeat(pt[:, None], H, 1) - test).mean(1)
        idx = np.abs(np.array(levels)[None, :] - sl[:, None]).argmin(axis=1)
        qa = np.zeros(N)
        for k, a in enumerate(levels):
            selk = idx == k
            qa[selk] = np.maximum(Q[m][a], 0.0)[selk]
        pin[:, j] = dist37.pinball_at(test, qa, sl)
        crp[:, j] = dist37.crps_approx(
            test, {a: np.maximum(Q[m][a], 0.0) for a in levels}, levels)
        pol = native_quantile_policy({a: np.maximum(Q[m][a], 0.0) for a in levels},
                                     pt, lq.astype(float), price, sl, cfg)
        ncost[:, j] = simulate(test, pol["s"], pol["s"] + pol["eoq"], lq, price,
                               cfg=cfg)["total_cost"]

    rho_mae = dist37.per_sku_rho(mae, ncost)
    rho_pin = dist37.per_sku_rho(pin, ncost)
    rho_crp = dist37.per_sku_rho(crp, ncost)

    pos = test.sum(axis=1) > 0
    common = pos & np.isfinite(rho_mae) & np.isfinite(rho_pin) & np.isfinite(rho_crp)
    n_common = int(common.sum())
    rm, rp_, rc = rho_mae[common], rho_pin[common], rho_crp[common]

    d_pin = rp_ - rm
    d_crp = rc - rm
    rng = np.random.default_rng(SEED)

    def paired(d):
        means = np.empty(B)
        for b in range(B):
            s = rng.integers(0, d.size, d.size)
            means[b] = d[s].mean()
        lo, hi = np.percentile(means, [2.5, 97.5])
        return {"mean": float(d.mean()), "median": float(np.median(d)),
                "lo": float(lo), "hi": float(hi),
                "p_le_zero": float((means <= 0).mean()),
                "excludes_zero": bool(lo > 0 or hi < 0), "B": B}

    res = {
        "N": int(N), "n_demand_active": int(pos.sum()), "n_common": n_common,
        "n_defined": {"mae": int((pos & np.isfinite(rho_mae)).sum()),
                      "pinball": int((pos & np.isfinite(rho_pin)).sum()),
                      "crps": int((pos & np.isfinite(rho_crp)).sum())},
        "common_set": {
            "mae": {"mean": float(rm.mean()), "median": float(np.median(rm))},
            "pinball": {"mean": float(rp_.mean()), "median": float(np.median(rp_))},
            "crps": {"mean": float(rc.mean()), "median": float(np.median(rc))}},
        "paired_pinball_minus_mae": paired(d_pin),
        "paired_crps_minus_mae": paired(d_crp),
        "note": "four quantile-emitting models; per-SKU rank correlations over "
                "four points have discrete support",
    }

    dp, dc = res["paired_pinball_minus_mae"], res["paired_crps_minus_mae"]
    cs = res["common_set"]
    L = ["METRIC COMPARISON ON THE COMMON SKU SET, WITH PAIRED INFERENCE",
         "=" * 76, "",
         f"demand-active SKUs {res['n_demand_active']:,}; correlation defined for "
         f"MAE {res['n_defined']['mae']:,}, pinball {res['n_defined']['pinball']:,}, "
         f"CRPS {res['n_defined']['crps']:,}; common set {n_common:,}.", "",
         "1. ON THE COMMON SET (same SKUs for all three metrics)",
         f"   {'metric':<28}{'median':>9}{'mean':>9}",
         f"   {'MAE':<28}{cs['mae']['median']:>+9.3f}{cs['mae']['mean']:>+9.3f}",
         f"   {'pinball at service level':<28}{cs['pinball']['median']:>+9.3f}"
         f"{cs['pinball']['mean']:>+9.3f}",
         f"   {'CRPS (5-quantile grid)':<28}{cs['crps']['median']:>+9.3f}"
         f"{cs['crps']['mean']:>+9.3f}", "",
         f"2. PAIRED PER-SKU DIFFERENCES (B = {B:,})",
         f"   pinball - MAE : mean {dp['mean']:+.3f}  median {dp['median']:+.3f}  "
         f"CI [{dp['lo']:+.3f}, {dp['hi']:+.3f}]  P(<=0) {dp['p_le_zero']:.4f}  "
         f"{'EXCLUDES 0' if dp['excludes_zero'] else 'includes 0'}",
         f"   CRPS - MAE    : mean {dc['mean']:+.3f}  median {dc['median']:+.3f}  "
         f"CI [{dc['lo']:+.3f}, {dc['hi']:+.3f}]  P(<=0) {dc['p_le_zero']:.4f}  "
         f"{'EXCLUDES 0' if dc['excludes_zero'] else 'includes 0'}", "",
         "3. READING",
         "   The Section 5.5 claim needs the pinball-minus-MAE difference to be",
         "   positive with an interval excluding zero on the same SKUs. Whether it",
         "   is, the line above decides; the four-model discrete support is a",
         "   stated limitation either way."]

    rep = "\n".join(L)
    (ROB / "pinball_common.txt").write_text(rep, encoding="utf-8")
    (ROB / "pinball_common.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(rep)


if __name__ == "__main__":
    main()
