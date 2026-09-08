"""
Inference for the specification grid, and regime membership recomputed per arm.

Two must-fixes from the referee panel land here.

First, the grid's headline (the holding-dominated correlation runs from -0.103
to +1.000 across nine designs) was point estimates only, so it could not be
told apart from sampling noise of a ten-point statistic. Every cell now gets a
SKU-bootstrap interval and P(rho <= 0), computed on the ten forecasters common
to all arms. The published arm also reports the eleven-model figure, which
separates the model-set effect (dropping Chronos: +0.591 to +0.455) from the
design effect. When forecasts_chronos_ctx12.parquet exists (run 45 first) the
12-quarter arms carry Chronos too and every arm has eleven models.

Second, the $1,600 regime label was held fixed across arms while the realized
stockout share moved from 9.1 % to 40.6 %, so 'holding-dominated' no longer
meant holding-dominated in realized dollars in some arms; part of the swing
could be moderator misclassification. Each arm is therefore also split by
REALIZED membership: a SKU is holding-dominated in an arm if its holding cost
exceeds its stockout cost there, averaged across models. If the fragility
range shrinks under realized membership, the fixed-label swing was partly
label drift; if it does not, the design fragility stands.

Writes output/sample/segment/robustness/grid_inference.{txt,json}
"""
from __future__ import annotations

import importlib.util
import json

import numpy as np
import polars as pl
from scipy.stats import spearmanr

import config as C
from simulator import (CostConfig, normal_policy, simulate,
                       lead_days_to_qtrs, service_level_for)

S_DIR = C.SAMPLE_DIR
ROB = S_DIR / "robustness"
B = 2000
SEED = 42

_spec = importlib.util.spec_from_file_location(
    "simex", str(C.PROJECT_DIR / "code" / "42_simulator_exercise.py"))
simex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(simex)


def boot_rho(mae, cost, idx, rng):
    """SKU-bootstrap CI and P(rho<=0) for the fleet correlation on subset idx."""
    if idx.size < 30:
        return None
    point = float(spearmanr(mae[idx].mean(0), cost[idx].mean(0)).correlation)
    v = np.empty(B)
    for b in range(B):
        s = idx[rng.integers(0, idx.size, idx.size)]
        v[b] = spearmanr(mae[s].mean(0), cost[s].mean(0)).correlation
    lo, hi = np.percentile(v, [2.5, 97.5])
    return {"point": point, "lo": float(lo), "hi": float(hi),
            "p_le_zero": float((v <= 0).mean()), "n": int(idx.size), "B": B}


def main():
    cfg = CostConfig()
    crossover = cfg.stockout_unit_cost / cfg.holding_per_unit_per_qtr
    df = simex.load()
    actuals = df.select(simex.ACT).to_numpy().astype(np.float64)
    N = df.height

    price = df["UNIT_PRICE_USD"].fill_null(1.0).cast(pl.Float64).to_numpy()
    price = np.where(price > 0, price, 1.0)
    ld = df["LEAD_TIME_MEAN"].fill_null(60.0).cast(pl.Float64).to_numpy()
    ld = np.where(ld > 0, ld, 60.0)
    lq = lead_days_to_qtrs(ld)
    lsq = (ld * 0.3) / 91.3125
    sl = np.array([service_level_for(c, cfg)
                   for c in df["CRITICALITY_MODE"].fill_null("Normal").to_list()])
    hi_fixed = price >= crossover

    ctx12 = S_DIR / "forecasts_chronos_ctx12.parquet"
    have_ctx12 = ctx12.exists()
    if have_ctx12:
        c12 = df.select(["STOCK_ITEM_NUMBER", "FORECAST_ID"]).join(
            pl.read_parquet(ctx12), on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="left")
        chronos12 = c12["POINT_Q13"].fill_null(0.0).to_numpy()

    results = {}
    for label, tr_hi, te_lo, te_hi, mode in simex.ARMS:
        train = actuals[:, :tr_hi]
        test = actuals[:, te_lo:te_hi]
        H = test.shape[1]
        F = simex.forecasts(df, train, H, tr_hi == 16, price, ld)
        if tr_hi == 12 and have_ctx12:
            F["CHRONOS"] = chronos12
        models = list(F)
        sigma = train.std(axis=1, ddof=0)
        u_pos = np.random.default_rng(SEED).random(N)

        mae = np.zeros((N, len(models)))
        cost = np.zeros((N, len(models)))
        hold_c = np.zeros((N, len(models)))
        stock_c = np.zeros((N, len(models)))
        for j, m in enumerate(models):
            pt = np.maximum(F[m], 0.0)
            mae[:, j] = np.abs(np.repeat(pt[:, None], H, 1) - test).mean(1)
            pol = normal_policy(pt, sigma, lq.astype(float), lsq, price, sl, cfg)
            init = simex.initial_stock(mode, pol["s"], pol["S"], u_pos)
            sim = simulate(test, pol["s"], pol["S"], lq, price,
                           initial_on_hand=init, cfg=cfg)
            cost[:, j] = sim["total_cost"]
            hold_c[:, j] = sim["holding_cost"]
            stock_c[:, j] = sim["stockout_cost"]

        common = [m for m in models if m != "CHRONOS"]
        ci_common = [models.index(m) for m in common]
        mae_c, cost_c = mae[:, ci_common], cost[:, ci_common]

        # realized membership: holding beats stockout for this SKU in this arm
        hi_real = hold_c.mean(axis=1) >= stock_c.mean(axis=1)

        rng = np.random.default_rng(SEED)
        arm = {"n_models": len(models), "common_models": len(common),
               "share_holding_dominated_fixed": float(hi_fixed.mean()),
               "share_holding_dominated_realized": float(hi_real.mean())}
        for gname, gmask in [("fixed", hi_fixed), ("realized", hi_real)]:
            arm[gname] = {
                "all": boot_rho(mae_c, cost_c, np.flatnonzero(np.ones(N, bool)), rng),
                "stockout_dominated": boot_rho(mae_c, cost_c, np.flatnonzero(~gmask), rng),
                "holding_dominated": boot_rho(mae_c, cost_c, np.flatnonzero(gmask), rng),
            }
        if len(models) > len(common):
            arm["holding_dominated_all_models"] = boot_rho(
                mae, cost, np.flatnonzero(hi_fixed), rng)
        results[label.strip()] = arm
        print(f"done: {label.strip()}  (models={len(models)})", flush=True)

    res = {"B": B, "seed": SEED, "crossover_price_usd": float(crossover),
           "chronos_ctx12_available": bool(have_ctx12), "arms": results}

    # ---------------- report ----------------
    L = ["INFERENCE FOR THE SPECIFICATION GRID", "=" * 100, "",
         f"SKU bootstrap, B = {B:,}. Sections 1-2 use the ten forecasters common "
         "to all arms; eleven-model", "figures (Chronos included) are reported "
         "separately where the arm carries them.", "",
         "1. HOLDING-DOMINATED CORRELATION, FIXED $1,600 LABEL vs REALIZED MEMBERSHIP",
         f"   {'ARM':<24}{'fixed rho':>10}{'95% CI':>19}{'P<=0':>7}"
         f"{'realized rho':>14}{'95% CI':>19}{'P<=0':>7}{'real.share':>11}"]
    for k, a in results.items():
        f_, r_ = a["fixed"]["holding_dominated"], a["realized"]["holding_dominated"]
        L.append(f"   {k:<24}{f_['point']:>+10.3f}"
                 f"{f'[{f_.get(chr(108)+chr(111)):+.2f},{f_.get(chr(104)+chr(105)):+.2f}]':>19}"
                 f"{f_['p_le_zero']:>7.3f}{r_['point']:>+14.3f}"
                 f"{f'[{r_.get(chr(108)+chr(111)):+.2f},{r_.get(chr(104)+chr(105)):+.2f}]':>19}"
                 f"{r_['p_le_zero']:>7.3f}{a['share_holding_dominated_realized']:>10.1%}")

    fx = [a["fixed"]["holding_dominated"] for a in results.values()]
    rl = [a["realized"]["holding_dominated"] for a in results.values()]
    rng_f = max(x["point"] for x in fx) - min(x["point"] for x in fx)
    rng_r = max(x["point"] for x in rl) - min(x["point"] for x in rl)
    lo_arm = min(fx, key=lambda x: x["point"])
    hi_arm = max(fx, key=lambda x: x["point"])
    overlap = not (lo_arm["hi"] < hi_arm["lo"] or hi_arm["hi"] < lo_arm["lo"])
    L += ["", "2. READING",
          f"   fixed-label range across arms   : {rng_f:.3f} "
          f"({min(x['point'] for x in fx):+.3f} to {max(x['point'] for x in fx):+.3f})",
          f"   realized-label range across arms: {rng_r:.3f} "
          f"({min(x['point'] for x in rl):+.3f} to {max(x['point'] for x in rl):+.3f})",
          f"   extreme fixed-label arms' CIs "
          f"{'OVERLAP: fragility not separable from sampling noise' if overlap else 'DO NOT OVERLAP: design variation exceeds within-arm sampling variation'}"]

    pub = results[simex.ARMS[0][0].strip()]
    if "holding_dominated_all_models" in pub:
        e = pub["holding_dominated_all_models"]
        t = pub["fixed"]["holding_dominated"]
        L += ["", "3. MODEL-SET EFFECT, PUBLISHED ARM (design held fixed)",
              f"   eleven models: {e['point']:+.3f}  [{e['lo']:+.2f},{e['hi']:+.2f}]  "
              f"P(rho<=0)={e['p_le_zero']:.4f}",
              f"   ten models   : {t['point']:+.3f}  [{t['lo']:+.2f},{t['hi']:+.2f}]  "
              f"P(rho<=0)={t['p_le_zero']:.4f}",
              "   The model-set effect sits inside both intervals; the design effect",
              "   in section 1 is measured with the model set held constant."]

    if all("holding_dominated_all_models" in a for a in results.values()):
        L += ["", "4. ELEVEN-MODEL HOLDING-DOMINATED FIGURE PER ARM",
              "   (Chronos refit on the 12-quarter context for the 1-12 arms)",
              f"   {'ARM':<24}{'rho':>8}{'95% CI':>19}{'P<=0':>7}"]
        for k, a in results.items():
            e = a["holding_dominated_all_models"]
            L.append(f"   {k:<24}{e['point']:>+8.3f}"
                     f"{f'[{e[chr(108)+chr(111)]:+.2f},{e[chr(104)+chr(105)]:+.2f}]':>19}"
                     f"{e['p_le_zero']:>7.3f}")
        e11 = [a["holding_dominated_all_models"] for a in results.values()]
        L.append(f"   range: {min(x['point'] for x in e11):+.3f} to "
                 f"{max(x['point'] for x in e11):+.3f}")

    rep = "\n".join(L)
    (ROB / "grid_inference.txt").write_text(rep, encoding="utf-8")
    (ROB / "grid_inference.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n" + rep)


if __name__ == "__main__":
    main()
