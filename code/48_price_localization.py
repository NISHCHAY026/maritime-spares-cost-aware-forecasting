"""
Where does the sign flip actually sit, and does it vanish when q* is flat?

Two must-fixes from the referee panel.

Localization: the paper derives a crossover at $1,600 and then DEFINES the
regime split there, so no ledger showed where the empirical sign change sits.
This sweeps the split price over a grid and reports the fleet correlation among
SKUs above each threshold, with SKU-bootstrap intervals, plus disjoint price
bands. The derivation is single-period and myopic; with multi-quarter holding
residence the effective boundary should sit BELOW $1,600, and the sweep shows
whether it does.

Falsification: the confound mechanism says the regime pattern exists because
q* = Cu/(Cu+Co) varies with price while MAE targets 0.5 everywhere. Make the
stockout penalty price-proportional, Cu = (0.25/4) x price per unit, and q*
is 0.5 for every SKU: the mechanism then predicts the price split stops
separating the correlation. Policies are unchanged (they depend on service
levels, not on the penalty), so this is a pure re-costing of the same runs.

Writes output/sample/segment/robustness/price_localization.{txt,json}
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
B = 1000
SEED = 42
THRESHOLDS = (100, 200, 400, 800, 1200, 1600, 2400, 3200, 6400, 12800)

_spec = importlib.util.spec_from_file_location(
    "simex", str(C.PROJECT_DIR / "code" / "42_simulator_exercise.py"))
simex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(simex)


def boot(mae, cost, idx, rng):
    if idx.size < 30:
        return None
    point = float(spearmanr(mae[idx].mean(0), cost[idx].mean(0)).correlation)
    v = np.empty(B)
    for b in range(B):
        s = idx[rng.integers(0, idx.size, idx.size)]
        v[b] = spearmanr(mae[s].mean(0), cost[s].mean(0)).correlation
    lo, hi = np.percentile(v, [2.5, 97.5])
    return {"point": point, "lo": float(lo), "hi": float(hi),
            "n": int(idx.size), "p_le_zero": float((v <= 0).mean())}


def main():
    cfg = CostConfig()
    h_qtr = cfg.holding_per_unit_per_qtr
    df = simex.load()
    actuals = df.select(simex.ACT).to_numpy().astype(np.float64)
    N = df.height
    train, test = actuals[:, :16], actuals[:, 16:20]
    H = test.shape[1]

    price = df["UNIT_PRICE_USD"].fill_null(1.0).cast(pl.Float64).to_numpy()
    price = np.where(price > 0, price, 1.0)
    ld = df["LEAD_TIME_MEAN"].fill_null(60.0).cast(pl.Float64).to_numpy()
    ld = np.where(ld > 0, ld, 60.0)
    lq = lead_days_to_qtrs(ld)
    lsq = (ld * 0.3) / 91.3125
    sl = np.array([service_level_for(c, cfg)
                   for c in df["CRITICALITY_MODE"].fill_null("Normal").to_list()])
    sigma = train.std(axis=1, ddof=0)

    F = simex.forecasts(df, train, H, True, price, ld)
    models = list(F)
    Mn = len(models)
    mae = np.zeros((N, Mn))
    cost = np.zeros((N, Mn))            # published costing, flat $100 penalty
    cost_flat = np.zeros((N, Mn))       # price-proportional penalty, q* = 0.5
    for j, m in enumerate(models):
        pt = np.maximum(F[m], 0.0)
        mae[:, j] = np.abs(np.repeat(pt[:, None], H, 1) - test).mean(1)
        pol = normal_policy(pt, sigma, lq.astype(float), lsq, price, sl, cfg)
        sim = simulate(test, pol["s"], pol["S"], lq, price, cfg=cfg)
        cost[:, j] = sim["total_cost"]
        # same run, re-costed: Cu = one quarter's holding of one unit
        cost_flat[:, j] = (sim["holding_cost"] + sim["ordering_cost"]
                           + sim["stockout_units"] * (h_qtr * price))

    rng = np.random.default_rng(SEED)
    res = {"B": B, "seed": SEED, "n_models": Mn,
           "holding_per_unit_per_qtr_rate": float(h_qtr),
           "thresholds": list(THRESHOLDS), "sweep": {}, "bands": {},
           "falsification": {}}

    # ---- 1. threshold sweep, published costing ----------------------------
    for p in THRESHOLDS:
        res["sweep"][str(p)] = {
            "above": boot(mae, cost, np.flatnonzero(price >= p), rng),
            "below": boot(mae, cost, np.flatnonzero(price < p), rng)}

    # ---- 2. disjoint price bands, published costing -----------------------
    qs = [0.0, 0.5, 0.8, 0.9, 0.95, 0.97, 0.99, 1.0]
    edges = np.quantile(price, qs)
    for k in range(len(edges) - 1):
        lo_e, hi_e = edges[k], edges[k + 1]
        m_ = (price >= lo_e) & (price < hi_e if k < len(edges) - 2 else price <= hi_e)
        res["bands"][f"{lo_e:.0f}-{hi_e:.0f}"] = boot(
            mae, cost, np.flatnonzero(m_), rng)

    # ---- 3. falsification: q* flat at 0.5 ---------------------------------
    tot = cost_flat.mean(1)
    for p in (400, 1600, 3200):
        res["falsification"][str(p)] = {
            "above": boot(mae, cost_flat, np.flatnonzero(price >= p), rng),
            "below": boot(mae, cost_flat, np.flatnonzero(price < p), rng)}
    res["falsification"]["all"] = boot(mae, cost_flat,
                                       np.flatnonzero(np.ones(N, bool)), rng)

    # ---------------- report ----------------
    def fmt(c):
        if c is None:
            return f"{'(n<30)':>34}"
        return (f"{c['point']:>+8.3f} [{c['lo']:+.2f},{c['hi']:+.2f}]"
                f" n={c['n']:>6,} P<=0 {c['p_le_zero']:.3f}")

    L = ["WHERE THE SIGN FLIP SITS, AND WHETHER IT VANISHES WHEN q* IS FLAT",
         "=" * 86, "",
         "1. FLEET rho(MAE, cost) AMONG SKUs ABOVE EACH SPLIT PRICE (published costing)",
         f"   {'split $':>8}   {'above':<44}{'below':<44}"]
    for p in THRESHOLDS:
        s = res["sweep"][str(p)]
        L.append(f"   {p:>8,}   {fmt(s['above']):<44}{fmt(s['below']):<44}")
    L += ["", "2. DISJOINT PRICE BANDS (published costing)"]
    for band, c in res["bands"].items():
        L.append(f"   ${band:<16} {fmt(c)}")
    L += ["", "3. FALSIFICATION: PRICE-PROPORTIONAL PENALTY, q* = 0.5 EVERYWHERE",
          "   The mechanism predicts the price split stops separating the sign.",
          f"   {'split $':>8}   {'above':<44}{'below':<44}"]
    for p in (400, 1600, 3200):
        s = res["falsification"][str(p)]
        L.append(f"   {p:>8,}   {fmt(s['above']):<44}{fmt(s['below']):<44}")
    L.append(f"   {'pooled':>8}   {fmt(res['falsification']['all'])}")

    rep = "\n".join(L)
    (ROB / "price_localization.txt").write_text(rep, encoding="utf-8")
    (ROB / "price_localization.json").write_text(json.dumps(res, indent=2),
                                                 encoding="utf-8")
    print(rep)


if __name__ == "__main__":
    main()
