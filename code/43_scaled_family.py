"""
The scale-family demonstration: the accuracy-cost 'relationship' among
forecasts that differ only by a multiplicative constant.

Why this experiment exists
--------------------------
The paper argues that on zero-median series MAE-rank is level-rank, so the
fleet-level correlation between method-mean accuracy and method-mean cost is
measuring forecast level, with its sign set by the cost regime. The cleanest
possible test of that claim uses forecasters with NO information differences at
all: one base forecaster (SBA, the operator's own method) multiplied by a scale
factor k. The six variants contain identical information about demand. Any
"accuracy-cost relationship" among them is, by construction, a level effect.

If the fleet-level statistic behaves the same way on this family as it does on
the eleven real forecasters (strongly positive where holding cost dominates,
negative where stockout cost dominates), then the regime pattern the literature
reports does not require any difference in forecast quality to appear.

Setup matches the published arm exactly: train quarters 1-16, test 17-20,
model-agnostic sigma from training actuals, normal-approximation (s, S) policy,
saturated start, published cost configuration.

Writes output/sample/segment/robustness/scaled_family.{txt,json}
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl
from scipy.stats import spearmanr

import config as C
from baselines import sba_next
from simulator import (CostConfig, normal_policy, simulate,
                       lead_days_to_qtrs, service_level_for)

S_DIR = C.SAMPLE_DIR
ROB = S_DIR / "robustness"
ACT = [f"ACTUALS_QTR{i:02d}" for i in range(1, 29)]
KS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)


def main():
    cfg = CostConfig()
    crossover = cfg.stockout_unit_cost / cfg.holding_per_unit_per_qtr

    sample = (pl.scan_parquet(S_DIR / "sample_skus.parquet").select(
        ["STOCK_ITEM_NUMBER", "FORECAST_ID", "UNIT_PRICE_USD", "LEAD_TIME_MEAN",
         "CRITICALITY_MODE", "SBA_ALPHA"])
        .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect())
    panel = (pl.scan_parquet(S_DIR / "panel.parquet")
             .select(["STOCK_ITEM_NUMBER", "FORECAST_ID"] + ACT)
             .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect())
    df = (sample.join(panel, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner")
                .sort(["STOCK_ITEM_NUMBER", "FORECAST_ID"]))
    a = df.select(ACT).to_numpy().astype(np.float64)
    train, test = a[:, :16], a[:, 16:20]
    H = test.shape[1]
    N = df.height

    alpha = df["SBA_ALPHA"].cast(pl.Float64).fill_null(0.1).to_numpy()
    base = np.maximum(sba_next(train, alpha), 0.0)
    price = df["UNIT_PRICE_USD"].fill_null(1.0).cast(pl.Float64).to_numpy()
    price = np.where(price > 0, price, 1.0)
    ld = df["LEAD_TIME_MEAN"].fill_null(60.0).cast(pl.Float64).to_numpy()
    ld = np.where(ld > 0, ld, 60.0)
    lq = lead_days_to_qtrs(ld)
    lsq = (ld * 0.3) / 91.3125
    sl = np.array([service_level_for(c, cfg)
                   for c in df["CRITICALITY_MODE"].fill_null("Normal").to_list()])
    sigma = train.std(axis=1, ddof=0)
    hi = price >= crossover

    mae = np.zeros((N, len(KS)))
    cost = np.zeros((N, len(KS)))
    for j, k in enumerate(KS):
        pt = k * base
        mae[:, j] = np.abs(np.repeat(pt[:, None], H, 1) - test).mean(1)
        pol = normal_policy(pt, sigma, lq.astype(float), lsq, price, sl, cfg)
        sim = simulate(test, pol["s"], pol["S"], lq, price, cfg=cfg)
        cost[:, j] = sim["total_cost"]

    def rho(mask):
        r, p = spearmanr(mae[mask].mean(0), cost[mask].mean(0))
        return float(r), float(p)

    r_all, p_all = rho(np.ones(N, bool))
    r_lo, p_lo = rho(~hi)
    r_hi, p_hi = rho(hi)

    # per-SKU: on how many SKUs does MAE rank the six variants exactly by k?
    active = base > 0                       # scaling a zero forecast produces ties
    zero_median = np.median(test, axis=1) == 0
    inc = np.all(np.diff(mae, axis=1) > 0, axis=1)      # MAE strictly increasing in k
    share_inc_zero_median = float(inc[active & zero_median].mean())
    share_inc_all_active = float(inc[active].mean())

    res = {
        "ks": list(KS), "N": int(N), "crossover_price_usd": float(crossover),
        "n_active": int(active.sum()),
        "share_zero_median": float(zero_median.mean()),
        "fleet_rho": {"all": {"rho": r_all, "p": p_all},
                      "stockout_dominated": {"rho": r_lo, "p": p_lo, "n": int((~hi).sum())},
                      "holding_dominated": {"rho": r_hi, "p": p_hi, "n": int(hi.sum())}},
        "mae_strictly_increasing_in_k": {
            "zero_median_active_skus": share_inc_zero_median,
            "all_active_skus": share_inc_all_active},
        "mean_mae_by_k": {str(k): float(mae[:, j].mean()) for j, k in enumerate(KS)},
        "mean_cost_by_k": {str(k): float(cost[:, j].mean()) for j, k in enumerate(KS)},
        "mean_cost_by_k_stockout_dom": {str(k): float(cost[~hi, j].mean())
                                        for j, k in enumerate(KS)},
        "mean_cost_by_k_holding_dom": {str(k): float(cost[hi, j].mean())
                                       for j, k in enumerate(KS)},
    }

    L = ["THE SCALE FAMILY: k x SBA FOR k IN " + str(KS), "=" * 78, "",
         "Six 'forecasters' that differ only by a multiplicative constant. They",
         "contain identical information about demand, so any accuracy-cost",
         "relationship among them is a level effect by construction.", "",
         f"1. FLEET-LEVEL rho(mean MAE, mean cost) ACROSS THE SIX VARIANTS",
         f"   all SKUs             {r_all:+.3f}  (p={p_all:.3f})",
         f"   stockout-dominated   {r_lo:+.3f}  (p={p_lo:.3f})   n={int((~hi).sum()):,}",
         f"   holding-dominated    {r_hi:+.3f}  (p={p_hi:.3f})   n={int(hi.sum()):,}", "",
         "2. MEAN MAE AND MEAN COST BY k",
         f"   {'k':>6}{'MAE':>10}{'cost all':>12}{'stockout-dom':>14}{'holding-dom':>13}"]
    for j, k in enumerate(KS):
        L.append(f"   {k:>6.2f}{mae[:, j].mean():>10.3f}{cost[:, j].mean():>12,.0f}"
                 f"{cost[~hi, j].mean():>14,.0f}{cost[hi, j].mean():>13,.0f}")
    L += ["", "3. PER-SKU: IS MAE STRICTLY INCREASING IN k?",
          f"   among SKUs with a positive base forecast and zero test median: "
          f"{100 * share_inc_zero_median:.1f} %",
          f"   among all SKUs with a positive base forecast: "
          f"{100 * share_inc_all_active:.1f} %",
          "", "4. READING",
          "   If the regime pattern (positive where holding dominates, negative where",
          "   stockouts dominate) appears here, it appears among forecasts with no",
          "   information differences, so it cannot be evidence about forecast quality."]

    rep = "\n".join(L)
    (ROB / "scaled_family.txt").write_text(rep, encoding="utf-8")
    (ROB / "scaled_family.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(rep)


if __name__ == "__main__":
    main()
