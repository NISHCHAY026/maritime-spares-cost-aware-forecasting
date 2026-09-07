"""
Does the accuracy-cost relationship depend on the cost regime?

Theodorou, Spiliotis & Assimakopoulos (2025, EJOR 322(2):414-426,
doi:10.1016/j.ejor.2024.12.033) find, on M5 retail data under an order-up-to
policy, that forecast accuracy is more relevant when holding cost is similar
to or larger than lost-sales cost, and that where lost-sales cost dominates
the preferable method may not be the most accurate one, especially for
intermittent products.

This tests that boundary out of domain: real maritime spare parts, an (s, S)
policy rather than order-up-to, and a deployed operator baseline.

Under the default cost configuration, per-unit-per-quarter holding cost is
(HOLDING_RATE/4) * unit_price and the stockout penalty is a flat
STOCKOUT_PER_UNIT_QTR, so the two are equal at a crossover unit price. SKUs
below it are lost-sales dominant; above it, holding dominant.

Writes output/sample/segment/robustness/cost_regime.{txt,json}
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl
from scipy.stats import spearmanr

import config as C
from simulator import CostConfig
from unified_stats import per_sku_kendall_tau, bootstrap_mean_ci

SAMPLE_DIR = C.SAMPLE_DIR
ROB_DIR = SAMPLE_DIR / "robustness"


def main():
    cfg = CostConfig()
    # Read the real CostConfig fields. An earlier version reached for
    # `holding_rate` / `stockout_per_unit_qtr`, neither of which exists, so it
    # silently used getattr fallbacks that happened to match the defaults.
    hold_rate = cfg.holding_rate_annual
    stockout = cfg.stockout_unit_cost
    crossover = stockout / cfg.holding_per_unit_per_qtr

    ps = pl.read_parquet(SAMPLE_DIR / "clean_per_sku.parquet")
    sk = (pl.read_parquet(SAMPLE_DIR / "sample_skus.parquet")
          .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"])
          .select(["STOCK_ITEM_NUMBER", "FORECAST_ID", "UNIT_PRICE_USD"]))
    d = ps.join(sk, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner").sort(["STOCK_ITEM_NUMBER", "FORECAST_ID"])
    models = [c[len("COST_"):] for c in d.columns if c.startswith("COST_")]
    price = d["UNIT_PRICE_USD"].fill_null(1.0).cast(pl.Float64).to_numpy()
    hi = price >= crossover

    total = np.mean([float(d[f"COST_{m}"].sum()) for m in models])
    hi_share = np.mean([float(d.filter(pl.Series(hi))[f"COST_{m}"].sum())
                        for m in models]) / total

    res = {
        "crossover_price_usd": float(crossover),
        "holding_rate_per_year": float(hold_rate),
        "stockout_per_unit_qtr": float(stockout),
        "share_skus_holding_dominant": float(hi.mean()),
        "share_cost_in_holding_dominant": float(hi_share),
        "groups": {},
    }

    L = ["COST-REGIME SPLIT OF THE ACCURACY-COST RELATIONSHIP", "=" * 72,
         f"holding = ({hold_rate:.2f}/4) * unit_price per unit-quarter; "
         f"stockout = ${stockout:,.0f} per unit-quarter",
         f"=> equal at unit price ${crossover:,.0f}", "",
         f"{100*hi.mean():.1f}% of SKUs are holding-dominant, and they carry "
         f"{100*hi_share:.1f}% of all simulated cost.", ""]

    for name, mask in [("lost_sales_dominant", ~hi), ("holding_dominant", hi)]:
        sub = d.filter(pl.Series(mask))
        mae_v = np.array([float(sub[f"MAE_{m}"].mean()) for m in models])
        cost_v = np.array([float(sub[f"COST_{m}"].mean()) for m in models])
        rho, p = spearmanr(mae_v, cost_v)

        mae_m = np.column_stack([sub[f"MAE_{m}"].to_numpy() for m in models])
        cost_m = np.column_stack([sub[f"COST_{m}"].to_numpy() for m in models])
        pos = sub["TEST_DMD"].to_numpy() > 0
        taus = per_sku_kendall_tau(mae_m, cost_m)
        ci = bootstrap_mean_ci(taus[pos]) if pos.sum() > 30 else None

        res["groups"][name] = {
            "n": int(sub.height),
            "fleet_spearman_rho": float(rho),
            "fleet_p": float(p),
            "most_accurate": models[int(np.argmin(mae_v))],
            "cheapest": models[int(np.argmin(cost_v))],
            "per_sku_tau_demand_active": ci,
        }
        L.append(f"{name}  n={sub.height:,}")
        L.append(f"   fleet-level Spearman rho(MAE, cost) = {rho:+.3f}  (p={p:.3f})")
        L.append(f"   most accurate = {models[int(np.argmin(mae_v))]}   "
                 f"cheapest = {models[int(np.argmin(cost_v))]}")
        if ci:
            L.append(f"   per-SKU tau (demand-active) = {ci['mean']:+.3f} "
                     f"[{ci['lo']:+.3f},{ci['hi']:+.3f}] N={ci['n_valid']:,}")
        L.append("")

    rep = "\n".join(L)
    (ROB_DIR / "cost_regime.txt").write_text(rep, encoding="utf-8")
    (ROB_DIR / "cost_regime.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(rep)


if __name__ == "__main__":
    main()
