"""
Ledger for the bootstrap seed sweep the paper quotes.

Section 5.4 reports that at B = 2,000 the holding-dominated lower bound ranged
across seeds and that B = 20,000 narrows the spread. Those ranges were first
computed in a throwaway script, so the paper quoted numbers with no artifact
behind them, which the referee audit flagged. This regenerates them into a
ledger the claims verifier can check.

Writes output/sample/segment/robustness/bootstrap_seed_sweep.json
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl
from scipy.stats import spearmanr

import config as C
from simulator import CostConfig

S = C.SAMPLE_DIR
ROB = S / "robustness"
SEEDS = (1, 7, 42, 2024, 99991)


def main():
    cfg = CostConfig()
    crossover = cfg.stockout_unit_cost / cfg.holding_per_unit_per_qtr
    ps = pl.read_parquet(S / "clean_per_sku.parquet")
    sk = (pl.read_parquet(S / "sample_skus.parquet")
          .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"])
          .select(["STOCK_ITEM_NUMBER", "FORECAST_ID", "UNIT_PRICE_USD"]))
    d = (ps.join(sk, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner")
           .sort(["STOCK_ITEM_NUMBER", "FORECAST_ID"]))
    models = [c[len("COST_"):] for c in d.columns if c.startswith("COST_")]
    mae = np.column_stack([d[f"MAE_{m}"].to_numpy().astype(float) for m in models])
    cost = np.column_stack([d[f"COST_{m}"].to_numpy().astype(float) for m in models])
    price = d["UNIT_PRICE_USD"].fill_null(1.0).cast(pl.Float64).to_numpy()
    idx = np.flatnonzero(price >= crossover)

    def sweep(B):
        rows = {}
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            v = np.empty(B)
            for b in range(B):
                s = idx[rng.integers(0, idx.size, idx.size)]
                v[b] = spearmanr(mae[s].mean(0), cost[s].mean(0)).correlation
            lo, hi = np.percentile(v, [2.5, 97.5])
            rows[str(sd)] = {"lo": float(lo), "hi": float(hi),
                             "p_le_zero": float((v <= 0).mean())}
        los = [r["lo"] for r in rows.values()]
        return {"per_seed": rows, "lo_min": min(los), "lo_max": max(los)}

    res = {"group": "holding_dominated", "n": int(idx.size),
           "crossover_price_usd": float(crossover), "seeds": list(SEEDS),
           "B2000": sweep(2_000), "B20000": sweep(20_000)}
    (ROB / "bootstrap_seed_sweep.json").write_text(json.dumps(res, indent=2),
                                                   encoding="utf-8")
    for k in ("B2000", "B20000"):
        r = res[k]
        print(f"{k}: lower bound ranges {r['lo_min']:+.3f} to {r['lo_max']:+.3f}")


if __name__ == "__main__":
    main()
