"""
Inference for every cost comparison in the paper.

Until now the paper reported cost as bare sample means and the fleet-level rank
correlations as point estimates over eleven model means. Simulated cost is
extremely right-skewed here (3.2 % of SKUs carry 82.3 % of it), so adjacent
models separated by a few dollars are almost certainly not distinguishable, and
the paper says so in prose without demonstrating it.

This supplies the demonstration with a PAIRED bootstrap over SKUs: each replicate
resamples SKUs with replacement and recomputes every model's mean on the SAME
resampled SKUs, which preserves the pairing that makes model-vs-model
comparisons informative.

Reported:
  1. mean cost per model with a percentile CI
  2. paired cost DIFFERENCES against the cheapest model, with CIs, and whether
     zero is excluded (i.e. whether the ordering is resolvable at all)
  3. the fleet-level Spearman rho(MAE, cost) with a CI and the bootstrap
     share of replicates at or below zero, overall and by cost regime, so the
     regime result is no longer a bare point estimate
  4. the deployed-policy gap on the overlap, with a CI

Writes output/sample/segment/robustness/inference.{txt,json}
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
# B was 2,000. At that size the 2.5th percentile of the rho bootstrap was not
# stable to the third decimal: across five seeds the holding-dominated lower
# bound ranged from +0.409 to +0.545, and the paper had quoted +0.527 off one
# of them. At B = 20,000 the same sweep spans +0.464 to +0.500, so the interval
# is reported to two decimals and the stable statistic — the bootstrap share of
# replicates at or below zero — is reported alongside it.
B = 20000
SEED = 42
CROSSOVER = 1600.0


def ci(v, lo=2.5, hi=97.5):
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def main():
    rng = np.random.default_rng(SEED)
    cfg = CostConfig()

    ps = pl.read_parquet(S / "clean_per_sku.parquet")
    sk = (pl.read_parquet(S / "sample_skus.parquet")
          .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"])
          .select(["STOCK_ITEM_NUMBER", "FORECAST_ID", "UNIT_PRICE_USD"]))
    d = ps.join(sk, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"],
                how="inner").sort(["STOCK_ITEM_NUMBER", "FORECAST_ID"])

    models = [c[len("COST_"):] for c in d.columns if c.startswith("COST_")]
    cost = np.column_stack([d[f"COST_{m}"].to_numpy().astype(float) for m in models])
    mae = np.column_stack([d[f"MAE_{m}"].to_numpy().astype(float) for m in models])
    price = d["UNIT_PRICE_USD"].fill_null(1.0).cast(pl.Float64).to_numpy()
    N, M = cost.shape
    hi = price >= CROSSOVER

    # deployed overlap
    pp = (pl.read_parquet(S / "policy_panel.parquet")
          .select(["STOCK_ITEM_NUMBER", "FORECAST_ID"])
          .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]))
    ov_keys = (d.select(["STOCK_ITEM_NUMBER", "FORECAST_ID"]).with_row_index("i")
               .join(pp, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner"))
    ov = ov_keys["i"].to_numpy()
    DEPLOYED = 1150.8459791916298

    # ---------------- bootstrap ----------------
    boot_cost = np.zeros((B, M))
    boot_rho_all = np.zeros(B)
    boot_rho_lo = np.zeros(B)   # stockout-dominated
    boot_rho_hi = np.zeros(B)   # holding-dominated
    boot_dep_gap = np.zeros(B)
    idx_lo = np.flatnonzero(~hi)
    idx_hi = np.flatnonzero(hi)

    for b in range(B):
        s_all = rng.integers(0, N, N)
        cb = cost[s_all].mean(axis=0)
        mb = mae[s_all].mean(axis=0)
        boot_cost[b] = cb
        boot_rho_all[b] = spearmanr(mb, cb).correlation

        s_lo = idx_lo[rng.integers(0, idx_lo.size, idx_lo.size)]
        boot_rho_lo[b] = spearmanr(mae[s_lo].mean(axis=0), cost[s_lo].mean(axis=0)).correlation
        s_hi = idx_hi[rng.integers(0, idx_hi.size, idx_hi.size)]
        boot_rho_hi[b] = spearmanr(mae[s_hi].mean(axis=0), cost[s_hi].mean(axis=0)).correlation

        s_ov = ov[rng.integers(0, ov.size, ov.size)]
        best_b = cost[s_ov].mean(axis=0).min()
        boot_dep_gap[b] = 100.0 * (best_b - DEPLOYED) / DEPLOYED

    point_cost = cost.mean(axis=0)
    order = np.argsort(point_cost)
    cheapest = order[0]

    # paired differences against the cheapest model
    diffs = {}
    for j in range(M):
        if j == cheapest:
            continue
        dv = boot_cost[:, j] - boot_cost[:, cheapest]
        lo, hi_ = ci(dv)
        diffs[models[j]] = {
            "point": float(point_cost[j] - point_cost[cheapest]),
            "lo": lo, "hi": hi_,
            "resolved": bool(lo > 0 or hi_ < 0),
        }

    res = {
        "B": B, "N": int(N), "models": models,
        "cheapest": models[cheapest],
        "cost": {models[j]: {"mean": float(point_cost[j]),
                             "lo": ci(boot_cost[:, j])[0],
                             "hi": ci(boot_cost[:, j])[1]} for j in range(M)},
        "diff_vs_cheapest": diffs,
        "fleet_rho": {
            "all": {"point": float(spearmanr(mae.mean(0), cost.mean(0)).correlation),
                    "lo": ci(boot_rho_all)[0], "hi": ci(boot_rho_all)[1],
                    "p_le_zero": float((boot_rho_all <= 0).mean())},
            "stockout_dominated": {
                "point": float(spearmanr(mae[~hi].mean(0), cost[~hi].mean(0)).correlation),
                "lo": ci(boot_rho_lo)[0], "hi": ci(boot_rho_lo)[1],
                "p_le_zero": float((boot_rho_lo <= 0).mean())},
            "holding_dominated": {
                "point": float(spearmanr(mae[hi].mean(0), cost[hi].mean(0)).correlation),
                "lo": ci(boot_rho_hi)[0], "hi": ci(boot_rho_hi)[1],
                "p_le_zero": float((boot_rho_hi <= 0).mean())},
        },
        "deployed_gap_pct": {
            "point": float(100.0 * (cost[ov].mean(axis=0).min() - DEPLOYED) / DEPLOYED),
            "lo": ci(boot_dep_gap)[0], "hi": ci(boot_dep_gap)[1],
            "n_overlap": int(ov.size)},
    }

    n_res = sum(1 for v in diffs.values() if v["resolved"])
    L = [f"PAIRED BOOTSTRAP INFERENCE  (B = {B:,}, resampling {N:,} SKUs, seed {SEED})",
         "=" * 78, "",
         "1. MEAN COST PER MODEL, 95 % percentile CI",
         f"   {'MODEL':<11}{'mean':>12}{'95 % CI':>26}"]
    for j in order:
        c_ = res["cost"][models[j]]
        L.append(f"   {models[j]:<11}{c_['mean']:>12,.0f}   [{c_['lo']:>9,.0f}, {c_['hi']:>9,.0f}]")
    L += ["",
          f"2. PAIRED DIFFERENCE vs the cheapest model ({models[cheapest]})",
          "   'resolved' means the 95 % CI on the paired difference excludes zero.",
          f"   {'MODEL':<11}{'diff':>10}{'95 % CI':>24}   resolved"]
    for j in order:
        if j == cheapest:
            continue
        v = diffs[models[j]]
        L.append(f"   {models[j]:<11}{v['point']:>10,.0f}   [{v['lo']:>9,.0f}, {v['hi']:>9,.0f}]   "
                 f"{'yes' if v['resolved'] else 'NO'}")
    L += ["", f"   {n_res} of {M-1} models are distinguishable from the cheapest.", "",
          "3. FLEET-LEVEL rho(MAE, cost) WITH A CI (bootstrapping SKUs, not models)"]
    for k, lab in [("all", "all SKUs"), ("stockout_dominated", "stockout-dominated"),
                   ("holding_dominated", "holding-dominated")]:
        r = res["fleet_rho"][k]
        excl = "excludes 0" if (r["lo"] > 0 or r["hi"] < 0) else "INCLUDES 0"
        L.append(f"   {lab:<22} {r['point']:+.3f}   [{r['lo']:+.2f}, {r['hi']:+.2f}]   "
                 f"{excl:<10}  P(rho<=0) = {r['p_le_zero']:.4f}")
    g = res["deployed_gap_pct"]
    L += ["", f"4. DEPLOYED GAP on the {g['n_overlap']:,}-SKU overlap (best model vs deployed)",
          f"   {g['point']:+.1f} %   [{g['lo']:+.1f} %, {g['hi']:+.1f} %]"]

    rep = "\n".join(L)
    (ROB / "inference.txt").write_text(rep, encoding="utf-8")
    (ROB / "inference.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(rep)


if __name__ == "__main__":
    main()
