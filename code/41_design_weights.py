"""
Design weights: re-estimate every headline number for the population, not the sample.

The defect
----------
02_stratified_sample.py draws an EQUAL-ALLOCATION stratified sample: it caps
every (SEGMENTATION x ABC) cell at ceil(target_n / n_cells) = 1,579 SKUs and
takes everything from cells smaller than the cap. That is a sensible design
for a model comparison, because it buys precision in the small, expensive
strata that a proportional sample would barely touch. It is NOT a sample whose
raw mean estimates a fleet quantity: Intermittent x C is 69 % of the panel and
10 % of the sample, so its sampling fraction is 1.0 % against 100 % for six
other cells. The unweighted mean therefore over-weights small dense strata by
up to two orders of magnitude, while the paper's prose describes a fleet.

The fix
-------
Attach the design weight w_h = N_h / n_h (inverse inclusion probability) and
re-estimate everything as a Horvitz-Thompson mean,

    Ybar_w = sum_i w_i y_i / sum_i w_i ,

with a design-consistent bootstrap: resample n_h SKUs with replacement WITHIN
each stratum, holding the weights fixed. Reported alongside the unweighted
figure so the reader can see which conclusions are properties of the fleet and
which are properties of the sample design.

Precision is not free. Kish's effective sample size,
    n_eff = (sum w)^2 / sum w^2 ,
is far below n whenever the weights are this dispersed, and the script reports
it rather than leaving the reader to assume 15,348 independent observations.

Writes output/sample/segment/robustness/design_weights.{txt,json}
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl
from scipy.stats import spearmanr

import config as C
from simulator import CostConfig
from unified_stats import per_sku_kendall_tau

S = C.SAMPLE_DIR
ROB = S / "robustness"
STRATA = ["SEGMENTATION_GRP", "ABC_GRP"]
B = 2000
SEED = 42
DEPLOYED_COST = 1150.8459791916298


def wmean(y, w):
    return float(np.sum(w * y) / np.sum(w))


def build_weights():
    """w_h = N_h / n_h on the sampling strata, joined onto the analysis frame."""
    panel = (pl.read_parquet(S / "panel.parquet")
             .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]))
    smp = (pl.read_parquet(S / "sample_skus.parquet")
           .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]))
    N_h = panel.group_by(STRATA).len().rename({"len": "N_h"})
    n_h = smp.group_by(STRATA).len().rename({"len": "n_h"})
    tab = (N_h.join(n_h, on=STRATA, how="inner")
              .with_columns((pl.col("N_h") / pl.col("n_h")).alias("W_H"))
              .sort(STRATA))
    keys = (smp.select(["STOCK_ITEM_NUMBER", "FORECAST_ID"] + STRATA)
               .join(tab, on=STRATA, how="inner"))
    return keys, tab, int(panel.height)


def main():
    cfg = CostConfig()
    crossover = cfg.stockout_unit_cost / cfg.holding_per_unit_per_qtr

    keys, tab, N_pop = build_weights()
    ps = pl.read_parquet(S / "clean_per_sku.parquet")
    sk = (pl.read_parquet(S / "sample_skus.parquet")
          .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"])
          .select(["STOCK_ITEM_NUMBER", "FORECAST_ID", "UNIT_PRICE_USD"]))
    d = (ps.join(sk, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner")
           .join(keys, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner")
           .sort(["STOCK_ITEM_NUMBER", "FORECAST_ID"]))

    models = [c[len("COST_"):] for c in d.columns if c.startswith("COST_")]
    mae = np.column_stack([d[f"MAE_{m}"].to_numpy().astype(float) for m in models])
    cost = np.column_stack([d[f"COST_{m}"].to_numpy().astype(float) for m in models])
    w = d["W_H"].to_numpy().astype(float)
    price = d["UNIT_PRICE_USD"].fill_null(1.0).cast(pl.Float64).to_numpy()
    hi = price >= crossover
    pos = d["TEST_DMD"].to_numpy() > 0
    n = w.size
    n_eff = float(w.sum() ** 2 / np.sum(w ** 2))

    # stratum membership as integer codes, for the within-stratum bootstrap
    codes = (d.select(STRATA).with_columns(
        pl.concat_str(STRATA, separator="|").alias("_k"))["_k"].to_numpy())
    uniq = {k: i for i, k in enumerate(sorted(set(codes.tolist())))}
    strat = np.array([uniq[k] for k in codes])
    strat_idx = [np.flatnonzero(strat == i) for i in range(len(uniq))]
    strat_name = [k.replace("|", " x ") for k in sorted(uniq, key=uniq.get)]

    def strat_resample(rng, mask=None):
        """Design-consistent bootstrap: resample n_h units within each stratum."""
        out = []
        for idx in strat_idx:
            sel = idx if mask is None else idx[mask[idx]]
            if sel.size:
                out.append(sel[rng.integers(0, sel.size, sel.size)])
        return np.concatenate(out) if out else np.array([], dtype=np.int64)

    # ---- per-model means, weighted and not --------------------------------
    per_model = {}
    for j, m in enumerate(models):
        per_model[m] = {
            "mae_unweighted": float(mae[:, j].mean()),
            "mae_weighted": wmean(mae[:, j], w),
            "cost_unweighted": float(cost[:, j].mean()),
            "cost_weighted": wmean(cost[:, j], w),
        }

    # ---- regime split, weighted and not -----------------------------------
    rng = np.random.default_rng(SEED)
    regime = {}
    for name, mask in [("all", np.ones(n, bool)),
                       ("stockout_dominated", ~hi), ("holding_dominated", hi)]:
        idx = np.flatnonzero(mask)
        rho_u = float(spearmanr(mae[idx].mean(0), cost[idx].mean(0)).correlation)
        mw = np.array([wmean(mae[idx, j], w[idx]) for j in range(len(models))])
        cw = np.array([wmean(cost[idx, j], w[idx]) for j in range(len(models))])
        rho_w = float(spearmanr(mw, cw).correlation)
        boot = np.empty(B)
        for b in range(B):
            s = strat_resample(rng, mask)
            bw = w[s]
            boot[b] = spearmanr(
                [wmean(mae[s, j], bw) for j in range(len(models))],
                [wmean(cost[s, j], bw) for j in range(len(models))]).correlation
        lo, hi_ = np.percentile(boot, [2.5, 97.5])

        # A weighted correlation can be one heavy stratum wearing a large
        # weight. Decompose it: rho within each stratum, and rho with each
        # stratum removed. If the estimate survives dropping the heaviest
        # cell it is a fleet fact; if it collapses, it is that cell's fact.
        within, loso = {}, {}
        for si, sidx in enumerate(strat_idx):
            keep = np.intersect1d(sidx, idx)
            if keep.size >= len(models):
                within[strat_name[si]] = {
                    "n": int(keep.size),
                    "population_weight_share": float(w[keep].sum() / w[idx].sum()),
                    "rho": float(spearmanr(mae[keep].mean(0),
                                           cost[keep].mean(0)).correlation)}
            drop = np.setdiff1d(idx, sidx)
            if drop.size >= len(models) and keep.size:
                loso[strat_name[si]] = float(spearmanr(
                    [wmean(mae[drop, j], w[drop]) for j in range(len(models))],
                    [wmean(cost[drop, j], w[drop]) for j in range(len(models))]
                ).correlation)

        regime[name] = {
            "n_sample": int(idx.size),
            "n_population_est": float(w[idx].sum()),
            "n_effective_kish": float(w[idx].sum() ** 2 / np.sum(w[idx] ** 2)),
            "rho_unweighted": rho_u, "rho_weighted": rho_w,
            "lo": float(lo), "hi": float(hi_),
            "excludes_zero": bool(lo > 0 or hi_ < 0),
            "p_le_zero": float((boot <= 0).mean()),
            "rho_within_stratum": within,
            "rho_leave_one_stratum_out": loso,
        }

    # ---- shares that the paper quotes as fleet facts -----------------------
    taus = per_sku_kendall_tau(mae, cost)
    fin = np.isfinite(taus)
    tau_pos = fin & pos

    # The paper's "3.2 % of SKUs carry 82.3 % of simulated cost" is the
    # holding-dominated group restated, not an independent tail statistic:
    # the same price threshold defines both. Weighted, both move together.
    cost_sku = cost.mean(1)
    cost_share_hi_u = float(cost_sku[hi].sum() / cost_sku.sum())
    cost_share_hi_w = float((w[hi] * cost_sku[hi]).sum() / (w * cost_sku).sum())

    shares = {
        "cost_share_in_holding_dominated_unweighted": cost_share_hi_u,
        "cost_share_in_holding_dominated_weighted": cost_share_hi_w,
        "holding_dominated_unweighted": float(hi.mean()),
        "holding_dominated_weighted": wmean(hi.astype(float), w),
        "demand_active_unweighted": float(pos.mean()),
        "demand_active_weighted": wmean(pos.astype(float), w),
        "tau_demand_active_unweighted": float(taus[tau_pos].mean()),
        "tau_demand_active_weighted": wmean(taus[tau_pos], w[tau_pos]),
        "tau_pooled_unweighted": float(taus[fin].mean()),
        "tau_pooled_weighted": wmean(taus[fin], w[fin]),
    }

    # ---- deployed gap on the overlap --------------------------------------
    pp = (pl.read_parquet(S / "policy_panel.parquet")
          .select(["STOCK_ITEM_NUMBER", "FORECAST_ID"])
          .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]))
    ovk = (d.select(["STOCK_ITEM_NUMBER", "FORECAST_ID"]).with_row_index("i")
             .join(pp, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner"))
    ov = ovk["i"].to_numpy()
    best_u = float(cost[ov].mean(0).min())
    cw_ov = np.array([wmean(cost[ov, j], w[ov]) for j in range(len(models))])
    best_w = float(cw_ov.min())
    deployed = {
        "n_overlap": int(ov.size),
        "gap_pct_unweighted": 100.0 * (best_u - DEPLOYED_COST) / DEPLOYED_COST,
        "gap_pct_weighted": 100.0 * (best_w - DEPLOYED_COST) / DEPLOYED_COST,
        "best_model_unweighted": models[int(np.argmin(cost[ov].mean(0)))],
        "best_model_weighted": models[int(np.argmin(cw_ov))],
        "weight_range_on_overlap": [float(w[ov].min()), float(w[ov].max())],
    }

    res = {"N_population": N_pop, "n_sample": int(n), "n_effective_kish": n_eff,
           "strata": STRATA, "B": B, "seed": SEED,
           "crossover_price_usd": float(crossover),
           "weight_table": tab.to_dicts(),
           "per_model": per_model, "regime": regime, "shares": shares,
           "deployed": deployed}

    L = ["DESIGN WEIGHTS: SAMPLE ESTIMATES vs POPULATION ESTIMATES", "=" * 78, "",
         f"The sample is equal-allocation stratified on {' x '.join(STRATA)}.",
         f"Population (panel) N = {N_pop:,}; sample n = {n:,}.",
         f"Design weights w_h = N_h / n_h run from {tab['W_H'].min():.2f} to "
         f"{tab['W_H'].max():.2f}, so the raw mean is not a fleet mean.",
         f"Kish effective sample size = {n_eff:,.0f} ({100 * n_eff / n:.1f} % of n): "
         "weighting buys",
         "unbiasedness for the population and pays for it in precision.", "",
         "1. STRATUM WEIGHTS",
         f"   {'SEGMENT':<14}{'ABC':<9}{'N_h':>9}{'n_h':>8}{'w_h':>8}{'sampled':>10}"]
    for r in tab.iter_rows(named=True):
        L.append(f"   {r['SEGMENTATION_GRP']:<14}{r['ABC_GRP']:<9}{r['N_h']:>9,}"
                 f"{r['n_h']:>8,}{r['W_H']:>8.2f}{r['n_h'] / r['N_h']:>9.2%}")

    L += ["", "2. PER-MODEL MEANS",
          f"   {'MODEL':<11}{'MAE samp':>10}{'MAE pop':>10}{'d %':>8}   "
          f"{'cost samp':>10}{'cost pop':>10}{'d %':>8}"]
    for m in models:
        v = per_model[m]
        dm = 100 * (v["mae_weighted"] - v["mae_unweighted"]) / v["mae_unweighted"]
        dc = 100 * (v["cost_weighted"] - v["cost_unweighted"]) / v["cost_unweighted"]
        L.append(f"   {m:<11}{v['mae_unweighted']:>10.3f}{v['mae_weighted']:>10.3f}"
                 f"{dm:>+8.1f}   {v['cost_unweighted']:>10,.0f}"
                 f"{v['cost_weighted']:>10,.0f}{dc:>+8.1f}")
    rk_u = sorted(models, key=lambda m: per_model[m]["cost_unweighted"])
    rk_w = sorted(models, key=lambda m: per_model[m]["cost_weighted"])
    L += ["", f"   cheapest, sample : {', '.join(rk_u)}",
          f"   cheapest, popn   : {', '.join(rk_w)}"]

    L += ["", "3. COST REGIME",
          f"   {'GROUP':<22}{'rho samp':>10}{'rho pop':>10}"
          f"{'95 % CI (pop)':>22}{'P(rho<=0)':>11}"]
    for k, lab in [("all", "all SKUs"), ("stockout_dominated", "stockout-dominated"),
                   ("holding_dominated", "holding-dominated")]:
        g = regime[k]
        L.append(f"   {lab:<22}{g['rho_unweighted']:>+10.3f}{g['rho_weighted']:>+10.3f}"
                 f"{f'[{g[chr(108) + chr(111)]:+.3f}, {g[chr(104) + chr(105)]:+.3f}]':>22}"
                 f"{g['p_le_zero']:>11.4f}")
        L.append(f"   {'  sample n / popn est':<22}{g['n_sample']:>10,}"
                 f"{g['n_population_est']:>10,.0f}   Kish n_eff "
                 f"{g['n_effective_kish']:,.0f}")

    L += ["", "   Is the weighted figure a fleet fact or one heavy stratum?",
          "   Leave-one-stratum-out on the weighted rho, largest strata first:"]
    for k, lab in [("stockout_dominated", "stockout-dominated"),
                   ("holding_dominated", "holding-dominated")]:
        g = regime[k]
        L.append(f"   {lab} (full {g['rho_weighted']:+.3f}):")
        rank = sorted(g["rho_within_stratum"].items(),
                      key=lambda kv: -kv[1]["population_weight_share"])[:4]
        for sname, v in rank:
            drop = g["rho_leave_one_stratum_out"].get(sname)
            ds = f"{drop:+.3f}" if drop is not None else "  n/a"
            L.append(f"      {sname:<24}{v['population_weight_share']:>7.1%} of weight, "
                     f"within {v['rho']:+.3f}, without {ds}")

    L += ["", "4. SHARES THE PAPER STATES AS FLEET FACTS",
          f"   {'QUANTITY':<34}{'sample':>12}{'population':>13}"]
    for lab, ku, kw in [
            ("holding-dominated SKU share", "holding_dominated_unweighted",
             "holding_dominated_weighted"),
            ("demand-active SKU share", "demand_active_unweighted",
             "demand_active_weighted"),
            ("cost share in holding-dominated", "cost_share_in_holding_dominated_unweighted",
             "cost_share_in_holding_dominated_weighted")]:
        L.append(f"   {lab:<34}{100 * shares[ku]:>11.1f}%{100 * shares[kw]:>12.1f}%")
    L += [f"   {'mean tau (demand-active)':<34}"
          f"{shares['tau_demand_active_unweighted']:>+12.3f}"
          f"{shares['tau_demand_active_weighted']:>+13.3f}",
          f"   {'mean tau (pooled)':<34}{shares['tau_pooled_unweighted']:>+12.3f}"
          f"{shares['tau_pooled_weighted']:>+13.3f}"]

    L += ["", f"5. DEPLOYED GAP on the {deployed['n_overlap']:,}-SKU overlap",
          f"   sample     {deployed['gap_pct_unweighted']:+.1f} %  "
          f"(cheapest {deployed['best_model_unweighted']})",
          f"   population {deployed['gap_pct_weighted']:+.1f} %  "
          f"(cheapest {deployed['best_model_weighted']})",
          f"   weights on the overlap span "
          f"{deployed['weight_range_on_overlap'][0]:.2f} to "
          f"{deployed['weight_range_on_overlap'][1]:.2f}.",
          "   Caveat: deployed-policy coverage is not itself a random subset of",
          "   the panel, so this estimates the population of panel SKUs that HAVE",
          "   a deployed policy, not the fleet."]

    rep = "\n".join(L)
    (ROB / "design_weights.txt").write_text(rep, encoding="utf-8")
    (ROB / "design_weights.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(rep)


if __name__ == "__main__":
    main()
