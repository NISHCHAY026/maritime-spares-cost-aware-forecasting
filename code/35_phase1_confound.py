"""
Phase 1: establish the MAE-level confound at the right unit of analysis.

The fleet-level rho(MAE, level) = +0.982 is computed over 11 models, which is
far too few points to rest a claim on. This does three things instead.

A. ANALYTIC. For a flat forecast f >= 0 on a test window d_1..d_H,
       MAE(f) = (1/H) * sum_t |d_t - f|
   is convex piecewise-linear with
       dMAE/df = (1/H) * ( #{t: f > d_t} - #{t: f < d_t} ).
   If more than half of the d_t are zero, then for every f > 0 the first count
   already exceeds the second, so MAE is STRICTLY INCREASING in f on f > 0.
   On such a SKU, ranking forecasters by MAE is exactly ranking them by
   forecast level, inverted. This is an identity, not a tendency.
   We report the share of SKUs where median(test) = 0.

B. PER-SKU EMPIRICS. For each SKU, across the 11 forecasters, Spearman
   correlation between model forecast level and model MAE, and between level
   and simulated cost. This is the per-SKU version of the fleet statistic, with
   n = 15,348 units instead of 11.

C. STRUCTURAL. Under the model-agnostic sigma the reorder point is
       s = D_bar*LT + z*sqrt(LT*sigma_D^2 + D_bar^2*sigma_LT^2)
   so the ONLY forecaster-specific input to the policy is D_bar. Cost is then a
   deterministic function of level given the SKU, which would make the confound
   structural rather than incidental. We test that directly by checking whether
   per-SKU cost is a monotone function of per-SKU level across models.

Writes output/sample/segment/robustness/phase1_confound.{txt,json}
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import polars as pl
from scipy.stats import spearmanr

import config as C
from baselines import sba as sba_fn, ses as ses_fn
from models_classical import croston, tsb, adida, imapa
from models_distributional import fit_zip, fit_hurdle_nb
from models_ml import LGBForecaster
from simulator import (CostConfig, normal_policy, simulate,
                       lead_days_to_qtrs, service_level_for)

S = C.SAMPLE_DIR
ROB = S / "robustness"
AC = [f"ACTUALS_QTR{i:02d}" for i in range(1, 29)]


def build():
    samp = (pl.scan_parquet(S / "sample_skus.parquet").select(
        ["STOCK_ITEM_NUMBER", "FORECAST_ID", "SEGMENTATION_GRP", "ABC_GRP",
         "STRATEGY_GRP", "VELOCITY_MODE", "CRITICALITY_MODE", "STOCKCLASS_MODE",
         "UNIT_PRICE_USD", "LEAD_TIME_MEAN", "USAGE_PER_YEAR_MEAN",
         "SBA_ALPHA", "SES_ALPHA"])
        .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect())
    pan = (pl.scan_parquet(S / "panel.parquet")
           .select(["STOCK_ITEM_NUMBER", "FORECAST_ID"] + AC)
           .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect())
    return samp.join(pan, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"],
                     how="inner").sort(["STOCK_ITEM_NUMBER", "FORECAST_ID"])


def main():
    cfg = CostConfig()
    df = build()
    act = df.select(AC).to_numpy().astype(float)
    train, test = act[:, :16], act[:, 16:20]
    N, H = df.height, test.shape[1]

    a = df["SBA_ALPHA"].cast(pl.Float64).fill_null(0.1).to_numpy()
    e = df["SES_ALPHA"].cast(pl.Float64).fill_null(0.5).to_numpy()
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
    F["SBA"] = sba_fn(train, a).forecast[:, -1]
    st = ses_fn(train, e)
    F["SES"] = e * train[:, -1] + (1 - e) * st[:, -1]
    F["MA"] = (train[:, -1] + train[:, -2]) / 2.0
    F["CROSTON"] = croston(train, a, sba=False).forecast[:, -1]
    F["TSB"] = tsb(train, a).forecast[:, -1]
    F["ADIDA"] = adida(train, e, k=None)["forecast_per_period"]
    F["IMAPA"] = imapa(train, e, k_max=None)["forecast_per_period"]
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
    cp = S / "forecasts_chronos_clean.parquet"
    if cp.exists():
        cd = df.select(["STOCK_ITEM_NUMBER", "FORECAST_ID"]).join(
            pl.read_parquet(cp), on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="left")
        F["CHRONOS"] = cd["POINT_Q17"].fill_null(0.0).to_numpy()

    models = list(F)
    M = len(models)
    lvl = np.column_stack([np.maximum(F[m], 0.0) for m in models])   # (N, M)
    mae = np.zeros((N, M))
    cost = np.zeros((N, M))
    for j, m in enumerate(models):
        pt = lvl[:, j]
        mae[:, j] = np.abs(np.repeat(pt[:, None], H, 1) - test).mean(1)
        pol = normal_policy(pt, sigma, lq.astype(float), lsq, price, sl, cfg)
        cost[:, j] = simulate(test, pol["s"], pol["S"], lq, price, cfg=cfg)["total_cost"]

    # ---------------- A. analytic condition ----------------
    med0 = np.median(test, axis=1) == 0.0
    allzero = test.sum(axis=1) == 0.0
    pos = ~allzero

    # ---------------- B. per-SKU rank correlations ----------------
    def per_sku_rho(X, Y):
        """Spearman across models, per SKU. NaN where either side is constant."""
        out = np.full(X.shape[0], np.nan)
        for i in range(X.shape[0]):
            x, y = X[i], Y[i]
            if np.all(x == x[0]) or np.all(y == y[0]):
                continue
            out[i] = spearmanr(x, y).correlation
        return out

    rho_lm = per_sku_rho(lvl, mae)     # level vs MAE
    rho_lc = per_sku_rho(lvl, cost)    # level vs cost
    rho_mc = per_sku_rho(mae, cost)    # MAE vs cost

    def summ(v, mask=None):
        w = v if mask is None else v[mask]
        w = w[np.isfinite(w)]
        if w.size == 0:
            return None
        return {"n": int(w.size), "mean": float(w.mean()),
                "median": float(np.median(w)),
                "frac_eq_plus1": float(np.mean(w > 0.999)),
                "frac_eq_minus1": float(np.mean(w < -0.999))}

    res = {
        "N": int(N), "models": models,
        "share_median_zero": float(med0.mean()),
        "share_all_zero": float(allzero.mean()),
        "share_median_zero_among_demand_active": float(med0[pos].mean()),
        "per_sku": {
            "level_vs_mae_all": summ(rho_lm),
            "level_vs_mae_demand_active": summ(rho_lm, pos),
            "level_vs_cost_all": summ(rho_lc),
            "level_vs_cost_demand_active": summ(rho_lc, pos),
            "mae_vs_cost_demand_active": summ(rho_mc, pos),
        },
    }

    L = ["PHASE 1 - THE MAE / FORECAST-LEVEL CONFOUND", "=" * 74, "",
         "A. ANALYTIC CONDITION",
         "   For a flat forecast f > 0, dMAE/df = (#{f > d_t} - #{f < d_t}) / H.",
         "   If median(d) = 0 then MAE is strictly increasing in f for all f > 0,",
         "   so MAE-rank IS level-rank (inverted) on that SKU, by identity.", "",
         f"   SKUs with median(test) = 0 : {100*med0.mean():.1f} %",
         f"   ... of which all-zero      : {100*allzero.mean():.1f} % of the sample",
         f"   among demand-active SKUs   : {100*med0[pos].mean():.1f} % still have median 0",
         "",
         "B. PER-SKU RANK CORRELATIONS ACROSS THE 11 FORECASTERS",
         "   (n = SKUs with variation on both sides, not n = 11 models)"]
    for k, lab in [("level_vs_mae_all", "level vs MAE   (all SKUs)"),
                   ("level_vs_mae_demand_active", "level vs MAE   (demand-active)"),
                   ("level_vs_cost_all", "level vs cost  (all SKUs)"),
                   ("level_vs_cost_demand_active", "level vs cost  (demand-active)"),
                   ("mae_vs_cost_demand_active", "MAE   vs cost  (demand-active)")]:
        d = res["per_sku"][k]
        if d is None:
            L.append(f"   {lab:<32} (no valid SKUs)"); continue
        L.append(f"   {lab:<32} mean {d['mean']:+.3f}  median {d['median']:+.3f}  "
                 f"share=+1 {100*d['frac_eq_plus1']:5.1f} %  n={d['n']:,}")

    L += ["",
          "C. STRUCTURAL READING",
          "   Under the model-agnostic sigma the policy sees the forecast only",
          "   through D_bar, so per-SKU cost is a deterministic function of",
          "   level. The level-vs-cost row above is that function's rank",
          "   signature: where it is +/-1 for essentially every SKU, no",
          "   point-error metric can add information about cost beyond level."]
    rep = "\n".join(L)
    (ROB / "phase1_confound.txt").write_text(rep, encoding="utf-8")
    (ROB / "phase1_confound.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(rep)


if __name__ == "__main__":
    main()
