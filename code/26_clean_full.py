"""
Audit fix #1, full clean-window analysis (11 models incl. Chronos).

train = QTR01-16, test = QTR17-20 (uncensored stable regime).
Produces every number the paper needs, on clean data:
  * per-model MAE / cost / fill (normal-approx policy, uniform sigma)
  * per-SKU Kendall tau split by demand activity (Finding 1)
  * predictive-quantile calibration + fill-matched native policy (Finding 2)
  * deployed-policy comparison on the 1,579-SKU overlap (Finding 3)
Dumps per-(SKU,model) MAE+cost arrays for figures.

LGBM uses lags (1,2,3,4) / rolls (2,4) so it gets ~11 training slices on
the 16-quarter window rather than ~4 (the crippled-LGBM critique).
Chronos read from forecasts_chronos_clean.parquet (run 25 first).
"""
from __future__ import annotations
import os, time, warnings
import numpy as np
import polars as pl
import config as C
from baselines import sba as sba_fn, ses as ses_fn, sba_next
from models_classical import croston, tsb, adida, imapa, croston_next, tsb_next
from models_distributional import (fit_zip, sample_zip, fit_hurdle_nb,
                                    sample_hurdle_nb, quantiles_from_samples)
from models_ml import LGBForecaster
from simulator import (CostConfig, normal_policy, simulate,
                       lead_days_to_qtrs, service_level_for)
from policy_native_quantile import native_quantile_policy
from unified_stats import per_sku_kendall_tau, bootstrap_mean_ci

SAMPLE_DIR = C.SAMPLE_DIR
ROB_DIR = SAMPLE_DIR / "robustness"
ACTUAL_COLS = [f"ACTUALS_QTR{i:02d}" for i in range(1, 29)]
TRAIN_HI, TEST_LO, TEST_HI = 16, 16, 20
QUANTS = (50, 80, 90, 95, 99)

# EQUAL_INFO=1 gives SBA / Croston / TSB the terminal state update, so every
# forecaster conditions on all 16 training quarters. Without it those three
# smooth only through quarter 15 (see the note in baselines.py) while SES, MA,
# ADIDA, IMAPA, ZIP, Hurdle-NB, LightGBM and Chronos use the full window.
# Outputs are suffixed so the two arms never overwrite each other.
EQUAL_INFO = os.environ.get("EQUAL_INFO", "").lower() in ("1", "true", "yes")
SUF = "_equalinfo" if EQUAL_INFO else ""


def main():
    t0 = time.time()
    cfg = CostConfig()
    sample = (pl.scan_parquet(SAMPLE_DIR / "sample_skus.parquet").select(
        ["STOCK_ITEM_NUMBER", "FORECAST_ID", "SEGMENTATION_GRP", "ABC_GRP",
         "STRATEGY_GRP", "VELOCITY_MODE", "CRITICALITY_MODE", "STOCKCLASS_MODE",
         "UNIT_PRICE_USD", "LEAD_TIME_MEAN", "USAGE_PER_YEAR_MEAN",
         "SBA_ALPHA", "SES_ALPHA"]).unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect())
    panel = (pl.scan_parquet(SAMPLE_DIR / "panel.parquet").select(
        ["STOCK_ITEM_NUMBER", "FORECAST_ID"] + ACTUAL_COLS).unique(
        subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect())
    df = sample.join(panel, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner").sort(["STOCK_ITEM_NUMBER", "FORECAST_ID"])
    N = df.height
    actuals = df.select(ACTUAL_COLS).to_numpy().astype(np.float64)
    train, test = actuals[:, :TRAIN_HI], actuals[:, TEST_LO:TEST_HI]
    H = test.shape[1]
    pos = test.sum(axis=1) > 0

    sba_a = df["SBA_ALPHA"].cast(pl.Float64).fill_null(0.1).to_numpy()
    ses_a = df["SES_ALPHA"].cast(pl.Float64).fill_null(0.5).to_numpy()
    price = np.where(df["UNIT_PRICE_USD"].fill_null(1.0).cast(pl.Float64).to_numpy() > 0,
                     df["UNIT_PRICE_USD"].fill_null(1.0).cast(pl.Float64).to_numpy(), 1.0)
    ld = df["LEAD_TIME_MEAN"].fill_null(60.0).cast(pl.Float64).to_numpy()
    ld = np.where(ld > 0, ld, 60.0)
    lq = lead_days_to_qtrs(ld); lsq = (ld * 0.3) / 91.3125
    sigma = train.std(axis=1, ddof=0)
    sl = np.array([service_level_for(c, cfg) for c in df["CRITICALITY_MODE"].fill_null("Normal").to_list()])

    def enc(col):
        s = df[col].fill_null("__NA__").cast(pl.Utf8); cats = sorted(set(s.to_list()))
        mp = {c: i for i, c in enumerate(cats)}; return np.array([mp[v] for v in s.to_list()], np.float64)

    F = {}
    # Insertion order fixes the column order of the dumped MAE/cost arrays;
    # keep it stable across both arms so the ledgers diff cleanly.
    F["SBA"] = {"pt": sba_next(train, sba_a) if EQUAL_INFO
                else sba_fn(train, sba_a).forecast[:, -1]}
    st = ses_fn(train, ses_a); F["SES"] = {"pt": ses_a*train[:, -1] + (1-ses_a)*st[:, -1]}
    F["MA"] = {"pt": (train[:, -1] + train[:, -2]) / 2.0}
    F["CROSTON"] = {"pt": croston_next(train, sba_a, sba=False) if EQUAL_INFO
                    else croston(train, sba_a, sba=False).forecast[:, -1]}
    F["TSB"] = {"pt": tsb_next(train, sba_a) if EQUAL_INFO
                else tsb(train, sba_a).forecast[:, -1]}
    F["ADIDA"] = {"pt": adida(train, ses_a, k=None)["forecast_per_period"]}
    F["IMAPA"] = {"pt": imapa(train, ses_a, k_max=None)["forecast_per_period"]}
    zp = fit_zip(train); zq = quantiles_from_samples(sample_zip(zp["pi"], zp["lam"], 1000, 42), qs=[q/100 for q in QUANTS])
    F["ZIP"] = {"pt": zp["mean"], "q": {q: zq[q/100] for q in QUANTS}}
    hp = fit_hurdle_nb(train); hq = quantiles_from_samples(sample_hurdle_nb(hp["p"], hp["mu"], hp["alpha"], 1000, 42), qs=[q/100 for q in QUANTS])
    F["HURDLE_NB"] = {"pt": hp["mean"], "q": {q: hq[q/100] for q in QUANTS}}
    static = {k: enc(c) for k, c in [("f_seg", "SEGMENTATION_GRP"), ("f_abc", "ABC_GRP"),
              ("f_strategy", "STRATEGY_GRP"), ("f_velocity", "VELOCITY_MODE"),
              ("f_criticality", "CRITICALITY_MODE"), ("f_stockclass", "STOCKCLASS_MODE")]}
    static["f_price"] = price; static["f_leadtime"] = ld
    static["f_usage"] = df["USAGE_PER_YEAR_MEAN"].fill_null(0.0).cast(pl.Float64).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lg = LGBForecaster(lags=(1, 2, 3, 4), rolls=(2, 4))   # fair on 16q train
        lg.fit(train, static, train_quantiles=[q/100 for q in QUANTS], seed=42)
        lp = lg.predict_recursive(train, static, h=H)
    F["LGBM"] = {"pt": lp["mean"][:, 0], "q": {q: lp["quantiles"].get(q/100, lp["mean"][:, 0]) for q in QUANTS}}

    # Chronos clean (from file)
    cpath = SAMPLE_DIR / "forecasts_chronos_clean.parquet"
    has_chronos = cpath.exists()
    if has_chronos:
        cdf = df.select(["STOCK_ITEM_NUMBER", "FORECAST_ID"]).join(
            pl.read_parquet(cpath), on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="left")
        F["CHRONOS"] = {"pt": cdf["POINT_Q17"].fill_null(0.0).to_numpy(),
                        "q": {q: cdf[f"Q_{q:02d}"].fill_null(0.0).to_numpy() for q in QUANTS}}

    models = list(F)
    mae = np.zeros((N, len(models))); cost = np.zeros((N, len(models))); fill = {}
    for j, m in enumerate(models):
        pt = np.maximum(F[m]["pt"], 0.0)
        mae[:, j] = np.abs(np.repeat(pt[:, None], H, 1) - test).mean(1)
        pol = normal_policy(pt, sigma, lq.astype(float), lsq, price, sl, cfg)
        sim = simulate(test, pol["s"], pol["S"], lq, price, cfg=cfg)
        cost[:, j] = sim["total_cost"]; fill[m] = sim["fill_rate"].mean()

    taus = per_sku_kendall_tau(mae, cost)
    ci_pos, ci_zero, ci_all = bootstrap_mean_ci(taus[pos]), bootstrap_mean_ci(taus[~pos]), bootstrap_mean_ci(taus)

    # calibration + fill-matched native
    def run(s, S, mask=slice(None)):
        return simulate(test[mask], s, S, lq[mask], price[mask], cfg=cfg)
    qmods = [m for m in models if "q" in F[m]]
    calib, matched = {}, {}
    for m in qmods:
        calib[m] = {q: float((test <= np.maximum(F[m]["q"][q], 0)[:, None]).mean()) for q in QUANTS}
        poln = normal_policy(np.maximum(F[m]["pt"], 0), sigma, lq.astype(float), lsq, price, sl, cfg)
        tgt = run(poln["s"], poln["S"])["fill_rate"].mean()
        qarr = {0.5: F[m]["q"][50], 0.8: F[m]["q"][80], 0.9: F[m]["q"][90], 0.95: F[m]["q"][95], 0.99: F[m]["q"][99]}
        polv = native_quantile_policy(qarr, np.maximum(F[m]["pt"], 0), lq.astype(float), price, sl, cfg)
        sb, eoq = polv["s"], polv["eoq"]
        lo, hi = 0.0, 8.0
        capped = run(hi*sb, hi*sb+eoq)["fill_rate"].mean() < tgt
        if not capped:
            for _ in range(40):
                mid = (lo+hi)/2
                if run(mid*sb, mid*sb+eoq)["fill_rate"].mean() < tgt: lo = mid
                else: hi = mid
            k = (lo+hi)/2
        else: k = hi
        cn = cost[:, models.index(m)].mean()
        cm = run(k*sb, k*sb+eoq)["total_cost"].mean()
        matched[m] = (k, capped, cn, cm, 100*(cm-cn)/cn)

    # deployed
    pp = (pl.scan_parquet(SAMPLE_DIR / "policy_panel.parquet").select(
        ["STOCK_ITEM_NUMBER", "FORECAST_ID", "DEPLOYED_NEW_MIN_MEAN", "DEPLOYED_NEW_MAX_MEAN"]).unique(
        subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect())
    ov = df.select(["STOCK_ITEM_NUMBER", "FORECAST_ID"]).with_row_index("idx").join(
        pp, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner")
    idx = ov["idx"].to_numpy()
    sdep = ov["DEPLOYED_NEW_MIN_MEAN"].fill_null(0.0).cast(pl.Float64).to_numpy()
    Sdep = np.maximum(ov["DEPLOYED_NEW_MAX_MEAN"].fill_null(0.0).cast(pl.Float64).to_numpy(), sdep+1.0)
    sd = run(sdep, Sdep, idx)
    dep_c, dep_f = sd["total_cost"].mean(), sd["fill_rate"].mean()
    ovc = {m: float(cost[idx, models.index(m)].mean()) for m in models}
    best = min(ovc, key=ovc.get)

    # dump per-(SKU,model) arrays
    dump = df.select(["STOCK_ITEM_NUMBER", "FORECAST_ID", "SEGMENTATION_GRP", "ABC_GRP"]).with_columns(
        pl.Series("TEST_DMD", test.sum(1)))
    for j, m in enumerate(models):
        dump = dump.with_columns([pl.Series(f"MAE_{m}", mae[:, j]), pl.Series(f"COST_{m}", cost[:, j])])
    dump.write_parquet(SAMPLE_DIR / f"clean_per_sku{SUF}.parquet", compression="zstd")

    L = ["CLEAN-WINDOW FULL (train QTR01-16, test QTR17-20, 11 models)"
         + ("  [EQUAL INFORMATION SET]" if EQUAL_INFO else ""), "=" * 70,
         f"N={N:,}  demand-active={100*pos.mean():.1f}%  chronos={'yes' if has_chronos else 'NO'}", "",
         "FINDING 1 — per-SKU Kendall tau:",
         f"  demand-active {ci_pos['mean']:+.3f} [{ci_pos['lo']:+.3f},{ci_pos['hi']:+.3f}] N={ci_pos['n_valid']:,}",
         f"  zero-demand   {ci_zero['mean']:+.3f} [{ci_zero['lo']:+.3f},{ci_zero['hi']:+.3f}] N={ci_zero['n_valid']:,}",
         f"  pooled        {ci_all['mean']:+.3f} [{ci_all['lo']:+.3f},{ci_all['hi']:+.3f}]", "",
         "FINDING 2 — calibration (coverage) + fill-matched native cost delta:",
         f"  {'MODEL':<10} {'Q90':>6} {'Q95':>6} {'Q99':>6}  | kappa  capped  matched-cost-delta"]
    for m in qmods:
        c = calib[m]; k, cap, cn, cm, d = matched[m]
        L.append(f"  {m:<10} {c[90]:>6.3f} {c[95]:>6.3f} {c[99]:>6.3f}  | {k:5.2f}  {str(cap):<5}  {d:+.1f}%")
    L += ["", f"FINDING 3 — deployed vs best ({best}) on overlap n={len(idx):,}:",
          f"  deployed ${dep_c:,.0f} fill {dep_f:.3f}   best ${ovc[best]:,.0f}  gap {100*(ovc[best]-dep_c)/dep_c:+.1f}%", "",
          "Per-model clean MAE / cost / fill (sorted by cost):"]
    for m in sorted(models, key=lambda x: cost[:, models.index(x)].mean()):
        j = models.index(m)
        L.append(f"  {m:<10} MAE {mae[:,j].mean():7.3f}  cost {cost[:,j].mean():9,.0f}  fill {fill[m]:.3f}")
    rep = "\n".join(L)
    (ROB_DIR / f"clean_full{SUF}.txt").write_text(rep, encoding="utf-8")

    # structured dump for figures
    import json
    js = {
        "N": int(N), "demand_active_frac": float(pos.mean()), "has_chronos": bool(has_chronos),
        "tau": {"pos": ci_pos, "zero": ci_zero, "all": ci_all},
        "per_model": {m: {"mae": float(mae[:, models.index(m)].mean()),
                          "cost": float(cost[:, models.index(m)].mean()),
                          "fill": float(fill[m])} for m in models},
        "calib": {m: calib[m] for m in qmods},
        "matched": {m: {"kappa": matched[m][0], "capped": bool(matched[m][1]),
                        "cost_normal": matched[m][2], "cost_matched": matched[m][3],
                        "delta_pct": matched[m][4]} for m in qmods},
        "deployed": {"cost": float(dep_c), "fill": float(dep_f), "n": int(len(idx)),
                     "best_model": best, "best_cost": float(ovc[best]),
                     "gap_pct": float(100*(ovc[best]-dep_c)/dep_c),
                     # per-model overlap costs, so Section 5.8's per-model
                     # figures have an artifact (referee audit, 2026-09)
                     "overlap_per_model": {m: float(v) for m, v in ovc.items()},
                     "n_models_beating_deployed": int(sum(v < dep_c for v in ovc.values()))},
    }
    (ROB_DIR / f"clean_full{SUF}.json").write_text(json.dumps(js, indent=2), encoding="utf-8")
    print(rep)
    print(f"\nDone {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
