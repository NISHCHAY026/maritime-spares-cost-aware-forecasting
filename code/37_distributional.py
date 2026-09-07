"""
Phase 1, point 4: does the predictive DISTRIBUTION open a channel that the
point forecast does not?

Under the normal-approximation policy the reorder point depends on the forecast
only through the scalar D_bar, so simulated cost is a deterministic function of
level and no point-error metric can add information (see 35/36). Under the
native-quantile policy the reorder point is s = LT * Q_alpha, so the policy
reads a quantile of the predictive distribution instead. That is a different
channel ONLY IF Q_alpha varies across models independently of the mean. If
Q_alpha is itself a monotone function of level, nothing is gained and the
confound simply reappears one level up.

So this measures, across the four quantile-emitting models (ZIP, Hurdle-NB,
LightGBM, Chronos):

  1. per-SKU rho(level, Q_alpha)  -- is there a second channel at all?
  2. per-SKU rho(metric, native-quantile cost) for MAE, pinball at the SKU's
     own service level, and a CRPS approximation over the five fitted
     quantiles.
  3. the structural check rho(Q_alpha, cost), which should be near +/-1 if the
     native policy is as deterministic in Q_alpha as the normal policy is in
     D_bar.

CAVEAT recorded in the output: only four models emit quantiles, so the
fleet-level rank correlations here rest on four points and are reported for
completeness rather than inference. The per-SKU statistics use n = 15,348.

Writes output/sample/segment/robustness/distributional.{txt,json}
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import polars as pl
from scipy.stats import spearmanr

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


def pinball_at(actual, f, q):
    d = actual - f[:, None]
    qq = q[:, None]
    return np.where(d >= 0, qq * d, (qq - 1.0) * d).mean(axis=1)


def crps_approx(actual, qmap, levels):
    """Mean pinball across the fitted quantile grid: a coarse CRPS."""
    tot = np.zeros(actual.shape[0])
    for a in levels:
        f = np.maximum(qmap[a], 0.0)
        tot += pinball_at(actual, f, np.full(actual.shape[0], a))
    return tot / len(levels)


def per_sku_rho(X, Y):
    out = np.full(X.shape[0], np.nan)
    for i in range(X.shape[0]):
        x, y = X[i], Y[i]
        if np.all(x == x[0]) or np.all(y == y[0]):
            continue
        out[i] = spearmanr(x, y).correlation
    return out


def sm(v, mask=None):
    w = v if mask is None else v[mask]
    w = w[np.isfinite(w)]
    return None if w.size == 0 else {
        "n": int(w.size), "mean": float(w.mean()),
        "median": float(np.median(w)),
        "frac_abs_gt_099": float(np.mean(np.abs(w) > 0.99))}


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

    Q = {}   # model -> {alpha: array}
    P = {}   # model -> point forecast

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
    if cp.exists():
        cd = df.select(["STOCK_ITEM_NUMBER", "FORECAST_ID"]).join(
            pl.read_parquet(cp), on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="left")
        Q["CHRONOS"] = {q / 100: cd[f"Q_{q:02d}"].fill_null(0.0).to_numpy() for q in QUANTS}
        P["CHRONOS"] = cd["POINT_Q17"].fill_null(0.0).to_numpy()

    models = list(Q)
    Mn = len(models)
    levels = [q / 100 for q in QUANTS]

    lvl = np.zeros((N, Mn)); qsl = np.zeros((N, Mn))
    mae = np.zeros((N, Mn)); pin = np.zeros((N, Mn)); crp = np.zeros((N, Mn))
    ncost = np.zeros((N, Mn))

    for j, m in enumerate(models):
        pt = np.maximum(P[m], 0.0)
        lvl[:, j] = pt
        mae[:, j] = np.abs(np.repeat(pt[:, None], H, 1) - test).mean(1)
        # the quantile the native policy actually reads: closest fitted level
        idx = np.abs(np.array(levels)[None, :] - sl[:, None]).argmin(axis=1)
        qa = np.zeros(N)
        for k, a in enumerate(levels):
            sel = idx == k
            qa[sel] = np.maximum(Q[m][a], 0.0)[sel]
        qsl[:, j] = qa
        pin[:, j] = pinball_at(test, qa, sl)
        crp[:, j] = crps_approx(test, {a: np.maximum(Q[m][a], 0.0) for a in levels}, levels)
        pol = native_quantile_policy({a: np.maximum(Q[m][a], 0.0) for a in levels},
                                     pt, lq.astype(float), price, sl, cfg)
        ncost[:, j] = simulate(test, pol["s"], pol["s"] + pol["eoq"], lq, price,
                               cfg=cfg)["total_cost"]

    pos = test.sum(axis=1) > 0
    res = {"models": models, "n_models": Mn, "N": int(N), "per_sku": {}, "fleet": {}}

    res["per_sku"]["level_vs_Qalpha"] = sm(per_sku_rho(lvl, qsl))
    res["per_sku"]["Qalpha_vs_cost"] = sm(per_sku_rho(qsl, ncost))
    res["per_sku"]["level_vs_cost"] = sm(per_sku_rho(lvl, ncost))
    res["per_sku"]["mae_vs_cost_active"] = sm(per_sku_rho(mae, ncost), pos)
    res["per_sku"]["pinball_vs_cost_active"] = sm(per_sku_rho(pin, ncost), pos)
    res["per_sku"]["crps_vs_cost_active"] = sm(per_sku_rho(crp, ncost), pos)

    def fleet(metric):
        mv = np.array([metric[:, j].mean() for j in range(Mn)])
        cv = np.array([ncost[:, j].mean() for j in range(Mn)])
        r, p = spearmanr(mv, cv)
        return {"rho": float(r), "p": float(p)}
    for nm, arr in [("mae", mae), ("pinball", pin), ("crps", crp), ("level", lvl)]:
        res["fleet"][nm] = fleet(arr)

    L = ["PHASE 1 POINT 4 - DOES THE PREDICTIVE DISTRIBUTION OPEN A CHANNEL?",
         "=" * 74, "",
         f"  models with quantiles: {', '.join(models)}  (only {Mn})",
         "  Fleet-level rows below therefore rest on 4 points and are shown for",
         "  completeness, not inference. Per-SKU rows use N = %s." % f"{N:,}", "",
         "1. IS Q_alpha A SEPARATE CHANNEL FROM THE POINT FORECAST?"]
    d = res["per_sku"]["level_vs_Qalpha"]
    L.append(f"   per-SKU rho(level, Q_alpha)   mean {d['mean']:+.3f}  median {d['median']:+.3f}  "
             f"|rho|>0.99 in {100*d['frac_abs_gt_099']:.1f} % of SKUs  n={d['n']:,}")
    L += ["", "2. STRUCTURAL: IS NATIVE COST DETERMINED BY Q_alpha?"]
    for k, lab in [("Qalpha_vs_cost", "rho(Q_alpha, native cost)"),
                   ("level_vs_cost", "rho(level,   native cost)")]:
        d = res["per_sku"][k]
        L.append(f"   {lab:<28} mean {d['mean']:+.3f}  median {d['median']:+.3f}  n={d['n']:,}")
    L += ["", "3. WHICH METRIC PREDICTS NATIVE-QUANTILE COST? (demand-active SKUs)"]
    for k, lab in [("mae_vs_cost_active", "MAE"),
                   ("pinball_vs_cost_active", "pinball at service level"),
                   ("crps_vs_cost_active", "CRPS (5-quantile grid)")]:
        d = res["per_sku"][k]
        if d:
            L.append(f"   {lab:<28} mean {d['mean']:+.3f}  median {d['median']:+.3f}  n={d['n']:,}")
    L += ["", "4. FLEET-LEVEL (4 points; indicative only)"]
    for nm in ("mae", "pinball", "crps", "level"):
        f_ = res["fleet"][nm]
        L.append(f"   rho({nm:<8}, native cost) = {f_['rho']:+.3f}  (p={f_['p']:.3f})")

    rep = "\n".join(L)
    (ROB / "distributional.txt").write_text(rep, encoding="utf-8")
    (ROB / "distributional.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(rep)


if __name__ == "__main__":
    main()
