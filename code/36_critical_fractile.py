"""
Phase 1, constructive half: the metric that is NOT confounded with level.

Section 35 shows MAE is minimised at the median of demand, which is zero for
77 % of these SKUs, so MAE-rank is level-rank. But the cost-minimising forecast
is not the median. In a newsvendor-type trade-off with underage cost Cu (a
stockout) and overage cost Co (holding), the optimal order quantity is the
demand quantile at the CRITICAL FRACTILE

    q* = Cu / (Cu + Co).

Here Cu = stockout_unit_cost (a flat $100 per unit short) and, per quarter,
Co = (holding_rate/4) * unit_price. So

    q*(price) = 100 / (100 + 0.0625 * price).

Two consequences worth stating precisely.

1. q* = 0.5 exactly when 0.0625*price = 100, i.e. price = $1,600. That is
   precisely the regime crossover found empirically. The cost regime boundary
   is the price at which the cost structure's required fractile equals the
   fractile MAE implicitly targets. MAE is "correct" at exactly one price.
2. Below the crossover q* > 0.5 (stock high), above it q* < 0.5 (stock low),
   and MAE targets 0.5 regardless. So MAE points the wrong way on BOTH sides,
   which is why the accuracy-cost correlation flips sign across the boundary
   rather than merely weakening.

This script tests the constructive claim: a pinball (quantile) loss evaluated
at each SKU's own critical fractile should predict simulated cost where MAE
does not.

    L_q(d, f) = q*(d - f)      if d >= f
                (1-q)*(f - d)  otherwise

Writes output/sample/segment/robustness/critical_fractile.{txt,json}
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


def pinball(actual, f, q):
    """Mean pinball loss over the horizon. actual (N,H), f (N,), q (N,)."""
    d = actual - f[:, None]
    qq = q[:, None]
    return np.where(d >= 0, qq * d, (qq - 1.0) * d).mean(axis=1)


def main():
    cfg = CostConfig()
    samp = (pl.scan_parquet(S / "sample_skus.parquet").select(
        ["STOCK_ITEM_NUMBER", "FORECAST_ID", "SEGMENTATION_GRP", "ABC_GRP",
         "STRATEGY_GRP", "VELOCITY_MODE", "CRITICALITY_MODE", "STOCKCLASS_MODE",
         "UNIT_PRICE_USD", "LEAD_TIME_MEAN", "USAGE_PER_YEAR_MEAN",
         "SBA_ALPHA", "SES_ALPHA"])
        .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect())
    pan = (pl.scan_parquet(S / "panel.parquet")
           .select(["STOCK_ITEM_NUMBER", "FORECAST_ID"] + AC)
           .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect())
    df = samp.join(pan, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"],
                   how="inner").sort(["STOCK_ITEM_NUMBER", "FORECAST_ID"])

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

    # critical fractile per SKU
    Cu = cfg.stockout_unit_cost
    Co = cfg.holding_per_unit_per_qtr * price
    qstar = Cu / (Cu + Co)
    crossover = Cu / cfg.holding_per_unit_per_qtr

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
    Mn = len(models)
    lvl = np.column_stack([np.maximum(F[m], 0.0) for m in models])
    mae = np.zeros((N, Mn)); pin = np.zeros((N, Mn)); cost = np.zeros((N, Mn))
    for j, m in enumerate(models):
        pt = lvl[:, j]
        mae[:, j] = np.abs(np.repeat(pt[:, None], H, 1) - test).mean(1)
        pin[:, j] = pinball(test, pt, qstar)
        pol = normal_policy(pt, sigma, lq.astype(float), lsq, price, sl, cfg)
        cost[:, j] = simulate(test, pol["s"], pol["S"], lq, price, cfg=cfg)["total_cost"]

    def per_sku_rho(X, Y):
        out = np.full(X.shape[0], np.nan)
        for i in range(X.shape[0]):
            x, y = X[i], Y[i]
            if np.all(x == x[0]) or np.all(y == y[0]):
                continue
            out[i] = spearmanr(x, y).correlation
        return out

    def sm(v, mask):
        w = v[mask]; w = w[np.isfinite(w)]
        return None if w.size == 0 else {"n": int(w.size), "mean": float(w.mean()),
                                         "median": float(np.median(w))}

    pos = test.sum(axis=1) > 0
    hi = price >= crossover
    r_mae = per_sku_rho(mae, cost)
    r_pin = per_sku_rho(pin, cost)
    r_pl = per_sku_rho(pin, lvl)
    r_ml = per_sku_rho(mae, lvl)

    def fleet(metric, mask):
        mv = np.array([metric[mask, j].mean() for j in range(Mn)])
        cv = np.array([cost[mask, j].mean() for j in range(Mn)])
        r, p = spearmanr(mv, cv)
        return float(r), float(p)

    res = {"crossover_price": float(crossover),
           "qstar_median": float(np.median(qstar)),
           "qstar_p10": float(np.percentile(qstar, 10)),
           "qstar_p90": float(np.percentile(qstar, 90)),
           "share_qstar_above_half": float(np.mean(qstar > 0.5)),
           "fleet": {}, "per_sku": {}}
    for nm, msk in [("all", np.ones(N, bool)), ("stockout_dominated", ~hi),
                    ("holding_dominated", hi)]:
        rm, pm_ = fleet(mae, msk); rp, pp_ = fleet(pin, msk)
        res["fleet"][nm] = {"mae_rho": rm, "mae_p": pm_,
                            "pinball_rho": rp, "pinball_p": pp_,
                            "n": int(msk.sum())}
    res["per_sku"] = {
        "mae_vs_cost_demand_active": sm(r_mae, pos),
        "pinball_vs_cost_demand_active": sm(r_pin, pos),
        "mae_vs_level_demand_active": sm(r_ml, pos),
        "pinball_vs_level_demand_active": sm(r_pl, pos),
    }

    L = ["PHASE 1 - CRITICAL FRACTILE: THE METRIC THE COST STRUCTURE IMPLIES",
         "=" * 74, "",
         f"  Cu = ${Cu:,.0f} per unit short;  Co = {cfg.holding_per_unit_per_qtr:.4f} * price per quarter",
         f"  q*(price) = Cu / (Cu + Co)  =>  q* = 0.5 exactly at price ${crossover:,.0f}",
         f"  which is the empirically observed regime crossover.", "",
         f"  q* median {np.median(qstar):.3f}   p10 {np.percentile(qstar,10):.3f}   "
         f"p90 {np.percentile(qstar,90):.3f}",
         f"  share of SKUs needing q* > 0.5 (stock ABOVE the median): {100*np.mean(qstar>0.5):.1f} %",
         "",
         "  MAE targets the 0.5 fractile for every SKU regardless of price, so it",
         "  is aligned with the cost structure at exactly one price and points the",
         "  wrong way on both sides of it.", "",
         "FLEET-LEVEL rho(metric, cost) ACROSS THE 11 FORECASTERS",
         f"  {'subset':<22}{'MAE':>18}{'pinball at q*':>20}"]
    for nm in ("all", "stockout_dominated", "holding_dominated"):
        d = res["fleet"][nm]
        L.append(f"  {nm:<22}{d['mae_rho']:+8.3f} (p={d['mae_p']:.3f}){d['pinball_rho']:+11.3f} (p={d['pinball_p']:.3f})")
    L += ["", "PER-SKU rho ACROSS THE 11 FORECASTERS (demand-active SKUs)"]
    for k, lab in [("mae_vs_cost_demand_active", "MAE     vs cost"),
                   ("pinball_vs_cost_demand_active", "pinball vs cost"),
                   ("mae_vs_level_demand_active", "MAE     vs level"),
                   ("pinball_vs_level_demand_active", "pinball vs level")]:
        d = res["per_sku"][k]
        if d:
            L.append(f"  {lab:<20} mean {d['mean']:+.3f}  median {d['median']:+.3f}  n={d['n']:,}")

    rep = "\n".join(L)
    (ROB / "critical_fractile.txt").write_text(rep, encoding="utf-8")
    (ROB / "critical_fractile.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(rep)


if __name__ == "__main__":
    main()
