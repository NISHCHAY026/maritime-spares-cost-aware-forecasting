"""
Is the accuracy-cost relationship mechanical?

Cost is 90.5% holding, and holding cost is close to monotone in forecast level.
So a correlation between accuracy and cost could be an artifact: a model that
forecasts high both misses more (on mostly-zero series) and holds more.

This decomposes it. For each forecaster we record mean forecast LEVEL, MAE,
total cost, and the holding / stockout components, then ask:
  rho(MAE, cost)               the headline
  rho(level, cost)             the mechanical channel
  partial rho(MAE, cost|level) what survives after removing it
and repeats the split by cost regime.
"""
from __future__ import annotations
import json, warnings
import numpy as np, polars as pl
from scipy.stats import spearmanr
import config as C
from baselines import sba as sba_fn, ses as ses_fn
from models_classical import croston, tsb, adida, imapa
from models_distributional import fit_zip, fit_hurdle_nb
from models_ml import LGBForecaster
from simulator import CostConfig, normal_policy, simulate, lead_days_to_qtrs, service_level_for

S = C.SAMPLE_DIR; ROB = S / "robustness"
AC = [f"ACTUALS_QTR{i:02d}" for i in range(1, 29)]

def partial(r_mc, r_ml, r_lc):
    d = np.sqrt(max(1 - r_ml**2, 1e-12) * max(1 - r_lc**2, 1e-12))
    return (r_mc - r_ml * r_lc) / d

def main():
    cfg = CostConfig()
    samp = (pl.scan_parquet(S/"sample_skus.parquet").select(
        ["STOCK_ITEM_NUMBER","FORECAST_ID","SEGMENTATION_GRP","ABC_GRP","STRATEGY_GRP",
         "VELOCITY_MODE","CRITICALITY_MODE","STOCKCLASS_MODE","UNIT_PRICE_USD",
         "LEAD_TIME_MEAN","USAGE_PER_YEAR_MEAN","SBA_ALPHA","SES_ALPHA"])
        .unique(subset=["STOCK_ITEM_NUMBER","FORECAST_ID"]).collect())
    pan = (pl.scan_parquet(S/"panel.parquet").select(["STOCK_ITEM_NUMBER","FORECAST_ID"]+AC)
           .unique(subset=["STOCK_ITEM_NUMBER","FORECAST_ID"]).collect())
    df = samp.join(pan, on=["STOCK_ITEM_NUMBER","FORECAST_ID"], how="inner").sort(["STOCK_ITEM_NUMBER", "FORECAST_ID"])
    act = df.select(AC).to_numpy().astype(float)
    train, test = act[:, :16], act[:, 16:20]
    N, H = df.height, test.shape[1]

    a = df["SBA_ALPHA"].cast(pl.Float64).fill_null(0.1).to_numpy()
    e = df["SES_ALPHA"].cast(pl.Float64).fill_null(0.5).to_numpy()
    price = df["UNIT_PRICE_USD"].fill_null(1.0).cast(pl.Float64).to_numpy()
    price = np.where(price > 0, price, 1.0)
    ld = df["LEAD_TIME_MEAN"].fill_null(60.0).cast(pl.Float64).to_numpy()
    ld = np.where(ld > 0, ld, 60.0)
    lq = lead_days_to_qtrs(ld); lsq = (ld*0.3)/91.3125
    sigma = train.std(axis=1, ddof=0)
    sl = np.array([service_level_for(c, cfg) for c in df["CRITICALITY_MODE"].fill_null("Normal").to_list()])

    def enc(col):
        s = df[col].fill_null("__NA__").cast(pl.Utf8)
        mp = {c:i for i,c in enumerate(sorted(set(s.to_list())))}
        return np.array([mp[v] for v in s.to_list()], np.float64)

    F = {}
    F["SBA"] = sba_fn(train, a).forecast[:, -1]
    st = ses_fn(train, e); F["SES"] = e*train[:, -1] + (1-e)*st[:, -1]
    F["MA"] = (train[:, -1] + train[:, -2]) / 2.0
    F["CROSTON"] = croston(train, a, sba=False).forecast[:, -1]
    F["TSB"] = tsb(train, a).forecast[:, -1]
    F["ADIDA"] = adida(train, e, k=None)["forecast_per_period"]
    F["IMAPA"] = imapa(train, e, k_max=None)["forecast_per_period"]
    F["ZIP"] = fit_zip(train)["mean"]
    F["HURDLE_NB"] = fit_hurdle_nb(train)["mean"]
    static = {k: enc(c) for k,c in [("f_seg","SEGMENTATION_GRP"),("f_abc","ABC_GRP"),
        ("f_strategy","STRATEGY_GRP"),("f_velocity","VELOCITY_MODE"),
        ("f_criticality","CRITICALITY_MODE"),("f_stockclass","STOCKCLASS_MODE")]}
    static["f_price"]=price; static["f_leadtime"]=ld
    static["f_usage"]=df["USAGE_PER_YEAR_MEAN"].fill_null(0.0).cast(pl.Float64).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lg = LGBForecaster(lags=(1,2,3,4), rolls=(2,4))
        lg.fit(train, static, train_quantiles=[0.5,0.8,0.9,0.95,0.99], seed=42)
        F["LGBM"] = lg.predict_recursive(train, static, h=H)["mean"][:, 0]
    cp = S/"forecasts_chronos_clean.parquet"
    if cp.exists():
        cd = df.select(["STOCK_ITEM_NUMBER","FORECAST_ID"]).join(
            pl.read_parquet(cp), on=["STOCK_ITEM_NUMBER","FORECAST_ID"], how="left")
        F["CHRONOS"] = cd["POINT_Q17"].fill_null(0.0).to_numpy()

    models = list(F)
    rec = {}
    per_sku = {}
    for m in models:
        pt = np.maximum(F[m], 0.0)
        mae = np.abs(np.repeat(pt[:,None], H, 1) - test).mean(1)
        pol = normal_policy(pt, sigma, lq.astype(float), lsq, price, sl, cfg)
        sim = simulate(test, pol["s"], pol["S"], lq, price, cfg=cfg)
        rec[m] = {"level": float(pt.mean()), "mae": float(mae.mean()),
                  "cost": float(sim["total_cost"].mean()),
                  "hold": float(sim["holding_cost"].mean()),
                  "stock": float(sim["stockout_cost"].mean())}
        per_sku[m] = {"mae": mae, "cost": sim["total_cost"], "hold": sim["holding_cost"],
                      "stock": sim["stockout_cost"], "pt": pt}

    L = ["MECHANICAL DECOMPOSITION OF THE ACCURACY-COST RELATIONSHIP", "="*74, "",
         f"{'MODEL':<11}{'level':>9}{'MAE':>9}{'cost':>11}{'holding':>11}{'stockout':>10}"]
    for m in sorted(models, key=lambda x: rec[x]["cost"]):
        r = rec[m]
        L.append(f"{m:<11}{r['level']:>9.3f}{r['mae']:>9.3f}{r['cost']:>11,.0f}{r['hold']:>11,.0f}{r['stock']:>10,.0f}")

    def block(name, mask):
        lev = np.array([per_sku[m]["pt"][mask].mean() for m in models])
        mae = np.array([per_sku[m]["mae"][mask].mean() for m in models])
        cost = np.array([per_sku[m]["cost"][mask].mean() for m in models])
        hold = np.array([per_sku[m]["hold"][mask].mean() for m in models])
        stock = np.array([per_sku[m]["stock"][mask].mean() for m in models])
        r_mc,p_mc = spearmanr(mae, cost); r_ml,_ = spearmanr(mae, lev); r_lc,_ = spearmanr(lev, cost)
        r_mh,p_mh = spearmanr(mae, hold); r_ms,p_ms = spearmanr(mae, stock)
        out = [f"", f"--- {name} (n={int(mask.sum()):,}) ---",
            f"  rho(MAE, cost)              = {r_mc:+.3f} (p={p_mc:.3f})   <- headline",
            f"  rho(level, cost)            = {r_lc:+.3f}              <- mechanical channel",
            f"  rho(MAE, level)             = {r_ml:+.3f}",
            f"  partial rho(MAE, cost|level)= {partial(r_mc,r_ml,r_lc):+.3f}   <- what survives",
            f"  rho(MAE, holding)           = {r_mh:+.3f} (p={p_mh:.3f})",
            f"  rho(MAE, stockout)          = {r_ms:+.3f} (p={p_ms:.3f})"]
        return out, {"rho_mae_cost": r_mc, "p": p_mc, "rho_level_cost": r_lc,
                     "rho_mae_level": r_ml, "partial": partial(r_mc,r_ml,r_lc),
                     "rho_mae_holding": r_mh, "rho_mae_stockout": r_ms, "n": int(mask.sum())}

    res = {}
    allm = np.ones(N, bool); hi = price >= 1600.0
    for nm, msk in [("ALL", allm), ("stockout-dominated", ~hi), ("holding-dominated", hi)]:
        lines, d = block(nm, msk); L += lines; res[nm] = d

    rep = "\n".join(L)
    (ROB/"mechanical.txt").write_text(rep, encoding="utf-8")
    (ROB/"mechanical.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(rep)

if __name__ == "__main__":
    main()
