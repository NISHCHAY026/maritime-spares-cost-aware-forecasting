"""
Does the cost result survive a policy that actually reorders?

The problem
-----------
The simulator starts every SKU with on-hand at S, the order-up-to level, and
runs four quarters. With demand this sparse, 83 % of SKUs never cross the
reorder point in that window, so the (s, S) logic never fires: the run measures
holding cost on an initial position, and 90.5 % of total cost is holding. A
referee can then argue that "simulated cost" here is largely a forecast-level
measure, which is awkward, because the paper's own finding is that MAE is a
forecast-level measure. If both sides of the correlation are level, the regime
result could be an artifact of a policy that never runs rather than a property
of the accuracy metric.

The test
--------
Two factors, crossed.

  Initial on-hand   S        saturated, as published
                    mid      uniform on [s, S], a random position in the cycle
                    s        exactly at the reorder point, so a review fires
                    zero     empty, the most demanding start

  Horizon           H = 4    train 1-16, test 17-20, as published
                    H = 8    train 1-12, test 13-20, still entirely uncensored

The H = 8 arms refit every forecaster on a 12-quarter window. Chronos is read
from a parquet produced for a 16-quarter context and cannot be reused there, so
the cross-arm comparison is computed on the ten forecasters present in every
arm, and the eleven-model figure is reported for the published arm alone.

For each arm we report whether the policy actually ran (share of SKUs placing an
order, orders per SKU), what the cost is made of, and then the statistics the
paper's claims rest on: the fleet-level rank correlation between MAE and cost,
overall and by cost regime, and the mean per-SKU Kendall tau.

Writes output/sample/segment/robustness/simulator_exercise.{txt,json}
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import polars as pl
from scipy.stats import spearmanr

import config as C
from baselines import sba_next, ses as ses_fn
from models_classical import adida, imapa, croston_next, tsb_next
from models_distributional import fit_zip, fit_hurdle_nb
from models_ml import LGBForecaster
from simulator import (CostConfig, normal_policy, simulate,
                       lead_days_to_qtrs, service_level_for)
from unified_stats import per_sku_kendall_tau

S_DIR = C.SAMPLE_DIR
ROB = S_DIR / "robustness"
ACT = [f"ACTUALS_QTR{i:02d}" for i in range(1, 29)]
SEED = 42

# (label, train_hi, test_lo, test_hi, init_mode)
ARMS = [
    ("published      S / H=4", 16, 16, 20, "S"),
    ("mid-cycle    mid / H=4", 16, 16, 20, "mid"),
    ("at reorder     s / H=4", 16, 16, 20, "s"),
    ("empty       zero / H=4", 16, 16, 20, "zero"),
    ("train12        S / H=4", 12, 12, 16, "S"),
    ("train12      mid / H=4", 12, 12, 16, "mid"),
    ("published      S / H=8", 12, 12, 20, "S"),
    ("mid-cycle    mid / H=8", 12, 12, 20, "mid"),
    ("empty       zero / H=8", 12, 12, 20, "zero"),
]


def load():
    sample = (pl.scan_parquet(S_DIR / "sample_skus.parquet").select(
        ["STOCK_ITEM_NUMBER", "FORECAST_ID", "SEGMENTATION_GRP", "ABC_GRP",
         "STRATEGY_GRP", "VELOCITY_MODE", "CRITICALITY_MODE", "STOCKCLASS_MODE",
         "UNIT_PRICE_USD", "LEAD_TIME_MEAN", "USAGE_PER_YEAR_MEAN",
         "SBA_ALPHA", "SES_ALPHA"])
        .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect())
    panel = (pl.scan_parquet(S_DIR / "panel.parquet")
             .select(["STOCK_ITEM_NUMBER", "FORECAST_ID"] + ACT)
             .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect())
    return (sample.join(panel, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner")
                  .sort(["STOCK_ITEM_NUMBER", "FORECAST_ID"]))


def forecasts(df, train, H, want_chronos, price, lead_days):
    """Every forecaster, on whatever training window it is handed."""
    sba_a = df["SBA_ALPHA"].cast(pl.Float64).fill_null(0.1).to_numpy()
    ses_a = df["SES_ALPHA"].cast(pl.Float64).fill_null(0.5).to_numpy()
    F = {}
    F["SBA"] = sba_next(train, sba_a)
    st = ses_fn(train, ses_a)
    F["SES"] = ses_a * train[:, -1] + (1 - ses_a) * st[:, -1]
    F["MA"] = (train[:, -1] + train[:, -2]) / 2.0
    F["CROSTON"] = croston_next(train, sba_a, sba=False)
    F["TSB"] = tsb_next(train, sba_a)
    F["ADIDA"] = adida(train, ses_a, k=None)["forecast_per_period"]
    F["IMAPA"] = imapa(train, ses_a, k_max=None)["forecast_per_period"]
    F["ZIP"] = fit_zip(train)["mean"]
    F["HURDLE_NB"] = fit_hurdle_nb(train)["mean"]

    def enc(col):
        s = df[col].fill_null("__NA__").cast(pl.Utf8)
        cats = sorted(set(s.to_list()))
        mp = {c: i for i, c in enumerate(cats)}
        return np.array([mp[v] for v in s.to_list()], np.float64)

    static = {k: enc(c) for k, c in [
        ("f_seg", "SEGMENTATION_GRP"), ("f_abc", "ABC_GRP"),
        ("f_strategy", "STRATEGY_GRP"), ("f_velocity", "VELOCITY_MODE"),
        ("f_criticality", "CRITICALITY_MODE"), ("f_stockclass", "STOCKCLASS_MODE")]}
    static["f_price"] = price
    static["f_leadtime"] = lead_days
    static["f_usage"] = df["USAGE_PER_YEAR_MEAN"].fill_null(0.0).cast(pl.Float64).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lg = LGBForecaster(lags=(1, 2, 3, 4), rolls=(2, 4))
        lg.fit(train, static,
               train_quantiles=[q / 100 for q in (50, 80, 90, 95, 99)],
               seed=SEED)
        F["LGBM"] = lg.predict_recursive(train, static, h=H)["mean"][:, 0]

    if want_chronos:
        cpath = S_DIR / "forecasts_chronos_clean.parquet"
        if cpath.exists():
            cdf = df.select(["STOCK_ITEM_NUMBER", "FORECAST_ID"]).join(
                pl.read_parquet(cpath), on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="left")
            F["CHRONOS"] = cdf["POINT_Q17"].fill_null(0.0).to_numpy()
    return F


def initial_stock(mode, s, S, u):
    """u is one uniform draw per SKU, shared by every model in the arm."""
    if mode == "S":
        return None                      # simulator default
    if mode == "s":
        return s.copy()
    if mode == "zero":
        return np.zeros_like(s)
    if mode == "mid":
        return s + u * np.maximum(S - s, 0.0)
    raise ValueError(mode)


def main():
    cfg = CostConfig()
    crossover = cfg.stockout_unit_cost / cfg.holding_per_unit_per_qtr
    df = load()
    actuals = df.select(ACT).to_numpy().astype(np.float64)
    N = df.height

    price = df["UNIT_PRICE_USD"].fill_null(1.0).cast(pl.Float64).to_numpy()
    price = np.where(price > 0, price, 1.0)
    ld = df["LEAD_TIME_MEAN"].fill_null(60.0).cast(pl.Float64).to_numpy()
    ld = np.where(ld > 0, ld, 60.0)
    lq = lead_days_to_qtrs(ld)
    lsq = (ld * 0.3) / 91.3125
    sl = np.array([service_level_for(c, cfg)
                   for c in df["CRITICALITY_MODE"].fill_null("Normal").to_list()])
    hi = price >= crossover

    results = {}
    L = ["DOES THE COST RESULT SURVIVE A POLICY THAT ACTUALLY REORDERS?", "=" * 86, "",
         f"N = {N:,} SKUs. Crossover between holding- and stockout-dominated is "
         f"${crossover:,.0f}.", ""]

    for label, tr_hi, te_lo, te_hi, mode in ARMS:
        train = actuals[:, :tr_hi]
        test = actuals[:, te_lo:te_hi]
        H = test.shape[1]
        want_chronos = (tr_hi == 16)
        F = forecasts(df, train, H, want_chronos, price, ld)
        models = list(F)
        sigma = train.std(axis=1, ddof=0)
        # one shared draw per arm, so all models see the same cycle position
        u_pos = np.random.default_rng(SEED).random(N)

        mae = np.zeros((N, len(models)))
        cost = np.zeros((N, len(models)))
        diag = {"order_share": [], "orders_per_sku": [], "hold_share": [],
                "stock_share": [], "order_cost_share": [], "fill": []}
        for j, m in enumerate(models):
            pt = np.maximum(F[m], 0.0)
            mae[:, j] = np.abs(np.repeat(pt[:, None], H, 1) - test).mean(1)
            pol = normal_policy(pt, sigma, lq.astype(float), lsq, price, sl, cfg)
            init = initial_stock(mode, pol["s"], pol["S"], u_pos)
            sim = simulate(test, pol["s"], pol["S"], lq, price,
                           initial_on_hand=init, cfg=cfg)
            cost[:, j] = sim["total_cost"]
            tot = sim["total_cost"].sum()
            diag["order_share"].append(float((sim["n_orders"] > 0).mean()))
            diag["orders_per_sku"].append(float(sim["n_orders"].mean()))
            diag["hold_share"].append(float(sim["holding_cost"].sum() / max(tot, 1e-9)))
            diag["stock_share"].append(float(sim["stockout_cost"].sum() / max(tot, 1e-9)))
            diag["order_cost_share"].append(float(sim["ordering_cost"].sum() / max(tot, 1e-9)))
            diag["fill"].append(float(sim["fill_rate"].mean()))

        # cross-arm statistics use the ten forecasters present in every arm
        common = [m for m in models if m != "CHRONOS"]
        ci = [models.index(m) for m in common]

        def rho(mask, idx):
            m_ = mae[np.ix_(mask, idx)].mean(0)
            c_ = cost[np.ix_(mask, idx)].mean(0)
            r, p = spearmanr(m_, c_)
            return float(r), float(p)

        allm = np.ones(N, bool)
        r_all, p_all = rho(allm, ci)
        r_lo, p_lo = rho(~hi, ci)
        r_hi, p_hi = rho(hi, ci)
        taus = per_sku_kendall_tau(mae[:, ci], cost[:, ci])
        pos = test.sum(1) > 0

        row = {
            "train_quarters": tr_hi, "H": H, "init": mode,
            "n_models": len(models), "n_models_common": len(common),
            "order_share_mean": float(np.mean(diag["order_share"])),
            "orders_per_sku_mean": float(np.mean(diag["orders_per_sku"])),
            "holding_share": float(np.mean(diag["hold_share"])),
            "stockout_share": float(np.mean(diag["stock_share"])),
            "ordering_share": float(np.mean(diag["order_cost_share"])),
            "fill_mean": float(np.mean(diag["fill"])),
            "mean_cost": float(cost[:, ci].mean()),
            "demand_active_share": float(pos.mean()),
            "rho_all": r_all, "p_all": p_all,
            "rho_stockout_dominated": r_lo, "p_stockout": p_lo,
            "rho_holding_dominated": r_hi, "p_holding": p_hi,
            "tau_demand_active": float(np.nanmean(taus[pos])),
            "cheapest": common[int(np.argmin(cost[:, ci].mean(0)))],
            "most_accurate": common[int(np.argmin(mae[:, ci].mean(0)))],
        }
        if "CHRONOS" in models:
            r11, _ = rho(hi, list(range(len(models))))
            row["rho_holding_dominated_11models"] = r11
        results[label.strip()] = row
        print(f"  done: {label}")

    # ---------------- report ----------------
    L += ["1. DID THE POLICY ACTUALLY RUN?",
          f"   {'ARM':<24}{'ordered':>9}{'orders/SKU':>12}{'holding':>10}"
          f"{'stockout':>10}{'fill':>8}{'mean cost':>12}"]
    for k, v in results.items():
        L.append(f"   {k:<24}{100*v['order_share_mean']:>8.1f}%{v['orders_per_sku_mean']:>12.2f}"
                 f"{100*v['holding_share']:>9.1f}%{100*v['stockout_share']:>9.1f}%"
                 f"{v['fill_mean']:>8.3f}{v['mean_cost']:>12,.0f}")

    L += ["", "2. DOES THE REGIME RESULT SURVIVE?  (ten forecasters common to every arm)",
          f"   {'ARM':<24}{'pooled':>10}{'stockout-dom':>15}{'holding-dom':>14}{'tau':>9}"]
    for k, v in results.items():
        L.append(f"   {k:<24}{v['rho_all']:>+10.3f}{v['rho_stockout_dominated']:>+15.3f}"
                 f"{v['rho_holding_dominated']:>+14.3f}{v['tau_demand_active']:>+9.3f}")

    L += ["", "3. WHICH MODEL WINS",
          f"   {'ARM':<24}{'most accurate':>16}{'cheapest':>14}"]
    for k, v in results.items():
        L.append(f"   {k:<24}{v['most_accurate']:>16}{v['cheapest']:>14}")

    base = results[ARMS[0][0].strip()]
    L += ["", "4. READING",
          f"   Published arm: {100*base['order_share_mean']:.1f} % of SKUs place an order and "
          f"cost is {100*base['holding_share']:.1f} % holding.",
          "   If the regime correlations in section 2 hold up in the arms where the",
          "   policy reorders and stockout cost carries real weight, the finding is a",
          "   property of the accuracy metric and not of an unexercised policy."]

    rep = "\n".join(L)
    (ROB / "simulator_exercise.txt").write_text(rep, encoding="utf-8")
    (ROB / "simulator_exercise.json").write_text(
        json.dumps({"crossover_price_usd": float(crossover), "N": int(N),
                    "seed": SEED, "arms": results}, indent=2), encoding="utf-8")
    print("\n" + rep)


if __name__ == "__main__":
    main()
