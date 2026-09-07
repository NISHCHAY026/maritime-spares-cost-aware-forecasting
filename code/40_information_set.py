"""
Equalise the information set across forecasters, and report what it changes.

The defect
----------
Croston, SBA and TSB update their smoothed state from actuals[t-1]:

    level[t]  = alpha * d[t-1] + (1 - alpha) * level[t-1]     (on a demand event)
    period[t] = alpha * interval[t-1] + (1 - alpha) * period[t-1]
    prob[t]   = beta * 1[d[t-1] > 0] + (1 - beta) * prob[t-1]    (TSB)

so the state at the last training column T-1 has consumed demand only through
T-2, and `forecast[:, -1]` never saw the final training quarter through the
smoothing channel. It enters only through the initialisation constants
(avg_nonzero, round_count, p0), whose influence decays as (1 - alpha)^T.

Every other forecaster in the comparison conditions on the full window: SES
gets an explicit extra step at the call site, MA reads the last two quarters
directly, ADIDA and IMAPA end with an SES step on the last aggregated block,
ZIP and Hurdle-NB fit moments over all 16 columns, and LightGBM and Chronos
take lags and context ending at the last quarter. Three of eleven models were
scored on a shorter history than their competitors.

The fix
-------
`sba_next`, `croston_next` and `tsb_next` run one further iteration of the SAME
recurrence (verified against the existing implementation carried one column
further with the initialisation held fixed). The state functions are untouched,
so `baselines.sba` still reproduces the deployed pipeline column for column.

This script diffs the two arms and re-derives every headline number that could
move: MAE, cost, per-SKU Kendall tau, the cost-regime Spearman correlations
with bootstrap intervals, and the deployed-policy gap.

Run 26_clean_full.py twice first (once plain, once with EQUAL_INFO=1).
Writes output/sample/segment/robustness/information_set.{txt,json}
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl
from scipy.stats import spearmanr

import config as C
from baselines import sba as sba_fn, sba_next
from models_classical import croston, tsb, croston_next, tsb_next
from simulator import CostConfig
from unified_stats import per_sku_kendall_tau

S = C.SAMPLE_DIR
ROB = S / "robustness"
AC = [f"ACTUALS_QTR{i:02d}" for i in range(1, 29)]
B = 2000
SEED = 42


def load_per_sku(suffix):
    ps = pl.read_parquet(S / f"clean_per_sku{suffix}.parquet")
    sk = (pl.read_parquet(S / "sample_skus.parquet")
          .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"])
          .select(["STOCK_ITEM_NUMBER", "FORECAST_ID", "UNIT_PRICE_USD"]))
    d = (ps.join(sk, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner")
           .sort(["STOCK_ITEM_NUMBER", "FORECAST_ID"]))
    return d, [c[len("COST_"):] for c in d.columns if c.startswith("COST_")]


def regime_stats(d, models, crossover):
    price = d["UNIT_PRICE_USD"].fill_null(1.0).cast(pl.Float64).to_numpy()
    hi = price >= crossover
    mae = np.column_stack([d[f"MAE_{m}"].to_numpy().astype(float) for m in models])
    cost = np.column_stack([d[f"COST_{m}"].to_numpy().astype(float) for m in models])
    rng = np.random.default_rng(SEED)
    out = {}
    for name, mask in [("all", np.ones(price.size, bool)),
                       ("stockout_dominated", ~hi), ("holding_dominated", hi)]:
        idx = np.flatnonzero(mask)
        rho, p = spearmanr(mae[idx].mean(0), cost[idx].mean(0))
        boot = np.empty(B)
        for b in range(B):
            s = idx[rng.integers(0, idx.size, idx.size)]
            boot[b] = spearmanr(mae[s].mean(0), cost[s].mean(0)).correlation
        lo, hi_ = np.percentile(boot, [2.5, 97.5])
        out[name] = {"n": int(idx.size), "rho": float(rho), "p": float(p),
                     "lo": float(lo), "hi": float(hi_),
                     "excludes_zero": bool(lo > 0 or hi_ < 0)}
    taus = per_sku_kendall_tau(mae, cost)
    pos = d["TEST_DMD"].to_numpy() > 0
    out["tau_demand_active"] = float(np.nanmean(taus[pos]))
    out["tau_pooled"] = float(np.nanmean(taus))
    return out


def ci_str(g):
    return f"[{g['lo']:+.3f},{g['hi']:+.3f}]"


def main():
    cfg = CostConfig()
    crossover = cfg.stockout_unit_cost / cfg.holding_per_unit_per_qtr

    # ---- 1. how far does the terminal update move the forecasts? ----------
    sample = (pl.scan_parquet(S / "sample_skus.parquet")
              .select(["STOCK_ITEM_NUMBER", "FORECAST_ID", "SBA_ALPHA"])
              .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect())
    panel = (pl.scan_parquet(S / "panel.parquet")
             .select(["STOCK_ITEM_NUMBER", "FORECAST_ID"] + AC)
             .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect())
    df = (sample.join(panel, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner")
                .sort(["STOCK_ITEM_NUMBER", "FORECAST_ID"]))
    train = df.select(AC).to_numpy().astype(np.float64)[:, :16]
    alpha = df["SBA_ALPHA"].cast(pl.Float64).fill_null(0.1).to_numpy()
    fired = train[:, -1] > 0

    shift = {}
    pairs = [("SBA", sba_fn(train, alpha).forecast[:, -1], sba_next(train, alpha)),
             ("CROSTON", croston(train, alpha, sba=False).forecast[:, -1],
              croston_next(train, alpha, sba=False)),
             ("TSB", tsb(train, alpha).forecast[:, -1], tsb_next(train, alpha))]
    for name, old, new in pairs:
        dv = new - old
        shift[name] = {"share_moved": float((np.abs(dv) > 1e-12).mean()),
                       "mean_abs_shift": float(np.abs(dv).mean()),
                       "mean_signed_shift": float(dv.mean()),
                       "mean_abs_shift_if_demand_fired": float(np.abs(dv[fired]).mean())}

    # ---- 2. diff the two ledgers ------------------------------------------
    base = json.loads((ROB / "clean_full.json").read_text(encoding="utf-8"))
    equal = json.loads((ROB / "clean_full_equalinfo.json").read_text(encoding="utf-8"))
    models = list(base["per_model"])
    per_model = {}
    for m in models:
        per_model[m] = {"mae_base": base["per_model"][m]["mae"],
                        "mae_equal": equal["per_model"][m]["mae"],
                        "cost_base": base["per_model"][m]["cost"],
                        "cost_equal": equal["per_model"][m]["cost"]}
    unchanged = [m for m in models
                 if per_model[m]["mae_base"] == per_model[m]["mae_equal"]
                 and per_model[m]["cost_base"] == per_model[m]["cost_equal"]]

    # ---- 3. re-derive the headline statistics on both arms ----------------
    d_base, mb = load_per_sku("")
    d_eq, me = load_per_sku("_equalinfo")
    assert mb == me, "model columns diverged between the two arms"
    reg_base = regime_stats(d_base, mb, crossover)
    reg_eq = regime_stats(d_eq, me, crossover)

    res = {"crossover_price_usd": float(crossover), "B": B, "seed": SEED,
           "affected_models": ["SBA", "CROSTON", "TSB"],
           "unchanged_models": unchanged,
           "share_demand_in_final_quarter": float(fired.mean()),
           "forecast_shift": shift, "per_model": per_model,
           "regime": {"base": reg_base, "equal_info": reg_eq},
           "deployed": {"gap_pct_base": base["deployed"]["gap_pct"],
                        "gap_pct_equal": equal["deployed"]["gap_pct"],
                        "best_base": base["deployed"]["best_model"],
                        "best_equal": equal["deployed"]["best_model"]},
           "rank_cost_base": sorted(models, key=lambda m: per_model[m]["cost_base"]),
           "rank_cost_equal": sorted(models, key=lambda m: per_model[m]["cost_equal"]),
           "rank_mae_base": sorted(models, key=lambda m: per_model[m]["mae_base"]),
           "rank_mae_equal": sorted(models, key=lambda m: per_model[m]["mae_equal"])}

    L = ["EQUAL INFORMATION SETS: SBA / CROSTON / TSB GIVEN THE FINAL QUARTER",
         "=" * 78, "",
         "Croston, SBA and TSB smooth from actuals[t-1], so reading forecast[:, -1]",
         "on a 16-quarter window produced a forecast whose state had consumed only",
         "15 quarters. The other eight forecasters condition on the full window.",
         "The terminal update runs one more iteration of the same recurrence.", "",
         "1. HOW FAR THE FORECAST MOVES",
         f"   {100 * fired.mean():.1f} % of SKUs had demand in the final training quarter.",
         f"   {'MODEL':<10}{'% moved':>10}{'mean |shift|':>15}"
         f"{'mean shift':>13}{'|shift| if fired':>19}"]
    for m in shift:
        v = shift[m]
        L.append(f"   {m:<10}{100 * v['share_moved']:>9.1f}%{v['mean_abs_shift']:>15.4f}"
                 f"{v['mean_signed_shift']:>+13.4f}"
                 f"{v['mean_abs_shift_if_demand_fired']:>19.4f}")

    L += ["", "2. EFFECT ON THE MODEL COMPARISON",
          f"   {'MODEL':<11}{'MAE base':>10}{'MAE eq':>10}{'d %':>8}   "
          f"{'cost base':>10}{'cost eq':>10}{'d %':>8}"]
    for m in models:
        v = per_model[m]
        dm = 100 * (v["mae_equal"] - v["mae_base"]) / v["mae_base"]
        dc = 100 * (v["cost_equal"] - v["cost_base"]) / v["cost_base"]
        tag = "" if m in unchanged else "   <-- affected"
        L.append(f"   {m:<11}{v['mae_base']:>10.3f}{v['mae_equal']:>10.3f}{dm:>+8.2f}   "
                 f"{v['cost_base']:>10,.0f}{v['cost_equal']:>10,.0f}{dc:>+8.2f}{tag}")
    L += ["", f"   {len(unchanged)} of {len(models)} models are bit-identical across the",
          "   two arms, which is the check that the patch touched nothing else."]

    L += ["", "3. HEADLINE STATISTICS, BOTH ARMS",
          f"   {'STATISTIC':<26}{'base':>14}{'equal-info':>16}"]
    for k, lab in [("all", "pooled rho"),
                   ("stockout_dominated", "stockout-dominated rho"),
                   ("holding_dominated", "holding-dominated rho")]:
        L.append(f"   {lab:<26}{reg_base[k]['rho']:>+14.3f}{reg_eq[k]['rho']:>+16.3f}")
        L.append(f"   {'  95 % CI':<26}{ci_str(reg_base[k]):>14}{ci_str(reg_eq[k]):>16}")
    L += [f"   {'tau (demand-active)':<26}{reg_base['tau_demand_active']:>+14.3f}"
          f"{reg_eq['tau_demand_active']:>+16.3f}",
          f"   {'tau (pooled)':<26}{reg_base['tau_pooled']:>+14.3f}"
          f"{reg_eq['tau_pooled']:>+16.3f}",
          f"   {'deployed gap %':<26}{base['deployed']['gap_pct']:>+14.2f}"
          f"{equal['deployed']['gap_pct']:>+16.2f}",
          f"   {'cheapest on overlap':<26}{base['deployed']['best_model']:>14}"
          f"{equal['deployed']['best_model']:>16}"]

    L += ["", "4. RANKINGS",
          f"   cost, base       : {', '.join(res['rank_cost_base'])}",
          f"   cost, equal-info : {', '.join(res['rank_cost_equal'])}",
          f"   MAE,  base       : {', '.join(res['rank_mae_base'])}",
          f"   MAE,  equal-info : {', '.join(res['rank_mae_equal'])}"]

    rep = "\n".join(L)
    (ROB / "information_set.txt").write_text(rep, encoding="utf-8")
    (ROB / "information_set.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(rep)


if __name__ == "__main__":
    main()
