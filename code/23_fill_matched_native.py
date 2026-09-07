"""
Stage-V2 audit fix: fill-matched native-quantile policy comparison.

The unmatched native-quantile result (18_run_simulator_v2.py) confounds a
cost reduction with a service-level reduction: native-quantile Chronos
ran 61 % cheaper but at 5.8 pp lower fill on demand-active SKUs. That is
a move along the cost-service frontier, not evidence that one policy
formula dominates.

This script makes the comparison apples-to-apples:

  For each of the 4 quantile-emitting models (ZIP, HNB, LGBM, Chronos):
    1. Simulate the normal-approximation policy -> record fleet fill F*.
    2. Build the native-quantile reorder point s_native (model's own
       predictive quantile of lead-time demand, default SL tier).
    3. Bisection-search a scalar safety-stock multiplier kappa >= 0 so
       the native-quantile policy hits fleet fill = F* (within 1e-3).
       Fill is monotone increasing in kappa, so bisection is valid.
    4. Report cost of the fill-matched native-quantile policy vs the
       normal-policy cost, AT THE SAME FLEET FILL.

Interpretation: the predictive distribution determines the *cross-SKU
allocation* of safety stock; kappa sets the *fleet-wide service level*.
Matching F* isolates the allocation effect — the genuine
'policy-formula' question.

If fill-matched native-quantile is still cheaper -> finding (2) holds.
If the saving evaporates -> finding (2) was a service-level artefact and
we report that.

Output:
  output/sample/segment/robustness/fill_matched_native.txt
  output/sample/segment/sim_results_native_matched.parquet
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl

import config as C
from simulator import (
    CostConfig, normal_policy, simulate, lead_days_to_qtrs, service_level_for,
)
from policy_native_quantile import native_quantile_policy

SAMPLE_DIR = C.SAMPLE_DIR
ROB_DIR = SAMPLE_DIR / "robustness"
ROB_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_END = 20
TEST_LEN = 28 - TRAIN_END
ACTUAL_COLS = [f"ACTUALS_QTR{i:02d}" for i in range(1, 29)]
QUANT_MODELS = ("ZIP", "HURDLE_NB", "LGBM", "CHRONOS")


def load_panel():
    sample = pl.scan_parquet(SAMPLE_DIR / "sample_skus.parquet").select([
        "STOCK_ITEM_NUMBER", "FORECAST_ID",
        "SEGMENTATION_GRP", "ABC_GRP", "CRITICALITY_MODE",
        "UNIT_PRICE_USD", "LEAD_TIME_MEAN",
    ]).unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect()
    panel = pl.scan_parquet(SAMPLE_DIR / "panel.parquet").select(
        ["STOCK_ITEM_NUMBER", "FORECAST_ID"] + ACTUAL_COLS
    ).unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect()
    return sample.join(panel, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner")


def collect_quant_models(panel):
    forecasts = {}
    fc_q21 = f"FORECAST_QTR{TRAIN_END + 1:02d}"
    for path in [SAMPLE_DIR / "forecasts_distributional.parquet",
                 SAMPLE_DIR / "forecasts_lgbm.parquet",
                 SAMPLE_DIR / "forecasts_chronos.parquet"]:
        if not path.exists():
            continue
        df = pl.read_parquet(path)
        for m in df["MODEL"].unique().to_list():
            if m not in QUANT_MODELS:
                continue
            mdf = df.filter(pl.col("MODEL") == m).unique(
                subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"])
            cols = ["STOCK_ITEM_NUMBER", "FORECAST_ID", "MAE_TEST", fc_q21]
            for q in (50, 80, 90, 95, 99):
                if f"Q_{q:02d}" in mdf.columns:
                    cols.append(f"Q_{q:02d}")
            j = panel.select(["STOCK_ITEM_NUMBER", "FORECAST_ID"]).join(
                mdf.select(cols), on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="left")
            entry = {"fc_q21": j[fc_q21].fill_null(0.0).to_numpy()}
            for q in (50, 80, 90, 95, 99):
                c = f"Q_{q:02d}"
                if c in j.columns:
                    entry[f"q{q}"] = j[c].fill_null(0.0).to_numpy()
            forecasts[m] = entry
    return forecasts


def main():
    t0 = time.time()
    cfg = CostConfig()
    panel = load_panel()
    N = panel.height
    print(f"Panel: {N:,}   SEGMENT_MODE={C.SEGMENT_MODE}")

    actuals = panel.select(ACTUAL_COLS).to_numpy().astype(np.float64)
    train = actuals[:, :TRAIN_END]
    test = actuals[:, TRAIN_END:]
    test_dmd = test.sum(axis=1)
    pos = test_dmd > 0

    unit_price = panel["UNIT_PRICE_USD"].fill_null(1.0).cast(pl.Float64).to_numpy()
    unit_price = np.where(unit_price > 0, unit_price, 1.0)
    lead_days = panel["LEAD_TIME_MEAN"].fill_null(60.0).cast(pl.Float64).to_numpy()
    lead_days = np.where(lead_days > 0, lead_days, 60.0)
    lead_qtrs = lead_days_to_qtrs(lead_days)
    lead_std_qtrs = (lead_days * 0.3) / 91.3125
    sigma = train.std(axis=1, ddof=0)
    crit = panel["CRITICALITY_MODE"].fill_null("Normal").to_list()
    sl = np.array([service_level_for(c, cfg) for c in crit])

    forecasts = collect_quant_models(panel)
    print(f"Quantile models: {list(forecasts)}\n")

    def run(s, S):
        return simulate(actuals_test=test, s=s, S=S, lead_qtrs=lead_qtrs,
                        unit_price=unit_price, cfg=cfg)

    lines = ["Fill-matched native-quantile policy comparison",
             "=" * 68,
             "Each native-quantile policy is scaled (safety-stock multiplier",
             "kappa) so its fleet fill rate matches the model's own",
             "normal-approximation policy. Cost is then compared at equal fill.",
             ""]
    rows = []
    for name, m in forecasts.items():
        # --- normal policy ---
        pol_n = normal_policy(
            forecast_mean_per_qtr=m["fc_q21"], forecast_std_per_qtr=sigma,
            lead_qtrs=lead_qtrs.astype(np.float64), lead_std_qtrs=lead_std_qtrs,
            unit_price=unit_price, service_level=sl, cfg=cfg)
        sim_n = run(pol_n["s"], pol_n["S"])
        target_fill = float(sim_n["fill_rate"].mean())

        # --- native base policy ---
        qarr = {0.50: m.get("q50", m["fc_q21"]), 0.80: m.get("q80", m["fc_q21"]),
                0.90: m.get("q90", m["fc_q21"]), 0.95: m.get("q95", m["fc_q21"]),
                0.99: m.get("q99", m["fc_q21"])}
        pol_v = native_quantile_policy(
            quantile_arr=qarr, forecast_mean=m["fc_q21"],
            lead_qtrs=lead_qtrs.astype(np.float64), unit_price=unit_price,
            service_level=sl, cfg=cfg)
        s_base, eoq = pol_v["s"], pol_v["eoq"]

        # --- bisection on kappa: fill(kappa) is monotone increasing ---
        lo, hi = 0.0, 8.0
        # ensure hi brackets the target
        f_hi = run(hi * s_base, hi * s_base + eoq)["fill_rate"].mean()
        matched = True
        if f_hi < target_fill:
            kappa = hi
            matched = False         # cannot reach target even at kappa=8
        else:
            for _ in range(45):
                mid = 0.5 * (lo + hi)
                f = run(mid * s_base, mid * s_base + eoq)["fill_rate"].mean()
                if f < target_fill:
                    lo = mid
                else:
                    hi = mid
            kappa = 0.5 * (lo + hi)

        s_m = kappa * s_base
        sim_m = run(s_m, s_m + eoq)
        fill_m = float(sim_m["fill_rate"].mean())

        cost_n_all = float(sim_n["total_cost"].mean())
        cost_m_all = float(sim_m["total_cost"].mean())
        cost_n_pos = float(sim_n["total_cost"][pos].mean())
        cost_m_pos = float(sim_m["total_cost"][pos].mean())
        fill_n_pos = float(sim_n["fill_rate"][pos].mean())
        fill_m_pos = float(sim_m["fill_rate"][pos].mean())

        rows.append({
            "MODEL": name, "kappa": kappa, "matched": matched,
            "target_fill": target_fill, "native_matched_fill": fill_m,
            "cost_normal_all": cost_n_all, "cost_native_all": cost_m_all,
            "delta_all_pct": 100 * (cost_m_all - cost_n_all) / cost_n_all,
            "cost_normal_pos": cost_n_pos, "cost_native_pos": cost_m_pos,
            "delta_pos_pct": 100 * (cost_m_pos - cost_n_pos) / cost_n_pos,
            "fill_normal_pos": fill_n_pos, "fill_native_pos": fill_m_pos,
        })
        flag = "" if matched else "  [UNMATCHED: kappa hit cap]"
        print(f"  {name:<10} kappa={kappa:5.2f}  fill {target_fill:.4f} -> "
              f"{fill_m:.4f}{flag}")
        print(f"             cost ALL  ${cost_n_all:>8,.0f} -> ${cost_m_all:>8,.0f}"
              f"  ({100*(cost_m_all-cost_n_all)/cost_n_all:+.1f}%)")
        print(f"             cost POS  ${cost_n_pos:>8,.0f} -> ${cost_m_pos:>8,.0f}"
              f"  ({100*(cost_m_pos-cost_n_pos)/cost_n_pos:+.1f}%)   "
              f"fill_pos {fill_n_pos:.3f} -> {fill_m_pos:.3f}")

    df = pl.DataFrame(rows)
    df.write_parquet(SAMPLE_DIR / "sim_results_native_matched.parquet",
                     compression="zstd")

    lines.append(df.to_pandas().round(3).to_string(index=False))
    lines.append("")
    lines.append("Reading: delta_*_pct < 0 means the fill-matched native-quantile")
    lines.append("policy is cheaper than the normal policy AT EQUAL FLEET FILL.")
    lines.append("If so, finding (2) holds: the predictive distribution allocates")
    lines.append("safety stock more cost-efficiently than the model-agnostic sigma.")
    (ROB_DIR / "fill_matched_native.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {ROB_DIR / 'fill_matched_native.txt'}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
