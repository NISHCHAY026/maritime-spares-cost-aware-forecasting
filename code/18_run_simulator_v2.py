"""
Stage-5 v2: re-run the cost simulator with two methodological fixes from
the v0.5 critique:

  (a) Uniform sigma_D proxy across all 11 forecasters: training-window
      actuals std is used for every model identically. The mixed
      proxy (residual-std for some + quantile-derived for others) used
      in robustness_v2 is dropped — it produced a 'sigma proxy
      determines the cost winner' artefact that the revised paper
      handles by simply standardising on one proxy and treating
      proxy choice as a design decision documented in §4.

  (b) Native-quantile (s,S) policy variant for the 4 models that emit
      predictive quantiles (ZIP, HNB, LGBM, Chronos). For each SKU we
      derive s = lead_qtrs * Q_alpha(demand) where alpha = SL target,
      then compare cost under this policy to cost under the
      normal-approximation policy.

Output:
  output/sample/segment/sim_results.parquet              normal-policy costs
  output/sample/segment/sim_results_native.parquet       native-quantile costs (4 models)
  output/sample/segment/sim_summary_v2.txt
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
TRAIN_END = 20
TEST_LEN  = 28 - TRAIN_END
ACTUAL_COLS = [f"ACTUALS_QTR{i:02d}" for i in range(1, 29)]


def load_panel():
    sample = pl.scan_parquet(SAMPLE_DIR / "sample_skus.parquet").select([
        "STOCK_ITEM_NUMBER", "FORECAST_ID",
        "SEGMENTATION_GRP", "ABC_GRP", "STRATEGY_GRP",
        "VELOCITY_MODE", "CRITICALITY_MODE",
        "UNIT_PRICE_USD", "LEAD_TIME_MEAN",
    ]).unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect()
    panel = pl.scan_parquet(SAMPLE_DIR / "panel.parquet").select(
        ["STOCK_ITEM_NUMBER", "FORECAST_ID"] + ACTUAL_COLS
    ).unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect()
    return sample.join(panel, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner")


def collect_models(panel):
    """
    Returns dict {MODEL: {fc_q21, mae, q50, q80, q90, q95, q99}}.
    Quantile columns missing for classical models.
    """
    sources = [
        SAMPLE_DIR / "forecasts_classical.parquet",
        SAMPLE_DIR / "forecasts_distributional.parquet",
        SAMPLE_DIR / "forecasts_lgbm.parquet",
        SAMPLE_DIR / "forecasts_chronos.parquet",
    ]
    forecasts = {}
    fc_col_q21 = f"FORECAST_QTR{TRAIN_END + 1:02d}"
    for path in sources:
        if not path.exists():
            continue
        df = pl.read_parquet(path)
        for m in df["MODEL"].unique().to_list():
            mdf = (
                df.filter(pl.col("MODEL") == m)
                  .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"])
            )
            sel_cols = ["STOCK_ITEM_NUMBER", "FORECAST_ID", "MAE_TEST", fc_col_q21]
            for q in (50, 80, 90, 95, 99):
                if f"Q_{q:02d}" in mdf.columns:
                    sel_cols.append(f"Q_{q:02d}")
            joined = panel.select(["STOCK_ITEM_NUMBER", "FORECAST_ID"]).join(
                mdf.select(sel_cols),
                on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="left"
            )
            entry = {
                "mae":    joined["MAE_TEST"].fill_null(0.0).to_numpy(),
                "fc_q21": joined[fc_col_q21].fill_null(0.0).to_numpy(),
            }
            for q in (50, 80, 90, 95, 99):
                col = f"Q_{q:02d}"
                if col in joined.columns:
                    entry[f"q{q:02d}"] = joined[col].fill_null(0.0).to_numpy()
            forecasts[m] = entry
    return forecasts


def main():
    t0 = time.time()
    cfg = CostConfig()
    panel = load_panel()
    print(f"Panel rows: {panel.height:,}    SEGMENT_MODE={C.SEGMENT_MODE}")

    actuals = panel.select(ACTUAL_COLS).to_numpy().astype(np.float64)
    train = actuals[:, :TRAIN_END]
    test  = actuals[:, TRAIN_END:]
    unit_price = panel["UNIT_PRICE_USD"].fill_null(1.0).cast(pl.Float64).to_numpy()
    unit_price = np.where(unit_price > 0, unit_price, 1.0)
    lead_days = panel["LEAD_TIME_MEAN"].fill_null(60.0).cast(pl.Float64).to_numpy()
    lead_days = np.where(lead_days > 0, lead_days, 60.0)
    lead_qtrs = lead_days_to_qtrs(lead_days)
    lead_std_qtrs = (lead_days * 0.3) / 91.3125
    sigma = train.std(axis=1, ddof=0)            # uniform sigma proxy
    crit = panel["CRITICALITY_MODE"].fill_null("Normal").to_list()
    sl = np.array([service_level_for(c, cfg) for c in crit])

    forecasts = collect_models(panel)
    print(f"Models: {list(forecasts)}\n")

    keys = panel.select([
        "STOCK_ITEM_NUMBER", "FORECAST_ID",
        "SEGMENTATION_GRP", "ABC_GRP",
        "STRATEGY_GRP", "VELOCITY_MODE", "CRITICALITY_MODE",
    ])

    # ----- Normal-policy simulation (for all models) -----
    print("=== Normal-approximation policy (uniform sigma) ===")
    rows_normal = []
    for name, m in forecasts.items():
        pol = normal_policy(
            forecast_mean_per_qtr=m["fc_q21"],
            forecast_std_per_qtr=sigma,
            lead_qtrs=lead_qtrs.astype(np.float64),
            lead_std_qtrs=lead_std_qtrs,
            unit_price=unit_price,
            service_level=sl,
            cfg=cfg,
        )
        sim = simulate(
            actuals_test=test, s=pol["s"], S=pol["S"],
            lead_qtrs=lead_qtrs, unit_price=unit_price, cfg=cfg,
        )
        block = keys.with_columns([
            pl.lit(name).alias("MODEL"),
            pl.Series("S_LO",          pol["s"]),
            pl.Series("S_HI",          pol["S"]),
            pl.Series("EOQ",           pol["eoq"]),
            pl.Series("HOLDING_COST",  sim["holding_cost"]),
            pl.Series("ORDERING_COST", sim["ordering_cost"]),
            pl.Series("STOCKOUT_UNITS", sim["stockout_units"]),
            pl.Series("STOCKOUT_COST", sim["stockout_cost"]),
            pl.Series("TOTAL_COST",    sim["total_cost"]),
            pl.Series("FILL_RATE",     sim["fill_rate"]),
            pl.Series("N_ORDERS",      sim["n_orders"]),
            pl.Series("MEAN_ON_HAND",  sim["mean_on_hand"]),
        ])
        rows_normal.append(block)
        print(f"  {name:<10}  med_total=${np.median(sim['total_cost']):>9,.2f}   "
              f"mean=${sim['total_cost'].mean():>9,.2f}   "
              f"fill={sim['fill_rate'].mean():.4f}")

    out_normal = pl.concat(rows_normal, how="vertical_relaxed")
    out_normal.write_parquet(SAMPLE_DIR / "sim_results.parquet", compression="zstd")
    print(f"\nWrote {SAMPLE_DIR / 'sim_results.parquet'}  ({out_normal.height:,} rows)")

    # ----- Native-quantile policy (for the 4 quantile-emitting models) -----
    print("\n=== Native-quantile policy (4 distributional/ML models) ===")
    rows_native = []
    for name, m in forecasts.items():
        if "q90" not in m:
            continue
        # Build the per-SKU service-level lookup using the model's own quantiles
        quantile_arr = {
            0.50: m.get("q50", m["fc_q21"]),
            0.80: m.get("q80", m["fc_q21"]),
            0.90: m.get("q90", m["fc_q21"]),
            0.95: m.get("q95", m["fc_q21"]),
            0.99: m.get("q99", m["fc_q21"]),
        }
        pol = native_quantile_policy(
            quantile_arr=quantile_arr,
            forecast_mean=m["fc_q21"],
            lead_qtrs=lead_qtrs.astype(np.float64),
            unit_price=unit_price,
            service_level=sl,
            cfg=cfg,
        )
        sim = simulate(
            actuals_test=test, s=pol["s"], S=pol["S"],
            lead_qtrs=lead_qtrs, unit_price=unit_price, cfg=cfg,
        )
        block = keys.with_columns([
            pl.lit(name).alias("MODEL"),
            pl.lit("native").alias("POLICY"),
            pl.Series("S_LO",          pol["s"]),
            pl.Series("S_HI",          pol["S"]),
            pl.Series("EOQ",           pol["eoq"]),
            pl.Series("HOLDING_COST",  sim["holding_cost"]),
            pl.Series("ORDERING_COST", sim["ordering_cost"]),
            pl.Series("STOCKOUT_UNITS", sim["stockout_units"]),
            pl.Series("STOCKOUT_COST", sim["stockout_cost"]),
            pl.Series("TOTAL_COST",    sim["total_cost"]),
            pl.Series("FILL_RATE",     sim["fill_rate"]),
            pl.Series("N_ORDERS",      sim["n_orders"]),
        ])
        rows_native.append(block)
        print(f"  {name:<10}  med_total=${np.median(sim['total_cost']):>9,.2f}   "
              f"mean=${sim['total_cost'].mean():>9,.2f}   "
              f"fill={sim['fill_rate'].mean():.4f}")

    if rows_native:
        out_native = pl.concat(rows_native, how="vertical_relaxed")
        out_native.write_parquet(SAMPLE_DIR / "sim_results_native.parquet",
                                  compression="zstd")
        print(f"\nWrote {SAMPLE_DIR / 'sim_results_native.parquet'}  "
              f"({out_native.height:,} rows)")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
