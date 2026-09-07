"""
Stage-5: run the (s, S) cost simulator for every model and compute
accuracy-vs-cost rank divergence.

For each model (and the deployed policy where Stockmax data exists):
  * compute (s, S) per SKU from the model's point forecast and a
    forecast-error std,
  * simulate 8 quarters under that policy,
  * record cost components and fill-rate per SKU.

Then aggregate to fleet level and compare to the deployed policy
(reconstructed from Stockmax New Min / New Max columns).

Output:
  output/sample/sim_results.parquet   one row per (SKU, model)
  output/sample/sim_summary.txt       fleet-level cost-by-model + correlation
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl

import config as C
from simulator import (
    CostConfig, normal_policy, simulate, lead_days_to_qtrs, service_level_for,
)


SAMPLE_DIR = C.SAMPLE_DIR
TRAIN_END = 20
TEST_LEN  = 28 - TRAIN_END
ACTUAL_COLS = [f"ACTUALS_QTR{i:02d}" for i in range(1, 29)]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

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


def collect_model_forecasts(panel: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """
    Return {model_name -> per-SKU frame with FORECAST_QTR21..28} for every
    forecast set we've produced in Stage-4.
    """
    sources = [
        ("classical",      SAMPLE_DIR / "forecasts_classical.parquet"),
        ("distributional", SAMPLE_DIR / "forecasts_distributional.parquet"),
        ("lgbm",           SAMPLE_DIR / "forecasts_lgbm.parquet"),
        ("chronos",        SAMPLE_DIR / "forecasts_chronos.parquet"),
    ]
    forecasts = {}
    for tag, path in sources:
        if not path.exists():
            continue
        df = pl.scan_parquet(path).collect()
        for model in df["MODEL"].unique().to_list():
            mdf = (
                df.filter(pl.col("MODEL") == model)
                  .select(
                      ["STOCK_ITEM_NUMBER", "FORECAST_ID", "MODEL"]
                      + [f"FORECAST_QTR{TRAIN_END + 1 + q:02d}" for q in range(TEST_LEN)]
                  )
                  .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"])
            )
            forecasts[model] = mdf
    return forecasts


def model_forecast_std(actuals_train: np.ndarray, model_name: str,
                        forecast_test: np.ndarray) -> np.ndarray:
    """
    Per-SKU forecast-error std proxy. We use training-window dispersion
    (std of training actuals) for the classical models since they don't
    expose an internal sigma; same for ZIP/HNB whose MoM variances we
    already used implicitly. For LightGBM with quantile heads we'd
    prefer (Q90-Q10)/2.56 — kept as future refinement; here we use
    training-actuals std for parity across models.
    """
    return actuals_train.std(axis=1, ddof=0)


# ---------------------------------------------------------------------------
# Simulate one model
# ---------------------------------------------------------------------------

def simulate_one_model(
    model: str,
    forecast_q21: np.ndarray,    # (N,) point forecast for first test quarter
    actuals_train: np.ndarray,
    actuals_test: np.ndarray,
    lead_qtrs: np.ndarray,
    lead_std_qtrs: np.ndarray,
    unit_price: np.ndarray,
    service_level: np.ndarray,
    cfg: CostConfig,
) -> dict:
    sigma = model_forecast_std(actuals_train, model, forecast_q21)
    pol = normal_policy(
        forecast_mean_per_qtr=forecast_q21,
        forecast_std_per_qtr=sigma,
        lead_qtrs=lead_qtrs.astype(np.float64),
        lead_std_qtrs=lead_std_qtrs,
        unit_price=unit_price,
        service_level=service_level,
        cfg=cfg,
    )
    sim = simulate(
        actuals_test=actuals_test,
        s=pol["s"], S=pol["S"],
        lead_qtrs=lead_qtrs, unit_price=unit_price, cfg=cfg,
    )
    return {**sim, **pol}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    cfg = CostConfig()
    print(f"Cost config: holding={cfg.holding_rate_annual:.2%}/yr  "
          f"K=${cfg.ordering_cost:.0f}  stockout=${cfg.stockout_unit_cost:.0f}/unit-qtr")
    print(f"Service-level tiers: critical={cfg.sl_critical}  "
          f"normal={cfg.sl_normal}  low={cfg.sl_low}\n")

    panel = load_panel()
    N = panel.height
    print(f"Panel rows: {N:,}")

    actuals = panel.select(ACTUAL_COLS).to_numpy().astype(np.float64)
    actuals_train = actuals[:, :TRAIN_END]
    actuals_test  = actuals[:, TRAIN_END:]

    unit_price = panel["UNIT_PRICE_USD"].fill_null(1.0).cast(pl.Float64).to_numpy()
    unit_price = np.where(unit_price > 0, unit_price, 1.0)
    lead_days  = panel["LEAD_TIME_MEAN"].fill_null(60.0).cast(pl.Float64).to_numpy()
    lead_days  = np.where(lead_days > 0, lead_days, 60.0)
    lead_qtrs  = lead_days_to_qtrs(lead_days)
    # Lead-time variability: assume CV=0.3 of mean for now (paper sensitivity
    # analysis later); converted to quarter units.
    lead_std_qtrs = (lead_days * 0.3) / 91.3125

    # Service level per SKU from criticality
    crit = panel["CRITICALITY_MODE"].fill_null("Normal").to_list()
    service_level = np.array([service_level_for(c, cfg) for c in crit])

    forecasts = collect_model_forecasts(panel)
    print(f"Models found: {list(forecasts)}\n")

    rows = []
    for model, fdf in forecasts.items():
        # Align to panel order
        joined = panel.select(["STOCK_ITEM_NUMBER", "FORECAST_ID"]).join(
            fdf, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="left",
        )
        if joined.height != N:
            print(f"  WARNING: {model} has {joined.height} rows, expected {N}")

        forecast_q21 = joined[f"FORECAST_QTR{TRAIN_END + 1:02d}"].fill_null(0.0).to_numpy()

        t1 = time.time()
        out = simulate_one_model(
            model, forecast_q21, actuals_train, actuals_test,
            lead_qtrs=lead_qtrs, lead_std_qtrs=lead_std_qtrs,
            unit_price=unit_price, service_level=service_level, cfg=cfg,
        )
        block = panel.select(["STOCK_ITEM_NUMBER", "FORECAST_ID",
                              "SEGMENTATION_GRP", "ABC_GRP",
                              "STRATEGY_GRP", "VELOCITY_MODE", "CRITICALITY_MODE"])
        block = block.with_columns([
            pl.lit(model).alias("MODEL"),
            pl.Series("S_LO",          out["s"]),
            pl.Series("S_HI",          out["S"]),
            pl.Series("EOQ",           out["eoq"]),
            pl.Series("HOLDING_COST",  out["holding_cost"]),
            pl.Series("ORDERING_COST", out["ordering_cost"]),
            pl.Series("STOCKOUT_UNITS", out["stockout_units"]),
            pl.Series("STOCKOUT_COST", out["stockout_cost"]),
            pl.Series("TOTAL_COST",    out["total_cost"]),
            pl.Series("FILL_RATE",     out["fill_rate"]),
            pl.Series("N_ORDERS",      out["n_orders"]),
            pl.Series("MEAN_ON_HAND",  out["mean_on_hand"]),
        ])
        rows.append(block)
        print(f"  {model:<10}  {time.time() - t1:5.2f}s   "
              f"med_total=${np.median(out['total_cost']):>9,.2f}   "
              f"mean_total=${out['total_cost'].mean():>9,.2f}   "
              f"med_fill={np.median(out['fill_rate']):.4f}   "
              f"sum_stockout_units={out['stockout_units'].sum():,.0f}")

    out_df = pl.concat(rows, how="vertical_relaxed")
    out_path = SAMPLE_DIR / "sim_results.parquet"
    out_df.write_parquet(out_path, compression="zstd")
    print(f"\nWrote {out_path}  ({out_df.height:,} rows)")

    # ---------------------------------------------------------------------
    # Aggregations + accuracy-vs-cost comparison
    # ---------------------------------------------------------------------
    print("\nFleet-level cost by model (mean across SKUs):")
    fleet = (
        out_df.group_by("MODEL")
              .agg([
                  pl.len().alias("n"),
                  pl.col("HOLDING_COST").mean().alias("hold_mean"),
                  pl.col("ORDERING_COST").mean().alias("order_mean"),
                  pl.col("STOCKOUT_UNITS").mean().alias("stockout_units_mean"),
                  pl.col("STOCKOUT_COST").mean().alias("stockout_cost_mean"),
                  pl.col("TOTAL_COST").mean().alias("total_mean"),
                  pl.col("TOTAL_COST").median().alias("total_med"),
                  pl.col("FILL_RATE").mean().alias("fill_mean"),
              ])
              .sort("total_mean")
    )
    print(fleet.to_pandas().to_string(index=False))

    # Pull MAE per model for accuracy ranking
    mae_blocks = []
    for tag, path in [
        ("classical",      SAMPLE_DIR / "forecasts_classical.parquet"),
        ("distributional", SAMPLE_DIR / "forecasts_distributional.parquet"),
        ("lgbm",           SAMPLE_DIR / "forecasts_lgbm.parquet"),
        ("chronos",        SAMPLE_DIR / "forecasts_chronos.parquet"),
    ]:
        if path.exists():
            mae_blocks.append(
                pl.scan_parquet(path)
                  .group_by("MODEL")
                  .agg(pl.col("MAE_TEST").mean().alias("mae_mean"))
                  .collect()
            )
    mae_df = pl.concat(mae_blocks, how="vertical_relaxed").sort("mae_mean")

    print("\nMAE-rank vs Cost-rank divergence:")
    accuracy_rank = (
        mae_df.with_row_index("acc_rank")
              .with_columns(pl.col("acc_rank").cast(pl.Int32) + 1)
    )
    cost_rank = (
        fleet.with_row_index("cost_rank")
             .with_columns(pl.col("cost_rank").cast(pl.Int32) + 1)
             .select(["MODEL", "cost_rank", "total_mean", "fill_mean"])
    )
    joined = accuracy_rank.join(cost_rank, on="MODEL", how="inner").sort("acc_rank")
    print(joined.to_pandas().to_string(index=False))

    # Spearman rank correlation
    if joined.height >= 3:
        from scipy.stats import spearmanr
        rho, p = spearmanr(joined["acc_rank"].to_numpy(),
                            joined["cost_rank"].to_numpy())
        print(f"\nSpearman rho(acc_rank, cost_rank) = {rho:+.3f}  (p={p:.3f})")

    # Persist summary
    lines = ["Stage-5 simulator summary",
             "=" * 60,
             f"Cost config: holding={cfg.holding_rate_annual:.2%}/yr  "
             f"K=${cfg.ordering_cost:.0f}  stockout=${cfg.stockout_unit_cost:.0f}/unit-qtr",
             f"Service tiers: critical={cfg.sl_critical}  normal={cfg.sl_normal}  low={cfg.sl_low}",
             f"Panel rows: {N}",
             "",
             "Fleet-level cost by model (mean across SKUs):",
             fleet.to_pandas().to_string(index=False),
             "",
             "MAE-rank vs Cost-rank:",
             joined.to_pandas().to_string(index=False),
             ]
    if joined.height >= 3:
        lines.append(f"\nSpearman rho(acc_rank, cost_rank) = {rho:+.3f}  (p={p:.3f})")
    (SAMPLE_DIR / "sim_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
