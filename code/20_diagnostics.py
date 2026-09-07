"""
Stage-7 v2 diagnostics: three quick analyses to defend the paper against
specific reviewer attack lines.

  D1. Selection-bias check on the deployed-policy overlap subset.
      Compare the wide ∩ policy SKUs to the wide-only SKUs on
      Segmentation × ABC × velocity × lead time × price. If the overlap
      systematically differs from the wide population, the deployed-vs-
      models cost gap claim has to soften.

  D2. Chronos MAE win composition (issue 17).
      Decompose Chronos-vs-MA MAE delta into (a) SKUs where both
      predict zero, (b) SKUs where only Chronos predicts non-zero,
      (c) SKUs where only MA predicts non-zero, (d) both non-zero.
      Quantify how much of Chronos's edge is from non-zero predictions
      on heavy-demand SKUs vs from zero predictions matching zero
      actuals.

  D3. Velocity D sensitivity (issue 14).
      Re-run summary stats with D-marked SKUs excluded. Does the
      headline tau / cost ranking change?

Outputs:
  output/sample/segment/diagnostics_d1_overlap_bias.txt
  output/sample/segment/diagnostics_d2_chronos_zero.txt
  output/sample/segment/diagnostics_d3_velocity_d.txt
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl

import config as C


SAMPLE_DIR = C.SAMPLE_DIR
TRAIN_END = 20
TEST_LEN  = 28 - TRAIN_END
ACTUAL_COLS = [f"ACTUALS_QTR{i:02d}" for i in range(1, 29)]


def d1_overlap_bias():
    """Compare overlap (wide ∩ policy) to wide-only on observable attrs."""
    sample = pl.read_parquet(SAMPLE_DIR / "sample_skus.parquet").unique(
        subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]
    )
    policy = pl.read_parquet(SAMPLE_DIR / "policy_panel.parquet").unique(
        subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]
    )
    overlap_keys = sample.join(
        policy.select(["STOCK_ITEM_NUMBER", "FORECAST_ID"]),
        on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner"
    ).select(["STOCK_ITEM_NUMBER", "FORECAST_ID"])
    sample = sample.with_columns(
        pl.col("STOCK_ITEM_NUMBER").is_in(
            overlap_keys["STOCK_ITEM_NUMBER"].to_list()
        ).alias("IN_OVERLAP")
    )

    n_total = sample.height
    n_overlap = sample["IN_OVERLAP"].sum()
    print(f"\nD1 — overlap-vs-wide comparison")
    print(f"  wide sample: {n_total:,} SKUs")
    print(f"  overlap (wide ∩ policy): {n_overlap:,} SKUs ({100*n_overlap/n_total:.1f}%)")

    cmp_rows = []
    for col, dtype in [
        ("SEGMENTATION_GRP", "cat"),
        ("ABC_GRP", "cat"),
        ("STRATEGY_GRP", "cat"),
        ("VELOCITY_MODE", "cat"),
        ("CRITICALITY_MODE", "cat"),
        ("UNIT_PRICE_USD", "num"),
        ("LEAD_TIME_MEAN", "num"),
        ("USAGE_PER_YEAR_MEAN", "num"),
    ]:
        if col not in sample.columns:
            continue
        if dtype == "cat":
            tab = (
                sample.group_by(["IN_OVERLAP", col]).len()
                      .pivot(index=col, on="IN_OVERLAP", values="len")
                      .fill_null(0)
                      .sort(col)
            )
            cols_now = tab.columns
            true_col = "true" if "true" in cols_now else (
                "True" if "True" in cols_now else cols_now[-1])
            false_col = "false" if "false" in cols_now else (
                "False" if "False" in cols_now else cols_now[1])
            tab = tab.with_columns([
                (pl.col(true_col) / pl.col(true_col).sum()
                  * 100).alias("pct_overlap"),
                (pl.col(false_col) / pl.col(false_col).sum()
                  * 100).alias("pct_wide"),
            ])
            cmp_rows.append((col, tab))
        else:
            stats = (
                sample.group_by("IN_OVERLAP").agg([
                    pl.col(col).mean().alias("mean"),
                    pl.col(col).median().alias("median"),
                    pl.col(col).std(ddof=0).alias("std"),
                    pl.col(col).quantile(0.95).alias("p95"),
                ]).sort("IN_OVERLAP")
            )
            cmp_rows.append((col, stats))

    lines = ["D1 - Selection-bias check on overlap subset"]
    lines.append("=" * 60)
    lines.append(f"wide N={n_total:,}  overlap N={n_overlap:,}  "
                  f"({100*n_overlap/n_total:.1f}%)")
    lines.append("")
    for col, tab in cmp_rows:
        lines.append(f"--- {col} ---")
        lines.append(tab.to_pandas().to_string(index=False))
        lines.append("")
    text = "\n".join(lines)
    (SAMPLE_DIR / "diagnostics_d1_overlap_bias.txt").write_text(text, encoding="utf-8")
    print(text[:1200])


def d2_chronos_zero():
    """Decompose Chronos-vs-MA MAE delta into zero/non-zero buckets."""
    chronos_path = SAMPLE_DIR / "forecasts_chronos.parquet"
    classical_path = SAMPLE_DIR / "forecasts_classical.parquet"
    if not chronos_path.exists():
        print("\nD2 - skipped (no Chronos forecasts yet)")
        return

    chronos = pl.read_parquet(chronos_path).select(
        ["STOCK_ITEM_NUMBER", "FORECAST_ID",
         "MAE_TEST", f"FORECAST_QTR{TRAIN_END + 1:02d}",
         "SEGMENTATION_GRP", "ABC_GRP"]
    ).unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"])
    classical = pl.read_parquet(classical_path)
    ma = (classical.filter(pl.col("MODEL") == "MA")
                    .select(["STOCK_ITEM_NUMBER", "FORECAST_ID",
                            "MAE_TEST", f"FORECAST_QTR{TRAIN_END + 1:02d}"])
                    .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"])
                    .rename({"MAE_TEST": "MAE_MA",
                              f"FORECAST_QTR{TRAIN_END + 1:02d}": "FC_Q21_MA"}))
    chronos = chronos.rename({
        "MAE_TEST": "MAE_CHRONOS",
        f"FORECAST_QTR{TRAIN_END + 1:02d}": "FC_Q21_CHRONOS",
    })

    df = chronos.join(ma, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner")
    df = df.with_columns([
        (pl.col("FC_Q21_MA") < 0.5).alias("MA_zero"),
        (pl.col("FC_Q21_CHRONOS") < 0.5).alias("CHRONOS_zero"),
        (pl.col("MAE_MA") - pl.col("MAE_CHRONOS")).alias("MAE_DELTA"),
    ])

    bucket = (
        df.with_columns([
            pl.when(pl.col("MA_zero") & pl.col("CHRONOS_zero")).then(pl.lit("both_zero"))
              .when(pl.col("MA_zero") & ~pl.col("CHRONOS_zero")).then(pl.lit("only_chronos_nonzero"))
              .when(~pl.col("MA_zero") & pl.col("CHRONOS_zero")).then(pl.lit("only_ma_nonzero"))
              .otherwise(pl.lit("both_nonzero")).alias("bucket")
        ])
    )
    summary = (
        bucket.group_by("bucket").agg([
            pl.len().alias("n_skus"),
            pl.col("MAE_MA").mean().alias("mae_ma_mean"),
            pl.col("MAE_CHRONOS").mean().alias("mae_chronos_mean"),
            pl.col("MAE_DELTA").mean().alias("delta_mean"),
            pl.col("MAE_DELTA").sum().alias("delta_sum"),
        ]).sort("delta_sum", descending=True)
    )
    total_delta = float(df["MAE_DELTA"].sum())
    summary = summary.with_columns(
        (pl.col("delta_sum") / total_delta * 100).alias("pct_of_total_delta")
    )
    print("\nD2 - Chronos vs MA MAE delta decomposition")
    print(summary.to_pandas().to_string(index=False))
    text = ("D2 - Chronos vs MA MAE delta decomposition\n"
            "=" * 60 + "\n"
            f"Total fleet MAE delta (MA - Chronos) = {total_delta:.2f}\n"
            f"Per-SKU mean delta = {df['MAE_DELTA'].mean():.4f}\n\n"
            + summary.to_pandas().to_string(index=False))
    (SAMPLE_DIR / "diagnostics_d2_chronos_zero.txt").write_text(text, encoding="utf-8")


def d3_velocity_d():
    """Re-run summary stats with D-marked SKUs excluded."""
    sample = pl.read_parquet(SAMPLE_DIR / "sample_skus.parquet").unique(
        subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]
    )
    n_total = sample.height
    n_d = (sample["VELOCITY_MODE"] == "D").sum()
    print(f"\nD3 - Velocity D sensitivity")
    print(f"  total wide sample: {n_total:,}")
    print(f"  D-marked: {n_d:,} ({100*n_d/n_total:.2f}%)")

    sim_path = SAMPLE_DIR / "sim_results.parquet"
    if not sim_path.exists():
        print("  skipped (no sim_results yet)")
        return
    sim = pl.read_parquet(sim_path)
    sim = sim.join(
        sample.select(["STOCK_ITEM_NUMBER", "VELOCITY_MODE"]).unique(),
        on="STOCK_ITEM_NUMBER", how="left"
    )

    full = (sim.group_by("MODEL").agg([
        pl.col("TOTAL_COST").mean().alias("cost_full"),
        pl.col("FILL_RATE").mean().alias("fill_full"),
    ]).sort("cost_full"))

    no_d = (sim.filter(pl.col("VELOCITY_MODE") != "D")
              .group_by("MODEL").agg([
                  pl.col("TOTAL_COST").mean().alias("cost_no_d"),
                  pl.col("FILL_RATE").mean().alias("fill_no_d"),
              ]).sort("cost_no_d"))

    cmp = full.join(no_d, on="MODEL", how="inner")
    cmp = cmp.with_columns(
        ((pl.col("cost_no_d") - pl.col("cost_full"))
          / pl.col("cost_full") * 100).alias("delta_pct")
    ).sort("cost_full")
    print(cmp.to_pandas().to_string(index=False))
    text = ("D3 - Velocity D sensitivity\n"
            "=" * 60 + "\n"
            f"D-marked SKUs in wide sample: {n_d:,} of {n_total:,}\n\n"
            + cmp.to_pandas().to_string(index=False))
    (SAMPLE_DIR / "diagnostics_d3_velocity_d.txt").write_text(text, encoding="utf-8")


def main():
    t0 = time.time()
    d1_overlap_bias()
    d2_chronos_zero()
    d3_velocity_d()
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
