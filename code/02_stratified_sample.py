"""
Stage-2: build the analytical panel + stratified sample + train/val/test split.

What this produces
------------------
output/sample/
  panel.parquet            one row per (PARTTYPENO, BRAND, SHIP) with the
                            attributes needed for stratification + modelling
  policy_panel.parquet     subset of panel that has a deployed (s,S) policy
                            in the Stockmax file — used for RQ1, RQ3, RQ4
  sample_skus.parquet      stratified-sampled SKU keys for the wide benchmark
                            (target ~30k SKUs)
  sample_skus_policy.parquet  stratified-sampled SKU keys with deployed
                            policy attached (target ~all that exist; bounded
                            by Stockmax coverage)
  splits.json              definition of train / val / test / rolling folds
  strata_summary.txt       counts per stratum, pre- and post-sample

Why two samples
---------------
The Stockmax file covers only a subset of the fleet (3 brands x 5 ships in
this dataset cut). RQ4 (cost of business-rule overrides vs deployed policy)
is necessarily limited to that subset. RQ1-3 (accuracy benchmarks, accuracy
vs cost divergence under a *simulated* policy) can use the full master x
forecast intersection — typically much larger.

Design decisions
----------------
* Sampling unit = (PARTTYPENO, BRAND, SHIP). The forecast_analysis file is
  keyed by (FORECAST_ID, STOCK_ITEM_NUMBER) without a ship column, so we
  attach ships from the master.  When more than one forecast run exists per
  SKU, we take the latest by FORECAST_ID.
* Strata = SEGMENTATION x ABC x STRATEGY x REGION_BUCKET. NULLs go to an
  explicit "Unknown" bucket — never silently dropped.
* Per-cell cap: target_per_cell = ceil(target_n / n_cells). Sparse cells
  contribute everything they have. Dense cells are subsampled with a fixed
  RNG seed so the run is reproducible.
* Train/val/test split is *over time*, not SKUs. Every sampled SKU
  contributes its own 28-quarter series; the same quarter-cut applies to
  all of them. This is how M5 / forecasting benchmarks are evaluated.

Run
---
    python 02_stratified_sample.py [--target-n 30000] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import polars as pl

import config as C


SAMPLE_DIR = C.SAMPLE_DIR

# ---------------------------------------------------------------------------
# Strata definitions
# ---------------------------------------------------------------------------

SEG_LEVELS = ["Smooth", "Erratic", "Intermittent", "Lumpy", "Unknown"]
ABC_LEVELS = ["A", "B", "C", "Unknown"]
STRATEGY_LEVELS = ["Strategy 1", "Strategy 2", "Strategy 3", "Strategy 4",
                   "Strategy 5", "Other", "Unknown"]
REGION_LEVELS = ["Africa", "Arab States", "Asia & Pacific", "Europe",
                 "Middle East", "North America/Caribbiean", "South America",
                 "All", "Unknown"]


def normalize_segmentation(c: str = "SEGMENTATION") -> pl.Expr:
    """Trim whitespace and bucket unknowns."""
    return (
        pl.col(c).cast(pl.Utf8, strict=False).str.strip_chars()
          .replace_strict(
              {s: s for s in SEG_LEVELS[:-1]},
              default="Unknown",
          )
          .alias("SEGMENTATION_GRP")
    )


def normalize_abc(c: str = "ABC") -> pl.Expr:
    return (
        pl.col(c).cast(pl.Utf8, strict=False).str.strip_chars()
          .replace_strict({"A": "A", "B": "B", "C": "C"}, default="Unknown")
          .alias("ABC_GRP")
    )


def normalize_strategy(c: str = "Strategy") -> pl.Expr:
    return (
        pl.col(c).cast(pl.Utf8, strict=False).str.strip_chars()
          .replace_strict(
              {s: s for s in STRATEGY_LEVELS[:-1]},
              default="Unknown",
          )
          .alias("STRATEGY_GRP")
    )


def normalize_region(c: str = "Region") -> pl.Expr:
    return (
        pl.col(c).cast(pl.Utf8, strict=False).str.strip_chars()
          .replace_strict(
              {s: s for s in REGION_LEVELS[:-1]},
              default="Unknown",
          )
          .alias("REGION_GRP")
    )


# ---------------------------------------------------------------------------
# Build panel
# ---------------------------------------------------------------------------

def build_latest_forecast() -> pl.LazyFrame:
    """One row per STOCK_ITEM_NUMBER, taking the most recent FORECAST_ID."""
    fcst = pl.scan_parquet(C.OUT_FILES["fcst_analysis"])
    latest_id = (
        fcst.group_by("STOCK_ITEM_NUMBER")
            .agg(pl.col("FORECAST_ID").max().alias("LATEST_FORECAST_ID"))
    )
    return (
        fcst.join(
            latest_id,
            left_on=["STOCK_ITEM_NUMBER", "FORECAST_ID"],
            right_on=["STOCK_ITEM_NUMBER", "LATEST_FORECAST_ID"],
            how="inner",
        )
    )


def build_master_per_sku() -> pl.LazyFrame:
    """
    parts_master is at (PARTID, SHIP) granularity. Aggregate to PARTTYPENO.

    Pricing rule (per user direction): use *last-purchase* price, not the
    rolling-average AVGPRICE. Coalesce in this priority:
        1. LASTPURCH_PRICEPERUNIT_USD_BRAND  (brand-level latest, STEP_7)
        2. LASTPURCH_PRICEPERUNIT_USD        (per-vessel latest, STEP_6)
        3. AVGPRICE                          (legacy fallback)
    Zeroes are treated as missing — the master frequently carries 0.0 for
    items whose price was never recorded.

    Slow-mover flag (per user direction): VELOCITY == 'C'.
    """
    m = pl.scan_parquet(C.OUT_FILES["spare_parts_list"])

    # Coalesce non-zero last-purchase price candidates per row first.
    nz = lambda c: pl.when(pl.col(c) > 0).then(pl.col(c)).otherwise(None)
    m = m.with_columns(
        pl.coalesce([
            nz("LASTPURCH_PRICEPERUNIT_USD_BRAND"),
            nz("LASTPURCH_PRICEPERUNIT_USD"),
            nz("AVGPRICE"),
        ]).alias("UNIT_PRICE_USD_ROW")
    )

    return (
        m.group_by("PARTTYPENO")
         .agg([
             pl.col("BRAND").n_unique().alias("N_BRANDS"),
             pl.col("SHIP").n_unique().alias("N_SHIPS"),

             # Last-purchase USD price, SKU-level summary.
             pl.col("UNIT_PRICE_USD_ROW").drop_nulls().median().alias("UNIT_PRICE_USD"),
             pl.col("UNIT_PRICE_USD_ROW").drop_nulls().min().alias("UNIT_PRICE_USD_MIN"),
             pl.col("UNIT_PRICE_USD_ROW").drop_nulls().max().alias("UNIT_PRICE_USD_MAX"),
             pl.col("UNIT_PRICE_USD_ROW").is_not_null().sum().alias("N_PRICE_OBS"),

             # Most-recent purchase date across vessels.
             pl.col("LASTPURCH_RECEIVEDDATE").drop_nulls().max().alias("LATEST_PURCH_DATE"),

             pl.col("AVERAGE_LEAD_TIME").drop_nulls().mean().alias("LEAD_TIME_MEAN"),
             pl.col("AVERAGEUSAGEPERYEAR").drop_nulls().mean().alias("USAGE_PER_YEAR_MEAN"),
             pl.col("VELOCITY").drop_nulls().mode().first().alias("VELOCITY_MODE"),
             pl.col("CRITICALITY").drop_nulls().mode().first().alias("CRITICALITY_MODE"),
             pl.col("STOCKCLASS").drop_nulls().mode().first().alias("STOCKCLASS_MODE"),
             pl.col("STATUS").drop_nulls().mode().first().alias("STATUS_MODE"),
         ])
         .with_columns(
             # Slow-mover := velocity C (per user direction).
             (pl.col("VELOCITY_MODE") == "C").alias("IS_SLOW_MOVER"),
         )
    )


def build_stockmax_per_sku() -> pl.LazyFrame:
    """
    Stockmax is at (Parttypeno, Ship, Region) granularity. Aggregate to
    Parttypeno, recording the dominant Strategy and the set of regions in
    which a deployed policy exists.
    """
    sm = pl.scan_parquet(C.OUT_FILES["stockmax"])
    return (
        sm.group_by("Parttypeno")
          .agg([
              pl.col("Strategy").drop_nulls().mode().first().alias("STRATEGY_MODE"),
              pl.col("Region").drop_nulls().unique().alias("REGIONS"),
              pl.col("Region").drop_nulls().n_unique().alias("N_REGIONS"),
              pl.col("Quarterly Forecast").drop_nulls().mean().alias("DEPLOYED_QTR_FCST_MEAN"),
              pl.col("RMSE").drop_nulls().mean().alias("DEPLOYED_RMSE_MEAN"),
              pl.col("New Min").drop_nulls().mean().alias("DEPLOYED_NEW_MIN_MEAN"),
              pl.col("New Max").drop_nulls().mean().alias("DEPLOYED_NEW_MAX_MEAN"),
          ])
          .rename({"Parttypeno": "PARTTYPENO"})
    )


def build_panel() -> pl.LazyFrame:
    fcst = build_latest_forecast()
    master = build_master_per_sku()
    sm = build_stockmax_per_sku()

    # Wide panel = forecast x master (inner)
    panel = fcst.join(master, left_on="STOCK_ITEM_NUMBER", right_on="PARTTYPENO", how="inner")
    # Attach stockmax (left) so we can split policy vs no-policy SKUs
    panel = panel.join(sm, left_on="STOCK_ITEM_NUMBER", right_on="PARTTYPENO", how="left")

    # SEGMENT restriction: filter to SKUs in the SEGMENT operating segment
    if C.SEGMENT_MODE:
        from segment_filter import segment_skus
        skus = list(segment_skus())
        panel = panel.filter(pl.col("STOCK_ITEM_NUMBER").is_in(skus))

    # Normalize strata fields
    panel = panel.with_columns([
        normalize_segmentation("SEGMENTATION"),
        normalize_abc("ABC"),
        normalize_strategy("STRATEGY_MODE"),
    ])
    return panel


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample(
    df: pl.DataFrame,
    strata_cols: list[str],
    target_n: int,
    seed: int,
) -> pl.DataFrame:
    """
    Equal-allocation per cell with hard cap. Cells smaller than the cap
    contribute everything they have; redistribution is one-shot (no second
    pass) — keeps the algorithm deterministic and easy to defend.
    """
    n_cells = df.select(strata_cols).unique().height
    per_cell = max(1, math.ceil(target_n / max(1, n_cells)))

    sampled = (
        df.with_columns(pl.int_range(0, pl.len()).shuffle(seed=seed).over(strata_cols).alias("_rn"))
          .filter(pl.col("_rn") < per_cell)
          .drop("_rn")
    )
    return sampled


# ---------------------------------------------------------------------------
# Train / val / test split (over quarters, not SKUs)
# ---------------------------------------------------------------------------

def build_splits(n_quarters: int = 28) -> dict:
    """
    Last 4 quarters = TEST, prior 4 = VAL, rest = TRAIN.
    Plus rolling-origin folds for sensitivity / DM tests.
    """
    test_qs  = list(range(n_quarters - 3, n_quarters + 1))            # 25..28
    val_qs   = list(range(n_quarters - 7, n_quarters - 3))            # 21..24
    train_qs = list(range(1, n_quarters - 7))                         # 1..20

    rolling_folds = []
    # 1-quarter-ahead rolling, expanding window, last 8 origins
    for origin in range(20, n_quarters):
        rolling_folds.append({
            "name": f"roll_origin_{origin:02d}",
            "train_qs": list(range(1, origin + 1)),
            "horizon_qs": [origin + 1],
        })
    # 4-quarter-ahead rolling, last 5 origins
    for origin in range(20, n_quarters - 3):
        rolling_folds.append({
            "name": f"roll4_origin_{origin:02d}",
            "train_qs": list(range(1, origin + 1)),
            "horizon_qs": list(range(origin + 1, origin + 5)),
        })

    return {
        "n_quarters": n_quarters,
        "fixed": {"train_qs": train_qs, "val_qs": val_qs, "test_qs": test_qs},
        "rolling": rolling_folds,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def strata_table(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    return (
        df.group_by(cols).len()
          .sort(cols)
          .rename({"len": "n"})
    )


def write_summary(
    path: Path,
    panel_n: int,
    policy_n: int,
    sample_n: int,
    sample_policy_n: int,
    table_strata_panel: pl.DataFrame,
    table_strata_sample: pl.DataFrame,
    target_n: int,
    seed: int,
) -> None:
    lines = []
    lines.append("Stage-2 sampling summary")
    lines.append("=" * 60)
    lines.append(f"target_n          = {target_n:,}")
    lines.append(f"seed              = {seed}")
    lines.append(f"panel rows        = {panel_n:,}   (forecast inner master)")
    lines.append(f"policy panel rows = {policy_n:,}   (panel inner stockmax)")
    lines.append(f"sample (wide)     = {sample_n:,}")
    lines.append(f"sample (policy)   = {sample_policy_n:,}")
    lines.append("")
    lines.append("Panel strata distribution:")
    lines.append(table_strata_panel.to_pandas().to_string(index=False))
    lines.append("")
    lines.append("Sample strata distribution:")
    lines.append(table_strata_sample.to_pandas().to_string(index=False))
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--target-n", type=int, default=30_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    t0 = time.time()
    print("Building panel...")
    panel_lf = build_panel()
    panel = panel_lf.collect(engine="streaming")
    print(f"  panel rows: {panel.height:,}   cols: {panel.width}")

    # The "policy panel" is the subset where Stockmax matched (REGIONS not null).
    policy_panel = panel.filter(pl.col("REGIONS").is_not_null())
    print(f"  policy panel rows: {policy_panel.height:,}")

    # ---------------------------------------------------------------------
    # Sample 1 — wide (no Strategy / Region requirement)
    # Strata: SEGMENTATION_GRP x ABC_GRP   (the always-present dimensions)
    # ---------------------------------------------------------------------
    wide_strata = ["SEGMENTATION_GRP", "ABC_GRP"]
    sample_wide = stratified_sample(panel, wide_strata, args.target_n, args.seed)
    print(f"  sample_wide rows: {sample_wide.height:,}")

    # ---------------------------------------------------------------------
    # Sample 2 — policy (full strata grid)
    # Strata: SEGMENTATION_GRP x ABC_GRP x STRATEGY_GRP
    # Region is per-row in stockmax, so we keep it as an attribute and
    # stratify on the SKU-level Strategy mode. (We can re-stratify by
    # Region downstream when we evaluate per-region cost.)
    # ---------------------------------------------------------------------
    policy_strata = ["SEGMENTATION_GRP", "ABC_GRP", "STRATEGY_GRP"]
    sample_policy = stratified_sample(
        policy_panel, policy_strata, args.target_n, args.seed
    )
    print(f"  sample_policy rows: {sample_policy.height:,}")

    # ---------------------------------------------------------------------
    # Persist
    # ---------------------------------------------------------------------
    panel.write_parquet(SAMPLE_DIR / "panel.parquet", compression="zstd")
    policy_panel.write_parquet(SAMPLE_DIR / "policy_panel.parquet", compression="zstd")
    sample_wide.write_parquet(SAMPLE_DIR / "sample_skus.parquet", compression="zstd")
    sample_policy.write_parquet(SAMPLE_DIR / "sample_skus_policy.parquet", compression="zstd")

    splits = build_splits(n_quarters=28)
    (SAMPLE_DIR / "splits.json").write_text(
        json.dumps(splits, indent=2), encoding="utf-8"
    )

    table_strata_panel = strata_table(panel, policy_strata)
    table_strata_sample = strata_table(sample_policy, policy_strata)
    write_summary(
        SAMPLE_DIR / "strata_summary.txt",
        panel_n=panel.height,
        policy_n=policy_panel.height,
        sample_n=sample_wide.height,
        sample_policy_n=sample_policy.height,
        table_strata_panel=table_strata_panel,
        table_strata_sample=table_strata_sample,
        target_n=args.target_n,
        seed=args.seed,
    )

    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"Outputs: {SAMPLE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
