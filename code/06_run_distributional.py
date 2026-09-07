"""
Stage-4 Tier-2: fit ZIP and Hurdle-NB per row on Q1-Q20, emit:
  - point forecasts for Q21-Q28 (constant horizon, like the classical
    models — predictive distribution is constant-rate intercept-only),
  - per-row predictive quantiles {0.5, 0.8, 0.9, 0.95, 0.99} for the
    cost simulator,
  - per-row n_samples=1000 draws (parquet binary blob via list column).

Output:
  output/sample/forecasts_distributional.parquet
    one row per (STOCK_ITEM_NUMBER, model)
    columns: keys + FORECAST_QTRn + ACTUALS_QTRn + per-quantile forecasts
             + MAE_TEST + RMSE_TEST.

Run:
    python 06_run_distributional.py
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl

import config as C
from models_distributional import (
    fit_zip, sample_zip,
    fit_hurdle_nb, sample_hurdle_nb,
    quantiles_from_samples,
)


SAMPLE_DIR = C.SAMPLE_DIR
TRAIN_END = 20
TEST_LEN  = 28 - TRAIN_END
N_SAMPLES = 1000
QUANTILES = (0.5, 0.8, 0.9, 0.95, 0.99)
ACTUAL_COLS = [f"ACTUALS_QTR{i:02d}" for i in range(1, 29)]


def load_sample_actuals():
    sample = pl.scan_parquet(SAMPLE_DIR / "sample_skus.parquet").select([
        "STOCK_ITEM_NUMBER", "FORECAST_ID",
        "SEGMENTATION", "ABC", "SEGMENTATION_GRP", "ABC_GRP",
        "SBA_ALPHA", "SES_ALPHA",
    ]).collect()
    panel = pl.scan_parquet(SAMPLE_DIR / "panel.parquet").select(
        ["STOCK_ITEM_NUMBER", "FORECAST_ID"] + ACTUAL_COLS
    ).collect()
    return sample.join(panel, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner")


def emit_block(name: str, point: np.ndarray, quantiles: dict,
               actuals_test: np.ndarray, keys: pl.DataFrame) -> pl.DataFrame:
    """Build a per-model output frame matching the Tier-1 schema + quantile cols."""
    h = actuals_test.shape[1]
    fc_h = np.repeat(point[:, None], h, axis=1)
    diff = fc_h - actuals_test
    mae  = np.abs(diff).mean(axis=1)
    rmse = np.sqrt((diff ** 2).mean(axis=1))

    block = keys.with_columns([pl.lit(name).alias("MODEL")])
    for q in range(h):
        block = block.with_columns([
            pl.Series(f"FORECAST_QTR{TRAIN_END + 1 + q:02d}", fc_h[:, q]),
            pl.Series(f"ACTUALS_QTR{TRAIN_END + 1 + q:02d}", actuals_test[:, q]),
        ])
    block = block.with_columns([
        pl.Series("MAE_TEST",  mae),
        pl.Series("RMSE_TEST", rmse),
    ])
    for q, vals in quantiles.items():
        block = block.with_columns(pl.Series(f"Q_{int(q*100):02d}", vals))
    return block


def main():
    t0 = time.time()
    print("Loading wide-sample actuals...")
    df = load_sample_actuals()
    N = df.height
    print(f"  rows: {N:,}")

    actuals = df.select(ACTUAL_COLS).to_numpy().astype(np.float64)
    train = actuals[:, :TRAIN_END]
    test  = actuals[:, TRAIN_END:]

    keys = df.select([
        "STOCK_ITEM_NUMBER", "FORECAST_ID",
        "SEGMENTATION", "ABC", "SEGMENTATION_GRP", "ABC_GRP",
        "SBA_ALPHA", "SES_ALPHA",
    ])

    # ----- ZIP -----
    t1 = time.time()
    zip_p = fit_zip(train)
    zip_samples = sample_zip(zip_p["pi"], zip_p["lam"], n_samples=N_SAMPLES, seed=42)
    zip_q = quantiles_from_samples(zip_samples, qs=QUANTILES)
    zip_block = emit_block("ZIP", zip_p["mean"], zip_q, test, keys)
    print(f"  ZIP   {time.time() - t1:5.2f}s   "
          f"mean pi={zip_p['pi'].mean():.3f}  mean lam={zip_p['lam'].mean():.3f}  "
          f"MAE median={np.median(zip_block['MAE_TEST'].to_numpy()):.3f}")

    # ----- HNB -----
    t1 = time.time()
    hnb_p = fit_hurdle_nb(train)
    hnb_samples = sample_hurdle_nb(hnb_p["p"], hnb_p["mu"], hnb_p["alpha"],
                                    n_samples=N_SAMPLES, seed=42)
    hnb_q = quantiles_from_samples(hnb_samples, qs=QUANTILES)
    hnb_block = emit_block("HURDLE_NB", hnb_p["mean"], hnb_q, test, keys)
    print(f"  HNB   {time.time() - t1:5.2f}s   "
          f"mean p={hnb_p['p'].mean():.3f}  mean mu={hnb_p['mu'].mean():.3f}  "
          f"mean alpha={hnb_p['alpha'].mean():.3f}  "
          f"MAE median={np.median(hnb_block['MAE_TEST'].to_numpy()):.3f}")

    # Persist
    out_df = pl.concat([zip_block, hnb_block], how="vertical_relaxed")
    out_path = SAMPLE_DIR / "forecasts_distributional.parquet"
    out_df.write_parquet(out_path, compression="zstd")
    print(f"\nWrote {out_path}  ({out_df.height:,} rows)")

    # ----- Per-segment summary -----
    print("\nPer-model x per-segmentation MAE (test window):")
    summary = (
        out_df.group_by(["MODEL", "SEGMENTATION_GRP"])
              .agg([
                  pl.len().alias("n"),
                  pl.col("MAE_TEST").median().alias("mae_med"),
                  pl.col("MAE_TEST").mean().alias("mae_mean"),
                  pl.col("RMSE_TEST").median().alias("rmse_med"),
                  pl.col("Q_95").median().alias("q95_med"),
              ])
              .sort(["SEGMENTATION_GRP", "mae_med"])
    )
    print(summary.to_pandas().to_string(index=False))

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
