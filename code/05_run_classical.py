"""
Stage-4: run the Tier-1 classical models on the wide sample and emit
forecasts for the held-out test window.

Layout of output:
  output/sample/forecasts_classical.parquet
    one row per (STOCK_ITEM_NUMBER, model)
    columns:
      MODEL                 in {SBA, SES, MA, CROSTON, TSB, ADIDA, IMAPA}
      STOCK_ITEM_NUMBER, FORECAST_ID, SEGMENTATION, ABC, SBA_ALPHA, SES_ALPHA
      ACTUALS_QTR21..28     ground truth for the test window
      FORECAST_QTR21..28    8-quarter point forecasts (constant for classical
                            models; horizon-decay limitation noted in the
                            paper's discussion section)
      MAE_TEST, RMSE_TEST   row-level test-window metrics

This is a *prototype* runner. Stage-5 adds rolling-origin folds, residual
bootstrapping for predictive distributions, and the (s,S) cost simulator.

Run:
    python 05_run_classical.py
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl

import config as C
from baselines import sba as sba_fn, ses as ses_fn, ma2 as ma_fn
from models_classical import croston, tsb, adida, imapa


SAMPLE_DIR = C.SAMPLE_DIR
TRAIN_END = 20            # Q1..Q20 train; Q21..Q28 test
TEST_LEN  = 28 - TRAIN_END

ACTUAL_COLS = [f"ACTUALS_QTR{i:02d}" for i in range(1, 29)]


def load_sample_actuals():
    """
    Pull the wide-sample SKUs and their full 28-quarter actuals from the panel.
    """
    sample = pl.scan_parquet(SAMPLE_DIR / "sample_skus.parquet").select([
        "STOCK_ITEM_NUMBER", "FORECAST_ID",
        "SEGMENTATION", "ABC", "SEGMENTATION_GRP", "ABC_GRP",
        "SBA_ALPHA", "SES_ALPHA",
    ]).collect()

    # The panel has actuals already (full 28 quarters per row, one row per
    # latest forecast run per SKU).
    panel = pl.scan_parquet(SAMPLE_DIR / "panel.parquet").select(
        ["STOCK_ITEM_NUMBER", "FORECAST_ID"] + ACTUAL_COLS
    ).collect()

    df = sample.join(panel, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner")
    return df


def constant_horizon(point_terminal: np.ndarray, h: int) -> np.ndarray:
    """Replicate a single-step terminal point forecast across h horizons."""
    return np.repeat(point_terminal[:, None], h, axis=1)


def make_forecast_block(
    name: str,
    actuals_train: np.ndarray,
    actuals_test: np.ndarray,
    alpha_sba: np.ndarray,
    alpha_ses: np.ndarray,
) -> dict:
    """
    Returns dict with keys:
      MODEL, FORECAST_QTR21..28, MAE_TEST, RMSE_TEST
    """
    h = actuals_test.shape[1]
    name_u = name.upper()

    if name_u == "SBA":
        state = sba_fn(actuals_train, alpha_sba)
        terminal = state.forecast[:, -1]
        fc = constant_horizon(terminal, h)
    elif name_u == "SES":
        # SES one-step-ahead is α·last_actual + (1-α)·last_forecast,
        # then constant for the rest of the horizon.
        ses_train = ses_fn(actuals_train, alpha_ses)
        last_a = actuals_train[:, -1]
        last_f = ses_train[:, -1]
        next_step = alpha_ses * last_a + (1 - alpha_ses) * last_f
        fc = constant_horizon(next_step, h)
    elif name_u == "MA":
        # 2-quarter MA: forecast = (last + second-last)/2, constant for h
        next_step = (actuals_train[:, -1] + actuals_train[:, -2]) / 2.0
        fc = constant_horizon(next_step, h)
    elif name_u == "CROSTON":
        state = croston(actuals_train, alpha_sba, sba=False)
        fc = constant_horizon(state.forecast[:, -1], h)
    elif name_u == "TSB":
        # Use SBA alpha as α; β = α (common default)
        state = tsb(actuals_train, alpha_sba)
        fc = constant_horizon(state.forecast[:, -1], h)
    elif name_u == "ADIDA":
        out = adida(actuals_train, alpha_ses, k=None)
        fc = constant_horizon(out["forecast_per_period"], h)
    elif name_u == "IMAPA":
        out = imapa(actuals_train, alpha_ses, k_max=None)
        fc = constant_horizon(out["forecast_per_period"], h)
    else:
        raise ValueError(f"unknown model: {name}")

    diff = fc - actuals_test
    mae = np.abs(diff).mean(axis=1)
    rmse = np.sqrt((diff ** 2).mean(axis=1))

    return {"MODEL": name_u, "FORECAST": fc, "MAE": mae, "RMSE": rmse}


def main():
    t0 = time.time()
    print("Loading wide-sample actuals...")
    df = load_sample_actuals()
    N = df.height
    print(f"  rows: {N:,}")

    actuals = df.select(ACTUAL_COLS).to_numpy().astype(np.float64)
    actuals_train = actuals[:, :TRAIN_END]
    actuals_test  = actuals[:, TRAIN_END:]
    sba_alpha = df["SBA_ALPHA"].cast(pl.Float64).fill_null(0.1).to_numpy()
    ses_alpha = df["SES_ALPHA"].cast(pl.Float64).fill_null(0.5).to_numpy()

    keys = df.select([
        "STOCK_ITEM_NUMBER", "FORECAST_ID",
        "SEGMENTATION", "ABC", "SEGMENTATION_GRP", "ABC_GRP",
        "SBA_ALPHA", "SES_ALPHA",
    ])

    models = ["SBA", "SES", "MA", "CROSTON", "TSB", "ADIDA", "IMAPA"]
    blocks = []
    for m in models:
        t1 = time.time()
        out = make_forecast_block(m, actuals_train, actuals_test, sba_alpha, ses_alpha)
        block = keys.with_columns([
            pl.lit(out["MODEL"]).alias("MODEL"),
        ])
        for q in range(TEST_LEN):
            block = block.with_columns(
                pl.Series(f"FORECAST_QTR{TRAIN_END + 1 + q:02d}", out["FORECAST"][:, q])
            )
        for q in range(TEST_LEN):
            block = block.with_columns(
                pl.Series(f"ACTUALS_QTR{TRAIN_END + 1 + q:02d}", actuals_test[:, q])
            )
        block = block.with_columns([
            pl.Series("MAE_TEST", out["MAE"]),
            pl.Series("RMSE_TEST", out["RMSE"]),
        ])
        blocks.append(block)
        print(f"  {m:<8} {time.time() - t1:6.2f}s   "
              f"MAE  median={np.median(out['MAE']):.3f}  mean={out['MAE'].mean():.3f}   "
              f"RMSE median={np.median(out['RMSE']):.3f}  mean={out['RMSE'].mean():.3f}")

    out_df = pl.concat(blocks, how="vertical_relaxed")
    out_path = SAMPLE_DIR / "forecasts_classical.parquet"
    out_df.write_parquet(out_path, compression="zstd")
    print(f"\nWrote {out_path}  ({out_df.height:,} rows)")

    # ----- Per-model x per-segment summary -----
    print("\nPer-model, per-segmentation MAE (test window):")
    summary = (
        out_df.group_by(["MODEL", "SEGMENTATION_GRP"])
              .agg([
                  pl.len().alias("n"),
                  pl.col("MAE_TEST").median().alias("mae_med"),
                  pl.col("MAE_TEST").mean().alias("mae_mean"),
                  pl.col("RMSE_TEST").median().alias("rmse_med"),
              ])
              .sort(["SEGMENTATION_GRP", "mae_med"])
    )
    print(summary.to_pandas().to_string(index=False))

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
