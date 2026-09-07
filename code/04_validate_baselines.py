"""
Stage-3b: reproduce SBA / SES / MA from raw quarterly actuals and validate
against the pipeline's emitted *_FORECAST_QTR01..28 columns.

If our reproduction agrees within tolerance for ~all rows, we have ground
truth that the deployed pipeline is doing what its docstring says, and we
can build new model wrappers (Croston, TSB, ADIDA, IMAPA, ZIP, Hurdle,
LightGBM, DeepAR, Chronos) on top with confidence.

Outputs:
  output/sample/baseline_validation.parquet   per-row max-abs diff per model
  output/sample/baseline_validation.txt       summary

Run:
    python 04_validate_baselines.py [--tol 1e-2] [--sample-n 0]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl

import config as C
from baselines import sba, ses, ma2


SAMPLE_DIR = C.SAMPLE_DIR


T = 28
ACTUAL_COLS    = [f"ACTUALS_QTR{i:02d}"          for i in range(1, T + 1)]
SBA_COLS       = [f"SBA_FORECAST_QTR{i:02d}"     for i in range(1, T + 1)]
SES_COLS       = [f"SES_FORECAST_QTR{i:02d}"     for i in range(1, T + 1)]
SMA_COLS       = [f"SMA_FORECAST_QTR{i:02d}"     for i in range(1, T + 1)]
SBA_INT_COLS   = [f"SBA_INTERVAL_QTR{i:02d}"     for i in range(1, T + 1)]


def load_arrays(sample_n: int = 0, seed: int = 42):
    """Load actuals, alphas, and pipeline-emitted forecasts as NumPy arrays."""
    lf = pl.scan_parquet(C.OUT_FILES["fcst_analysis"])
    if sample_n > 0:
        df = lf.collect().sample(n=sample_n, seed=seed)
    else:
        df = lf.collect()

    actuals = df.select(ACTUAL_COLS).to_numpy().astype(np.float64)
    sba_pipeline = df.select(SBA_COLS).to_numpy().astype(np.float64)
    ses_pipeline = df.select(SES_COLS).to_numpy().astype(np.float64)
    sma_pipeline = df.select(SMA_COLS).to_numpy().astype(np.float64)

    sba_alpha = df["SBA_ALPHA"].cast(pl.Float64).to_numpy()
    ses_alpha = df["SES_ALPHA"].cast(pl.Float64).to_numpy()

    # Detect per-row training window length: the pipeline emits non-null
    # SBA_INTERVAL only within the active window. Count non-nulls per row.
    sba_int_arr = df.select(SBA_INT_COLS).to_numpy().astype(np.float64)
    valid_mask = np.isfinite(sba_int_arr)               # (N, T)
    window_lengths = valid_mask.sum(axis=1).astype(np.int64)
    # If a row has 0 valid intervals (e.g., all-zero actuals), default to T.
    window_lengths = np.where(window_lengths == 0, T, window_lengths)

    keys = df.select([
        "FORECAST_ID", "STOCK_ITEM_NUMBER",
        "SEGMENTATION", "ABC",
        "SBA_ALPHA", "SES_ALPHA",
    ]).with_columns(pl.Series("WINDOW_LEN", window_lengths))

    return (keys, actuals, sba_pipeline, ses_pipeline, sma_pipeline,
            sba_alpha, ses_alpha, window_lengths, valid_mask)


def per_row_metrics(
    name: str, ours: np.ndarray, theirs: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> dict:
    """
    Compare row-by-row within the per-row valid window.
    valid_mask: (N, T) bool — True where the comparison should be counted.
    NaNs on either side are also excluded.
    """
    finite = np.isfinite(ours) & np.isfinite(theirs)
    if valid_mask is not None:
        finite = finite & valid_mask
    diff = np.where(finite, np.abs(ours - theirs), 0.0)
    # For rows with no valid columns, defend against /0 in mean.
    n_valid = finite.sum(axis=1).clip(min=1)
    max_abs = diff.max(axis=1)
    mean_abs = diff.sum(axis=1) / n_valid
    sum_abs = diff.sum(axis=1)
    return {
        "name": name,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "sum_abs": sum_abs,
        "n_valid": n_valid,
    }


def summarize(name: str, max_abs: np.ndarray, tol: float) -> str:
    pct = lambda thr: 100.0 * (max_abs <= thr).mean()
    qs = np.quantile(max_abs, [0.5, 0.9, 0.99, 0.999, 1.0])
    return (
        f"  {name:<5}  "
        f"<= {tol:g}: {pct(tol):6.2f}%   "
        f"<= 0.001: {pct(0.001):6.2f}%   "
        f"<= 0.01:  {pct(0.01):6.2f}%   "
        f"<= 0.1:   {pct(0.1):6.2f}%   "
        f"<= 1.0:   {pct(1.0):6.2f}%   "
        f"max-abs quantiles (50/90/99/99.9/100): "
        f"{qs[0]:.4f} / {qs[1]:.4f} / {qs[2]:.4f} / {qs[3]:.4f} / {qs[4]:.4f}"
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tol", type=float, default=1e-2,
                   help="absolute tolerance for considering a row matched")
    p.add_argument("--sample-n", type=int, default=0,
                   help="run on a random sample (0 = all rows)")
    args = p.parse_args(argv)

    t0 = time.time()
    print("Loading arrays...")
    (keys, actuals, sba_pipe, ses_pipe, sma_pipe,
     sba_alpha, ses_alpha, window_lengths, valid_mask) = \
        load_arrays(sample_n=args.sample_n)
    N = actuals.shape[0]
    print(f"  rows: {N:,}    actuals shape: {actuals.shape}    "
          f"window-length distribution: "
          f"min={window_lengths.min()}, "
          f"med={np.median(window_lengths):.0f}, "
          f"max={window_lengths.max()}")
    print(f"  loaded in {time.time() - t0:.1f}s")

    print("\nReproducing baselines (with per-row window lengths)...")
    t1 = time.time()
    sba_ours = sba(actuals, sba_alpha, window_lengths=window_lengths).forecast
    print(f"  SBA   done in {time.time() - t1:.1f}s")
    t1 = time.time()
    ses_ours = ses(actuals, ses_alpha)
    print(f"  SES   done in {time.time() - t1:.1f}s")
    t1 = time.time()
    ma_ours  = ma2(actuals)
    print(f"  MA    done in {time.time() - t1:.1f}s")

    print("\nMeasuring agreement (within per-row valid window)...")
    sba_m = per_row_metrics("SBA", sba_ours, sba_pipe, valid_mask=valid_mask)
    ses_m = per_row_metrics("SES", ses_ours, ses_pipe, valid_mask=valid_mask)
    ma_m  = per_row_metrics("MA",  ma_ours,  sma_pipe, valid_mask=valid_mask)

    # Assemble per-row report
    out = keys.with_columns([
        pl.Series("SBA_MAX_ABS_DIFF",  sba_m["max_abs"]),
        pl.Series("SBA_MEAN_ABS_DIFF", sba_m["mean_abs"]),
        pl.Series("SES_MAX_ABS_DIFF",  ses_m["max_abs"]),
        pl.Series("SES_MEAN_ABS_DIFF", ses_m["mean_abs"]),
        pl.Series("MA_MAX_ABS_DIFF",   ma_m["max_abs"]),
        pl.Series("MA_MEAN_ABS_DIFF",  ma_m["mean_abs"]),
    ])
    out.write_parquet(SAMPLE_DIR / "baseline_validation.parquet", compression="zstd")

    # ----- Summary -----
    lines = []
    lines.append("Baseline reproduction — validation report")
    lines.append("=" * 72)
    lines.append(f"rows                : {N:,}")
    lines.append(f"tolerance (--tol)   : {args.tol:g}")
    lines.append(f"actuals shape       : {actuals.shape}")
    lines.append("")
    lines.append("Per-model agreement (max-abs forecast diff per row):")
    lines.append(summarize("SBA", sba_m["max_abs"], args.tol))
    lines.append(summarize("SES", ses_m["max_abs"], args.tol))
    lines.append(summarize("MA",  ma_m["max_abs"],  args.tol))

    # Per-segmentation drilldown for SBA (the most complex model)
    lines.append("")
    lines.append("SBA agreement by SEGMENTATION (% rows with max-abs <= 0.01):")
    seg_df = (
        out.group_by("SEGMENTATION")
           .agg([
               pl.len().alias("n"),
               (pl.col("SBA_MAX_ABS_DIFF") <= 0.01).mean().alias("agree_001"),
               (pl.col("SBA_MAX_ABS_DIFF") <= 0.1).mean().alias("agree_01"),
               pl.col("SBA_MAX_ABS_DIFF").quantile(0.99).alias("p99"),
           ])
           .sort("n", descending=True)
    )
    lines.append(seg_df.to_pandas().to_string(index=False))

    report_path = SAMPLE_DIR / "baseline_validation.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  wrote {report_path}")
    print()
    print("\n".join(lines))
    print(f"\nDone in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
