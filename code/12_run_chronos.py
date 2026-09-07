"""
Stage-4 Tier-3 (part 2): Chronos zero-shot forecasting.

Run Amazon's Chronos-T5 foundation model in zero-shot mode on the wide
sample. No fine-tuning, just give it the 20-quarter training history as
context and ask for 8-quarter predictions plus predictive samples.

Output schema matches forecasts_classical/distributional/lgbm so the
Stage-5 simulator can ingest without changes:
  output/sample/forecasts_chronos.parquet

Run:
    python 12_run_chronos.py [--sample-n 100] [--model amazon/chronos-t5-small]
                              [--batch 64] [--n-samples 50]

Performance: chronos-t5-small (~50M params) on CPU runs at ~5-10s per
batch of 64 series predicting 8 quarters with 50 samples.  Full
16,325-SKU run takes ~40 min on CPU. Pass --sample-n N to subsample
for a quick sanity check.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import polars as pl
import torch

from chronos import ChronosPipeline

import config as C


SAMPLE_DIR = C.SAMPLE_DIR
TRAIN_END = 20
TEST_LEN  = 28 - TRAIN_END
ACTUAL_COLS = [f"ACTUALS_QTR{i:02d}" for i in range(1, 29)]
QUANTILES = (0.5, 0.8, 0.9, 0.95, 0.99)


def load_panel(sample_n: int = 0, seed: int = 42):
    sample = pl.scan_parquet(SAMPLE_DIR / "sample_skus.parquet").select([
        "STOCK_ITEM_NUMBER", "FORECAST_ID",
        "SEGMENTATION", "ABC", "SEGMENTATION_GRP", "ABC_GRP",
        "SBA_ALPHA", "SES_ALPHA",
    ]).unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect()
    panel = pl.scan_parquet(SAMPLE_DIR / "panel.parquet").select(
        ["STOCK_ITEM_NUMBER", "FORECAST_ID"] + ACTUAL_COLS
    ).unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect()
    df = sample.join(panel, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner")
    if sample_n > 0 and df.height > sample_n:
        df = df.sample(n=sample_n, seed=seed)
    return df


def run_chronos(
    actuals_train: np.ndarray,    # (N, T_train)
    pipe: ChronosPipeline,
    batch_size: int,
    h: int,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      mean_forecast: (N, h)
      samples_q     : (N, h, n_samples)
    """
    N = actuals_train.shape[0]
    means = np.zeros((N, h))
    samples_full = np.zeros((N, h, n_samples), dtype=np.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ctx_list = [
            torch.tensor(actuals_train[i].astype(np.float32))
            for i in range(start, end)
        ]
        # Chronos returns shape (B, n_samples, h). API arg is `inputs`,
        # not `context`, in chronos-forecasting >= 1.4.
        out = pipe.predict(
            inputs=ctx_list,
            prediction_length=h,
            num_samples=n_samples,
            limit_prediction_length=False,
        )
        out = out.detach().cpu().numpy().astype(np.float32)
        # out: (B, n_samples, h) -> permute to (B, h, n_samples)
        out = np.transpose(out, (0, 2, 1))
        # Clamp negatives to zero (demand cannot be negative)
        out = np.clip(out, 0, None)
        samples_full[start:end] = out
        means[start:end] = out.mean(axis=2)
        if (start // batch_size) % 4 == 0:
            print(f"    batch {start // batch_size + 1}/"
                  f"{(N + batch_size - 1) // batch_size}  rows {start}..{end}")

    return means, samples_full


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample-n", type=int, default=0,
                   help="subsample for quick validation; 0 = full sample")
    p.add_argument("--model", default="amazon/chronos-t5-small",
                   help="HF model id (small / base / large)")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--n-samples", type=int, default=50,
                   help="predictive samples per series")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    t0 = time.time()
    print(f"Loading panel (sample_n={args.sample_n})...")
    df = load_panel(sample_n=args.sample_n, seed=args.seed)
    N = df.height
    print(f"  rows: {N:,}")

    actuals = df.select(ACTUAL_COLS).to_numpy().astype(np.float32)
    train = actuals[:, :TRAIN_END]
    test  = actuals[:, TRAIN_END:]

    print(f"\nLoading Chronos {args.model} on CPU...")
    pipe = ChronosPipeline.from_pretrained(
        args.model,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    print(f"  loaded in {time.time() - t0:.1f}s")

    print(f"\nRunning zero-shot inference (batch={args.batch}, "
          f"n_samples={args.n_samples})...")
    t1 = time.time()
    mean_forecast, samples_full = run_chronos(
        actuals_train=train, pipe=pipe,
        batch_size=args.batch, h=TEST_LEN, n_samples=args.n_samples,
    )
    print(f"  inference done in {time.time() - t1:.1f}s "
          f"({(time.time() - t1) / N * 1000:.1f}ms / SKU)")

    # Aggregate to quantiles + use h=0 (= QTR21) for simulator
    qhat = {
        q: np.quantile(samples_full[:, 0, :], q, axis=1)  # (N,) at first horizon
        for q in QUANTILES
    }

    # Per-row MAE / RMSE on the 8-quarter test window
    diff = mean_forecast - test
    mae  = np.abs(diff).mean(axis=1)
    rmse = np.sqrt((diff ** 2).mean(axis=1))

    keys = df.select([
        "STOCK_ITEM_NUMBER", "FORECAST_ID",
        "SEGMENTATION", "ABC", "SEGMENTATION_GRP", "ABC_GRP",
        "SBA_ALPHA", "SES_ALPHA",
    ])
    out = keys.with_columns([pl.lit("CHRONOS").alias("MODEL")])
    for q in range(TEST_LEN):
        out = out.with_columns([
            pl.Series(f"FORECAST_QTR{TRAIN_END + 1 + q:02d}", mean_forecast[:, q]),
            pl.Series(f"ACTUALS_QTR{TRAIN_END + 1 + q:02d}", test[:, q]),
        ])
    out = out.with_columns([
        pl.Series("MAE_TEST",  mae),
        pl.Series("RMSE_TEST", rmse),
    ])
    for q, vals in qhat.items():
        out = out.with_columns(pl.Series(f"Q_{int(q*100):02d}", vals))

    out_path = SAMPLE_DIR / "forecasts_chronos.parquet"
    out.write_parquet(out_path, compression="zstd")
    print(f"\nWrote {out_path}  ({out.height:,} rows)")

    # Per-segment summary
    print("\nChronos MAE by segmentation:")
    summary = (
        out.group_by("SEGMENTATION_GRP")
           .agg([
               pl.len().alias("n"),
               pl.col("MAE_TEST").median().alias("mae_med"),
               pl.col("MAE_TEST").mean().alias("mae_mean"),
               pl.col("Q_95").median().alias("q95_med"),
           ])
           .sort("mae_mean", descending=True)
    )
    print(summary.to_pandas().to_string(index=False))

    print(f"\nTotal: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
