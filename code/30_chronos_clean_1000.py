"""
Chronos clean-window re-run at 1,000 seeded sample paths.

Replaces the 30-sample run behind forecasts_chronos_clean.parquet. Thirty
draws makes the empirical 0.99 quantile essentially the largest draw, which
is downward-biased for a right-skewed count predictive; ZIP and Hurdle-NB are
sampled 1,000 times, so the old calibration comparison confounded tail shape
with Monte Carlo sample size. This run puts Chronos on the same footing and
seeds the sampler so the result is reproducible.

Context = QTR01-16, predict QTR17-20. Writes
output/sample/segment/forecasts_chronos_clean_1000.parquet (a NEW file; the
30-sample file is left in place until the comparison is done).

Checkpoints every CKPT_EVERY batches so an interrupted run resumes instead of
restarting. Roughly 4 h on 16 CPU threads at batch 8.
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl
import torch
from chronos import ChronosPipeline

import config as C

SAMPLE_DIR = C.SAMPLE_DIR
ACTUAL_COLS = [f"ACTUALS_QTR{i:02d}" for i in range(1, 29)]
TRAIN_HI = 16
TEST_LO, TEST_HI = 16, 20
H = TEST_HI - TEST_LO
QUANTS = (50, 80, 90, 95, 99)

BATCH = 8            # measured optimum on this CPU; larger is slower
N_SAMPLES = 1000     # parity with ZIP / Hurdle-NB
SEED = 42
CKPT_EVERY = 100

OUT = SAMPLE_DIR / "forecasts_chronos_clean_1000.parquet"
CKPT = SAMPLE_DIR / "_chronos1000_ckpt.npz"


def main():
    t0 = time.time()
    torch.set_num_threads(torch.get_num_threads())

    sample = (pl.scan_parquet(SAMPLE_DIR / "sample_skus.parquet")
              .select(["STOCK_ITEM_NUMBER", "FORECAST_ID", "SEGMENTATION_GRP", "ABC_GRP"])
              .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect())
    panel = (pl.scan_parquet(SAMPLE_DIR / "panel.parquet")
             .select(["STOCK_ITEM_NUMBER", "FORECAST_ID"] + ACTUAL_COLS)
             .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect())
    df = sample.join(panel, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner")
    N = df.height
    actuals = df.select(ACTUAL_COLS).to_numpy().astype(np.float32)
    ctx = actuals[:, :TRAIN_HI]
    test = actuals[:, TEST_LO:TEST_HI]

    # Only step 0 samples are needed for the quantiles, but the mean uses all
    # H steps, so keep the full cube. This MUST be float32: float16 overflows
    # above 65504 and Chronos draws exceed that on a handful of high-volume
    # SKUs, which silently turns their sample mean into inf.
    samples = np.zeros((N, H, N_SAMPLES), dtype=np.float32)
    start = 0
    if CKPT.exists():
        z = np.load(CKPT)
        if int(z["n"]) == N and int(z["s"]) == N_SAMPLES:
            samples = z["samples"]
            start = int(z["done"])
            print(f"resuming from checkpoint at row {start:,}")

    print(f"Chronos clean-window 1000-sample: N={N:,}  ctx=16q  h={H}  "
          f"samples={N_SAMPLES}  seed={SEED}  batch={BATCH}")

    pipe = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small", device_map="cpu", torch_dtype=torch.float32)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    nb = 0
    for s in range(start, N, BATCH):
        e = min(s + BATCH, N)
        # Re-seed per batch so the result does not depend on where a resume
        # happened to restart.
        torch.manual_seed(SEED + s)
        out = pipe.predict(inputs=[torch.tensor(ctx[i]) for i in range(s, e)],
                           prediction_length=H, num_samples=N_SAMPLES,
                           limit_prediction_length=False)
        arr = out.detach().cpu().numpy().astype(np.float32)   # (b, S, H)
        arr = np.clip(np.transpose(arr, (0, 2, 1)), 0, None)  # (b, H, S)
        samples[s:e] = arr.astype(np.float16)
        nb += 1
        if nb % CKPT_EVERY == 0:
            np.savez_compressed(CKPT, samples=samples, done=e, n=N, s=N_SAMPLES)
            el = time.time() - t0
            rate = (e - start) / max(el, 1e-9)
            eta = (N - e) / max(rate, 1e-9) / 3600.0
            print(f"  {e:,}/{N:,}  {100*e/N:5.1f}%  elapsed {el/3600:.2f} h  ETA {eta:.2f} h",
                  flush=True)

    samples_f = samples.astype(np.float32)
    mean_fc = samples_f.mean(axis=2)
    mae = np.abs(mean_fc - test).mean(axis=1)
    q_at = {q: np.quantile(samples_f[:, 0, :], q / 100.0, axis=1) for q in QUANTS}

    out_df = df.select(["STOCK_ITEM_NUMBER", "FORECAST_ID", "SEGMENTATION_GRP",
                        "ABC_GRP"]).with_columns([
        pl.lit("CHRONOS").alias("MODEL"),
        pl.Series("POINT_Q17", mean_fc[:, 0]),
        pl.Series("MAE_TEST", mae),
    ])
    for q in QUANTS:
        out_df = out_df.with_columns(pl.Series(f"Q_{q:02d}", q_at[q]))
    out_df.write_parquet(OUT, compression="zstd")

    print(f"\nwrote {OUT.name}  ({N:,} rows)  {(time.time()-t0)/3600:.2f} h")
    print(f"  mean MAE {mae.mean():.4f}")
    for q in QUANTS:
        print(f"  Q_{q:02d}: zero-frac {np.mean(q_at[q] == 0):.4f}  mean {q_at[q].mean():.3f}")
    if CKPT.exists():
        CKPT.unlink()


if __name__ == "__main__":
    main()
