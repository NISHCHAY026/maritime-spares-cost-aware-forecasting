"""
Chronos on the 12-quarter context, so every arm of the specification grid can
carry eleven models.

The referee panel's first must-fix: the nine-design fragility grid runs on ten
forecasters because Chronos existed only for the 16-quarter context, so the
grid conflates a model-set change (11 -> 10 moves the holding-dominated
correlation from +0.591 to +0.455) with the design change it is meant to
measure. This run produces Chronos forecasts from context QTR01-12 predicting
QTR13-20, same estimator as the published run: chronos-t5-small, 1,000 seeded
sample paths, per-batch seeding, checkpointed.

Writes output/sample/segment/forecasts_chronos_ctx12.parquet
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
TRAIN_HI = 12
TEST_LO, TEST_HI = 12, 20
H = TEST_HI - TEST_LO          # 8: covers both the 13-16 and 13-20 test arms
QUANTS = (50, 80, 90, 95, 99)

BATCH = 8
N_SAMPLES = 1000
SEED = 42
CKPT_EVERY = 100

OUT = SAMPLE_DIR / "forecasts_chronos_ctx12.parquet"
CKPT = SAMPLE_DIR / "_chronos_ctx12_ckpt.npz"


def main():
    t0 = time.time()
    sample = (pl.scan_parquet(SAMPLE_DIR / "sample_skus.parquet")
              .select(["STOCK_ITEM_NUMBER", "FORECAST_ID"])
              .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect())
    panel = (pl.scan_parquet(SAMPLE_DIR / "panel.parquet")
             .select(["STOCK_ITEM_NUMBER", "FORECAST_ID"] + ACTUAL_COLS)
             .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect())
    df = (sample.join(panel, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner")
                .sort(["STOCK_ITEM_NUMBER", "FORECAST_ID"]))
    N = df.height
    actuals = df.select(ACTUAL_COLS).to_numpy().astype(np.float32)
    ctx = actuals[:, :TRAIN_HI]

    # float32 throughout: float16 overflowed on high-volume SKUs in an earlier
    # run and silently produced inf means (reproducibility note, item 2).
    samples = np.zeros((N, H, N_SAMPLES), dtype=np.float32)
    start = 0
    if CKPT.exists():
        z = np.load(CKPT)
        if int(z["n"]) == N and int(z["s"]) == N_SAMPLES:
            samples = z["samples"].astype(np.float32)
            start = int(z["done"])
            print(f"resuming from checkpoint at row {start:,}", flush=True)

    print(f"Chronos ctx-12 run: N={N:,}  ctx=12q  h={H}  samples={N_SAMPLES}  "
          f"seed={SEED}  batch={BATCH}", flush=True)

    pipe = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small", device_map="cpu", torch_dtype=torch.float32)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    nb = 0
    for s in range(start, N, BATCH):
        e = min(s + BATCH, N)
        torch.manual_seed(SEED + s)     # resume-invariant
        out = pipe.predict(inputs=[torch.tensor(ctx[i]) for i in range(s, e)],
                           prediction_length=H, num_samples=N_SAMPLES,
                           limit_prediction_length=False)
        arr = out.detach().cpu().numpy().astype(np.float32)   # (b, S, H)
        samples[s:e] = np.clip(np.transpose(arr, (0, 2, 1)), 0, None)
        nb += 1
        if nb % CKPT_EVERY == 0:
            np.savez_compressed(CKPT, samples=samples, done=e, n=N, s=N_SAMPLES)
            el = time.time() - t0
            rate = (e - start) / max(el, 1e-9)
            eta = (N - e) / max(rate, 1e-9) / 3600.0
            print(f"  {e:,}/{N:,}  {100*e/N:5.1f}%  elapsed {el/3600:.2f} h  "
                  f"ETA {eta:.2f} h", flush=True)

    mean_fc = samples.mean(axis=2)                            # (N, 8)
    q_at = {q: np.quantile(samples[:, 0, :], q / 100.0, axis=1) for q in QUANTS}

    out_df = df.select(["STOCK_ITEM_NUMBER", "FORECAST_ID"]).with_columns([
        pl.lit("CHRONOS").alias("MODEL"),
        # flat-point convention, parity with POINT_Q17 of the 16q run
        pl.Series("POINT_Q13", mean_fc[:, 0]),
    ])
    for h in range(H):
        out_df = out_df.with_columns(pl.Series(f"MEAN_H{h+1}", mean_fc[:, h]))
    for q in QUANTS:
        out_df = out_df.with_columns(pl.Series(f"Q_{q:02d}", q_at[q]))
    out_df.write_parquet(OUT, compression="zstd")

    print(f"\nwrote {OUT.name}  ({N:,} rows)  {(time.time()-t0)/3600:.2f} h", flush=True)
    if CKPT.exists():
        CKPT.unlink()


if __name__ == "__main__":
    main()
