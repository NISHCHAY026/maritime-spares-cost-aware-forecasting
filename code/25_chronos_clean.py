"""
Clean-window Chronos: context = QTR01-16, predict QTR17-20.
Mirrors 12_run_chronos but on the uncensored window (audit fix #1).
Writes output/sample/segment/forecasts_chronos_clean.parquet with
point forecast, MAE on QTR17-20, and Q_50/80/90/95/99.
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
TRAIN_HI = 16          # context = QTR01-16
TEST_LO, TEST_HI = 16, 20   # predict QTR17-20
H = TEST_HI - TEST_LO
QUANTS = (50, 80, 90, 95, 99)
BATCH = 32
N_SAMPLES = 30


def main():
    t0 = time.time()
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
    print(f"Chronos clean-window: N={N:,}  context=16q  predict={H}q")

    pipe = ChronosPipeline.from_pretrained("amazon/chronos-t5-small",
                                           device_map="cpu", torch_dtype=torch.float32)
    samples_full = np.zeros((N, H, N_SAMPLES), dtype=np.float32)
    for s in range(0, N, BATCH):
        e = min(s + BATCH, N)
        out = pipe.predict(inputs=[torch.tensor(ctx[i]) for i in range(s, e)],
                           prediction_length=H, num_samples=N_SAMPLES,
                           limit_prediction_length=False)
        out = np.clip(np.transpose(out.detach().cpu().numpy().astype(np.float32), (0, 2, 1)), 0, None)
        samples_full[s:e] = out
        if (s // BATCH) % 40 == 0:
            print(f"  {s}/{N}")
    mean_fc = samples_full.mean(axis=2)
    mae = np.abs(mean_fc - test).mean(axis=1)
    q_at = {q: np.quantile(samples_full[:, 0, :], q / 100, axis=1) for q in QUANTS}

    out = df.select(["STOCK_ITEM_NUMBER", "FORECAST_ID", "SEGMENTATION_GRP", "ABC_GRP"]).with_columns([
        pl.lit("CHRONOS").alias("MODEL"),
        pl.Series("POINT_Q17", mean_fc[:, 0]),
        pl.Series("MAE_TEST", mae),
    ])
    for q in QUANTS:
        out = out.with_columns(pl.Series(f"Q_{q:02d}", q_at[q]))
    out.write_parquet(SAMPLE_DIR / "forecasts_chronos_clean.parquet", compression="zstd")
    print(f"wrote forecasts_chronos_clean.parquet  ({N:,} rows)  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
