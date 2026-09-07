"""
Stage-4 Tier-3 (part 1): global LightGBM forecaster on the wide sample.

Approach
--------
* Train ONE LightGBM jointly across all SKUs in the wide sample, using
  per-(SKU, quarter) lag/rolling features + SKU-level static features
  (segmentation, ABC, velocity, lead time, price).
* Two heads: occurrence (binary) + size (regression on positive obs).
* Quantile heads at {0.5, 0.8, 0.9, 0.95, 0.99} for predictive
  distribution.
* Recursive 8-step forecast for the test window.

Output
------
  output/sample/forecasts_lgbm.parquet
    one row per SKU with MODEL='LGBM' and the same schema as the Tier-2
    distributional output: keys + FORECAST_QTR21..28 + ACTUALS_QTR21..28
    + Q_50/80/90/95/99 + MAE_TEST + RMSE_TEST.

  output/sample/lgbm_feature_importance.txt
    sorted feature importances from the size head (gain-based).
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl

import config as C
from models_ml import LGBForecaster


SAMPLE_DIR = C.SAMPLE_DIR
TRAIN_END = 20
TEST_LEN  = 28 - TRAIN_END
QUANTILES = (0.5, 0.8, 0.9, 0.95, 0.99)
ACTUAL_COLS = [f"ACTUALS_QTR{i:02d}" for i in range(1, 29)]


def encode_categorical(values: pl.Series) -> np.ndarray:
    """Stable integer encoding (sorted unique → 0..K-1)."""
    s = values.fill_null("__NA__").cast(pl.Utf8)
    cats = sorted(set(s.to_list()))
    mapping = {c: i for i, c in enumerate(cats)}
    return np.array([mapping[v] for v in s.to_list()], dtype=np.float64)


def load_panel():
    """Wide sample with full 28-quarter actuals and SKU static features."""
    sample = pl.scan_parquet(SAMPLE_DIR / "sample_skus.parquet").select([
        "STOCK_ITEM_NUMBER", "FORECAST_ID",
        "SEGMENTATION", "ABC", "SEGMENTATION_GRP", "ABC_GRP",
        "STRATEGY_GRP", "VELOCITY_MODE", "CRITICALITY_MODE",
        "STOCKCLASS_MODE",
        "UNIT_PRICE_USD", "LEAD_TIME_MEAN", "USAGE_PER_YEAR_MEAN",
        "IS_SLOW_MOVER",
        "SBA_ALPHA", "SES_ALPHA",
    ]).collect()
    panel = pl.scan_parquet(SAMPLE_DIR / "panel.parquet").select(
        ["STOCK_ITEM_NUMBER", "FORECAST_ID"] + ACTUAL_COLS
    ).collect()
    return sample.join(panel, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner")


def main():
    t0 = time.time()
    print("Loading panel...")
    df = load_panel()
    N = df.height
    print(f"  rows: {N:,}")

    actuals = df.select(ACTUAL_COLS).to_numpy().astype(np.float64)
    train = actuals[:, :TRAIN_END]
    test  = actuals[:, TRAIN_END:]

    static = {
        "f_seg":         encode_categorical(df["SEGMENTATION_GRP"]),
        "f_abc":         encode_categorical(df["ABC_GRP"]),
        "f_strategy":    encode_categorical(df["STRATEGY_GRP"]),
        "f_velocity":    encode_categorical(df["VELOCITY_MODE"]),
        "f_criticality": encode_categorical(df["CRITICALITY_MODE"]),
        "f_stockclass":  encode_categorical(df["STOCKCLASS_MODE"]),
        "f_price":       df["UNIT_PRICE_USD"].fill_null(0.0).cast(pl.Float64).to_numpy(),
        "f_leadtime":    df["LEAD_TIME_MEAN"].fill_null(0.0).cast(pl.Float64).to_numpy(),
        "f_usage":       df["USAGE_PER_YEAR_MEAN"].fill_null(0.0).cast(pl.Float64).to_numpy(),
        "f_slow_mover":  df["IS_SLOW_MOVER"].fill_null(False).cast(pl.Float64).to_numpy(),
    }

    print("Fitting LightGBM (occur + size + 5 quantile heads)...")
    t1 = time.time()
    fcst = LGBForecaster()
    fcst.fit(train, static, train_quantiles=QUANTILES, seed=42)
    print(f"  fit done in {time.time() - t1:.1f}s   "
          f"occur best={fcst.occur_model.best_iteration if fcst.occur_model else 0}   "
          f"size  best={fcst.size_model.best_iteration if fcst.size_model else 0}")

    print("Predicting test window recursively (h=8)...")
    t1 = time.time()
    pred = fcst.predict_recursive(train, static, h=TEST_LEN)
    fc_h = pred["mean"]
    print(f"  predict done in {time.time() - t1:.1f}s")

    # Metrics
    diff = fc_h - test
    mae  = np.abs(diff).mean(axis=1)
    rmse = np.sqrt((diff ** 2).mean(axis=1))

    keys = df.select([
        "STOCK_ITEM_NUMBER", "FORECAST_ID",
        "SEGMENTATION", "ABC", "SEGMENTATION_GRP", "ABC_GRP",
        "SBA_ALPHA", "SES_ALPHA",
    ])

    out = keys.with_columns([pl.lit("LGBM").alias("MODEL")])
    for q in range(TEST_LEN):
        out = out.with_columns([
            pl.Series(f"FORECAST_QTR{TRAIN_END + 1 + q:02d}", fc_h[:, q]),
            pl.Series(f"ACTUALS_QTR{TRAIN_END + 1 + q:02d}", test[:, q]),
        ])
    out = out.with_columns([
        pl.Series("MAE_TEST",  mae),
        pl.Series("RMSE_TEST", rmse),
    ])
    for q, vals in pred["quantiles"].items():
        out = out.with_columns(pl.Series(f"Q_{int(q*100):02d}", vals))

    out_path = SAMPLE_DIR / "forecasts_lgbm.parquet"
    out.write_parquet(out_path, compression="zstd")
    print(f"\nWrote {out_path}  ({out.height:,} rows)")

    # Feature importance
    fi = pred["feature_importance_size"]
    fi_sorted = sorted(fi.items(), key=lambda kv: kv[1], reverse=True)
    fi_lines = ["Feature importance (gain-based, size head):"]
    for name, gain in fi_sorted:
        fi_lines.append(f"  {name:<20} {gain:>14.0f}")
    fi_text = "\n".join(fi_lines)
    (SAMPLE_DIR / "lgbm_feature_importance.txt").write_text(fi_text, encoding="utf-8")
    print("\n" + fi_text)

    # Per-segment summary
    print("\nLightGBM MAE by segmentation:")
    summary = (
        out.group_by("SEGMENTATION_GRP")
           .agg([
               pl.len().alias("n"),
               pl.col("MAE_TEST").median().alias("mae_med"),
               pl.col("MAE_TEST").mean().alias("mae_mean"),
               pl.col("RMSE_TEST").median().alias("rmse_med"),
               pl.col("Q_95").median().alias("q95_med"),
           ])
           .sort("mae_mean", descending=True)
    )
    print(summary.to_pandas().to_string(index=False))

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
