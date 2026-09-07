"""
Stage-7 v2: per-SKU LightGBM as a robustness check addressing reviewer
critique #5 (asymmetric model evaluation).

The global LGBM cross-learns across all SKUs. Reviewers will object that
its cost advantage may come from cross-SKU information rather than from
being a better forecaster for any given SKU.

We re-fit LGBM *per-SKU* with the same lag/rolling features (no static
encodings — they're meaningless for a single-SKU model). Compare its
test-window MAE and cost to the global LGBM and to the strongest
classical baseline (SES) on the same SKUs.

Output:
  output/sample/segment/forecasts_lgbm_per_sku.parquet
  output/sample/segment/lgbm_global_vs_local.txt
"""

from __future__ import annotations

import time
import warnings

import lightgbm as lgb
import numpy as np
import polars as pl

import config as C
from models_ml import build_features, DEFAULT_LAGS, DEFAULT_ROLLS


SAMPLE_DIR = C.SAMPLE_DIR
TRAIN_END = 20
TEST_LEN  = 28 - TRAIN_END
ACTUAL_COLS = [f"ACTUALS_QTR{i:02d}" for i in range(1, 29)]


def load_panel():
    sample = pl.scan_parquet(SAMPLE_DIR / "sample_skus.parquet").select([
        "STOCK_ITEM_NUMBER", "FORECAST_ID",
        "SEGMENTATION", "ABC", "SEGMENTATION_GRP", "ABC_GRP",
    ]).collect()
    panel = pl.scan_parquet(SAMPLE_DIR / "panel.parquet").select(
        ["STOCK_ITEM_NUMBER", "FORECAST_ID"] + ACTUAL_COLS
    ).collect()
    return sample.join(panel, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner")


def fit_local_lgbm_one(actuals_train: np.ndarray) -> float:
    """
    Fit one LGBM per SKU on its own (lag, rolling) features. Predict the
    next quarter (Q21) and return the point forecast.

    With only ~12-16 training-row samples per SKU (T - max_lag), tree-based
    models struggle. We use shallow trees + small leaves to avoid
    overfitting to single-SKU noise.
    """
    T = len(actuals_train)
    max_lag = max(DEFAULT_LAGS + DEFAULT_ROLLS)
    if T <= max_lag + 2:
        # Not enough rows; fall back to mean of training
        return float(actuals_train.mean())

    # Build (M, F) feature matrix and (M,) target
    Xs, ys = [], []
    actuals_2d = actuals_train.reshape(1, -1)
    for t in range(max_lag, T):
        X_t, _ = build_features(actuals_2d[:, :t], static={},
                                 lags=DEFAULT_LAGS, rolls=DEFAULT_ROLLS)
        Xs.append(X_t)
        ys.append(actuals_train[t])
    X = np.vstack(Xs)
    y = np.array(ys, dtype=np.float64)

    if X.shape[0] < 4:
        return float(y.mean())

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 7,
        "min_data_in_leaf": 1,
        "feature_fraction": 0.9,
        "verbose": -1,
        "num_threads": 1,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_set = lgb.Dataset(X, label=y)
        model = lgb.train(params, train_set, num_boost_round=80)

    # Predict next-step from the latest window
    X_pred, _ = build_features(actuals_2d, static={},
                                lags=DEFAULT_LAGS, rolls=DEFAULT_ROLLS)
    yhat = float(model.predict(X_pred)[0])
    return max(yhat, 0.0)


def main():
    t0 = time.time()
    df = load_panel()
    N = df.height
    print(f"SEGMENT panel rows: {N:,}    SEGMENT_MODE={C.SEGMENT_MODE}")

    actuals = df.select(ACTUAL_COLS).to_numpy().astype(np.float64)
    train = actuals[:, :TRAIN_END]
    test  = actuals[:, TRAIN_END:]

    print("Fitting per-SKU LightGBM (this is the slow part — one model "
          "per SKU)...")
    t1 = time.time()
    yhat = np.zeros(N)
    for i in range(N):
        if i and i % 500 == 0:
            elapsed = time.time() - t1
            eta = elapsed * (N - i) / i
            print(f"  {i:>6,}/{N:,}   elapsed={elapsed:.0f}s   eta={eta:.0f}s",
                  flush=True)
        yhat[i] = fit_local_lgbm_one(train[i])
    print(f"  done in {time.time() - t1:.0f}s "
          f"({(time.time() - t1) / N * 1000:.1f}ms / SKU)")

    # Constant-horizon forecast
    fc_h = np.repeat(yhat[:, None], TEST_LEN, axis=1)
    diff = fc_h - test
    mae = np.abs(diff).mean(axis=1)
    rmse = np.sqrt((diff ** 2).mean(axis=1))

    keys = df.select([
        "STOCK_ITEM_NUMBER", "FORECAST_ID",
        "SEGMENTATION", "ABC", "SEGMENTATION_GRP", "ABC_GRP",
    ])
    out = keys.with_columns([pl.lit("LGBM_LOCAL").alias("MODEL")])
    for q in range(TEST_LEN):
        out = out.with_columns([
            pl.Series(f"FORECAST_QTR{TRAIN_END + 1 + q:02d}", fc_h[:, q]),
            pl.Series(f"ACTUALS_QTR{TRAIN_END + 1 + q:02d}", test[:, q]),
        ])
    out = out.with_columns([
        pl.Series("MAE_TEST",  mae),
        pl.Series("RMSE_TEST", rmse),
    ])
    out_path = SAMPLE_DIR / "forecasts_lgbm_per_sku.parquet"
    out.write_parquet(out_path, compression="zstd")
    print(f"\nWrote {out_path}  ({out.height:,} rows)")

    # Comparison table: per-segment MAE for global vs local LGBM vs SES
    print("\n=== Per-segment MAE: LGBM_LOCAL vs LGBM_GLOBAL vs SES ===")
    blocks = [out.select(["STOCK_ITEM_NUMBER", "SEGMENTATION_GRP",
                          pl.col("MAE_TEST").alias("LGBM_LOCAL")])]
    for src, mname in [
        (SAMPLE_DIR / "forecasts_lgbm.parquet", "LGBM"),
        (SAMPLE_DIR / "forecasts_classical.parquet", "SES"),
    ]:
        if not src.exists():
            continue
        df_src = pl.read_parquet(src)
        if "MODEL" in df_src.columns:
            df_src = df_src.filter(pl.col("MODEL") == mname)
        df_src = (df_src
                   .select(["STOCK_ITEM_NUMBER", "SEGMENTATION_GRP",
                           pl.col("MAE_TEST").alias(mname)])
                   .unique(subset=["STOCK_ITEM_NUMBER"])
                   .rename({"SEGMENTATION_GRP": "SEGMENTATION_GRP_other"}))
        blocks.append(df_src.select(["STOCK_ITEM_NUMBER", mname]))
    full = blocks[0]
    for b in blocks[1:]:
        full = full.join(b, on="STOCK_ITEM_NUMBER", how="inner")

    summary = (
        full.group_by("SEGMENTATION_GRP")
            .agg([
                pl.len().alias("n"),
                pl.col("LGBM_LOCAL").mean().alias("LGBM_LOCAL"),
                pl.col("LGBM").mean().alias("LGBM_GLOBAL"),
                pl.col("SES").mean().alias("SES"),
            ])
            .sort("SEGMENTATION_GRP")
    )
    print(summary.to_pandas().to_string(index=False))
    text = ("Per-SKU LightGBM (LGBM_LOCAL) vs Global LightGBM (LGBM_GLOBAL) "
            "vs SES on SEGMENT\n"
            f"N = {N:,} SKUs\n\n"
            + summary.to_pandas().to_string(index=False))
    (SAMPLE_DIR / "lgbm_global_vs_local.txt").write_text(text, encoding="utf-8")
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
