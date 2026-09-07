"""
Stage-6 v2 — reviewer-defence robustness:
  (1) Replace the parity sigma_D proxy (training-window residual std) with
      a predictive-quantile-derived sigma for the three models that emit
      native quantiles (ZIP, HNB, LGBM). Re-run the (s, S) simulator and
      report whether the headline cost ranking changes.
  (2) Pairwise Diebold-Mariano tests across the 10 forecasters using
      squared-error losses on the 8-quarter test window. Reports a
      symmetric matrix of DM statistics (positive => row-model worse than
      col-model) and a sign matrix at p < 0.05.

Outputs:
  output/sample/sim_results_qsigma.parquet   per-(SKU, model) sim with quantile sigma
  output/sample/dm_matrix.parquet            symmetric DM stat matrix
  output/sample/dm_signif.parquet            sign matrix at p<0.05
  output/sample/robustness_v2.txt            summary report
"""

from __future__ import annotations

import time
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm, t

import config as C
from simulator import (
    CostConfig, normal_policy, simulate, lead_days_to_qtrs, service_level_for,
)


SAMPLE_DIR = C.SAMPLE_DIR
TRAIN_END = 20
TEST_LEN  = 28 - TRAIN_END
ACTUAL_COLS = [f"ACTUALS_QTR{i:02d}" for i in range(1, 29)]


def load_panel():
    sample = pl.scan_parquet(SAMPLE_DIR / "sample_skus.parquet").select([
        "STOCK_ITEM_NUMBER", "FORECAST_ID",
        "SEGMENTATION_GRP", "ABC_GRP",
        "VELOCITY_MODE", "CRITICALITY_MODE",
        "UNIT_PRICE_USD", "LEAD_TIME_MEAN",
    ]).unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect()
    panel = pl.scan_parquet(SAMPLE_DIR / "panel.parquet").select(
        ["STOCK_ITEM_NUMBER", "FORECAST_ID"] + ACTUAL_COLS
    ).unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect()
    return sample.join(panel, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner")


def load_model(panel: pl.DataFrame, source_path: Path,
                model_name: str | None = None) -> dict:
    """
    Returns aligned arrays (NaN-safe joins back to panel order) per model:
      mae, point_q21, forecast_test (N, 8), q90 (N,) if available, q50 (N,)
    """
    df = pl.read_parquet(source_path)
    if model_name:
        df = df.filter(pl.col("MODEL") == model_name)
    df = df.unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"])
    fc_cols = [f"FORECAST_QTR{TRAIN_END + 1 + q:02d}" for q in range(TEST_LEN)]
    keep_cols = ["STOCK_ITEM_NUMBER", "FORECAST_ID", "MAE_TEST"] + fc_cols
    has_q = "Q_50" in df.columns
    if has_q:
        keep_cols += ["Q_50", "Q_80", "Q_90", "Q_95", "Q_99"]
    aligned = panel.select(["STOCK_ITEM_NUMBER", "FORECAST_ID"]).join(
        df.select(keep_cols),
        on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="left",
    )
    forecast_test = aligned.select(fc_cols).fill_null(0.0).to_numpy().astype(np.float64)
    out = {
        "mae":    aligned["MAE_TEST"].fill_null(0.0).to_numpy(),
        "fc":     forecast_test,
        "fc_q21": forecast_test[:, 0],
    }
    if has_q:
        out["q50"] = aligned["Q_50"].fill_null(0.0).to_numpy()
        out["q90"] = aligned["Q_90"].fill_null(0.0).to_numpy()
    return out


def collect_models(panel):
    """All 10 models with their forecast arrays."""
    models = {}
    df_classical = pl.scan_parquet(SAMPLE_DIR / "forecasts_classical.parquet").collect()
    for m in df_classical["MODEL"].unique().to_list():
        models[m] = load_model(panel, SAMPLE_DIR / "forecasts_classical.parquet",
                                model_name=m)

    df_dist = pl.scan_parquet(SAMPLE_DIR / "forecasts_distributional.parquet").collect()
    for m in df_dist["MODEL"].unique().to_list():
        models[m] = load_model(panel, SAMPLE_DIR / "forecasts_distributional.parquet",
                                model_name=m)

    models["LGBM"] = load_model(panel, SAMPLE_DIR / "forecasts_lgbm.parquet")

    chronos_path = SAMPLE_DIR / "forecasts_chronos.parquet"
    if chronos_path.exists():
        models["CHRONOS"] = load_model(panel, chronos_path)
    return models


# ---------------------------------------------------------------------------
# (1) Quantile-derived sigma_D
# ---------------------------------------------------------------------------

def quantile_sigma(q90: np.ndarray, q50: np.ndarray) -> np.ndarray:
    """
    sigma_hat = (Q90 - Q50) / Phi^-1(0.90)
              = (Q90 - Q50) / 1.2816

    Symmetric-normal approximation.  Good enough for downstream policy
    sizing, robust to long-tailed predictive distributions, and identical
    in expectation to the residual-std proxy when the predictive
    distribution is roughly Gaussian.
    """
    z90 = float(norm.ppf(0.90))
    return np.maximum((q90 - q50) / z90, 0.0)


def simulate_with_sigma(panel, models, cfg, sigma_strategy: str) -> pl.DataFrame:
    """
    sigma_strategy:
      'residual'  - training-window residual std (Stage-5 default)
      'quantile'  - quantile-derived for {ZIP, HNB, LGBM}, residual otherwise
    """
    actuals = panel.select(ACTUAL_COLS).to_numpy().astype(np.float64)
    train = actuals[:, :TRAIN_END]
    test  = actuals[:, TRAIN_END:]
    unit_price = panel["UNIT_PRICE_USD"].fill_null(1.0).cast(pl.Float64).to_numpy()
    unit_price = np.where(unit_price > 0, unit_price, 1.0)
    lead_days = panel["LEAD_TIME_MEAN"].fill_null(60.0).cast(pl.Float64).to_numpy()
    lead_qtrs = lead_days_to_qtrs(np.where(lead_days > 0, lead_days, 60.0))
    lead_std_qtrs = (lead_days * 0.3) / 91.3125
    sigma_residual = train.std(axis=1, ddof=0)
    crit = panel["CRITICALITY_MODE"].fill_null("Normal").to_list()
    sl = np.array([service_level_for(c, cfg) for c in crit])

    rows = []
    for name, m in models.items():
        if sigma_strategy == "quantile" and "q90" in m and "q50" in m:
            sigma = quantile_sigma(m["q90"], m["q50"])
            # Floor at residual std to avoid degenerate zero-spread cases
            sigma = np.maximum(sigma, sigma_residual * 0.1)
        else:
            sigma = sigma_residual

        pol = normal_policy(
            forecast_mean_per_qtr=m["fc_q21"],
            forecast_std_per_qtr=sigma,
            lead_qtrs=lead_qtrs.astype(np.float64),
            lead_std_qtrs=lead_std_qtrs,
            unit_price=unit_price,
            service_level=sl,
            cfg=cfg,
        )
        sim = simulate(
            actuals_test=test, s=pol["s"], S=pol["S"],
            lead_qtrs=lead_qtrs, unit_price=unit_price, cfg=cfg,
        )
        block = panel.select([
            "STOCK_ITEM_NUMBER", "FORECAST_ID",
            "SEGMENTATION_GRP", "ABC_GRP",
        ]).with_columns([
            pl.lit(name).alias("MODEL"),
            pl.lit(sigma_strategy).alias("SIGMA"),
            pl.Series("TOTAL_COST",    sim["total_cost"]),
            pl.Series("FILL_RATE",     sim["fill_rate"]),
            pl.Series("STOCKOUT_UNITS", sim["stockout_units"]),
            pl.Series("HOLDING_COST",  sim["holding_cost"]),
        ])
        rows.append(block)
    return pl.concat(rows, how="vertical_relaxed")


# ---------------------------------------------------------------------------
# (2) Diebold-Mariano test (Newey-West-free, h=1 small-T variant)
# ---------------------------------------------------------------------------

def dm_pair(e_a: np.ndarray, e_b: np.ndarray) -> tuple[float, float]:
    """
    Squared-error DM. e_a, e_b are (N_obs,) arrays of forecast errors
    flattened across (SKUs x horizons).
    DM stat ~ N(0,1) for large samples; here we use t-dist with df=N-1
    for robustness given the cross-SKU pooling.
    """
    d = e_a**2 - e_b**2
    n = d.size
    mu = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return 0.0, 1.0
    stat = mu / se
    p = 2 * (1.0 - t.cdf(abs(stat), df=n - 1))
    return float(stat), float(p)


def dm_matrix(panel, models) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Returns (stat_df, signif_df) — symmetric square frames indexed by
    model name. Sign convention: matrix[i, j] > 0 means row-i model has
    LARGER squared error than col-j model.
    """
    actuals_test = panel.select(
        [f"ACTUALS_QTR{TRAIN_END + 1 + q:02d}" for q in range(TEST_LEN)]
    ).to_numpy().astype(np.float64)

    errors = {
        m: (info["fc"] - actuals_test).flatten()
        for m, info in models.items()
    }
    names = list(errors.keys())
    N = len(names)
    stat = np.zeros((N, N))
    sig  = np.zeros((N, N))
    for i, j in combinations(range(N), 2):
        s, p = dm_pair(errors[names[i]], errors[names[j]])
        stat[i, j] = s
        stat[j, i] = -s
        sig[i, j]  = (1 if p < 0.05 else 0) * np.sign(s)
        sig[j, i]  = -sig[i, j]

    stat_df = pl.DataFrame(
        {"MODEL": names, **{names[k]: stat[:, k] for k in range(N)}}
    )
    sig_df = pl.DataFrame(
        {"MODEL": names, **{names[k]: sig[:, k].astype(np.int64) for k in range(N)}}
    )
    return stat_df, sig_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    cfg = CostConfig()
    panel = load_panel()
    models = collect_models(panel)
    print(f"Panel rows: {panel.height:,}   models: {list(models)}")

    # ---------------------------------------------------------------------
    # (1) Sigma comparison
    # ---------------------------------------------------------------------
    print("\nSimulating with residual-std sigma...")
    sim_res = simulate_with_sigma(panel, models, cfg, "residual")
    print("Simulating with quantile-derived sigma (ZIP/HNB/LGBM)...")
    sim_q   = simulate_with_sigma(panel, models, cfg, "quantile")
    sim_q.write_parquet(SAMPLE_DIR / "sim_results_qsigma.parquet", compression="zstd")

    fleet_res = (
        sim_res.group_by("MODEL")
              .agg(pl.col("TOTAL_COST").mean().alias("cost_residual"),
                   pl.col("FILL_RATE").mean().alias("fill_residual"))
    )
    fleet_q = (
        sim_q.group_by("MODEL")
              .agg(pl.col("TOTAL_COST").mean().alias("cost_quantile"),
                   pl.col("FILL_RATE").mean().alias("fill_quantile"))
    )
    cmp = (
        fleet_res.join(fleet_q, on="MODEL", how="inner")
                .with_columns(((pl.col("cost_quantile") - pl.col("cost_residual"))
                                / pl.col("cost_residual") * 100).alias("delta_pct"))
                .sort("cost_quantile")
    )
    print("\nQuantile sigma vs residual sigma (mean total cost / SKU):")
    print(cmp.to_pandas().to_string(index=False))

    # ---------------------------------------------------------------------
    # (2) Diebold-Mariano
    # ---------------------------------------------------------------------
    print("\nComputing pairwise Diebold-Mariano tests...")
    stat_df, sig_df = dm_matrix(panel, models)
    stat_df.write_parquet(SAMPLE_DIR / "dm_matrix.parquet", compression="zstd")
    sig_df.write_parquet(SAMPLE_DIR / "dm_signif.parquet", compression="zstd")
    names = stat_df["MODEL"].to_list()
    print("\nDM statistic matrix (row vs col, positive => row has larger SqE):")
    print(stat_df.to_pandas().set_index("MODEL").round(2).to_string())
    print("\nSign at p<0.05  (+1: row worse, -1: row better, 0: NS):")
    print(sig_df.to_pandas().set_index("MODEL").to_string())

    # Persist text summary
    lines = ["Robustness v2 summary",
             "=" * 60,
             "(1) Quantile sigma vs residual sigma (mean total cost / SKU):",
             cmp.to_pandas().to_string(index=False),
             "",
             "(2) DM statistic matrix:",
             stat_df.to_pandas().set_index("MODEL").round(2).to_string(),
             "",
             "DM significance matrix (alpha=0.05):",
             sig_df.to_pandas().set_index("MODEL").to_string()]
    (SAMPLE_DIR / "robustness_v2.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
