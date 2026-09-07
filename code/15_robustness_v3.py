"""
Stage-6 v3 reinforcement: two more reviewer-defence sensitivities.

  R5. Lead-time CV sensitivity. The default CV(LT) = 0.3 is an industry
      proxy. Re-run the simulator under CV(LT) in {0.1, 0.3, 0.5} to
      check that the cost-rank winner identity is invariant.

  R6. Per-(Segmentation x ABC) bootstrap CI on rho. For each cell with
      n >= 30 SKUs, resample SKUs with replacement (B=200), recompute
      Spearman rho across forecasters, and report the 2.5/97.5
      percentile. Defends the cells with extreme rho values
      (Erratic x B = -0.69, Erratic x Unknown = -0.85, etc) against
      'small-n artefact' criticism.

Outputs:
  output/sample/robustness/lt_cv_sensitivity.parquet
  output/sample/robustness/lt_cv_sensitivity.txt
  output/sample/robustness/per_seg_abc_bootstrap.parquet
  output/sample/robustness/per_seg_abc_bootstrap.txt
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl
from scipy.stats import spearmanr

import config as C
from simulator import (
    CostConfig, normal_policy, simulate, lead_days_to_qtrs, service_level_for,
)


SAMPLE_DIR = C.SAMPLE_DIR
ROB_DIR    = SAMPLE_DIR / "robustness"
ROB_DIR.mkdir(parents=True, exist_ok=True)

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


def collect_models(panel):
    forecasts = {}
    for path in [
        SAMPLE_DIR / "forecasts_classical.parquet",
        SAMPLE_DIR / "forecasts_distributional.parquet",
        SAMPLE_DIR / "forecasts_lgbm.parquet",
        SAMPLE_DIR / "forecasts_chronos.parquet",
    ]:
        if not path.exists():
            continue
        df = pl.read_parquet(path)
        for model in df["MODEL"].unique().to_list():
            mdf = (
                df.filter(pl.col("MODEL") == model)
                  .select(["STOCK_ITEM_NUMBER", "FORECAST_ID", "MAE_TEST",
                          f"FORECAST_QTR{TRAIN_END + 1:02d}"])
                  .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"])
            )
            joined = panel.select(["STOCK_ITEM_NUMBER", "FORECAST_ID"]).join(
                mdf, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="left"
            )
            forecasts[model] = {
                "mae":    joined["MAE_TEST"].fill_null(0.0).to_numpy(),
                "fc_q21": joined[f"FORECAST_QTR{TRAIN_END + 1:02d}"].fill_null(0.0).to_numpy(),
            }
    return forecasts


# ---------------------------------------------------------------------------
# R5. Lead-time CV sensitivity
# ---------------------------------------------------------------------------

def r5_lt_cv(panel, models, cfg: CostConfig):
    actuals = panel.select(ACTUAL_COLS).to_numpy().astype(np.float64)
    train = actuals[:, :TRAIN_END]
    test  = actuals[:, TRAIN_END:]
    unit_price = panel["UNIT_PRICE_USD"].fill_null(1.0).cast(pl.Float64).to_numpy()
    unit_price = np.where(unit_price > 0, unit_price, 1.0)
    lead_days = panel["LEAD_TIME_MEAN"].fill_null(60.0).cast(pl.Float64).to_numpy()
    lead_days = np.where(lead_days > 0, lead_days, 60.0)
    lead_qtrs = lead_days_to_qtrs(lead_days)
    sigma = train.std(axis=1, ddof=0)
    crit = panel["CRITICALITY_MODE"].fill_null("Normal").to_list()
    sl = np.array([service_level_for(c, cfg) for c in crit])

    rows = []
    for cv in [0.1, 0.3, 0.5]:
        lead_std_qtrs = (lead_days * cv) / 91.3125
        for name, m in models.items():
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
            rows.append({
                "lt_cv": cv,
                "MODEL": name,
                "mean_total":   float(sim["total_cost"].mean()),
                "median_total": float(np.median(sim["total_cost"])),
                "mean_fill":    float(sim["fill_rate"].mean()),
                "mean_holding": float(sim["holding_cost"].mean()),
                "mae":          float(m["mae"].mean()),
            })
    df = pl.DataFrame(rows)
    df.write_parquet(ROB_DIR / "lt_cv_sensitivity.parquet", compression="zstd")

    # Spearman per cv
    summary_lines = ["R5 — Lead-time CV sensitivity"]
    summary_lines.append("=" * 60)
    for cv in [0.1, 0.3, 0.5]:
        sub = df.filter(pl.col("lt_cv") == cv).sort("mean_total")
        cost_ranks = sub.with_row_index("cr").select(["MODEL", "cr"])
        mae_ranks = sub.sort("mae").with_row_index("mr").select(["MODEL", "mr"])
        joined = cost_ranks.join(mae_ranks, on="MODEL", how="inner")
        rho, p = spearmanr(joined["mr"].to_numpy(), joined["cr"].to_numpy())
        summary_lines.append(
            f"  CV(LT)={cv}  best_cost={sub['MODEL'][0]:<10} "
            f"best_acc={sub.sort('mae')['MODEL'][0]:<10} "
            f"rho={rho:+.3f} (p={p:.3f})  "
            f"mean_cost_best=${sub['mean_total'][0]:,.0f}"
        )
    summary_lines.append("")
    summary_lines.append("Per-model mean total cost by CV(LT):")
    pivot = df.pivot(index="MODEL", on="lt_cv", values="mean_total").to_pandas()
    pivot = pivot.set_index("MODEL")
    sort_col = pivot.columns[1] if len(pivot.columns) > 1 else pivot.columns[0]
    pivot = pivot.sort_values(sort_col)
    summary_lines.append(pivot.round(0).astype(int).to_string())
    out = "\n".join(summary_lines)
    (ROB_DIR / "lt_cv_sensitivity.txt").write_text(out, encoding="utf-8")
    print(out)


# ---------------------------------------------------------------------------
# R6. Per-(Seg x ABC) bootstrap CI on rho
# ---------------------------------------------------------------------------

def _rho_per_cell(sub_sim_cost: dict[str, np.ndarray],
                   sub_mae:      dict[str, float]) -> float:
    """Per-cell rho given cost arrays per model and mean MAE per model."""
    if len(sub_sim_cost) < 5:
        return float("nan")
    cost_mean = {m: float(np.mean(v)) for m, v in sub_sim_cost.items()}
    cost_rank = {m: r + 1 for r, m in enumerate(sorted(cost_mean, key=cost_mean.get))}
    mae_rank  = {m: r + 1 for r, m in enumerate(sorted(sub_mae, key=sub_mae.get))}
    common = sorted(set(cost_rank) & set(mae_rank))
    if len(common) < 5:
        return float("nan")
    a = np.array([mae_rank[m] for m in common])
    c = np.array([cost_rank[m] for m in common])
    rho, _ = spearmanr(a, c)
    return float(rho)


def r6_per_seg_abc_bootstrap(panel, models, cfg: CostConfig,
                              n_boot: int = 200, seed: int = 42, min_n: int = 30):
    """
    For each (Seg × ABC) cell with n_skus >= min_n, bootstrap rho with
    SKU-level resampling (B=n_boot).
    """
    # Pre-compute per-SKU cost per model (once)
    sim = pl.read_parquet(SAMPLE_DIR / "sim_results.parquet")
    cost_wide = (
        sim.pivot(index=["STOCK_ITEM_NUMBER", "FORECAST_ID"],
                   on="MODEL", values="TOTAL_COST")
    )
    keys = panel.select(["STOCK_ITEM_NUMBER", "FORECAST_ID",
                         "SEGMENTATION_GRP", "ABC_GRP"])
    cost_with_seg = keys.join(cost_wide, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"],
                                 how="inner")

    # Per-(SKU, model) MAE across all forecast files
    mae_blocks = []
    for path in [
        SAMPLE_DIR / "forecasts_classical.parquet",
        SAMPLE_DIR / "forecasts_distributional.parquet",
        SAMPLE_DIR / "forecasts_lgbm.parquet",
        SAMPLE_DIR / "forecasts_chronos.parquet",
    ]:
        if not path.exists():
            continue
        df = pl.read_parquet(path)
        for m in df["MODEL"].unique().to_list():
            mdf = (
                df.filter(pl.col("MODEL") == m)
                  .select(["STOCK_ITEM_NUMBER", "FORECAST_ID",
                          pl.col("MAE_TEST").alias(m)])
                  .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"])
            )
            mae_blocks.append(mdf)
    mae_wide = mae_blocks[0]
    for b in mae_blocks[1:]:
        mae_wide = mae_wide.join(b, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"],
                                  how="full", coalesce=True)
    full = cost_with_seg.join(mae_wide, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"],
                                 how="inner",
                                 suffix="_mae")

    model_names = list(models.keys())
    cost_cols = model_names                                # cost columns
    mae_cols  = [f"{m}_mae" if f"{m}_mae" in full.columns else m
                 for m in model_names]

    rng = np.random.default_rng(seed)
    rows = []
    for seg in ["Smooth", "Erratic", "Intermittent", "Lumpy", "Unknown"]:
        for abc in ["A", "B", "C", "Unknown"]:
            cell = full.filter((pl.col("SEGMENTATION_GRP") == seg)
                                & (pl.col("ABC_GRP") == abc))
            n = cell.height
            if n < min_n:
                rows.append({"segment": seg, "abc": abc, "n_skus": n,
                              "rho_point": None, "rho_lo": None, "rho_hi": None,
                              "n_boot_valid": 0})
                continue
            cost_arr = cell.select(cost_cols).to_numpy()         # (N, M)
            mae_arr  = cell.select(mae_cols ).to_numpy()
            point_rho = _rho_per_cell(
                {m: cost_arr[:, i] for i, m in enumerate(model_names)},
                {m: float(np.mean(mae_arr[:, i])) for i, m in enumerate(model_names)},
            )
            rhos = []
            for _ in range(n_boot):
                idx = rng.integers(0, n, size=n)
                boot_cost = {m: cost_arr[idx, i] for i, m in enumerate(model_names)}
                boot_mae  = {m: float(np.mean(mae_arr[idx, i])) for i, m in enumerate(model_names)}
                r = _rho_per_cell(boot_cost, boot_mae)
                if not np.isnan(r):
                    rhos.append(r)
            rhos = np.array(rhos) if rhos else np.array([np.nan])
            lo, hi = (np.quantile(rhos, [0.025, 0.975])
                       if rhos.size >= 10 else (np.nan, np.nan))
            rows.append({"segment": seg, "abc": abc, "n_skus": n,
                          "rho_point": float(point_rho),
                          "rho_lo": float(lo), "rho_hi": float(hi),
                          "n_boot_valid": int(rhos.size)})

    df = pl.DataFrame(rows)
    df.write_parquet(ROB_DIR / "per_seg_abc_bootstrap.parquet", compression="zstd")
    text = ("R6 — Per-(Seg x ABC) bootstrap CI on rho\n"
            f"(B={n_boot}, seed={seed}, min n_skus={min_n})\n\n"
            + df.with_columns([pl.col("rho_point").round(3),
                                pl.col("rho_lo").round(3),
                                pl.col("rho_hi").round(3)])
                 .to_pandas().to_string(index=False))
    (ROB_DIR / "per_seg_abc_bootstrap.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    cfg = CostConfig()
    panel = load_panel()
    models = collect_models(panel)
    print(f"Panel: {panel.height:,}  models: {list(models)}\n")

    print("=" * 60)
    r5_lt_cv(panel, models, cfg)
    print("\n" + "=" * 60)
    r6_per_seg_abc_bootstrap(panel, models, cfg, n_boot=200, seed=42, min_n=30)
    print(f"\nTotal: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
