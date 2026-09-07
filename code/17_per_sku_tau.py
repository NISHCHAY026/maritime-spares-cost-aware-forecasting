"""
Stage-7 v2: per-SKU Kendall's tau analysis — the new central statistic
for the revised paper.

The Stage-5 / v0.5 paper expressed the headline as Spearman rho computed
over n=11 forecasters. The bootstrap CI was [-0.41, +0.48], reflecting
the inherent low power of an 11-point rank correlation. Reviewer
critique correctly flagged this as inadequate.

We replace it with: the *mean per-SKU* Kendall's tau between MAE-rank
and cost-rank across the 11 forecasters. Power scales with the number
of SKUs (15.3 k for SEGMENT-wide; 1,500-6,300 per cell), not 11.

Also reported (defends the framing):
  * fleet-mean MAE and cost per model (point estimates, no CI claim)
  * per-SKU dominance pairs: P(best-MAE model = best-cost model)
  * model identity frequency: how often each model is best-MAE / best-cost
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl

import config as C
from unified_stats import (
    per_sku_kendall_tau,
    bootstrap_mean_ci,
    per_sku_dominance_pairs,
    fleet_rank_table,
    cell_tau_with_ci,
)


SAMPLE_DIR = C.SAMPLE_DIR
ROB_DIR = SAMPLE_DIR / "robustness"
ROB_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_END = 20
TEST_LEN = 28 - TRAIN_END
ACTUAL_COLS = [f"ACTUALS_QTR{i:02d}" for i in range(1, 29)]


def load_per_sku_arrays() -> tuple[pl.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """
    Returns:
      keys           : (N, key cols) DataFrame for joining back
      mae_per_sku    : (N, M) test-window MAE per (SKU, model)
      cost_per_sku   : (N, M) total cost per (SKU, model) from sim_results
      model_names    : list of M model names, column order matches arrays
    """
    sim = pl.read_parquet(SAMPLE_DIR / "sim_results.parquet")
    cost_wide = sim.pivot(index=["STOCK_ITEM_NUMBER", "FORECAST_ID",
                                  "SEGMENTATION_GRP", "ABC_GRP"],
                           on="MODEL", values="TOTAL_COST")

    # Pull MAE per (SKU, model) from forecast files
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
            mae_blocks.append(
                df.filter(pl.col("MODEL") == m)
                  .select(["STOCK_ITEM_NUMBER", "FORECAST_ID",
                          pl.col("MAE_TEST").alias(m)])
                  .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"])
            )
    mae_wide = mae_blocks[0]
    for b in mae_blocks[1:]:
        mae_wide = mae_wide.join(b, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"],
                                  how="full", coalesce=True)

    # Inner-join cost & mae on (SKU, FORECAST_ID); enforces every SKU has
    # a value in BOTH frames (drops the 1-2 SKUs where mismatch occurs).
    full = cost_wide.join(mae_wide, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"],
                            how="inner", suffix="_mae")

    # Identify the model columns from cost_wide (these are the canonical
    # M model names in column order)
    base_cols = ["STOCK_ITEM_NUMBER", "FORECAST_ID",
                 "SEGMENTATION_GRP", "ABC_GRP"]
    cost_cols = [c for c in cost_wide.columns if c not in base_cols]
    # Corresponding MAE column names from the join (suffix _mae if collision)
    mae_cols = []
    for m in cost_cols:
        if m in full.columns and m + "_mae" in full.columns:
            mae_cols.append(m + "_mae")
        elif m + "_mae" in full.columns:
            mae_cols.append(m + "_mae")
        else:
            mae_cols.append(m)

    keys = full.select(base_cols)
    mae_per_sku = full.select(mae_cols).to_numpy()
    cost_per_sku = full.select(cost_cols).to_numpy()
    return keys, mae_per_sku, cost_per_sku, cost_cols


def main():
    t0 = time.time()
    print(f"Loading per-SKU arrays from {SAMPLE_DIR}...")
    keys, mae_per_sku, cost_per_sku, model_names = load_per_sku_arrays()
    N, M = mae_per_sku.shape
    print(f"  N (SKUs) = {N:,}    M (models) = {M}")
    print(f"  models: {model_names}")

    # ----- (1) Fleet-level point estimates (no CI claim) -----
    fleet = fleet_rank_table(mae_per_sku, cost_per_sku, model_names)
    print("\n=== Fleet-level point estimates (mean across SKUs) ===")
    print(f"{'Model':<12} {'mean MAE':>10} {'mean cost':>14}")
    for m in sorted(fleet["mean_mae"], key=fleet["mean_mae"].get):
        print(f"  {m:<10} {fleet['mean_mae'][m]:>10.3f} "
              f"{fleet['mean_cost'][m]:>13,.2f}")

    # ----- (2) Per-SKU dominance -----
    dom = per_sku_dominance_pairs(mae_per_sku, cost_per_sku, model_names)
    print(f"\n=== Per-SKU dominance ===")
    print(f"  P(best-MAE model = best-cost model) = {dom['agreement_rate']:.4f}")
    print(f"  Best-MAE model frequency:")
    for m, n in sorted(dom["best_mae_freq"].items(), key=lambda kv: -kv[1]):
        print(f"    {m:<12} {n:>6,}  ({100*n/N:5.2f}%)")
    print(f"  Best-cost model frequency:")
    for m, n in sorted(dom["best_cost_freq"].items(), key=lambda kv: -kv[1]):
        print(f"    {m:<12} {n:>6,}  ({100*n/N:5.2f}%)")

    # ----- (3) Per-SKU Kendall's tau (the new central statistic) -----
    print("\n=== Per-SKU Kendall's tau across forecasters ===")
    print("  computing tau per SKU ...", end=" ", flush=True)
    t1 = time.time()
    taus = per_sku_kendall_tau(mae_per_sku, cost_per_sku)
    print(f"done in {time.time() - t1:.1f}s")

    print("  bootstrap (1000 resamples of SKUs) ...", end=" ", flush=True)
    t1 = time.time()
    overall_ci = bootstrap_mean_ci(taus, n_boot=1000, seed=42)
    print(f"done in {time.time() - t1:.1f}s")
    print(f"\n  Overall:  mean tau = {overall_ci['mean']:+.4f}   "
          f"95% CI [{overall_ci['lo']:+.4f}, {overall_ci['hi']:+.4f}]   "
          f"(N={overall_ci['n_valid']:,})")

    # ----- (4) Per-(Segmentation x ABC) tau with CI -----
    print("\n=== Per-(Segmentation x ABC) Kendall's tau ===")
    seg_levels = ["Smooth", "Erratic", "Intermittent", "Lumpy", "Unknown"]
    abc_levels = ["A", "B", "C", "Unknown"]
    rows = []
    for seg in seg_levels:
        for abc in abc_levels:
            mask = (
                (keys["SEGMENTATION_GRP"] == seg) &
                (keys["ABC_GRP"] == abc)
            ).to_numpy()
            n = mask.sum()
            if n < 30:
                rows.append({"segment": seg, "abc": abc, "n_skus": int(n),
                              "mean_tau": None, "lo": None, "hi": None})
                continue
            sub_ci = cell_tau_with_ci(
                mae_per_sku[mask], cost_per_sku[mask],
                n_boot=500, seed=42,
            )
            rows.append({"segment": seg, "abc": abc, "n_skus": int(n),
                          "mean_tau": float(sub_ci["mean"]),
                          "lo": float(sub_ci["lo"]),
                          "hi": float(sub_ci["hi"])})
    df_cell = pl.DataFrame(rows)
    df_cell.write_parquet(ROB_DIR / "per_sku_tau_cells.parquet", compression="zstd")
    pretty = df_cell.with_columns([
        pl.col("mean_tau").round(3),
        pl.col("lo").round(3),
        pl.col("hi").round(3),
    ])
    print(pretty.to_pandas().to_string(index=False))

    # ----- (5) Per-Segment marginal tau with CI -----
    print("\n=== Per-Segment Kendall's tau (collapsed across ABC) ===")
    seg_rows = []
    for seg in seg_levels:
        mask = (keys["SEGMENTATION_GRP"] == seg).to_numpy()
        n = mask.sum()
        if n < 30:
            continue
        ci = cell_tau_with_ci(mae_per_sku[mask], cost_per_sku[mask],
                                n_boot=1000, seed=42)
        seg_rows.append({"segment": seg, "n_skus": int(n),
                          "mean_tau": float(ci["mean"]),
                          "lo": float(ci["lo"]), "hi": float(ci["hi"])})
    df_seg = pl.DataFrame(seg_rows)
    df_seg.write_parquet(ROB_DIR / "per_sku_tau_segments.parquet", compression="zstd")
    pretty_seg = df_seg.with_columns([
        pl.col("mean_tau").round(3),
        pl.col("lo").round(3),
        pl.col("hi").round(3),
    ])
    print(pretty_seg.to_pandas().to_string(index=False))

    # ----- Persist text summary -----
    lines = ["Per-SKU Kendall tau analysis (SEGMENT, revised v2)",
             "=" * 60,
             f"N (SKUs) = {N:,}    M (models) = {M}",
             f"Models: {model_names}",
             "",
             "Fleet-mean MAE and cost per model:",
             pl.DataFrame({
                 "MODEL": list(fleet["mean_mae"].keys()),
                 "mean_mae":  [fleet["mean_mae"][m]  for m in fleet["mean_mae"]],
                 "mean_cost": [fleet["mean_cost"][m] for m in fleet["mean_mae"]],
             }).sort("mean_cost").to_pandas().to_string(index=False),
             "",
             f"Per-SKU dominance: P(best-MAE = best-cost) = {dom['agreement_rate']:.4f}",
             "",
             f"Overall mean per-SKU tau = {overall_ci['mean']:+.4f}",
             f"  95% bootstrap CI = [{overall_ci['lo']:+.4f}, {overall_ci['hi']:+.4f}]",
             f"  N_valid = {overall_ci['n_valid']:,}",
             "",
             "Per-(Segment x ABC) tau:",
             pretty.to_pandas().to_string(index=False),
             "",
             "Per-Segment tau (collapsed):",
             pretty_seg.to_pandas().to_string(index=False),
             ]
    (ROB_DIR / "per_sku_tau_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nWrote {ROB_DIR / 'per_sku_tau_summary.txt'}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
