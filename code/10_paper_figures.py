"""
Stage-8: generate the five paper figures from the Stage-4..6 outputs.

Outputs:
  docs/paper/figures/fig1_segment_mae.png     per-segment MAE box-plots, 10 models
  docs/paper/figures/fig2_acc_vs_cost.png     scatter, one point per model
  docs/paper/figures/fig3_segment_heatmap.png segmentation x model cost-rank heatmap
  docs/paper/figures/fig4_cost_grid.png       cost-config sweep rho heatmap
  docs/paper/figures/fig5_deployed_vs_models.png  bar chart, models vs deployed

All figures use a neutral, journal-friendly palette and 300 DPI output.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

import config as C


SAMPLE_DIR = C.SAMPLE_DIR
ROB_DIR    = SAMPLE_DIR / "robustness"
FIG_DIR    = C.PROJECT_DIR / "docs" / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Stable model order for all figures (accuracy ranking from §6.1)
MODEL_ORDER = ["CHRONOS", "MA", "SES", "IMAPA", "ADIDA", "TSB",
               "HURDLE_NB", "ZIP", "SBA", "CROSTON", "LGBM"]


def style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    plt.rcParams["axes.spines.top"]   = False
    plt.rcParams["axes.spines.right"] = False


def fig1_segment_mae():
    """MAE distribution box-plots, faceted by segment, all 10 models."""
    blocks = []
    for path in [
        SAMPLE_DIR / "forecasts_classical.parquet",
        SAMPLE_DIR / "forecasts_distributional.parquet",
        SAMPLE_DIR / "forecasts_lgbm.parquet",
        SAMPLE_DIR / "forecasts_chronos.parquet",
    ]:
        if path.exists():
            blocks.append(pl.read_parquet(path).select(
                ["MODEL", "SEGMENTATION_GRP", "MAE_TEST"]
            ))
    df = pl.concat(blocks, how="vertical_relaxed").to_pandas()
    df = df[df["MAE_TEST"].between(0, df["MAE_TEST"].quantile(0.99))]
    df["MODEL"] = df["MODEL"].astype("category").cat.set_categories(MODEL_ORDER)

    g = sns.catplot(
        data=df, kind="box", col="SEGMENTATION_GRP",
        x="MODEL", y="MAE_TEST", order=MODEL_ORDER,
        col_order=["Smooth", "Intermittent", "Erratic", "Lumpy", "Unknown"],
        col_wrap=3, height=2.6, aspect=1.4, sharey=False,
        showfliers=False, color="#4C72B0",
    )
    g.set_xticklabels(rotation=45, ha="right")
    g.set_titles("{col_name}")
    g.set_axis_labels("", "Test MAE")
    g.figure.suptitle("Test-window MAE by model and segmentation", y=1.02)
    g.figure.savefig(FIG_DIR / "fig1_segment_mae.png", dpi=300, bbox_inches="tight")
    plt.close(g.figure)
    print(f"  fig1 done")


def fig2_acc_vs_cost():
    """Scatter of fleet-mean MAE vs fleet-mean total cost; one point per model."""
    sim = pl.read_parquet(SAMPLE_DIR / "sim_results.parquet")
    cost = (
        sim.group_by("MODEL")
           .agg(pl.col("TOTAL_COST").mean().alias("cost_mean"))
           .to_pandas()
    )
    blocks = []
    for path in [
        SAMPLE_DIR / "forecasts_classical.parquet",
        SAMPLE_DIR / "forecasts_distributional.parquet",
        SAMPLE_DIR / "forecasts_lgbm.parquet",
        SAMPLE_DIR / "forecasts_chronos.parquet",
    ]:
        if path.exists():
            blocks.append(pl.read_parquet(path).select(["MODEL", "MAE_TEST"]))
    mae = (
        pl.concat(blocks, how="vertical_relaxed")
          .group_by("MODEL")
          .agg(pl.col("MAE_TEST").mean().alias("mae_mean"))
          .to_pandas()
    )
    df = mae.merge(cost, on="MODEL")

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.scatter(df["mae_mean"], df["cost_mean"], s=80, color="#4C72B0",
               edgecolor="black", zorder=3)
    for _, row in df.iterrows():
        ax.annotate(row["MODEL"],
                    (row["mae_mean"], row["cost_mean"]),
                    textcoords="offset points", xytext=(7, 4),
                    fontsize=9)
    ax.set_xlabel("Fleet-mean test MAE  (lower = more accurate)")
    ax.set_ylabel("Fleet-mean total cost (USD)  (lower = cheaper)")
    ax.set_title("Accuracy vs cost across 11 forecasters\n"
                  "Spearman rho = +0.04  (p = 0.92)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_acc_vs_cost.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig2 done")


def fig3_segment_heatmap():
    """Heatmap: segmentation x model, cell value = cost-rank within segment."""
    sim = pl.read_parquet(SAMPLE_DIR / "sim_results.parquet")
    seg_cost = (
        sim.group_by(["SEGMENTATION_GRP", "MODEL"])
           .agg(pl.col("TOTAL_COST").mean().alias("c"))
           .to_pandas()
    )
    seg_cost["rank"] = seg_cost.groupby("SEGMENTATION_GRP")["c"].rank(method="min")
    pivot = seg_cost.pivot(index="SEGMENTATION_GRP", columns="MODEL", values="rank")
    pivot = pivot.reindex(["Smooth", "Intermittent", "Erratic", "Lumpy", "Unknown"])
    pivot = pivot.reindex(columns=MODEL_ORDER)

    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn_r", cbar=False,
                linewidths=0.5, ax=ax,
                annot_kws={"size": 9})
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"Cost rank by model within each segmentation group "
                  f"(1 = cheapest, {pivot.shape[1]} = most expensive)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_segment_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig3 done")


def fig4_cost_grid():
    """Distribution of rho across the 80-cell cost-config grid."""
    if not (ROB_DIR / "cost_grid.parquet").exists():
        print("  fig4 skipped (no cost_grid.parquet)")
        return
    g = pl.read_parquet(ROB_DIR / "cost_grid.parquet").to_pandas()

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    # Left: rho distribution histogram
    axes[0].hist(g["rho"], bins=20, color="#4C72B0", edgecolor="black")
    axes[0].axvline(0, color="grey", linestyle="--", linewidth=1)
    axes[0].set_xlabel(r"Spearman $\rho$(MAE-rank, cost-rank)")
    axes[0].set_ylabel("Count of cost configs")
    axes[0].set_title(f"rho across 80 cost configs\n"
                      f"median = {g['rho'].median():+.2f}, "
                      f"range [{g['rho'].min():+.2f}, {g['rho'].max():+.2f}]")

    # Right: best-cost-model frequency
    best_counts = g["best_cost_model"].value_counts()
    axes[1].barh(best_counts.index[::-1], best_counts.values[::-1],
                 color="#4C72B0", edgecolor="black")
    axes[1].set_xlabel("Configs in which model is cost-rank #1")
    axes[1].set_ylabel("")
    axes[1].set_title("Cost-rank winner across configs")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_cost_grid.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig4 done")


def fig5_deployed_vs_models():
    """Bar chart: model-derived costs ascending, deployed shown red on the right."""
    txt = (ROB_DIR / "deployed_compare.txt").read_text(encoding="utf-8")
    # Parse the trailing table — easier to re-load from parquet if we kept one,
    # but Stage-6 only persisted text. Re-derive from sim by restricting to
    # the same 561-SKU overlap.
    sim = pl.read_parquet(SAMPLE_DIR / "sim_results.parquet")
    pp = pl.read_parquet(SAMPLE_DIR / "policy_panel.parquet").select(
        ["STOCK_ITEM_NUMBER", "FORECAST_ID",
         "DEPLOYED_NEW_MIN_MEAN", "DEPLOYED_NEW_MAX_MEAN"]
    ).unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"])
    overlap = (
        sim.join(pp, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner")
    )
    cost = (
        overlap.group_by("MODEL")
               .agg(pl.col("TOTAL_COST").mean().alias("c"))
               .sort("c").to_pandas()
    )

    # Deployed cost from the saved file — extract via simple re-simulation we
    # already wrote in 09_robustness; here we just hardcode the computed
    # value for plotting purposes (it is also in deployed_compare.txt).
    # Read it back robustly:
    dep_cost = None
    for line in txt.splitlines():
        if "DEPLOYED" in line:
            parts = [p for p in line.split() if p]
            try:
                dep_cost = float(parts[2])
                break
            except (ValueError, IndexError):
                continue
    if dep_cost is None:
        print("  fig5 skipped (deployed cost not parseable)")
        return

    labels = list(cost["MODEL"]) + ["DEPLOYED"]
    values = list(cost["c"]) + [dep_cost]
    colors = ["#4C72B0"] * len(cost) + ["#C44E52"]

    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    bars = ax.bar(labels, values, color=colors, edgecolor="black")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 30, f"${v:,.0f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Mean total cost / SKU (USD), 8-quarter test window")
    ax.set_title(f"Deployed normal-approximation policy vs model-derived "
                  f"policies (n={overlap['STOCK_ITEM_NUMBER'].n_unique()} SKUs)")
    ax.set_ylim(0, max(values) * 1.18)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_deployed_vs_models.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig5 done")


def main():
    style()
    print("Generating paper figures...")
    fig1_segment_mae()
    fig2_acc_vs_cost()
    fig3_segment_heatmap()
    fig4_cost_grid()
    fig5_deployed_vs_models()
    print(f"\nAll figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
