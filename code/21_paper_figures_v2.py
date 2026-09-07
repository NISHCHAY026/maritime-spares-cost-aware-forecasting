"""
Stage-V2 paper figures (SEGMENT revision).

The figure set is reorganised around the new headline findings:

  fig1_tau_distribution.png   per-SKU Kendall tau histogram + per-segment
                              CIs. Replaces fig2 (acc-vs-cost scatter as
                              the headline figure).
  fig2_acc_vs_cost.png        accuracy vs cost scatter — same axes as
                              before but now labelled with mean tau and
                              its 95% CI; supports the positive-rho
                              story rather than the divergence story.
  fig3_seg_heatmap.png        cost rank by Segmentation x Model. Story
                              shifts to: 'no single model dominates' —
                              still informative but no longer the main
                              figure.
  fig4_policy_compare.png     bar chart of mean cost by model under
                              normal vs native-quantile policy. THE
                              new central figure.
  fig5_deployed_compare.png   deployed vs each model on the 1,579-SKU
                              overlap (SEGMENT).
  fig6_seg_abc_tau.png        per-(Seg x ABC) Kendall tau heatmap with
                              CI annotations. Replaces v0.5 fig6.
  fig7_lgbm_local_global.png  per-SKU LGBM vs global LGBM by segment.
                              Reviewer-defence figure for asymmetric-
                              training critique.
"""

from __future__ import annotations

import json
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
FIG_DIR    = C.PROJECT_DIR / "docs" / "paper" / "figures_v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["CHRONOS", "MA", "SES", "IMAPA", "ADIDA", "TSB",
               "HURDLE_NB", "ZIP", "SBA", "CROSTON", "LGBM"]
SEG_ORDER = ["Smooth", "Erratic", "Intermittent", "Lumpy", "Unknown"]
ABC_ORDER = ["A", "B", "C", "Unknown"]


def style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    plt.rcParams["axes.spines.top"]   = False
    plt.rcParams["axes.spines.right"] = False


def fig1_tau_distribution():
    """Per-SKU tau histogram + per-segment CI bars."""
    seg = pl.read_parquet(ROB_DIR / "per_sku_tau_segments.parquet").to_pandas()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8),
                              gridspec_kw={"width_ratios": [1, 1]})

    # Left: per-segment tau with CIs (point + error bars)
    seg = seg.set_index("segment").reindex(SEG_ORDER).dropna(how="all").reset_index()
    y = np.arange(len(seg))
    axes[0].errorbar(
        seg["mean_tau"], y,
        xerr=[seg["mean_tau"] - seg["lo"], seg["hi"] - seg["mean_tau"]],
        fmt="o", color="#4C72B0", ecolor="#4C72B0", capsize=4,
        markersize=8, linewidth=1.5,
    )
    axes[0].axvline(0, color="grey", linestyle="--", linewidth=1)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{s} (n={int(n):,})"
                              for s, n in zip(seg["segment"], seg["n_skus"])])
    axes[0].set_xlabel(r"Mean per-SKU Kendall's $\tau$")
    axes[0].set_xlim(-0.1, 1.0)
    axes[0].set_title(r"Per-segment $\tau$ with 95% bootstrap CI"
                       "\n(SKU-level resampling, B=1000)")

    # Right: distribution panel — cell-level dots
    cells = pl.read_parquet(ROB_DIR / "per_sku_tau_cells.parquet").to_pandas()
    cells = cells.dropna(subset=["mean_tau"])
    axes[1].errorbar(
        cells["mean_tau"], np.arange(len(cells)),
        xerr=[cells["mean_tau"] - cells["lo"], cells["hi"] - cells["mean_tau"]],
        fmt="o", color="#55A868", ecolor="#55A868", capsize=2,
        markersize=4, linewidth=0.8,
    )
    axes[1].axvline(0, color="grey", linestyle="--", linewidth=1)
    axes[1].set_yticks(np.arange(len(cells)))
    axes[1].set_yticklabels([f"{r['segment']} × {r['abc']} (n={int(r['n_skus']):,})"
                              for _, r in cells.iterrows()],
                             fontsize=8)
    axes[1].set_xlabel(r"Mean per-SKU Kendall's $\tau$")
    axes[1].set_xlim(-0.1, 1.0)
    axes[1].set_title(r"Per-(Segment $\times$ ABC) $\tau$, B=500")
    axes[1].invert_yaxis()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_tau_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  fig1 done")


def fig2_acc_vs_cost():
    """Scatter of fleet-mean MAE vs fleet-mean cost (normal-policy), N=15k."""
    sim = pl.read_parquet(SAMPLE_DIR / "sim_results.parquet")
    cost = (sim.group_by("MODEL").agg(pl.col("TOTAL_COST").mean().alias("c"))
                .to_pandas())
    blocks = []
    for path in [
        SAMPLE_DIR / "forecasts_classical.parquet",
        SAMPLE_DIR / "forecasts_distributional.parquet",
        SAMPLE_DIR / "forecasts_lgbm.parquet",
        SAMPLE_DIR / "forecasts_chronos.parquet",
    ]:
        if path.exists():
            blocks.append(pl.read_parquet(path).select(["MODEL", "MAE_TEST"]))
    mae = (pl.concat(blocks, how="vertical_relaxed")
              .group_by("MODEL").agg(pl.col("MAE_TEST").mean().alias("m"))
              .to_pandas())
    df = mae.merge(cost, on="MODEL")

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.scatter(df["m"], df["c"], s=80, color="#4C72B0", edgecolor="black", zorder=3)
    for _, row in df.iterrows():
        ax.annotate(row["MODEL"], (row["m"], row["c"]),
                    textcoords="offset points", xytext=(7, 4), fontsize=9)
    ax.set_xlabel("Fleet-mean test MAE  (lower = more accurate)")
    ax.set_ylabel("Fleet-mean total cost (USD)  (lower = cheaper)")

    # Pull mean tau from summary file
    tau_path = ROB_DIR / "per_sku_tau_summary.txt"
    tau_str = ""
    if tau_path.exists():
        for ln in tau_path.read_text().splitlines():
            if ln.strip().startswith("Overall mean per-SKU tau"):
                tau_str = ln.strip()
                break

    ax.set_title("Accuracy vs cost across 11 forecasters "
                  "(SEGMENT, normal-approx policy)\n"
                  "per-SKU Kendall's tau = +0.49 on demand-active SKUs "
                  "(95% CI [+0.47, +0.50], N=5,215)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_acc_vs_cost.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  fig2 done")


def fig3_seg_heatmap():
    sim = pl.read_parquet(SAMPLE_DIR / "sim_results.parquet")
    seg_cost = (sim.group_by(["SEGMENTATION_GRP", "MODEL"])
                   .agg(pl.col("TOTAL_COST").mean().alias("c"))
                   .to_pandas())
    seg_cost["rank"] = seg_cost.groupby("SEGMENTATION_GRP")["c"].rank(method="min")
    pivot = seg_cost.pivot(index="SEGMENTATION_GRP", columns="MODEL", values="rank")
    pivot = pivot.reindex(SEG_ORDER).reindex(columns=MODEL_ORDER)
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn_r", cbar=False,
                linewidths=0.5, ax=ax, annot_kws={"size": 9})
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_title(f"Cost rank by model within each segmentation group "
                  f"(1=cheapest, {pivot.shape[1]}=most expensive)  SEGMENT-only")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_seg_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  fig3 done")


def fig4_policy_compare():
    """
    Native-quantile policy: naive (unmatched) vs fill-matched.
    Three bars per quantile-emitting model:
      normal-approx cost | native-quantile cost (naive, default alpha)
                         | native-quantile cost (fill-matched)
    The naive bar looks cheap; the fill-matched bar shows the advantage
    disappears once service level is held equal.
    """
    sim_n = pl.read_parquet(SAMPLE_DIR / "sim_results.parquet")
    sim_q = pl.read_parquet(SAMPLE_DIR / "sim_results_native.parquet")
    matched = pl.read_parquet(SAMPLE_DIR / "sim_results_native_matched.parquet").to_pandas()

    n = (sim_n.group_by("MODEL").agg(pl.col("TOTAL_COST").mean().alias("normal"))
              .to_pandas())
    q = (sim_q.group_by("MODEL").agg(pl.col("TOTAL_COST").mean().alias("naive"))
              .to_pandas())

    models = ["ZIP", "HURDLE_NB", "LGBM", "CHRONOS"]
    rows = []
    for m in models:
        normal = float(n.loc[n["MODEL"] == m, "normal"].iloc[0])
        naive = float(q.loc[q["MODEL"] == m, "naive"].iloc[0])
        mrow = matched[matched["MODEL"] == m].iloc[0]
        rows.append({"MODEL": m, "normal": normal, "naive": naive,
                     "matched": float(mrow["cost_native_all"]),
                     "kappa_capped": not bool(mrow["matched"])})
    df = pd_df(rows)

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    x = np.arange(len(df))
    w = 0.27
    ax.bar(x - w, df["normal"], w, label="Normal-approx policy",
           color="#4C72B0", edgecolor="black")
    ax.bar(x,      df["naive"],  w, label="Native-quantile (naive, lower fill)",
           color="#DD8452", edgecolor="black")
    ax.bar(x + w,  df["matched"], w, label="Native-quantile (fill-matched)",
           color="#55A868", edgecolor="black")

    ymax = 6500
    ax.set_ylim(0, ymax)
    for i, r in df.iterrows():
        # naive delta
        dn = (r["naive"] - r["normal"]) / r["normal"] * 100
        ax.text(i, min(r["naive"], ymax) + 80, f"{dn:+.0f}%",
                ha="center", va="bottom", fontsize=7.5, color="#9F4A1E")
        # matched delta (may be off-scale)
        dm = (r["matched"] - r["normal"]) / r["normal"] * 100
        if r["matched"] > ymax:
            ax.annotate(f"${r['matched']:,.0f}\n({dm:+.0f}%)",
                        xy=(i + w, ymax), xytext=(i + w, ymax - 900),
                        ha="center", fontsize=7.5, color="#9F2D2D",
                        arrowprops=dict(arrowstyle="-|>", color="#9F2D2D"))
        else:
            ax.text(i + w, r["matched"] + 80, f"{dm:+.0f}%",
                    ha="center", va="bottom", fontsize=7.5,
                    color="#2A6F45" if dm < 0 else "#9F2D2D")
        if r["kappa_capped"]:
            ax.text(i + w, 250, "κ capped", ha="center", fontsize=6.5,
                    rotation=90, color="white", weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(df["MODEL"], rotation=0)
    ax.set_ylabel("Mean total cost / SKU (USD)")
    ax.set_title("Native-quantile policy: the naive cost saving is a service cut\n"
                  "At matched fill rate the advantage disappears (SEGMENT, 15,348 SKUs)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_policy_compare.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  fig4 done")


def pd_df(rows):
    import pandas as pd
    return pd.DataFrame(rows)


def fig8_calibration():
    """Predictive-quantile calibration: empirical coverage vs nominal."""
    tcols = [f"ACTUALS_QTR{i:02d}" for i in range(21, 29)]
    levels = [0.50, 0.80, 0.90, 0.95, 0.99]
    qcols = {0.50: "Q_50", 0.80: "Q_80", 0.90: "Q_90", 0.95: "Q_95", 0.99: "Q_99"}
    srcs = {
        "ZIP":       (SAMPLE_DIR / "forecasts_distributional.parquet", "ZIP"),
        "HURDLE_NB": (SAMPLE_DIR / "forecasts_distributional.parquet", "HURDLE_NB"),
        "LGBM":      (SAMPLE_DIR / "forecasts_lgbm.parquet", "LGBM"),
        "CHRONOS":   (SAMPLE_DIR / "forecasts_chronos.parquet", "CHRONOS"),
    }
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.plot([0.5, 1.0], [0.5, 1.0], "--", color="grey", linewidth=1,
            label="perfect calibration")
    palette = {"ZIP": "#4C72B0", "HURDLE_NB": "#DD8452",
               "LGBM": "#55A868", "CHRONOS": "#C44E52"}
    for name, (path, model) in srcs.items():
        if not path.exists():
            continue
        df = pl.read_parquet(path).filter(pl.col("MODEL") == model)
        actuals = df.select(tcols).to_numpy()
        cov = []
        for q in levels:
            qc = qcols[q]
            if qc not in df.columns:
                cov.append(np.nan); continue
            qv = df[qc].to_numpy()[:, None]
            cov.append(float((actuals <= qv).mean()))
        ax.plot(levels, cov, "o-", color=palette[name], label=name,
                markersize=5, linewidth=1.5)
    ax.set_xlabel("Nominal quantile level  q")
    ax.set_ylabel("Empirical coverage  P(actual ≤ predicted Q$_q$)")
    ax.set_title("Predictive-quantile calibration (SEGMENT test window)\n"
                  "Lines above the diagonal: distribution mis-shapen;\n"
                  "Q99 below 0.99: upper tail too thin for safety stock")
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig8_calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  fig8 done")


def pd_isna(x):
    try:
        import math
        return math.isnan(x)
    except (TypeError, ValueError):
        return x is None


def fig5_deployed_compare():
    """Deployed vs models bar chart on SEGMENT overlap."""
    txt = (ROB_DIR / "deployed_compare.txt").read_text(encoding="utf-8")
    sim = pl.read_parquet(SAMPLE_DIR / "sim_results.parquet")
    pp = pl.read_parquet(SAMPLE_DIR / "policy_panel.parquet").select(
        ["STOCK_ITEM_NUMBER", "FORECAST_ID"]
    ).unique()
    overlap = sim.join(pp, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner")
    cost = (overlap.group_by("MODEL")
                    .agg(pl.col("TOTAL_COST").mean().alias("c"))
                    .sort("c").to_pandas())
    dep_cost = None
    for line in txt.splitlines():
        if "DEPLOYED" in line:
            parts = [p for p in line.split() if p]
            try:
                dep_cost = float(parts[2]); break
            except (ValueError, IndexError):
                continue
    if dep_cost is None:
        return
    labels = list(cost["MODEL"]) + ["DEPLOYED"]
    values = list(cost["c"]) + [dep_cost]
    colors = ["#4C72B0"] * len(cost) + ["#C44E52"]
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    bars = ax.bar(labels, values, color=colors, edgecolor="black")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 30, f"${v:,.0f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Mean total cost / SKU (USD), 8-quarter test window")
    ax.set_title(f"Deployed normal-approximation vs model-derived policies\n"
                  f"SEGMENT overlap, n={overlap['STOCK_ITEM_NUMBER'].n_unique()} SKUs")
    ax.set_ylim(0, max(values) * 1.18)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_deployed_compare.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  fig5 done")


def fig6_seg_abc_tau():
    cells = pl.read_parquet(ROB_DIR / "per_sku_tau_cells.parquet").to_pandas()
    cells = cells.dropna(subset=["mean_tau"])
    pivot = cells.pivot(index="segment", columns="abc", values="mean_tau")
    pivot = pivot.reindex(index=SEG_ORDER, columns=ABC_ORDER)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, vmin=-0.2, vmax=1.0,
                cbar_kws={"label": r"Mean per-SKU $\tau$"},
                linewidths=0.5, ax=ax, annot_kws={"size": 10})
    ax.set_xlabel("ABC class")
    ax.set_ylabel("Segmentation")
    ax.set_title(r"Mean per-SKU Kendall's $\tau$(MAE-rank, cost-rank) by Segment $\times$ ABC"
                  "\n(SEGMENT; cells with n<30 left blank)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_seg_abc_tau.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  fig6 done")


def fig7_lgbm_local_global():
    if not (SAMPLE_DIR / "lgbm_global_vs_local.txt").exists():
        return
    # Re-derive from forecast files for cleaner data
    blocks = []
    for src, m_label in [
        (SAMPLE_DIR / "forecasts_lgbm_per_sku.parquet", "LGBM_LOCAL"),
        (SAMPLE_DIR / "forecasts_lgbm.parquet", "LGBM_GLOBAL"),
        (SAMPLE_DIR / "forecasts_classical.parquet", "SES"),
        (SAMPLE_DIR / "forecasts_chronos.parquet", "CHRONOS"),
    ]:
        if not src.exists():
            continue
        df = pl.read_parquet(src)
        if "MODEL" in df.columns and src.name != "forecasts_lgbm_per_sku.parquet":
            df = df.filter(pl.col("MODEL") == m_label)
        df = (df.unique(subset=["STOCK_ITEM_NUMBER"])
                .group_by("SEGMENTATION_GRP")
                .agg(pl.col("MAE_TEST").mean().alias(m_label))
                .sort("SEGMENTATION_GRP"))
        blocks.append(df)
    full = blocks[0]
    for b in blocks[1:]:
        full = full.join(b, on="SEGMENTATION_GRP", how="full", coalesce=True)
    pdf = full.to_pandas().set_index("SEGMENTATION_GRP").reindex(SEG_ORDER)
    cols = [c for c in ["LGBM_LOCAL", "LGBM_GLOBAL", "SES", "CHRONOS"]
            if c in pdf.columns]
    pdf = pdf[cols]
    fig, ax = plt.subplots(figsize=(7, 4))
    pdf.plot(kind="bar", ax=ax, color=["#4C72B0", "#55A868", "#C44E52", "#8172B3"],
              edgecolor="black", width=0.85)
    ax.set_ylabel("Fleet-mean test MAE")
    ax.set_xlabel("")
    ax.set_title("Per-SKU vs global LightGBM (cross-learning ablation)\n"
                  "SEGMENT, by segmentation; SES + Chronos for context")
    plt.setp(ax.get_xticklabels(), rotation=0)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_lgbm_local_global.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  fig7 done")


def main():
    style()
    print("Generating SEGMENT paper figures...")
    fig1_tau_distribution()
    fig2_acc_vs_cost()
    fig3_seg_heatmap()
    fig4_policy_compare()
    fig5_deployed_compare()
    fig6_seg_abc_tau()
    fig7_lgbm_local_global()
    fig8_calibration()
    print(f"\nAll figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
