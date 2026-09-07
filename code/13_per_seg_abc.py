"""
Stage-6 reinforcement: per-(Segmentation x ABC) double-stratification of
the Spearman rho(MAE-rank, cost-rank). Goal: confirm that the
segment-localized accuracy-cost divergence holds *within* ABC tiers as
well — i.e. it isn't just a confounder where Lumpy SKUs happen to be
higher-ABC.

Outputs:
  output/sample/robustness/per_seg_abc.parquet
  output/sample/robustness/per_seg_abc.txt
  docs/paper/figures/fig6_seg_abc_heatmap.png
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl
from scipy.stats import spearmanr

import config as C


SAMPLE_DIR = C.SAMPLE_DIR
ROB_DIR    = SAMPLE_DIR / "robustness"
FIG_DIR    = C.PROJECT_DIR / "docs" / "paper" / "figures"
ROB_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = 20
TEST_LEN  = 28 - TRAIN_END


def load_inputs() -> tuple[pl.DataFrame, dict[str, dict[str, float]]]:
    """Returns (sim_results_with_segments, model_seg_abc_mae_table)."""
    sim = pl.read_parquet(SAMPLE_DIR / "sim_results.parquet")

    # MAE per (Model, SEGMENTATION_GRP, ABC_GRP)
    blocks = []
    for path in [
        SAMPLE_DIR / "forecasts_classical.parquet",
        SAMPLE_DIR / "forecasts_distributional.parquet",
        SAMPLE_DIR / "forecasts_lgbm.parquet",
        SAMPLE_DIR / "forecasts_chronos.parquet",
    ]:
        if path.exists():
            blocks.append(pl.read_parquet(path).select(
                ["MODEL", "SEGMENTATION_GRP", "ABC_GRP", "MAE_TEST"]
            ))
    mae = pl.concat(blocks, how="vertical_relaxed")
    return sim, mae


def main():
    t0 = time.time()
    sim, mae = load_inputs()

    seg_levels = ["Smooth", "Erratic", "Intermittent", "Lumpy", "Unknown"]
    abc_levels = ["A", "B", "C", "Unknown"]

    rows = []
    for seg in seg_levels:
        for abc in abc_levels:
            sub_sim = sim.filter(
                (pl.col("SEGMENTATION_GRP") == seg) & (pl.col("ABC_GRP") == abc)
            )
            sub_mae = mae.filter(
                (pl.col("SEGMENTATION_GRP") == seg) & (pl.col("ABC_GRP") == abc)
            )
            n_skus = sub_sim["STOCK_ITEM_NUMBER"].n_unique()
            if n_skus < 10:
                rows.append({"segment": seg, "abc": abc, "n_skus": n_skus,
                              "rho": None, "p": None, "n_models": 0})
                continue

            cost_rank = (
                sub_sim.group_by("MODEL").agg(pl.col("TOTAL_COST").mean().alias("c"))
                       .sort("c").with_row_index("rank")
            )
            mae_rank = (
                sub_mae.group_by("MODEL").agg(pl.col("MAE_TEST").mean().alias("m"))
                       .sort("m").with_row_index("rank")
            )
            joined = (
                cost_rank.rename({"rank": "cost_rank"})
                          .join(mae_rank.rename({"rank": "mae_rank"}), on="MODEL", how="inner")
            )
            if joined.height < 5:
                rows.append({"segment": seg, "abc": abc, "n_skus": n_skus,
                              "rho": None, "p": None, "n_models": joined.height})
                continue
            rho, p = spearmanr(
                joined["mae_rank"].to_numpy(),
                joined["cost_rank"].to_numpy(),
            )
            rows.append({"segment": seg, "abc": abc, "n_skus": n_skus,
                          "rho": float(rho), "p": float(p),
                          "n_models": joined.height})

    df = pl.DataFrame(rows)
    df.write_parquet(ROB_DIR / "per_seg_abc.parquet", compression="zstd")

    # Text summary
    pretty = df.with_columns([
        pl.col("rho").round(3),
        pl.col("p").round(3),
    ])
    out = "Per-(Segment x ABC) Spearman rho across 11 forecasters\n"
    out += "(cells with n_skus<10 are reported as null; 5+ models required)\n\n"
    out += pretty.to_pandas().to_string(index=False)
    (ROB_DIR / "per_seg_abc.txt").write_text(out, encoding="utf-8")
    print(out)

    # Heatmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    pivot = (
        df.filter(pl.col("rho").is_not_null())
          .pivot(index="segment", on="abc", values="rho")
          .to_pandas()
          .set_index("segment")
          .reindex(index=seg_levels, columns=abc_levels)
    )

    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, vmin=-1, vmax=1, cbar_kws={"label": r"Spearman $\rho$"},
                linewidths=0.5, ax=ax,
                annot_kws={"size": 10})
    ax.set_xlabel("ABC class")
    ax.set_ylabel("Segmentation")
    ax.set_title(r"Spearman $\rho$(MAE-rank, cost-rank) by Segment x ABC,"
                  "\n11 forecasters; cells with <10 SKUs blank")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_seg_abc_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfig6 saved to {FIG_DIR / 'fig6_seg_abc_heatmap.png'}")
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
