"""
Stage-6 reinforcement: vary service-level tiers and re-run the
simulator across all 11 forecasters. Reinforces finding (3) — that the
deployed-policy comparison is robust to SL choice — and complements
finding (1) by showing whether the segment-localised divergence depends
on SL.

Scenarios:
  default     Critical 0.99 / Normal 0.97 / Low 0.95   (paper baseline)
  conservative all 0.99
  aggressive   all 0.90
  highly-tiered Critical 0.995 / Normal 0.97 / Low 0.90
  flat-95      all 0.95

Outputs:
  output/sample/robustness/sl_sensitivity.parquet      cost per (model, scenario)
  output/sample/robustness/sl_sensitivity.txt          text table
  docs/paper/figures/fig7_sl_sensitivity.png           cost-by-scenario chart
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import polars as pl
from scipy.stats import spearmanr

import config as C
from simulator import (
    CostConfig, normal_policy, simulate, lead_days_to_qtrs, service_level_for,
)


SAMPLE_DIR = C.SAMPLE_DIR
ROB_DIR    = SAMPLE_DIR / "robustness"
FIG_DIR    = C.PROJECT_DIR / "docs" / "paper" / "figures"

TRAIN_END = 20
TEST_LEN  = 28 - TRAIN_END
ACTUAL_COLS = [f"ACTUALS_QTR{i:02d}" for i in range(1, 29)]

SCENARIOS = {
    "default":      dict(sl_critical=0.99,  sl_normal=0.97, sl_low=0.95),
    "conservative": dict(sl_critical=0.99,  sl_normal=0.99, sl_low=0.99),
    "aggressive":   dict(sl_critical=0.90,  sl_normal=0.90, sl_low=0.90),
    "highly_tiered": dict(sl_critical=0.995, sl_normal=0.97, sl_low=0.90),
    "flat_95":      dict(sl_critical=0.95,  sl_normal=0.95, sl_low=0.95),
}


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
                  .select(["STOCK_ITEM_NUMBER", "FORECAST_ID", "MAE_TEST"]
                          + [f"FORECAST_QTR{TRAIN_END + 1:02d}"])
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


def simulate_scenario(panel, models, cfg):
    actuals = panel.select(ACTUAL_COLS).to_numpy().astype(np.float64)
    train = actuals[:, :TRAIN_END]
    test  = actuals[:, TRAIN_END:]
    unit_price = panel["UNIT_PRICE_USD"].fill_null(1.0).cast(pl.Float64).to_numpy()
    unit_price = np.where(unit_price > 0, unit_price, 1.0)
    lead_days = panel["LEAD_TIME_MEAN"].fill_null(60.0).cast(pl.Float64).to_numpy()
    lead_qtrs = lead_days_to_qtrs(np.where(lead_days > 0, lead_days, 60.0))
    lead_std_qtrs = (lead_days * 0.3) / 91.3125
    sigma = train.std(axis=1, ddof=0)
    crit = panel["CRITICALITY_MODE"].fill_null("Normal").to_list()
    sl = np.array([service_level_for(c, cfg) for c in crit])

    rows = []
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
            "MODEL": name,
            "mean_total":    float(sim["total_cost"].mean()),
            "median_total":  float(np.median(sim["total_cost"])),
            "mean_fill":     float(sim["fill_rate"].mean()),
            "mean_holding":  float(sim["holding_cost"].mean()),
            "mean_stockout": float(sim["stockout_units"].mean()),
            "mae":           float(m["mae"].mean()),
        })
    return pl.DataFrame(rows)


def main():
    t0 = time.time()
    panel = load_panel()
    models = collect_models(panel)
    print(f"Panel rows: {panel.height:,}    models: {list(models)}\n")

    all_rows = []
    rho_summary = []
    for name, kwargs in SCENARIOS.items():
        cfg = replace(CostConfig(), **kwargs)
        t1 = time.time()
        df = simulate_scenario(panel, models, cfg)
        df = df.with_columns(pl.lit(name).alias("scenario")).sort("mean_total")
        all_rows.append(df)
        # Spearman across this scenario
        cost_rank = df.with_row_index("cost_rank").select(["MODEL", "cost_rank"])
        mae_rank = df.sort("mae").with_row_index("mae_rank").select(["MODEL", "mae_rank"])
        joined = cost_rank.join(mae_rank, on="MODEL", how="inner")
        rho, p = spearmanr(joined["mae_rank"].to_numpy(),
                            joined["cost_rank"].to_numpy())
        rho_summary.append({
            "scenario": name,
            "sl_critical": kwargs["sl_critical"],
            "sl_normal":   kwargs["sl_normal"],
            "sl_low":      kwargs["sl_low"],
            "rho": float(rho),
            "p":   float(p),
            "best_cost": df["MODEL"][0],
            "best_acc":  df.sort("mae")["MODEL"][0],
            "mean_total_best": float(df["mean_total"][0]),
            "fill_mean": float(df["mean_fill"].mean()),
        })
        print(f"  scenario {name:<14}  best_cost={df['MODEL'][0]:<10} "
              f"best_acc={df.sort('mae')['MODEL'][0]:<10} "
              f"rho={rho:+.3f}  fill_mean={df['mean_fill'].mean():.3f}  "
              f"({time.time() - t1:.1f}s)")

    out = pl.concat(all_rows, how="vertical_relaxed")
    out.write_parquet(ROB_DIR / "sl_sensitivity.parquet", compression="zstd")

    summary = pl.DataFrame(rho_summary)
    text = "Service-level sensitivity\n"
    text += "Scenarios: default = Crit 0.99/Norm 0.97/Low 0.95\n"
    text += "           conservative = all 0.99\n"
    text += "           aggressive   = all 0.90\n"
    text += "           highly_tiered = 0.995/0.97/0.90\n"
    text += "           flat_95      = all 0.95\n\n"
    text += summary.to_pandas().to_string(index=False)
    (ROB_DIR / "sl_sensitivity.txt").write_text(text, encoding="utf-8")
    print("\n" + text)

    # Figure: cost by model x scenario heatmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    pivot = (
        out.pivot(index="MODEL", on="scenario", values="mean_total")
           .to_pandas()
           .set_index("MODEL")
    )
    pivot = pivot[list(SCENARIOS.keys())]
    pivot = pivot.sort_values("default")
    fig, ax = plt.subplots(figsize=(8, 4.0))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn_r",
                cbar_kws={"label": "Mean total cost / SKU (USD)"},
                linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_xlabel("Service-level scenario")
    ax.set_ylabel("Forecaster")
    ax.set_title("Mean total cost / SKU under varying service-level tiers")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_sl_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfig7 saved to {FIG_DIR / 'fig7_sl_sensitivity.png'}")
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
