"""
Stage-6 robustness checks for the accuracy-vs-cost finding.

Runs four analyses in one script:

  R1. Cost-config sweep.   Vary holding rate, ordering cost K, and stockout
                           penalty across a grid; for each combination
                           recompute the (acc-rank, cost-rank) Spearman
                           rho. The headline finding is robust if rho
                           stays near zero across configs.

  R2. Per-segment.         Recompute rho within each SEGMENTATION_GRP and
                           ABC_GRP bucket — does the divergence hold
                           inside Intermittent / Lumpy / Smooth / etc?

  R3. Bootstrap CI.        Resample SKUs with replacement, recompute
                           rho per resample, report 95% CI. If 0 sits
                           comfortably inside, accuracy and cost ranks
                           are statistically indistinguishable from
                           independent.

  R4. Deployed policy.     For the 6,281 SKUs in the policy panel
                           (Stockmax intersection), simulate the
                           deployed New Min / New Max directly and
                           compare to every model's cost.

Outputs:
  output/sample/robustness/cost_grid.parquet
  output/sample/robustness/per_segment.txt
  output/sample/robustness/bootstrap_ci.txt
  output/sample/robustness/deployed_compare.txt
"""

from __future__ import annotations

import time
from itertools import product
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Loaders (lifted from 08_run_simulator with dedupe)
# ---------------------------------------------------------------------------

def load_panel():
    sample = pl.scan_parquet(SAMPLE_DIR / "sample_skus.parquet").select([
        "STOCK_ITEM_NUMBER", "FORECAST_ID",
        "SEGMENTATION_GRP", "ABC_GRP", "STRATEGY_GRP",
        "VELOCITY_MODE", "CRITICALITY_MODE",
        "UNIT_PRICE_USD", "LEAD_TIME_MEAN",
    ]).unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect()
    panel = pl.scan_parquet(SAMPLE_DIR / "panel.parquet").select(
        ["STOCK_ITEM_NUMBER", "FORECAST_ID"] + ACTUAL_COLS
    ).unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect()
    return sample.join(panel, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner")


def load_forecasts() -> dict[str, pl.DataFrame]:
    sources = [
        SAMPLE_DIR / "forecasts_classical.parquet",
        SAMPLE_DIR / "forecasts_distributional.parquet",
        SAMPLE_DIR / "forecasts_lgbm.parquet",
        SAMPLE_DIR / "forecasts_chronos.parquet",
    ]
    forecasts = {}
    for path in sources:
        if not path.exists():
            continue
        df = pl.scan_parquet(path).collect()
        for model in df["MODEL"].unique().to_list():
            mdf = (
                df.filter(pl.col("MODEL") == model)
                  .select(
                      ["STOCK_ITEM_NUMBER", "FORECAST_ID", "MAE_TEST"]
                      + [f"FORECAST_QTR{TRAIN_END + 1 + q:02d}" for q in range(TEST_LEN)]
                  )
                  .unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"])
            )
            forecasts[model] = mdf
    return forecasts


def aligned_inputs(panel: pl.DataFrame, forecasts: dict[str, pl.DataFrame]):
    """Return per-SKU arrays (actuals, prices, etc.) and a function that
    yields the model's first-quarter forecast aligned to panel order."""
    actuals = panel.select(ACTUAL_COLS).to_numpy().astype(np.float64)
    train = actuals[:, :TRAIN_END]
    test  = actuals[:, TRAIN_END:]
    unit_price = panel["UNIT_PRICE_USD"].fill_null(1.0).cast(pl.Float64).to_numpy()
    unit_price = np.where(unit_price > 0, unit_price, 1.0)
    lead_days = panel["LEAD_TIME_MEAN"].fill_null(60.0).cast(pl.Float64).to_numpy()
    lead_days = np.where(lead_days > 0, lead_days, 60.0)
    lead_qtrs = lead_days_to_qtrs(lead_days)
    lead_std_qtrs = (lead_days * 0.3) / 91.3125
    sigma = train.std(axis=1, ddof=0)

    def get_q21(model: str) -> np.ndarray:
        joined = panel.select(["STOCK_ITEM_NUMBER", "FORECAST_ID"]).join(
            forecasts[model], on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="left",
        )
        return joined[f"FORECAST_QTR{TRAIN_END + 1:02d}"].fill_null(0.0).to_numpy()

    return {
        "train": train, "test": test, "sigma": sigma,
        "unit_price": unit_price, "lead_qtrs": lead_qtrs,
        "lead_std_qtrs": lead_std_qtrs, "get_q21": get_q21,
    }


# ---------------------------------------------------------------------------
# Simulate every model under a given cost config
# ---------------------------------------------------------------------------

def simulate_all_models(panel: pl.DataFrame, forecasts: dict, inputs: dict,
                        cfg: CostConfig) -> pl.DataFrame:
    crit = panel["CRITICALITY_MODE"].fill_null("Normal").to_list()
    service_level = np.array([service_level_for(c, cfg) for c in crit])

    rows = []
    for model in forecasts:
        forecast_q21 = inputs["get_q21"](model)
        pol = normal_policy(
            forecast_mean_per_qtr=forecast_q21,
            forecast_std_per_qtr=inputs["sigma"],
            lead_qtrs=inputs["lead_qtrs"].astype(np.float64),
            lead_std_qtrs=inputs["lead_std_qtrs"],
            unit_price=inputs["unit_price"],
            service_level=service_level,
            cfg=cfg,
        )
        sim = simulate(
            actuals_test=inputs["test"],
            s=pol["s"], S=pol["S"],
            lead_qtrs=inputs["lead_qtrs"], unit_price=inputs["unit_price"], cfg=cfg,
        )
        block = panel.select([
            "STOCK_ITEM_NUMBER", "FORECAST_ID",
            "SEGMENTATION_GRP", "ABC_GRP",
        ]).with_columns([
            pl.lit(model).alias("MODEL"),
            pl.Series("TOTAL_COST", sim["total_cost"]),
            pl.Series("FILL_RATE",  sim["fill_rate"]),
            pl.Series("STOCKOUT_UNITS", sim["stockout_units"]),
            pl.Series("HOLDING_COST", sim["holding_cost"]),
        ])
        rows.append(block)
    return pl.concat(rows, how="vertical_relaxed")


def acc_cost_rank_corr(sim_df: pl.DataFrame,
                       mae_by_model: dict[str, float]) -> tuple[float, float]:
    """Spearman ρ between MAE-rank and total-cost-rank across models."""
    fleet_cost = (
        sim_df.group_by("MODEL")
              .agg(pl.col("TOTAL_COST").mean().alias("total"))
              .sort("total")
    )
    cost_order = fleet_cost["MODEL"].to_list()
    mae_order = sorted(mae_by_model.keys(), key=lambda m: mae_by_model[m])
    common = [m for m in mae_order if m in cost_order]
    acc_ranks = {m: i + 1 for i, m in enumerate(mae_order) if m in common}
    cost_ranks = {m: cost_order.index(m) + 1 for m in common}
    a = np.array([acc_ranks[m] for m in common])
    c = np.array([cost_ranks[m] for m in common])
    rho, p = spearmanr(a, c)
    return float(rho), float(p)


# ---------------------------------------------------------------------------
# R1. Cost-config sweep
# ---------------------------------------------------------------------------

def r1_cost_grid(panel, forecasts, inputs, mae_by_model):
    holdings   = [0.15, 0.20, 0.25, 0.30, 0.35]
    orderings  = [25.0, 50.0, 100.0, 250.0]
    stockouts  = [50.0, 100.0, 250.0, 500.0]
    rows = []
    for h, k, p in product(holdings, orderings, stockouts):
        cfg = CostConfig(holding_rate_annual=h, ordering_cost=k,
                          stockout_unit_cost=p)
        sim_df = simulate_all_models(panel, forecasts, inputs, cfg)
        rho, pval = acc_cost_rank_corr(sim_df, mae_by_model)
        cost_means = (
            sim_df.group_by("MODEL").agg(pl.col("TOTAL_COST").mean().alias("c"))
                  .sort("c")
        )
        rows.append({
            "holding": h, "K": k, "stockout": p,
            "rho": rho, "p": pval,
            "best_cost_model": cost_means["MODEL"][0],
            "best_acc_model": min(mae_by_model, key=mae_by_model.get),
            "spread_cost": cost_means["c"][-1] - cost_means["c"][0],
        })
    df = pl.DataFrame(rows)
    df.write_parquet(ROB_DIR / "cost_grid.parquet", compression="zstd")
    print("R1 — cost-config sweep")
    print(f"  configs run: {len(rows)}")
    print(f"  rho stats   : "
          f"min={df['rho'].min():+.3f}  median={df['rho'].median():+.3f}  "
          f"max={df['rho'].max():+.3f}  |rho|<=0.3 in "
          f"{(df['rho'].abs() <= 0.3).sum()}/{len(rows)} configs")
    print(f"  best-cost model is LGBM in "
          f"{(df['best_cost_model'] == 'LGBM').sum()}/{len(rows)} configs;"
          f"\n                   MA  in "
          f"{(df['best_cost_model'] == 'MA').sum()} configs")
    return df


# ---------------------------------------------------------------------------
# R2. Per-segment correlation
# ---------------------------------------------------------------------------

def r2_per_segment(panel, forecasts, inputs, mae_by_model, cfg: CostConfig):
    """Within each SEGMENTATION_GRP, recompute Spearman rho on the
    fleet-level cost ranking restricted to that segment."""
    sim_df = simulate_all_models(panel, forecasts, inputs, cfg)
    # Per-segment MAE rank & cost rank
    rows = []
    for seg in sorted(sim_df["SEGMENTATION_GRP"].unique().to_list()):
        sub = sim_df.filter(pl.col("SEGMENTATION_GRP") == seg)
        # Need MAE per (model, segment) — pull from the forecast frames
        seg_mae = {}
        for model, fdf in forecasts.items():
            joined = panel.select(["STOCK_ITEM_NUMBER", "FORECAST_ID",
                                   "SEGMENTATION_GRP"]).join(
                fdf, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner"
            ).filter(pl.col("SEGMENTATION_GRP") == seg)
            if joined.height == 0:
                continue
            seg_mae[model] = float(joined["MAE_TEST"].mean())
        if len(seg_mae) < 3:
            continue
        rho, p = acc_cost_rank_corr(sub, seg_mae)
        rows.append({
            "segment": seg, "n_skus": sub["STOCK_ITEM_NUMBER"].n_unique(),
            "n_models": len(seg_mae), "rho": rho, "p": p,
        })

    df = pl.DataFrame(rows)
    out = "R2 — per-segment Spearman rho(acc_rank, cost_rank)\n"
    out += df.to_pandas().to_string(index=False)
    (ROB_DIR / "per_segment.txt").write_text(out, encoding="utf-8")
    print("\n" + out)
    return df


# ---------------------------------------------------------------------------
# R3. Bootstrap CI on the headline rho
# ---------------------------------------------------------------------------

def r3_bootstrap(panel, forecasts, inputs, mae_by_model, cfg: CostConfig,
                 n_boot: int = 200, seed: int = 42):
    sim_df = simulate_all_models(panel, forecasts, inputs, cfg)
    cost_wide = (
        sim_df.pivot(index=["STOCK_ITEM_NUMBER", "FORECAST_ID"],
                     on="MODEL", values="TOTAL_COST")
    )
    # Convert MAE per (sku, model) similarly
    mae_blocks = []
    for model, fdf in forecasts.items():
        mae_blocks.append(
            fdf.select(["STOCK_ITEM_NUMBER", "FORECAST_ID",
                        pl.col("MAE_TEST").alias(model)])
        )
    mae_wide = mae_blocks[0]
    for b in mae_blocks[1:]:
        mae_wide = mae_wide.join(b, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"],
                                  how="full", coalesce=True)

    joined = cost_wide.join(mae_wide, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"],
                              how="inner",
                              suffix="_mae")
    models = list(forecasts.keys())
    cost_arr = joined.select([m for m in models]).to_numpy()
    # Note pivot used model names as cols; mae cols come back as model_mae
    mae_cols = [f"{m}_mae" if f"{m}_mae" in joined.columns else m for m in models]
    mae_arr = joined.select(mae_cols).to_numpy()
    N, M = cost_arr.shape

    rng = np.random.default_rng(seed)
    rhos = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        cost_means = cost_arr[idx].mean(axis=0)
        mae_means  = mae_arr[idx].mean(axis=0)
        cost_rank = np.argsort(np.argsort(cost_means)) + 1
        mae_rank  = np.argsort(np.argsort(mae_means)) + 1
        rho, _ = spearmanr(mae_rank, cost_rank)
        rhos[b] = rho
    lo, hi = np.quantile(rhos, [0.025, 0.975])
    msg = (
        f"R3 — bootstrap (n={n_boot}, seed={seed}, panel resampled with replacement)\n"
        f"  rho mean   = {rhos.mean():+.3f}\n"
        f"  rho 95% CI = [{lo:+.3f}, {hi:+.3f}]\n"
        f"  fraction of resamples with rho > 0 : {(rhos > 0).mean():.3f}\n"
    )
    print("\n" + msg)
    (ROB_DIR / "bootstrap_ci.txt").write_text(msg, encoding="utf-8")
    return rhos


# ---------------------------------------------------------------------------
# R4. Deployed-policy comparison
# ---------------------------------------------------------------------------

def r4_deployed(panel, forecasts, inputs, cfg: CostConfig):
    """Simulate the deployed New Min / New Max from Stockmax for SKUs in
    both the wide sample and the policy sample."""
    pp = pl.scan_parquet(SAMPLE_DIR / "policy_panel.parquet").select([
        "STOCK_ITEM_NUMBER", "FORECAST_ID",
        "DEPLOYED_NEW_MIN_MEAN", "DEPLOYED_NEW_MAX_MEAN",
    ]).unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"]).collect()

    pp_panel = panel.join(pp, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner")
    if pp_panel.height == 0:
        msg = "R4 — no overlap between wide sample and policy panel."
        print("\n" + msg)
        (ROB_DIR / "deployed_compare.txt").write_text(msg, encoding="utf-8")
        return None

    actuals = pp_panel.select(ACTUAL_COLS).to_numpy().astype(np.float64)
    test = actuals[:, TRAIN_END:]
    unit_price = pp_panel["UNIT_PRICE_USD"].fill_null(1.0).cast(pl.Float64).to_numpy()
    unit_price = np.where(unit_price > 0, unit_price, 1.0)
    lead_days = pp_panel["LEAD_TIME_MEAN"].fill_null(60.0).cast(pl.Float64).to_numpy()
    lead_qtrs = lead_days_to_qtrs(np.where(lead_days > 0, lead_days, 60.0))
    s_dep = pp_panel["DEPLOYED_NEW_MIN_MEAN"].fill_null(0.0).cast(pl.Float64).to_numpy()
    S_dep = pp_panel["DEPLOYED_NEW_MAX_MEAN"].fill_null(0.0).cast(pl.Float64).to_numpy()
    # Floor S above s
    S_dep = np.maximum(S_dep, s_dep + 1.0)

    sim_dep = simulate(
        actuals_test=test, s=s_dep, S=S_dep,
        lead_qtrs=lead_qtrs, unit_price=unit_price, cfg=cfg,
    )

    # Restrict the model sims to this overlap and average
    keys_sub = pp_panel.select(["STOCK_ITEM_NUMBER", "FORECAST_ID"])
    sub_idx = (
        panel.with_row_index("idx")
             .join(keys_sub, on=["STOCK_ITEM_NUMBER", "FORECAST_ID"], how="inner")
             ["idx"].to_numpy()
    )
    crit = panel["CRITICALITY_MODE"].fill_null("Normal").to_list()
    service_level = np.array([service_level_for(c, cfg) for c in crit])

    rows = [{
        "policy": "DEPLOYED",
        "n_skus": len(sub_idx),
        "mean_total":   float(sim_dep["total_cost"].mean()),
        "median_total": float(np.median(sim_dep["total_cost"])),
        "mean_fill":    float(sim_dep["fill_rate"].mean()),
        "mean_stockout_units": float(sim_dep["stockout_units"].mean()),
    }]
    for model in forecasts:
        forecast_q21 = inputs["get_q21"](model)
        pol = normal_policy(
            forecast_mean_per_qtr=forecast_q21[sub_idx],
            forecast_std_per_qtr=inputs["sigma"][sub_idx],
            lead_qtrs=inputs["lead_qtrs"][sub_idx].astype(np.float64),
            lead_std_qtrs=inputs["lead_std_qtrs"][sub_idx],
            unit_price=inputs["unit_price"][sub_idx],
            service_level=service_level[sub_idx],
            cfg=cfg,
        )
        sim = simulate(
            actuals_test=inputs["test"][sub_idx],
            s=pol["s"], S=pol["S"],
            lead_qtrs=inputs["lead_qtrs"][sub_idx],
            unit_price=inputs["unit_price"][sub_idx],
            cfg=cfg,
        )
        rows.append({
            "policy": model,
            "n_skus": len(sub_idx),
            "mean_total":   float(sim["total_cost"].mean()),
            "median_total": float(np.median(sim["total_cost"])),
            "mean_fill":    float(sim["fill_rate"].mean()),
            "mean_stockout_units": float(sim["stockout_units"].mean()),
        })
    df = pl.DataFrame(rows).sort("mean_total")
    out = (
        f"R4 — deployed-policy comparison on {len(sub_idx):,} SKUs (wide ∩ policy panel)\n\n"
        + df.to_pandas().to_string(index=False)
    )
    (ROB_DIR / "deployed_compare.txt").write_text(out, encoding="utf-8")
    print("\n" + out)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    panel = load_panel()
    print(f"Panel rows: {panel.height:,}")

    forecasts = load_forecasts()
    print(f"Models: {list(forecasts)}")

    inputs = aligned_inputs(panel, forecasts)

    # Per-model fleet MAE — used everywhere as the accuracy reference.
    mae_by_model = {
        m: float(forecasts[m]["MAE_TEST"].mean()) for m in forecasts
    }

    cfg_default = CostConfig()

    print("\n" + "=" * 60)
    r1_cost_grid(panel, forecasts, inputs, mae_by_model)

    print("\n" + "=" * 60)
    r2_per_segment(panel, forecasts, inputs, mae_by_model, cfg_default)

    print("\n" + "=" * 60)
    r3_bootstrap(panel, forecasts, inputs, mae_by_model, cfg_default,
                  n_boot=200, seed=42)

    print("\n" + "=" * 60)
    r4_deployed(panel, forecasts, inputs, cfg_default)

    print(f"\nTotal: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
