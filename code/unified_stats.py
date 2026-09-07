"""
Per-SKU rank-correlation machinery for the revised paper.

Key design choices addressing reviewer attack lines from Critique v0.5:

(1) The headline claim is now expressed as the *mean per-SKU Kendall's
    tau* between MAE-rank and cost-rank of M forecasters, computed over
    N SKUs. Power scales with N (15-16 k), not with M (11). 95 % CIs
    come from SKU-level bootstrapping.

(2) The sigma_D proxy is now uniformly the per-SKU training-actuals
    standard deviation across all M forecasters (no mixing of quantile-
    derived sigma for some models and residual-std for others). The
    'sigma proxy matters' finding becomes a methodological footnote
    explaining why we chose a single proxy.

(3) Per-(Segment x ABC) tau uses the same per-SKU machinery restricted
    to that cell, with bootstrap CIs from SKU-level resampling.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, rankdata


def per_sku_kendall_tau(
    mae_per_sku: np.ndarray,         # (N, M)
    cost_per_sku: np.ndarray,        # (N, M)
) -> np.ndarray:
    """
    Returns (N,) array of Kendall's tau values, one per SKU.

    For each SKU i:
      tau_i = kendalltau(rank(mae_per_sku[i]), rank(cost_per_sku[i]))[0]

    Implementation note: scipy.stats.kendalltau is O(M log M) per call,
    so N invocations is fine for M=11. We use the tau-b variant (default)
    which handles ties correctly — though with 11 models, ties on cost
    would be a degenerate case.
    """
    N, M = mae_per_sku.shape
    out = np.empty(N, dtype=np.float64)
    for i in range(N):
        m = mae_per_sku[i]
        c = cost_per_sku[i]
        finite = np.isfinite(m) & np.isfinite(c)
        if finite.sum() < 3:
            out[i] = np.nan
            continue
        tau, _ = kendalltau(m[finite], c[finite])
        out[i] = tau
    return out


def bootstrap_mean_ci(
    values: np.ndarray,              # (N,)
    n_boot: int = 1000,
    seed: int = 42,
    qs: tuple = (0.025, 0.975),
) -> dict:
    """
    SKU-level bootstrap CI on the mean.
    Returns: {mean, lo, hi, n_valid}
    """
    finite = np.isfinite(values)
    valid = values[finite]
    N = valid.size
    if N == 0:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "n_valid": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        means[b] = valid[idx].mean()
    lo, hi = np.quantile(means, qs)
    return {"mean": float(valid.mean()),
            "lo": float(lo), "hi": float(hi),
            "n_valid": int(N)}


def per_sku_dominance_pairs(
    mae_per_sku: np.ndarray,
    cost_per_sku: np.ndarray,
    model_names: list[str],
) -> dict:
    """
    For each SKU, identify which model is best on MAE and which is best
    on cost. Returns frequencies + agreement rate.
    """
    N, M = mae_per_sku.shape
    best_mae_idx = np.nanargmin(mae_per_sku, axis=1)
    best_cost_idx = np.nanargmin(cost_per_sku, axis=1)
    agreement = (best_mae_idx == best_cost_idx).mean()

    # Identity counts
    mae_counts = np.bincount(best_mae_idx, minlength=M)
    cost_counts = np.bincount(best_cost_idx, minlength=M)

    return {
        "best_mae_freq":  {model_names[i]: int(mae_counts[i])  for i in range(M)},
        "best_cost_freq": {model_names[i]: int(cost_counts[i]) for i in range(M)},
        "agreement_rate": float(agreement),
        "n_skus": int(N),
    }


def fleet_rank_table(
    mae_per_sku: np.ndarray,
    cost_per_sku: np.ndarray,
    model_names: list[str],
) -> dict:
    """
    Fleet-level point estimates: mean MAE and mean cost per model.
    Returns dicts {model -> mean}.
    """
    M = len(model_names)
    mean_mae  = {model_names[i]: float(np.nanmean(mae_per_sku[:, i]))  for i in range(M)}
    mean_cost = {model_names[i]: float(np.nanmean(cost_per_sku[:, i])) for i in range(M)}
    return {"mean_mae": mean_mae, "mean_cost": mean_cost}


def cell_tau_with_ci(
    mae_per_sku: np.ndarray,         # (N, M)  filtered to one cell
    cost_per_sku: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Compute mean per-SKU tau within one cell, plus 95% bootstrap CI from
    SKU-level resampling.
    """
    taus = per_sku_kendall_tau(mae_per_sku, cost_per_sku)
    return bootstrap_mean_ci(taus, n_boot=n_boot, seed=seed)
