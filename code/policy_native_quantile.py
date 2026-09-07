"""
Native-quantile (s, S) policy for models that emit predictive quantiles.

Reviewer critique #7: the normal-approximation (s, S) formula wastes
distributional information for ZIP / HNB / LGBM / Chronos. This module
provides an alternative policy derivation that uses the model's native
predictive quantile of lead-time demand directly:

  s = lead_qtrs * Q_alpha[demand]
  Q_alpha is read from the model's per-SKU predictive distribution at
  alpha = service_level.

EOQ + S construction unchanged.

For models without native quantiles (the seven classical Croston-family
forecasters), we fall back to the normal-approximation policy.

This lets the paper compare two policy derivations head-to-head, with
the same simulator and the same uniform sigma proxy elsewhere.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from simulator import CostConfig, normal_policy


def quantile_lookup(quantile_arr: dict[float, np.ndarray],
                     target_q: float) -> np.ndarray:
    """
    Pick the closest available quantile column to `target_q` and return it.
    quantile_arr is {0.5: arr, 0.8: arr, 0.9: arr, 0.95: arr, 0.99: arr}.
    """
    keys = sorted(quantile_arr.keys())
    diffs = [abs(k - target_q) for k in keys]
    closest = keys[int(np.argmin(diffs))]
    return quantile_arr[closest]


def native_quantile_policy(
    quantile_arr: dict[float, np.ndarray],   # {0.5: (N,), ...}
    forecast_mean: np.ndarray,                # (N,) point forecast
    lead_qtrs: np.ndarray,                    # (N,)
    unit_price: np.ndarray,                   # (N,)
    service_level: np.ndarray,                # (N,) per-SKU SL target
    cfg: CostConfig,
) -> dict:
    """
    s = lead_qtrs * Q_alpha[demand], where Q_alpha is the model's
        predictive quantile at the SKU's service_level (closest available).
    EOQ from the standard EOQ formula on annual mean demand.
    S = s + max(EOQ, 1).
    """
    N = forecast_mean.shape[0]
    # Per-SKU lookup: pick the right quantile column for each SKU.
    # Vectorise by mapping unique SL values to columns.
    s = np.zeros(N)
    for sl_value in np.unique(service_level):
        mask = service_level == sl_value
        if not mask.any():
            continue
        # Find the closest quantile available
        keys = sorted(quantile_arr.keys())
        diffs = [abs(k - sl_value) for k in keys]
        closest = keys[int(np.argmin(diffs))]
        # Lead-time demand at the chosen quantile
        s[mask] = lead_qtrs[mask] * quantile_arr[closest][mask]

    s = np.maximum(s, 0.0)

    # EOQ stays formula-driven from mean
    annual_demand = forecast_mean * cfg.n_quarters_per_year
    h_per_unit_per_year = cfg.holding_rate_annual * unit_price
    h = np.where(h_per_unit_per_year > 0, h_per_unit_per_year, 1e-9)
    eoq = np.sqrt(2.0 * np.maximum(annual_demand, 0.0) * cfg.ordering_cost / h)
    eoq = np.where(np.isfinite(eoq), np.maximum(eoq, 1.0), 1.0)

    S = s + eoq
    return {"s": s, "S": S, "eoq": eoq, "z": np.full(N, np.nan)}
