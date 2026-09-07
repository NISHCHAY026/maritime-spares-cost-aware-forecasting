"""
Tier-1 classical intermittent-demand forecasters.

All four methods share the (N, T) actuals layout used by `baselines.py`.
They produce a per-period level series and a single point forecast for the
horizon.  Predictive distributions for the cost simulator come later via
residual bootstrap (Stage-5) — keeping the model code lean here.

References
----------
Croston (1972)             — lazy smoothing of level + interval
Syntetos & Boylan (2005)   — SBA bias correction (already in baselines.py)
Teunter, Syntetos, Babai (2011) — TSB: smooth probability of demand
Nikolopoulos et al. (2011) — ADIDA / IMAPA: aggregate-disaggregate
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Croston / SBA share state — define once
# ---------------------------------------------------------------------------

@dataclass
class CrostonState:
    level: np.ndarray
    periodicity: np.ndarray
    interval: np.ndarray
    forecast: np.ndarray


def croston(actuals: np.ndarray, alpha: np.ndarray,
            window_lengths: np.ndarray | None = None,
            sba: bool = False) -> CrostonState:
    """
    Classic Croston (1972).
        forecast = level / periodicity
    With sba=True: Syntetos-Boylan bias-corrected variant.
        forecast = (1 - α/2) · level / periodicity

    Both share the lazy update rule: level + periodicity smooth ONLY on
    demand events; carry forward otherwise. Time alignment matches the
    deployed pipeline (level/periodicity use lagged actual + lagged
    interval; interval uses current actual). See `baselines.sba` for the
    detailed timing rationale.
    """
    N, T = actuals.shape
    a = alpha[:, None]

    if window_lengths is None:
        T_eff = np.full(N, T, dtype=np.float64)
        in_window = np.ones_like(actuals, dtype=bool)
    else:
        T_eff = window_lengths.astype(np.float64)
        cols = np.arange(T)[None, :]
        in_window = cols < window_lengths[:, None]

    nonzero = (actuals > 0) & in_window
    n_nonzero = nonzero.sum(axis=1).astype(np.float64)
    sum_nonzero = np.where(nonzero, actuals, 0.0).sum(axis=1)
    avg_nonzero = np.where(n_nonzero > 0, sum_nonzero / np.maximum(n_nonzero, 1), 0.0)
    round_count = np.where(n_nonzero > 0, T_eff / np.maximum(n_nonzero, 1), T_eff)

    level       = np.zeros_like(actuals)
    periodicity = np.zeros_like(actuals)
    interval    = np.zeros_like(actuals)
    forecast    = np.zeros_like(actuals)

    level[:, 0]       = avg_nonzero
    periodicity[:, 0] = round_count
    interval[:, 0]    = round_count
    correction = (1.0 - alpha / 2.0) if sba else np.ones_like(alpha)
    forecast[:, 0]    = correction * level[:, 0] / np.maximum(periodicity[:, 0], 1e-12)

    a_flat = alpha
    corr_flat = (1.0 - a_flat / 2.0) if sba else np.ones_like(a_flat)
    for t in range(1, T):
        d_curr = actuals[:, t]
        d_prev = actuals[:, t-1]
        had_curr = d_curr > 0
        had_prev = d_prev > 0

        interval[:, t] = np.where(had_curr, 1.0, interval[:, t-1] + 1.0)
        level[:, t] = np.where(
            had_prev,
            a_flat * d_prev + (1.0 - a_flat) * level[:, t-1],
            level[:, t-1],
        )
        periodicity[:, t] = np.where(
            had_prev,
            a_flat * interval[:, t-1] + (1.0 - a_flat) * periodicity[:, t-1],
            periodicity[:, t-1],
        )
        forecast[:, t] = corr_flat * level[:, t] / np.maximum(periodicity[:, t], 1e-12)

    return CrostonState(level=level, periodicity=periodicity, interval=interval, forecast=forecast)


# ---------------------------------------------------------------------------
# TSB (Teunter-Syntetos-Babai 2011)
# ---------------------------------------------------------------------------

@dataclass
class TSBState:
    level: np.ndarray
    probability: np.ndarray
    forecast: np.ndarray


def tsb(actuals: np.ndarray, alpha: np.ndarray, beta: np.ndarray | None = None,
        window_lengths: np.ndarray | None = None) -> TSBState:
    """
    Teunter-Syntetos-Babai. Two parameters:
      α — smooths demand size when demand fires.
      β — smooths probability of demand EVERY period (this is the
          difference from Croston/SBA, which smooth interval lazily).

    Forecast at every t:
        forecast[t] = level[t] · probability[t]

    If β is not supplied, default β = α (common practice).
    """
    if beta is None:
        beta = alpha
    N, T = actuals.shape
    a = alpha
    b = beta

    if window_lengths is None:
        T_eff = np.full(N, T, dtype=np.float64)
        in_window = np.ones_like(actuals, dtype=bool)
    else:
        T_eff = window_lengths.astype(np.float64)
        cols = np.arange(T)[None, :]
        in_window = cols < window_lengths[:, None]

    nonzero = (actuals > 0) & in_window
    n_nonzero = nonzero.sum(axis=1).astype(np.float64)
    sum_nonzero = np.where(nonzero, actuals, 0.0).sum(axis=1)
    avg_nonzero = np.where(n_nonzero > 0, sum_nonzero / np.maximum(n_nonzero, 1), 0.0)
    p0 = np.where(T_eff > 0, n_nonzero / np.maximum(T_eff, 1), 0.0)

    level = np.zeros_like(actuals)
    prob  = np.zeros_like(actuals)
    fc    = np.zeros_like(actuals)
    level[:, 0] = avg_nonzero
    prob[:, 0]  = p0
    fc[:, 0]    = level[:, 0] * prob[:, 0]

    for t in range(1, T):
        d_prev = actuals[:, t-1]
        had_prev = d_prev > 0

        prob[:, t]  = b * had_prev.astype(np.float64) + (1.0 - b) * prob[:, t-1]
        level[:, t] = np.where(
            had_prev,
            a * d_prev + (1.0 - a) * level[:, t-1],
            level[:, t-1],
        )
        fc[:, t] = level[:, t] * prob[:, t]

    return TSBState(level=level, probability=prob, forecast=fc)


# ---------------------------------------------------------------------------
# ADIDA / IMAPA (aggregate-disaggregate)
# ---------------------------------------------------------------------------

def _aggregate_block_sums(actuals: np.ndarray, k: int) -> np.ndarray:
    """
    Sum actuals into non-overlapping k-period blocks aligned to the END of
    the series (most recent block is full; oldest may be truncated and is
    dropped).  Returns (N, n_blocks).
    """
    N, T = actuals.shape
    if k <= 1:
        return actuals.copy()
    n_blocks = T // k
    if n_blocks == 0:
        return actuals.sum(axis=1, keepdims=True)
    trim = T - n_blocks * k
    blocks = actuals[:, trim:].reshape(N, n_blocks, k).sum(axis=2)
    return blocks


def _ses_last(values: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Simple exponential smoothing on (N, M); return the LAST forecast,
    which is the standard SES one-step-ahead point estimate.
    """
    N, M = values.shape
    if M == 0:
        return np.zeros(N)
    if M == 1:
        return values[:, 0].copy()
    a = alpha
    fc = values[:, 0].copy()
    for t in range(1, M):
        fc = a * values[:, t-1] + (1.0 - a) * fc
    # One more step using the latest actual
    fc = a * values[:, -1] + (1.0 - a) * fc
    return fc


def adida(
    actuals: np.ndarray,
    alpha: np.ndarray,
    k: np.ndarray | int | None = None,
    window_lengths: np.ndarray | None = None,
) -> dict:
    """
    Aggregate-Disaggregate Intermittent Demand Approach (Nikolopoulos 2011).

    Steps:
      1. Aggregate actuals into k-period blocks (last block aligned).
      2. SES forecast for the next aggregated value.
      3. Disaggregate by dividing by k (uniform spread).

    `k` may be:
      - None: auto-pick per row as round(ADI), clipped to [1, T/3].
      - int: same k for all rows.
      - (N,) array: per-row k.

    Returns dict with:
      forecast_per_period: (N,) per-period forecast for the next period
      forecast_block:       (N,) raw aggregated-level forecast
      k_used:               (N,) k chosen
    """
    N, T = actuals.shape
    if window_lengths is None:
        T_eff = np.full(N, T, dtype=np.int64)
    else:
        T_eff = window_lengths.astype(np.int64)

    if k is None:
        nonzero = actuals > 0
        if window_lengths is not None:
            cols = np.arange(T)[None, :]
            mask = cols < window_lengths[:, None]
            nonzero = nonzero & mask
        n_nonzero = nonzero.sum(axis=1)
        adi = np.where(n_nonzero > 0, T_eff / np.maximum(n_nonzero, 1), T_eff)
        k_used = np.clip(np.round(adi).astype(np.int64), 1, np.maximum(T_eff // 3, 1))
    elif np.isscalar(k):
        k_used = np.full(N, int(k), dtype=np.int64)
    else:
        k_used = np.asarray(k, dtype=np.int64)

    # Per-row aggregation (varying k requires a loop, but it's cheap per row)
    block_fc = np.zeros(N)
    for ki in np.unique(k_used):
        idx = np.where(k_used == ki)[0]
        if len(idx) == 0:
            continue
        sub = actuals[idx]
        blocks = _aggregate_block_sums(sub, int(ki))
        block_fc[idx] = _ses_last(blocks, alpha[idx])

    per_period = block_fc / k_used.astype(np.float64)
    return {"forecast_per_period": per_period, "forecast_block": block_fc, "k_used": k_used}


def imapa(
    actuals: np.ndarray,
    alpha: np.ndarray,
    k_max: np.ndarray | int | None = None,
    window_lengths: np.ndarray | None = None,
) -> dict:
    """
    Multiple Aggregation Prediction Algorithm (Nikolopoulos 2011).

    For each k in [1, k_max], run ADIDA and average the per-period forecasts.

    `k_max` may be None (auto from ADI), int, or (N,) array. When auto:
        k_max = max(2, round(ADI))
    """
    N, T = actuals.shape
    if window_lengths is None:
        T_eff = np.full(N, T, dtype=np.int64)
    else:
        T_eff = window_lengths.astype(np.int64)

    if k_max is None:
        nonzero = actuals > 0
        if window_lengths is not None:
            cols = np.arange(T)[None, :]
            mask = cols < window_lengths[:, None]
            nonzero = nonzero & mask
        n_nonzero = nonzero.sum(axis=1)
        adi = np.where(n_nonzero > 0, T_eff / np.maximum(n_nonzero, 1), T_eff)
        k_max_arr = np.clip(np.round(adi).astype(np.int64), 2, np.maximum(T_eff // 3, 2))
    elif np.isscalar(k_max):
        k_max_arr = np.full(N, int(k_max), dtype=np.int64)
    else:
        k_max_arr = np.asarray(k_max, dtype=np.int64)

    # Average ADIDA forecasts across k = 1..k_max[i] per row.
    sum_fc = np.zeros(N)
    counts = np.zeros(N, dtype=np.int64)
    for ki in range(1, int(k_max_arr.max()) + 1):
        active = ki <= k_max_arr
        if not active.any():
            continue
        # Build a temporary k array equal to ki for active rows
        sub_idx = np.where(active)[0]
        sub_actuals = actuals[sub_idx]
        sub_alpha = alpha[sub_idx]
        out = adida(sub_actuals, sub_alpha, k=ki,
                    window_lengths=(window_lengths[sub_idx] if window_lengths is not None else None))
        sum_fc[sub_idx] += out["forecast_per_period"]
        counts[sub_idx] += 1

    per_period = np.where(counts > 0, sum_fc / np.maximum(counts, 1), 0.0)
    return {"forecast_per_period": per_period, "k_max_used": k_max_arr}


# ---------------------------------------------------------------------------
# Terminal updates: equalise the information set across forecasters
# ---------------------------------------------------------------------------
#
# See the note in baselines.py. Croston, SBA and TSB update LEVEL (and
# PERIODICITY / PROBABILITY) from actuals[t-1], so `forecast[:, -1]` has not
# consumed the final training observation through the smoothing channel. Every
# other forecaster in the comparison has. These helpers run one further
# iteration of the same recurrence so all methods see the same window.

def croston_next(actuals: np.ndarray, alpha: np.ndarray,
                 window_lengths: np.ndarray | None = None,
                 sba: bool = False) -> np.ndarray:
    """One-step-ahead Croston/SBA forecast conditioned on the full window."""
    st = croston(actuals, alpha, window_lengths=window_lengths, sba=sba)
    d_last = actuals[:, -1]
    had_last = d_last > 0
    level = np.where(had_last, alpha * d_last + (1.0 - alpha) * st.level[:, -1],
                     st.level[:, -1])
    period = np.where(had_last,
                      alpha * st.interval[:, -1] + (1.0 - alpha) * st.periodicity[:, -1],
                      st.periodicity[:, -1])
    correction = (1.0 - alpha / 2.0) if sba else np.ones_like(alpha)
    return correction * level / np.maximum(period, 1e-12)


def tsb_next(actuals: np.ndarray, alpha: np.ndarray,
             beta: np.ndarray | None = None,
             window_lengths: np.ndarray | None = None) -> np.ndarray:
    """One-step-ahead TSB forecast conditioned on the full window."""
    if beta is None:
        beta = alpha
    st = tsb(actuals, alpha, beta=beta, window_lengths=window_lengths)
    d_last = actuals[:, -1]
    had_last = d_last > 0
    prob = beta * had_last.astype(np.float64) + (1.0 - beta) * st.probability[:, -1]
    level = np.where(had_last, alpha * d_last + (1.0 - alpha) * st.level[:, -1],
                     st.level[:, -1])
    return level * prob
