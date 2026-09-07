"""
Reference implementations of the three baselines deployed in the production
pipeline (SBA, SES, MA).  Vectorised across SKUs with NumPy.

Conventions match STEP_18 of the SQL pipeline:

  ACTUALS         shape (N, T), float64, T quarterly observations per SKU
  ALPHA           shape (N,),   per-row smoothing parameter
  ROUND_COUNT     shape (N,),   the segmentation ADI = T / n_nonzero
  AVG_NONZERO     shape (N,),   mean of non-zero actuals (init level)

SBA (Syntetos-Boylan)
---------------------
Initialisation (t=0):
    LEVEL[:,0]       = AVG_NONZERO
    PERIODICITY[:,0] = ROUND_COUNT
    INTERVAL[:,0]    = ROUND_COUNT          # per pipeline doc
For t >= 1:
    INTERVAL[:,t]    = 1 if actuals[:,t] > 0 else INTERVAL[:,t-1] + 1
    if actuals[:,t] > 0:
        LEVEL[:,t]       = α * actuals[:,t] + (1 - α) * LEVEL[:,t-1]
        PERIODICITY[:,t] = α * INTERVAL[:,t] + (1 - α) * PERIODICITY[:,t-1]
    else:
        LEVEL[:,t]       = LEVEL[:,t-1]
        PERIODICITY[:,t] = PERIODICITY[:,t-1]
    FORECAST[:,t]   = (1 - α/2) * LEVEL[:,t] / PERIODICITY[:,t]
FORECAST[:,0] uses the init state similarly.

Note: the pipeline smooths PERIODICITY against the *post-reset* INTERVAL
(=1) on a demand event, not against the gap-since-last-demand. This is a
deviation from textbook Croston/SBA but is what the deployed code does, so
we replicate it exactly here.

SES
---
    FORECAST[:,0] = actuals[:,0]
    FORECAST[:,1] = actuals[:,1]                                     (warmup)
    FORECAST[:,t] = α * actuals[:,t-1] + (1 - α) * FORECAST[:,t-1]   t >= 2

MA (2-quarter)
--------------
    FORECAST[:,0] = 0
    FORECAST[:,1] = 0
    FORECAST[:,t] = (actuals[:,t-1] + actuals[:,t-2]) / 2            t >= 2
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SBAState:
    level: np.ndarray
    periodicity: np.ndarray
    interval: np.ndarray
    forecast: np.ndarray


def sba(actuals: np.ndarray, alpha: np.ndarray,
        window_lengths: np.ndarray | None = None) -> SBAState:
    """
    actuals: (N, T)
    alpha:   (N,)
    window_lengths: (N,) per-row training window length, optional.
                    The pipeline runs each forecast over a user-specified
                    quarter window (typically <=28). Init values
                    (avg_nonzero, round_count) are computed using only
                    the first window_lengths[i] quarters of row i. Outside
                    that window the iteration just sees zero-padded
                    actuals; the caller masks unused outputs.
                    If None, full T is used per row.
    Returns level, periodicity, interval, forecast — all (N, T)
    """
    N, T = actuals.shape
    a = alpha[:, None]                                          # (N, 1) for broadcasting

    # Per-row effective T for init purposes
    if window_lengths is None:
        T_eff = np.full(N, T, dtype=np.float64)
        in_window = np.ones_like(actuals, dtype=bool)
    else:
        T_eff = window_lengths.astype(np.float64)
        cols = np.arange(T)[None, :]                            # (1, T)
        in_window = cols < window_lengths[:, None]              # (N, T)

    # Init using in-window portion only
    nonzero = (actuals > 0) & in_window
    n_nonzero = nonzero.sum(axis=1).astype(np.float64)          # (N,)
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
    forecast[:, 0]    = (1.0 - alpha / 2.0) * level[:, 0] / np.maximum(periodicity[:, 0], 1e-12)

    # The pipeline (verified empirically) updates the three series with two
    # different time alignments:
    #   * INTERVAL[t] depends on actuals[t]      (current quarter)
    #   * LEVEL[t], PERIODICITY[t] depend on actuals[t-1] AND interval[t-1]
    #     (previous quarter)
    # See `04_validate_baselines.py` for the trace that revealed this — the
    # pipeline's docstring (STEP_18) doesn't make this distinction explicit.
    a_flat = alpha
    for t in range(1, T):
        d_curr = actuals[:, t]
        d_prev = actuals[:, t-1]
        had_curr = d_curr > 0
        had_prev = d_prev > 0

        # INTERVAL: based on current
        interval[:, t] = np.where(had_curr, 1.0, interval[:, t-1] + 1.0)

        # LEVEL / PERIODICITY: based on previous
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

        forecast[:, t] = (1.0 - a_flat / 2.0) * level[:, t] / np.maximum(periodicity[:, t], 1e-12)

    return SBAState(level=level, periodicity=periodicity, interval=interval, forecast=forecast)


def ses(actuals: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Simple exponential smoothing per pipeline STEP_18."""
    N, T = actuals.shape
    a = alpha
    fc = np.zeros_like(actuals)
    fc[:, 0] = actuals[:, 0]
    if T > 1:
        fc[:, 1] = actuals[:, 1]
    for t in range(2, T):
        fc[:, t] = a * actuals[:, t-1] + (1.0 - a) * fc[:, t-1]
    return fc


def ma2(actuals: np.ndarray) -> np.ndarray:
    """
    2-quarter moving average per pipeline STEP_18 (labelled '6 MONTH MA').

    The pipeline initialises QTR01 and QTR02 with the actuals themselves
    (same warmup pattern as SES) — the docstring only states the n>=3
    recurrence, but the deployed code carries actuals through positions 1
    and 2.
    """
    N, T = actuals.shape
    fc = np.zeros_like(actuals)
    if T >= 1:
        fc[:, 0] = actuals[:, 0]
    if T >= 2:
        fc[:, 1] = actuals[:, 1]
    if T > 2:
        fc[:, 2:] = (actuals[:, 1:T-1] + actuals[:, 0:T-2]) / 2.0
    return fc


# ---------------------------------------------------------------------------
# Terminal update: equalise the information set across forecasters
# ---------------------------------------------------------------------------
#
# The recursion above sets LEVEL[t] and PERIODICITY[t] from actuals[t-1], so
# the state at the last training column T-1 has consumed demand only through
# T-2. Reading `forecast[:, -1]` therefore produces a forecast that never saw
# the final training quarter through the smoothing channel; it enters only via
# the initialisation constants, whose weight decays as (1-alpha)^T.
#
# SES already gets an explicit extra step at the call site
# (`alpha*train[:, -1] + (1-alpha)*ses(...)[:, -1]`), and MA, ADIDA, IMAPA,
# ZIP, Hurdle-NB, LightGBM and Chronos all condition on the full window. Only
# SBA, Croston and TSB were short by one observation.
#
# `sba_next` runs one further iteration of the SAME recurrence, so the returned
# forecast conditions on actuals[:, :T]. The state functions themselves are
# left untouched: `sba()` still reproduces the deployed pipeline column for
# column, which is what 04_validate_baselines.py checks.

def sba_next(actuals: np.ndarray, alpha: np.ndarray,
             window_lengths: np.ndarray | None = None) -> np.ndarray:
    """One-step-ahead SBA forecast that has consumed the full window."""
    st = sba(actuals, alpha, window_lengths=window_lengths)
    d_last = actuals[:, -1]
    had_last = d_last > 0
    level = np.where(had_last, alpha * d_last + (1.0 - alpha) * st.level[:, -1],
                     st.level[:, -1])
    period = np.where(had_last,
                      alpha * st.interval[:, -1] + (1.0 - alpha) * st.periodicity[:, -1],
                      st.periodicity[:, -1])
    return (1.0 - alpha / 2.0) * level / np.maximum(period, 1e-12)
