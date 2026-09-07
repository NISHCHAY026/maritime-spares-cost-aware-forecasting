"""
Inventory cost simulator for the accuracy-vs-cost benchmark.

Workflow per SKU and per model:
  1. Derive an (s, S) policy from the model's point + spread forecast and the
     SKU's lead time, using the deployed normal-approximation formula:
        s = D_bar * LT + z * sqrt(LT * sigma_D^2 + D_bar^2 * sigma_LT^2)
        EOQ = sqrt(2 * D_annual * K / h_per_unit_per_year)
        S   = s + max(EOQ, 1)
  2. Simulate H quarters of demand (the test window) under that policy:
        - quarterly review,
        - integer-quarter lead time (= ceil(LT_days / 91.3125), min 1),
        - LOST SALES: unmet demand is charged once per unit short and does
          NOT carry into the next quarter,
        - holding cost on positive on-hand at end of quarter.
  3. Record cost components, fill-rate, on-hand quantiles, and order count.

Design choices (documented in §4.2 of the paper)
------------------------------------------------
* Quarterly review & quarterly lead-time grain. Daily would be more
  realistic but quarterly matches the forecast cadence and the deployed
  pipeline's planning horizon.
* Lost sales, not backorders. This docstring previously claimed backorders
  with a per-unit-per-quarter delay penalty; the code has never done that.
  `short` is accumulated for costing and discarded (see step 2 below).
* Initial on-hand = S (saturated start). This is consequential on a short
  window: 83% of SKUs never reorder across four quarters, so most of the
  sample measures holding an initial position rather than replenishment.
  Pass initial_on_hand explicitly for the zero-start sensitivity.
* fill_rate is 1.0 by DEFINITION when a SKU has no realised demand (61% of
  the sample here), so mean fill is largely definitional. Use it to match
  service between policies, not as a headline.
* service_level enters only as z = Phi^-1(SL) in the safety-stock term, i.e.
  as a cycle-service (Type-1) quantity, despite the tiers being specified as
  availability targets.
* Service level is the input; stockout cost is computed both as an
  explicit per-unit penalty (so we can report a $-figure) and as fill-rate
  (so we can check the constraint was met).
* Predictive uncertainty per model:
    classical (SBA/Croston/...) -> training-window residual std
    distributional (ZIP/HNB)    -> sqrt(variance from MoM fit)
    ML (LightGBM)               -> (Q90 - Q10) / 2.56  if quantiles available,
                                   else training residual std
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Cost configuration
# ---------------------------------------------------------------------------

@dataclass
class CostConfig:
    holding_rate_annual: float = 0.25       # 25 %/yr (Silver-Pyke-Peterson default)
    ordering_cost: float = 50.0              # $/order
    stockout_unit_cost: float = 100.0        # $ per unit short (charged once, lost sales)
    n_quarters_per_year: int = 4

    # Criticality-tiered service-level targets
    sl_critical: float = 0.99
    sl_normal:   float = 0.97
    sl_low:      float = 0.95

    @property
    def holding_per_unit_per_qtr(self) -> float:
        # holding_rate_annual is fraction of unit value, applied per year.
        # We multiply by unit_price downstream; here we just give the
        # quarterly fraction. holding_cost = on_hand * unit_price * h_per_qtr.
        return self.holding_rate_annual / self.n_quarters_per_year


# ---------------------------------------------------------------------------
# Service-level lookup
# ---------------------------------------------------------------------------

def service_level_for(criticality: str | None, cfg: CostConfig) -> float:
    """Map a criticality string to a target service level.

    Note: normal_policy applies this as z = Phi^-1(SL), i.e. as a Type-1
    (cycle service) quantity, not as a Type-2 fill-rate target.
    """
    if criticality is None:
        return cfg.sl_normal
    c = str(criticality).strip().lower()
    if c in {"critical", "high", "1"}:
        return cfg.sl_critical
    if c in {"low", "3"}:
        return cfg.sl_low
    return cfg.sl_normal


# ---------------------------------------------------------------------------
# Policy derivation: normal-approximation (s, S, EOQ)
# ---------------------------------------------------------------------------

def normal_policy(
    forecast_mean_per_qtr: np.ndarray,    # (N,) D_bar
    forecast_std_per_qtr:  np.ndarray,    # (N,) sigma_D
    lead_qtrs:             np.ndarray,    # (N,) LT in quarters
    lead_std_qtrs:         np.ndarray,    # (N,) sigma_LT
    unit_price:            np.ndarray,    # (N,) USD per unit
    service_level:         np.ndarray,    # (N,) target SL, applied as Type-1 via z
    cfg: CostConfig,
) -> dict:
    """
    Vectorised computation of (s, S, EOQ) for the whole sample.
    """
    N = len(forecast_mean_per_qtr)
    z = norm.ppf(np.clip(service_level, 1e-3, 1 - 1e-3))

    lt_demand = forecast_mean_per_qtr * lead_qtrs
    safety = z * np.sqrt(
        lead_qtrs * forecast_std_per_qtr**2
        + forecast_mean_per_qtr**2 * lead_std_qtrs**2
    )
    s = np.maximum(lt_demand + safety, 0.0)

    annual_demand = forecast_mean_per_qtr * cfg.n_quarters_per_year
    h_per_unit_per_year = cfg.holding_rate_annual * unit_price
    # Guard against zero unit_price -> infinite EOQ; use 1 unit as floor.
    h = np.where(h_per_unit_per_year > 0, h_per_unit_per_year, 1e-9)
    eoq = np.sqrt(2.0 * np.maximum(annual_demand, 0.0) * cfg.ordering_cost / h)
    eoq = np.where(np.isfinite(eoq), np.maximum(eoq, 1.0), 1.0)

    S = s + eoq
    return {"s": s, "S": S, "eoq": eoq, "z": z, "lt_demand": lt_demand, "safety": safety}


# ---------------------------------------------------------------------------
# Vectorised period-by-period simulator
# ---------------------------------------------------------------------------

def simulate(
    actuals_test:   np.ndarray,    # (N, H)
    s:              np.ndarray,    # (N,)
    S:              np.ndarray,    # (N,)
    lead_qtrs:      np.ndarray,    # (N,) integer quarters
    unit_price:     np.ndarray,    # (N,)
    initial_on_hand: np.ndarray | None = None,
    cfg: CostConfig | None = None,
) -> dict:
    """
    Vectorised (s, S) simulation across N SKUs and H horizons.

    Returns dict with arrays of length N:
      holding_cost, ordering_cost, stockout_units, stockout_cost,
      total_cost, fill_rate, n_orders, mean_on_hand, end_on_hand
    """
    if cfg is None:
        cfg = CostConfig()
    N, H = actuals_test.shape
    s = np.asarray(s, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    lead_qtrs = np.asarray(lead_qtrs, dtype=np.int64)
    lead_qtrs = np.maximum(lead_qtrs, 1)         # min lead = 1 quarter
    unit_price = np.asarray(unit_price, dtype=np.float64)

    # Saturated initial on-hand if not supplied.
    if initial_on_hand is None:
        on_hand = S.copy()
    else:
        on_hand = np.asarray(initial_on_hand, dtype=np.float64).copy()

    # Pending orders matrix: arrivals_at[t][i] = qty arriving at start of qtr t.
    # Allow up to H + max_lead steps so orders placed near the end can land
    # outside the window without IndexErrors.
    max_lead = int(lead_qtrs.max())
    arrivals = np.zeros((H + max_lead + 1, N), dtype=np.float64)

    holding_cost = np.zeros(N)
    n_orders = np.zeros(N, dtype=np.int64)
    stockout_units = np.zeros(N)
    on_hand_trace = np.zeros((N, H), dtype=np.float64)

    h_per_qtr = cfg.holding_per_unit_per_qtr

    for t in range(H):
        # 1. Receive arrivals
        on_hand += arrivals[t]

        # 2. Observe demand, fulfil on-hand first
        d = actuals_test[:, t]
        served = np.minimum(on_hand, d)
        short = d - served
        stockout_units += short
        on_hand -= served                                # on_hand >= 0 maintained

        # 3. Inventory position = on-hand + outstanding on-order
        on_order = arrivals[t + 1: t + max_lead + 1].sum(axis=0)
        position = on_hand + on_order

        # 4. Place an order to bring up to S, if below s
        below = position < s
        order_qty = np.where(below, np.maximum(S - position, 0.0), 0.0)

        # Schedule arrival
        for li in np.unique(lead_qtrs):
            sel = (lead_qtrs == li) & below
            if not sel.any():
                continue
            arrival_t = t + int(li)
            if arrival_t < arrivals.shape[0]:
                arrivals[arrival_t][sel] += order_qty[sel]

        n_orders += below.astype(np.int64)

        # 5. End-of-quarter holding cost (use end-of-period on-hand)
        holding_cost += on_hand * unit_price * h_per_qtr
        on_hand_trace[:, t] = on_hand

    total_demand = actuals_test.sum(axis=1)
    fill_rate = np.where(
        total_demand > 0,
        1.0 - stockout_units / np.maximum(total_demand, 1e-12),
        1.0,
    )
    ordering_cost = n_orders.astype(np.float64) * cfg.ordering_cost
    stockout_cost = stockout_units * cfg.stockout_unit_cost
    total_cost = holding_cost + ordering_cost + stockout_cost

    return {
        "holding_cost": holding_cost,
        "ordering_cost": ordering_cost,
        "stockout_units": stockout_units,
        "stockout_cost": stockout_cost,
        "total_cost": total_cost,
        "fill_rate": fill_rate,
        "n_orders": n_orders,
        "mean_on_hand": on_hand_trace.mean(axis=1),
        "end_on_hand": on_hand,
    }


# ---------------------------------------------------------------------------
# Helper: lead-time conversion
# ---------------------------------------------------------------------------

DAYS_PER_QTR = 91.3125


def lead_days_to_qtrs(days: np.ndarray) -> np.ndarray:
    """Convert lead-time-in-days to integer quarters (min 1)."""
    out = np.ceil(np.asarray(days, dtype=np.float64) / DAYS_PER_QTR)
    return np.maximum(out.astype(np.int64), 1)
