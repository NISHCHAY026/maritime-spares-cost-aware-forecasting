"""
Tier-3 ML wrappers: LightGBM (and later DeepAR / Chronos).

Design choices for the LightGBM wrapper
---------------------------------------
* **Global model** — one LightGBM trained across ALL SKUs in the sample,
  with SKU-level static features (segmentation, ABC, velocity, lead time,
  price) + per-quarter dynamic features (lags, rolling stats, intermittency
  signals). This is the conventional 'global cross-learning' setup
  documented in Bandara et al (2020), Salinas et al (2020 — DeepAR), and
  M5 winners. Per-SKU local LightGBM would be prohibitive at 16k SKUs.
* **Two heads**: one regression head on demand size, one classification
  head on demand occurrence. Joint forecast = P(occur) * E[size | occur].
  This mirrors the Hurdle-NB structure but with non-parametric heads,
  giving LightGBM access to whatever interaction structure the features
  expose.
* **Recursive multi-step prediction** — predict QTR21 with features built
  from Q1-Q20, then append the prediction to history and predict QTR22,
  etc. Simpler than direct-multi-output and adequate for h=8.
* **Quantile heads (optional)** — for predictive distribution we use
  LightGBM's quantile objective on the size head, fit separately at each
  target quantile {0.5, 0.8, 0.9, 0.95, 0.99}. Cheap because the dataset
  is tabular and small (<400k rows).

The feature builder is reusable for any tabular learner; DeepAR / Chronos
will use the raw series instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import lightgbm as lgb
import numpy as np


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

DEFAULT_LAGS = (1, 2, 3, 4, 8, 12)
DEFAULT_ROLLS = (4, 8, 12)


def build_features(
    actuals_history: np.ndarray,        # (N, T_history)
    static: dict[str, np.ndarray],      # {feat: (N,) array}
    lags: Iterable[int] = DEFAULT_LAGS,
    rolls: Iterable[int] = DEFAULT_ROLLS,
) -> tuple[np.ndarray, list[str]]:
    """
    Build a (N, F) feature matrix using actuals_history as the lookback.

    Features:
      - lag_k          actuals_history[:, -k]                for k in lags
      - roll_mean_w    mean of last w quarters
      - roll_std_w     std  of last w quarters
      - roll_nz_w      count of non-zero quarters in last w
      - q_since_demand number of quarters since most recent demand
      - n_demands      total non-zero count in history
      - mean_size_pos  mean of positive observations
      - + every static feature
    """
    N, T = actuals_history.shape
    cols: list[np.ndarray] = []
    names: list[str] = []

    for k in lags:
        col = actuals_history[:, -k] if k <= T else np.zeros(N)
        cols.append(col)
        names.append(f"lag_{k}")

    for w in rolls:
        if w <= T:
            window = actuals_history[:, -w:]
        else:
            window = actuals_history
        cols.extend([
            window.mean(axis=1),
            window.std(axis=1, ddof=0),
            (window > 0).sum(axis=1).astype(np.float64),
        ])
        names.extend([f"roll_mean_{w}", f"roll_std_{w}", f"roll_nz_{w}"])

    # Quarters since last demand (T if never demand)
    nz_mask = actuals_history > 0
    if nz_mask.any():
        last_idx = np.where(nz_mask, np.arange(T), -1).max(axis=1)
        q_since = np.where(last_idx >= 0, T - 1 - last_idx, T)
    else:
        q_since = np.full(N, T)
    cols.append(q_since.astype(np.float64))
    names.append("q_since_demand")

    cols.append((actuals_history > 0).sum(axis=1).astype(np.float64))
    names.append("n_demands")

    pos_vals = np.where(nz_mask, actuals_history, np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_pos = np.nanmean(pos_vals, axis=1)
    cols.append(np.nan_to_num(mean_pos))
    names.append("mean_size_pos")

    for k, v in static.items():
        cols.append(v.astype(np.float64))
        names.append(k)

    X = np.stack(cols, axis=1)
    return X, names


# ---------------------------------------------------------------------------
# LightGBM trainer
# ---------------------------------------------------------------------------

# Fixed so results do not depend on how many cores happen to be free.
DETERMINISTIC_THREADS = 8


def _seeded(params: dict, seed: int) -> dict:
    """Pin every RNG LightGBM exposes, so a given seed reproduces exactly."""
    return {
        **params,
        "seed": seed,
        "bagging_seed": seed + 1,
        "feature_fraction_seed": seed + 2,
        "data_random_seed": seed + 3,
        "extra_seed": seed + 4,
    }

@dataclass
class LGBForecaster:
    lags:       tuple = DEFAULT_LAGS
    rolls:      tuple = DEFAULT_ROLLS
    # feature_fraction / bagging_fraction / bagging_freq are stochastic, so
    # every head needs its seeds pinned. LightGBM is ALSO non-deterministic
    # across runs by default when multithreaded: "deterministic" requires
    # force_row_wise (or force_col_wise), and the thread count must be fixed,
    # or tree construction depends on machine load. Without this the same seed
    # gave MAE 8.526 / 8.555 / 8.987 on three consecutive runs.
    occur_params: dict = field(default_factory=lambda: {
        "objective": "binary",
        "metric":    "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": DETERMINISTIC_THREADS,
    })
    size_params: dict = field(default_factory=lambda: {
        "objective": "regression",
        "metric":    "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": DETERMINISTIC_THREADS,
    })
    n_rounds:   int = 400
    early_stop: int = 30

    feat_names: list[str] = field(default_factory=list)
    occur_model: lgb.Booster | None = None
    size_model:  lgb.Booster | None = None
    quantile_models: dict = field(default_factory=dict)

    def fit(
        self,
        actuals_train: np.ndarray,    # (N, T_train)
        static: dict[str, np.ndarray],
        valid_frac: float = 0.15,
        train_quantiles: tuple = (0.5, 0.8, 0.9, 0.95, 0.99),
        seed: int = 42,
    ) -> None:
        """
        Build a per-(SKU, quarter) training table from actuals_train. For
        each row at quarter t in [max_lag .. T_train-1], the features come
        from actuals_train[:, :t] and the target is actuals_train[:, t].
        """
        N, T = actuals_train.shape
        max_lag = max(self.lags + self.rolls)
        Xs, ys = [], []
        for t in range(max_lag, T):
            Xt, names = build_features(
                actuals_train[:, :t], static, self.lags, self.rolls
            )
            Xs.append(Xt)
            ys.append(actuals_train[:, t])
        X = np.vstack(Xs)
        y = np.concatenate(ys)
        self.feat_names = names

        # Train/valid split — last quarters of each SKU as validation.
        rng = np.random.default_rng(seed)
        n_quarters = T - max_lag
        valid_quarters = max(1, int(round(n_quarters * valid_frac)))
        valid_mask = np.zeros(len(y), dtype=bool)
        # mark the last `valid_quarters` quarter-slices as validation
        valid_offset = (n_quarters - valid_quarters) * N
        valid_mask[valid_offset:] = True

        X_tr, X_vl = X[~valid_mask], X[valid_mask]
        y_tr, y_vl = y[~valid_mask], y[valid_mask]

        # Occur head
        y_tr_o = (y_tr > 0).astype(np.float64)
        y_vl_o = (y_vl > 0).astype(np.float64)
        d_tr = lgb.Dataset(X_tr, label=y_tr_o, feature_name=self.feat_names)
        d_vl = lgb.Dataset(X_vl, label=y_vl_o, feature_name=self.feat_names,
                            reference=d_tr)
        self.occur_model = lgb.train(
            _seeded(self.occur_params, seed), d_tr, num_boost_round=self.n_rounds,
            valid_sets=[d_vl], valid_names=["valid"],
            callbacks=[lgb.early_stopping(self.early_stop, verbose=False),
                       lgb.log_evaluation(0)],
        )

        # Size head — train only on positive-target rows
        pos_tr = y_tr > 0
        pos_vl = y_vl > 0
        if pos_tr.sum() < 50:
            self.size_model = None
            return

        d_tr = lgb.Dataset(X_tr[pos_tr], label=y_tr[pos_tr],
                            feature_name=self.feat_names)
        d_vl = lgb.Dataset(X_vl[pos_vl], label=y_vl[pos_vl],
                            feature_name=self.feat_names, reference=d_tr)
        self.size_model = lgb.train(
            _seeded(self.size_params, seed), d_tr, num_boost_round=self.n_rounds,
            valid_sets=[d_vl], valid_names=["valid"],
            callbacks=[lgb.early_stopping(self.early_stop, verbose=False),
                       lgb.log_evaluation(0)],
        )

        # Quantile heads — fit one per quantile on positive-target rows.
        for q in train_quantiles:
            params_q = _seeded({**self.size_params,
                                "objective": "quantile", "alpha": float(q),
                                "metric": "quantile"}, seed)
            d_tr_q = lgb.Dataset(X_tr[pos_tr], label=y_tr[pos_tr],
                                  feature_name=self.feat_names)
            d_vl_q = lgb.Dataset(X_vl[pos_vl], label=y_vl[pos_vl],
                                  feature_name=self.feat_names,
                                  reference=d_tr_q)
            self.quantile_models[float(q)] = lgb.train(
                params_q, d_tr_q, num_boost_round=self.n_rounds,
                valid_sets=[d_vl_q], valid_names=["valid"],
                callbacks=[lgb.early_stopping(self.early_stop, verbose=False),
                           lgb.log_evaluation(0)],
            )

    def predict_recursive(
        self,
        actuals_train: np.ndarray,
        static: dict[str, np.ndarray],
        h: int,
    ) -> dict:
        """
        Recursive multi-step. At each horizon step:
          1. Build features from current history.
          2. Predict P(occur) and E[size | occur].
          3. Joint mean = P(occur) * E[size | occur].
          4. Append predicted mean to history; advance.
        Quantile heads predicted at h=1 only (held constant across the
        horizon — cheap and adequate for stockout-cost simulation).
        """
        N, T = actuals_train.shape
        history = actuals_train.copy()
        means = np.zeros((N, h))
        for step in range(h):
            X, _ = build_features(history, static, self.lags, self.rolls)
            p_occur = self.occur_model.predict(X) if self.occur_model is not None else np.zeros(N)
            mu_size = (
                self.size_model.predict(X)
                if self.size_model is not None else np.zeros(N)
            )
            # Clamp to non-negative
            mu_size = np.maximum(mu_size, 0.0)
            mean_step = p_occur * mu_size
            means[:, step] = mean_step
            history = np.concatenate([history, mean_step[:, None]], axis=1)

        # Quantile heads — at h=1 features (the most-recent state).
        X_h1, _ = build_features(actuals_train, static, self.lags, self.rolls)

        # P(occur) must come from the h=1 features too. The loop variable
        # above holds the value from the LAST horizon step, which is a
        # different information set from the one qhat is built on.
        p_h1 = (self.occur_model.predict(X_h1)
                if self.occur_model is not None else np.zeros(N))
        p_h1 = np.clip(p_h1, 0.0, 1.0)

        # The quantile heads are fit on positive-target rows only, so they
        # estimate the size quantile CONDITIONAL on occurrence. The demand
        # itself is a hurdle mixture: zero with prob 1-p, size otherwise.
        # Its alpha-quantile is therefore
        #     Q_alpha = 0                              if alpha <= 1 - p
        #             = F_size^-1( (alpha-(1-p)) / p )  otherwise
        # NOT qhat * 1{p > 0.5}, which collapses the whole distribution to a
        # point mass at zero for every SKU with p <= 0.5 and leaves the
        # surviving SKUs on an un-remapped conditional quantile.
        levels = np.array(sorted(self.quantile_models.keys()), dtype=np.float64)
        qcond = np.column_stack([
            np.maximum(self.quantile_models[q].predict(X_h1), 0.0) for q in levels
        ])
        # LightGBM fits each quantile independently, so they can cross.
        qcond = np.maximum.accumulate(qcond, axis=1)

        # Interpolate F_size^-1 on the trained grid, anchored at (0, 0).
        xs = np.concatenate([[0.0], levels])
        ys = np.concatenate([np.zeros((N, 1)), qcond], axis=1)

        quantiles = {}
        for alpha in levels:
            a_cond = np.where(p_h1 > 0.0, (alpha - (1.0 - p_h1)) / np.maximum(p_h1, 1e-12), 0.0)
            a_cond = np.clip(a_cond, 0.0, 1.0)
            j = np.clip(np.searchsorted(xs, a_cond, side="right") - 1, 0, len(xs) - 2)
            x0, x1 = xs[j], xs[j + 1]
            y0 = ys[np.arange(N), j]
            y1 = ys[np.arange(N), j + 1]
            w = np.where(x1 > x0, (a_cond - x0) / np.maximum(x1 - x0, 1e-12), 0.0)
            qa = y0 + w * (y1 - y0)
            # Above the top trained level, hold the top conditional quantile.
            qa = np.where(a_cond >= xs[-1], ys[:, -1], qa)
            # Exactly zero where the mixture puts alpha below the zero atom.
            quantiles[float(alpha)] = np.where(alpha <= (1.0 - p_h1), 0.0, np.maximum(qa, 0.0))
        return {
            "mean": means,
            "p_occur_step1": p_h1,
            "size_step1": mu_size,
            "quantiles": quantiles,
            "feature_importance_size": (
                dict(zip(self.feat_names, self.size_model.feature_importance(importance_type="gain")))
                if self.size_model else {}
            ),
        }
