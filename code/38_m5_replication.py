"""
Phase 4: does the MAE / forecast-level confound replicate on public M5 data?

Section 35 showed, on the maritime panel, that MAE is close to a monotone
transform of forecast level: median per-SKU rank correlation +1.000, with an
analytic reason (MAE is minimised at the median, which is zero for 77 % of
these SKUs). If that is a property of intermittent demand rather than of this
operator, it must reappear on the M5 competition data.

This uses the per-series classical results already computed for a separate
project (results/m5_classical.parquet: 30,490 series, eight forecasters, each
with out-of-sample MAE and bias). Forecast level is recovered from bias and the
realised out-of-sample mean, so no re-forecasting is needed.

The claim under test is about the METRIC, not about cost, so no inventory
simulation is required and none is run: this is a clean, public-data test of
the confound alone.

Writes output/sample/segment/robustness/m5_replication.{txt,json}
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

import config as C

ROB = C.SAMPLE_DIR / "robustness"
# The M5 inputs come from a separate project and are not part of this release.
# Point M5_DIR at a directory holding m5_classical.parquet and m5_weekly.parquet
# to reproduce Section 5.8; the script exits cleanly if they are absent.
_M5_DIR = Path(os.environ.get("M5_DIR", C.PROJECT_DIR / "data" / "m5"))
M5 = _M5_DIR / "m5_classical.parquet"
WEEKLY = _M5_DIR / "m5_weekly.parquet"
MODELS = ["naive", "sma", "ses", "croston", "sba", "tsb", "adida", "mapa"]


def per_series_rho(X, Y):
    out = np.full(X.shape[0], np.nan)
    for i in range(X.shape[0]):
        x, y = X[i], Y[i]
        if np.all(x == x[0]) or np.all(y == y[0]):
            continue
        out[i] = spearmanr(x, y).correlation
    return out


def summ(v, mask=None):
    w = v if mask is None else v[mask]
    w = w[np.isfinite(w)]
    if w.size == 0:
        return None
    return {"n": int(w.size), "mean": float(w.mean()),
            "median": float(np.median(w)),
            "frac_eq_plus1": float(np.mean(w > 0.999)),
            "frac_gt_09": float(np.mean(w > 0.9))}


def main():
    if not M5.exists():
        print(f"ERROR: {M5} not found"); return 1
    d = pl.read_parquet(M5)
    n = d.height

    mae = np.column_stack([d[f"{m}_oos_mae"].to_numpy().astype(float) for m in MODELS])
    bias = np.column_stack([d[f"{m}_oos_bias"].to_numpy().astype(float) for m in MODELS])
    oos_mean = d["oos_mean"].to_numpy().astype(float)

    # Recover forecast level from bias. Determine the sign convention
    # empirically: the correct one yields non-negative levels for these
    # non-negative-demand series.
    lvl_plus = oos_mean[:, None] + bias
    lvl_minus = oos_mean[:, None] - bias
    neg_plus = float(np.mean(lvl_plus < -1e-9))
    neg_minus = float(np.mean(lvl_minus < -1e-9))
    if neg_plus <= neg_minus:
        lvl, conv, negfrac = lvl_plus, "level = oos_mean + bias  (bias = forecast - actual)", neg_plus
    else:
        lvl, conv, negfrac = lvl_minus, "level = oos_mean - bias  (bias = actual - forecast)", neg_minus

    rho = per_series_rho(lvl, mae)

    cls = d["class"].cast(pl.Utf8).fill_null("Unknown").to_numpy()

    # Conditional test. The analytic condition from Section 35 is
    # median(test) = 0, under which MAE is monotone in forecast level. Recover
    # the out-of-sample window from the weekly panel by finding the horizon H
    # whose trailing mean reproduces oos_mean, then evaluate the condition.
    wk = pl.read_parquet(WEEKLY).select(["unique_id", "week_idx", "y"])
    piv = (wk.sort(["unique_id", "week_idx"])
           .group_by("unique_id", maintain_order=True).agg(pl.col("y")))
    pos = {u: i for i, u in enumerate(piv["unique_id"].to_list())}
    ys = [np.asarray(v, dtype=float) for v in piv["y"].to_list()]
    order = np.array([pos[u] for u in d["unique_id"].to_list()])
    best_H, best_frac = None, -1.0
    for H in range(1, 40):
        m = np.array([ys[i][-H:].mean() for i in order])
        fr = float(np.mean(np.abs(m - oos_mean) < 1e-6))
        if fr > best_frac:
            best_H, best_frac = H, fr
    med0 = np.array([np.median(ys[i][-best_H:]) == 0 for i in order])

    res = {
        "source": str(M5), "n_series": int(n), "models": MODELS,
        "level_convention": conv, "neg_level_fraction": negfrac,
        "overall": summ(rho), "by_class": {},
    }
    for c in sorted(set(cls)):
        res["by_class"][c] = summ(rho, cls == c)
    res["inferred_oos_horizon_weeks"] = int(best_H)
    res["horizon_match_fraction"] = float(best_frac)
    res["share_median_zero"] = float(med0.mean())
    res["conditional"] = {"condition_met": summ(rho, med0),
                          "condition_not_met": summ(rho, ~med0)}

    L = ["PHASE 4 - DOES THE MAE / LEVEL CONFOUND REPLICATE ON PUBLIC M5 DATA?",
         "=" * 74, "",
         f"  source : {M5.name}  ({n:,} series, {len(MODELS)} forecasters)",
         f"  models : {', '.join(MODELS)}",
         f"  level  : {conv}",
         f"           negative-level fraction under this convention: {100*negfrac:.3f} %",
         "",
         "  Per-series Spearman correlation between forecast LEVEL and MAE,",
         "  computed across the eight forecasters. n is series with variation on",
         "  both sides.", ""]
    o = res["overall"]
    L.append(f"  {'ALL SERIES':<16} mean {o['mean']:+.3f}  median {o['median']:+.3f}  "
             f"share=+1 {100*o['frac_eq_plus1']:5.1f} %  share>0.9 {100*o['frac_gt_09']:5.1f} %  n={o['n']:,}")
    L.append("")
    L.append("  By demand class (Syntetos-Boylan):")
    for c, s in res["by_class"].items():
        if s is None:
            L.append(f"  {c:<16} (no valid series)"); continue
        L.append(f"  {c:<16} mean {s['mean']:+.3f}  median {s['median']:+.3f}  "
                 f"share=+1 {100*s['frac_eq_plus1']:5.1f} %  n={s['n']:,}")

    L += ["",
          f"  Inferred out-of-sample horizon: {best_H} weeks "
          f"(trailing mean reproduces oos_mean on {100*best_frac:.1f} % of series)",
          f"  Share of M5 series meeting the analytic condition median(OOS)=0: "
          f"{100*med0.mean():.1f} %   [maritime: 77.3 % at H=4]", "",
          "  CONDITIONAL ON THE ANALYTIC CONDITION:"]
    for k, lab in [("condition_met", "median(OOS) == 0  (condition MET)"),
                   ("condition_not_met", "median(OOS) >  0  (not met)")]:
        c_ = res["conditional"][k]
        L.append(f"    {lab:<36} n={c_['n']:>6,}  mean {c_['mean']:+.3f}  "
                 f"median {c_['median']:+.3f}  share=+1 {100*c_['frac_eq_plus1']:5.1f} %")
    L += ["", "  Interpretation: where this correlation is at or near +1, ranking",
          "  forecasters by MAE is ranking them by how much they forecast, so MAE",
          "  cannot separate forecast quality from forecast level on those series."]

    rep = "\n".join(L)
    (ROB / "m5_replication.txt").write_text(rep, encoding="utf-8")
    (ROB / "m5_replication.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
