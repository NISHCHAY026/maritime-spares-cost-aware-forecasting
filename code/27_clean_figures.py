"""
Clean-window paper figures (audit fix #1). Reads:
  output/sample/segment/robustness/clean_full.json   (scalars)
  output/sample/segment/clean_per_sku.parquet         (per-SKU MAE+cost per model)
  output/sample/segment/panel.parquet                 (for the censoring diagnostic)
Writes to docs/paper/figures_clean/.
"""
from __future__ import annotations
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import config as C

SAMPLE_DIR = C.SAMPLE_DIR
ROB_DIR = SAMPLE_DIR / "robustness"
FIG = C.PROJECT_DIR / "docs" / "paper" / "figures_clean"
FIG.mkdir(parents=True, exist_ok=True)
ACTUAL_COLS = [f"ACTUALS_QTR{i:02d}" for i in range(1, 29)]


def style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def figA_censoring():
    panel = pl.read_parquet(SAMPLE_DIR / "panel.parquet").unique(subset=["STOCK_ITEM_NUMBER", "FORECAST_ID"])
    frac = [(panel[c].to_numpy() > 0).mean() for c in ACTUAL_COLS]
    mean = [panel[c].to_numpy().mean() for c in ACTUAL_COLS]
    q = np.arange(1, 29)
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    ax.bar(q, frac, color="#4C72B0", edgecolor="black", width=0.7)
    ax.set_ylabel("Fraction of SKUs with nonzero demand", color="#4C72B0")
    ax.set_xlabel("Quarter position in panel (QTR01 = oldest)")
    ax2 = ax.twinx(); ax2.plot(q, mean, "o-", color="#C44E52", markersize=4); ax2.set_ylabel("Mean demand / SKU", color="#C44E52")
    ax2.spines["top"].set_visible(False)
    ax.axvspan(0.5, 16.5, alpha=0.08, color="green")
    ax.axvspan(16.5, 20.5, alpha=0.12, color="blue")
    ax.axvspan(20.5, 28.5, alpha=0.12, color="red")
    ax.text(8, max(frac)*0.95, "train Q1-16", ha="center", fontsize=8)
    ax.text(18.5, max(frac)*0.95, "clean\ntest\nQ17-20", ha="center", fontsize=7)
    ax.text(24.5, max(frac)*0.95, "censored / original test Q21-28", ha="center", fontsize=8, color="#9F2D2D")
    ax.set_title("Demand profile across panel quarters: the original test window (Q21-28)\n"
                 "is right-censored (recent transactions not yet recorded)")
    fig.tight_layout(); fig.savefig(FIG / "figA_censoring.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print("  figA censoring done")


def figB_tau(js):
    t = js["tau"]
    labels = ["demand-active\n(n=%s)" % f"{t['pos']['n_valid']:,}",
              "zero-demand\n(n=%s)" % f"{t['zero']['n_valid']:,}", "pooled"]
    means = [t["pos"]["mean"], t["zero"]["mean"], t["all"]["mean"]]
    los = [t[k]["mean"]-t[k]["lo"] for k in ("pos", "zero", "all")]
    his = [t[k]["hi"]-t[k]["mean"] for k in ("pos", "zero", "all")]
    colors = ["#4C72B0", "#CCCCCC", "#999999"]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.bar(range(3), means, yerr=[los, his], color=colors, edgecolor="black", capsize=5)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels)
    ax.set_ylabel(r"Mean per-SKU Kendall's $\tau$ (MAE-rank vs cost-rank)")
    ax.set_ylim(0, 1.05)
    ax.axhline(t["pos"]["mean"], color="#4C72B0", ls="--", lw=0.8)
    ax.set_title("Accuracy↔cost agreement is moderate on demand-active SKUs;\n"
                 "the zero-demand value is mechanical and excluded")
    for i, m in enumerate(means):
        ax.text(i, m+0.03, f"{m:+.2f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "figB_tau.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print("  figB tau done")


def figC_acc_cost(js):
    pm = js["per_model"]
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    for m, d in pm.items():
        ax.scatter(d["mae"], d["cost"], s=70, color="#4C72B0", edgecolor="black", zorder=3)
        ax.annotate(m, (d["mae"], d["cost"]), textcoords="offset points", xytext=(6, 3), fontsize=8)
    ax.set_xlabel("Fleet-mean test MAE (lower = more accurate)")
    ax.set_ylabel("Fleet-mean total cost (USD)")
    ax.set_title("Accuracy vs cost, clean window (train Q1-16 / test Q17-20)\n"
                 r"demand-active $\tau$ = %+.2f [%+.2f, %+.2f]" %
                 (js["tau"]["pos"]["mean"], js["tau"]["pos"]["lo"], js["tau"]["pos"]["hi"]))
    fig.tight_layout(); fig.savefig(FIG / "figC_acc_cost.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print("  figC acc-cost done")


def figD_calibration(js):
    levels = [0.5, 0.8, 0.9, 0.95, 0.99]
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.plot([0.5, 1], [0.5, 1], "--", color="grey", label="perfect")
    pal = {"ZIP": "#4C72B0", "HURDLE_NB": "#DD8452", "LGBM": "#55A868", "CHRONOS": "#C44E52"}
    for m, cov in js["calib"].items():
        ax.plot(levels, [cov[str(int(l*100))] for l in levels], "o-",
                color=pal.get(m, "#333"), label=m, markersize=5)
    ax.set_xlabel("Nominal quantile q"); ax.set_ylabel("Empirical coverage")
    ax.set_title("Predictive-quantile calibration (clean window)\nQ99 still below 0.99: tails too thin")
    ax.legend(fontsize=8, loc="lower right"); ax.set_xlim(0.45, 1.02); ax.set_ylim(0.45, 1.02)
    fig.tight_layout(); fig.savefig(FIG / "figD_calibration.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print("  figD calibration done")


def figE_fillmatched(js):
    mm = js["matched"]
    models = list(mm)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(models)); w = 0.38
    normal = [mm[m]["cost_normal"] for m in models]
    matched = [mm[m]["cost_matched"] for m in models]
    ax.bar(x - w/2, normal, w, label="Normal-approx", color="#4C72B0", edgecolor="black")
    ax.bar(x + w/2, [min(v, 6000) for v in matched], w, label="Native-quantile (fill-matched)",
           color="#55A868", edgecolor="black")
    ax.set_ylim(0, 6500)
    for i, m in enumerate(models):
        d = mm[m]["delta_pct"]
        yv = min(matched[i], 6000)
        ax.text(i + w/2, yv + 80, (f"{d:+.0f}%" + ("\n(capped)" if mm[m]["capped"] else "")),
                ha="center", fontsize=7, color="#9F2D2D" if d > 0 else "#2A6F45")
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Mean total cost / SKU (USD)")
    ax.set_title("Fill-matched native-quantile vs normal-approx (clean window)\n"
                 "at equal fill the native-quantile advantage does not appear")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "figE_fillmatched.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print("  figE fill-matched done")


def figF_deployed(js):
    pm = js["per_model"]; dep = js["deployed"]
    order = sorted(pm, key=lambda m: pm[m]["cost"])
    labels = order + ["DEPLOYED"]
    vals = [pm[m]["cost"] for m in order] + [dep["cost"]]
    colors = ["#4C72B0"]*len(order) + ["#C44E52"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    b = ax.bar(labels, vals, color=colors, edgecolor="black")
    for bar, v in zip(b, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+20, f"${v:,.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Mean total cost / SKU (USD)")
    ax.set_title(f"Deployed vs model policies, clean window (overlap n={dep['n']:,})\n"
                 f"best model {dep['best_model']} beats deployed by {dep['gap_pct']:.0f}%")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout(); fig.savefig(FIG / "figF_deployed.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print("  figF deployed done")


def main():
    style()
    figA_censoring()
    js = json.loads((ROB_DIR / "clean_full.json").read_text())
    figB_tau(js); figC_acc_cost(js); figD_calibration(js); figE_fillmatched(js); figF_deployed(js)
    print(f"All clean figures in {FIG}")


if __name__ == "__main__":
    main()
