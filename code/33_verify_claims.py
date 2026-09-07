"""
Claims verifier: cross-check the paper's assertions against the result ledgers.

Written after a referee panel found that the paper contained numbers no file
supported, numbers from superseded runs, and two claims its own data
contradicted. Eyeballing did not catch those; this does.

Each claim declares:
  required  - substrings that MUST appear (the correct, current value)
  forbidden - substrings that MUST NOT appear (known-stale or disproved values)

Run after any edit to the paper, and after any re-run of the analysis.
Exit code 0 = clean, 1 = discrepancies.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import config as C

ROB = C.SAMPLE_DIR / "robustness"
PAPER_DIR = C.PROJECT_DIR / "docs" / "paper"
MD = PAPER_DIR / "arxiv_v1.md"
TEX = PAPER_DIR / "arxiv_v1.tex"


def load(name):
    p = ROB / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def norm(s: str) -> str:
    """Fold LaTeX/markdown number formatting so one rule matches both files."""
    s = s.replace("{,}", ",").replace("\$", "$").replace("\%", "%")
    s = s.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"\s+", " ", s)
    return s


def main():
    cf = load("clean_full.json")
    t1 = load("tier1.json")
    cw = load("censored_window.json")
    cr = load("cost_regime.json")
    if cf is None:
        print("ERROR: clean_full.json missing; run 26_clean_full.py first")
        return 1

    pm, mt, cal, dep = cf["per_model"], cf["matched"], cf["calib"], cf["deployed"]

    claims = []

    def add(name, required=(), forbidden=(), note=""):
        claims.append({"name": name, "required": list(required),
                       "forbidden": list(forbidden), "note": note})

    # --- sample -----------------------------------------------------------
    add("sample size", [f'{cf["N"]:,}'], [])
    add("demand-active count", ["5,929"], [])

    # --- Finding 1 --------------------------------------------------------
    add("tau demand-active", ["+0.387", "+0.39"], [])
    if t1:
        r = t1.get("tau_rmsse", {}).get("mean")
        if r is not None:
            add("RMSSE tau", [f"{r:+.3f}"[:6], f"{r:.3f}".lstrip("0")],
                ["+0.178", "0.178"],
                "paper previously quoted +0.178 from the pre-fix run")
    add("MASE not sold as independent check", [],
        ["barely moves when we switch from MAE",
         "robust to the error metric for absolute error"],
        "MASE is a positive per-SKU rescaling of MAE; tau is identical by construction")

    # --- Finding 2 (the one that inverted) --------------------------------
    add("LGBM fill-matched", [f'{mt["LGBM"]["delta_pct"]:.0f}'.lstrip("-")],
        ["+204", "204 %", "204%"],
        "LGBM now REACHES target fill and saves cost")
    add("Chronos fill-matched", [],
        ["+66 %", "+66%", "66 %"],
        "Chronos now reaches target at kappa 3.83")
    add("no capped claims", [],
        ["no (capped)", "at the cap, not matched", "target unreachable"],
        "nothing is capped any more")
    add("Q99 coverage", [f'{cal["LGBM"]["99"]:.3f}'],
        ["roughly 94", "about 94", "only about 94 %"],
        "LGBM Q99 is now 0.995, Chronos 0.961")
    add("under-dispersed narrative gone", [],
        ["under-dispersed LightGBM and Chronos", "upper tails are too thin",
         "tails too thin"],
        "disproved: the failures were a gating bug and Monte Carlo sample size")

    # --- Finding 3 --------------------------------------------------------
    add("deployed cost", [f'{dep["cost"]:,.0f}'], [])
    add("deployed gap", ["-50.7", "50.7", "51 %"], [])
    add("not every model", [], ["by every model-driven policy",
                                "every\nmodel-driven policy beats"],
        "LightGBM loses to deployed on the overlap")
    if t1 and "service_level_scenarios" in str(t1):
        pass
    add("service-level scenario count", [],
        ["five service-level"],
        "the code runs four scenarios, not five")

    # --- per-model headline numbers ---------------------------------------
    add("Chronos MAE", [f'{pm["CHRONOS"]["mae"]:.3f}'[:5]], ["6.55 "])
    add("Chronos cost", [f'{pm["CHRONOS"]["cost"]:,.0f}'], ["2,109"])
    add("fleet-level rho", ["+0.018"], ["-0.045"],
        "pooled fleet correlation is +0.018 under canonical ordering")
    add("LGBM cost", [f'{pm["LGBM"]["cost"]:,.0f}'], ["2,784"])

    # --- censoring contribution -------------------------------------------
    if cw:
        # Section 5.6 legitimately reports that the NAIVE split makes the moving
        # average cheapest; only attributing that flip to censoring is forbidden.
        add("censoring claim withdrawn", [],
            ["flips the cheapest forecaster",
             "into the most expensive once corrected"],
            "controlled A/B shows the ranking does NOT change under censoring")

    # --- simulator disclosure (audited against simulator.py) --------------
    add("lost sales disclosed", ["lost, not backordered"],
        ["backorders allowed", "backorders with a per-unit-per-quarter"],
        "simulator.py discards `short`; it has never modelled backorders")
    add("saturated start disclosed", ["starts saturated", "83.0 %"], [],
        "initial_on_hand defaults to S; 83% of SKUs never reorder in 4 quarters")
    add("definitional fill disclosed", ["1.0 by definition", "61.4 %"], [],
        "simulate() sets fill_rate=1.0 when total demand is zero")
    add("cost composition disclosed", ["90.5 %"], [],
        "cost is 90.5% holding, so the ranking is close to a bias ranking")

    # --- cost regime, and the decomposition that qualifies it -------------
    if cr:
        g = cr["groups"]
        add("cost-regime split reported",
            [f'{g["holding_dominant"]["fleet_spearman_rho"]:.3f}'.lstrip("0")],
            ["+0.673", "0.673"],
            "regime rho moved to +0.591 (p=0.056) under canonical row ordering")
    me = load("mechanical.json")
    if me:
        add("MAE-level confound disclosed",
            ["+0.982", "confound"],
            [], "MAE and forecast level rank-correlate at +0.982; the regime "
                "result does not survive conditioning on level")
        add("partial correlation reported", ["+0.000"], [],
            "partial rho(MAE,cost|level) is exactly 0.000 in the "
            "holding-dominated group")
        add("cost components reported", ["+0.827", "-0.509"], [],
            "level raises holding cost and lowers stockout cost")

    # --- Phase 1: the MAE / level confound --------------------------------
    pc = load("phase1_confound.json")
    if pc:
        add("analytic condition disclosed", ["77.3"], [],
            "share of SKUs with median(test)=0, where MAE is monotone in level")
        add("per-SKU confound reported", ["78.6"], [],
            "78.6% of SKUs have level-MAE rank correlation exactly +1")
    cfr = load("critical_fractile.json")
    if cfr:
        add("critical fractile derived", ["0.981", "critical fractile"], [],
            "q* = Cu/(Cu+Co) equals 0.5 at a unit price of 1600")
        add("pinball remedy refuted", ["-0.491"], [],
            "scoring at q* inherits the confound with the sign reversed")
    ds = load("distributional.json")
    if ds:
        add("distributional ordering reported", ["+0.800", "+0.400"], [],
            "pinball at the service level beats MAE; generic CRPS is worse")
    m5 = load("m5_replication.json")
    if m5:
        add("M5 non-replication reported", ["+0.194", "10.1"], [],
            "the confound does not generalise; condition holds for 10.1% of M5")
        add("M5 conditional mechanism reported", ["+0.614"], [],
            "mechanism reappears where the analytic condition is met")

    # --- inference (paired bootstrap over SKUs) ---------------------------
    inf = load("inference.json")
    if inf:
        add("cost ordering unresolvable", ["only one of ten"], [],
            "only LGBM is distinguishable from the cheapest under a paired bootstrap")
        # Required strings are matched against the WHOLE file, including the
        # reproducibility note, so a superseded value recorded there as a
        # correction would satisfy a stale rule. Both endpoints below are the
        # current two-decimal ones for that reason.
        add("regime CI reported", ["+0.49"], [],
            "holding-dominated rho CI [+0.49, +1.00] excludes zero at B=20,000")
        add("stockout regime not claimed negative", ["-0.76"], [],
            "stockout-dominated CI includes zero; no negative relationship claimed")
        add("deployed gap CI reported", ["-60.4"], [],
            "deployed gap -50.7% CI [-60.4%, -41.0%]")

    # --- equal information sets (roadmap item 4) ---------------------------
    inf_set = load("information_set.json")
    if inf_set:
        add("information-set fix reported", ["2.12", "bit-identical"],
            ["15-quarter effective information set while the other models",
             "may contribute to their accuracy ranking"],
            "SBA/Croston/TSB now get the terminal update; the old limitation "
            "must not still be stated as an open one")
        add("information-set fix leaves conclusions", ["-50.9"], [],
            "deployed gap moves from -50.7 % to -50.9 % under the equalised arm")

    # --- design weights (roadmap item 5) -----------------------------------
    dw = load("design_weights.json")
    if dw:
        add("design weights reported", ["96.12", "3,127"],
            ["Design-weighted estimates are future work",
             "We report unweighted statistics throughout"],
            "w_h spans 1.00 to 96.12 and Kish n_eff is 3,127; the paper no "
            "longer defers weighting to future work")
        add("weighted regime reported", ["+0.991", "-0.882"], [],
            "Horvitz-Thompson regime correlations")
        add("weighted stratum caveat reported", ["-0.182", "68"], [],
            "the weighted stockout-dominated rho collapses to -0.182 without "
            "Intermittent x C, which is 68 % of the fleet by weight")
        add("weighted fleet shares reported", ["14.8", "62.1"], [],
            "demand-active share and holding-dominated cost share, weighted")

    # --- bootstrap Monte Carlo error ---------------------------------------
    if inf:
        add("MC error disclosed", ["+0.409", "20,000"],
            ["+0.527"],
            "the B=2,000 lower bound was seed-unstable; the paper must not "
            "still quote +0.527 as the interval endpoint")
        add("stable bootstrap statistic reported", ["0.0014"], [],
            "P(rho <= 0) is reported alongside the percentile interval")

    # --- run --------------------------------------------------------------
    # The reproducibility note deliberately quotes superseded values ("moved
    # from +204 % to -59.6 %"). Forbidden-string checks run on the body only,
    # so recording a correction is not mistaken for still making the claim.
    def split_body(raw: str) -> str:
        for marker in ("## Reproducibility and analysis note",
                       "\\section*{Reproducibility and analysis note}"):
            i = raw.find(marker)
            if i != -1:
                return raw[:i]
        return raw

    texts, bodies = {}, {}
    for label, p in (("md", MD), ("tex", TEX)):
        if p.exists():
            raw = p.read_text(encoding="utf-8")
            texts[label] = norm(raw)
            bodies[label] = norm(split_body(raw))

    fails = []
    print("CLAIMS VERIFICATION")
    print("=" * 74)
    for c in claims:
        bad = []
        for label, txt in texts.items():
            for f in c["forbidden"]:
                if norm(f) in bodies.get(label, txt):
                    bad.append(f"{label}: FORBIDDEN present (body) -> {f!r}")
            if c["required"]:
                if not any(norm(r) in txt for r in c["required"]):
                    bad.append(f"{label}: required missing -> {c['required']}")
        status = "FAIL" if bad else "ok"
        if bad:
            fails.append((c["name"], bad, c["note"]))
        print(f"  [{status:>4}] {c['name']}")
        for b in bad:
            print(f"          {b}")
        if bad and c["note"]:
            print(f"          note: {c['note']}")

    print("=" * 74)
    print(f"{len(claims)-len(fails)}/{len(claims)} claims clean; {len(fails)} need attention")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
