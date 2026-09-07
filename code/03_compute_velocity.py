"""
Recompute velocity per the rules and validate against the legacy VELOCITY
column in the master.

Outputs:
  output/sample/velocity.parquet         (PARTID, SHIP) -> VELOCITY_COMPUTED + intermediates
  output/sample/velocity_validation.txt  agreement vs legacy + coverage stats

Run:
    python 03_compute_velocity.py [--as-of 2026-04-25]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

import config as C
from velocity import compute_velocity


SAMPLE_DIR = C.SAMPLE_DIR



def confusion_matrix(df: pl.DataFrame, legacy: str, computed: str) -> pl.DataFrame:
    return (
        df.group_by([legacy, computed]).len()
          .pivot(index=legacy, on=computed, values="len")
          .fill_null(0)
          .sort(legacy)
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--as-of", default=None,
                   help="cutoff date YYYY-MM-DD (default: max sane TRANSDATE)")
    args = p.parse_args(argv)

    t0 = time.time()
    print("Computing velocity...")
    vel = compute_velocity(
        log_path=C.OUT_FILES["spare_part_log"],
        master_path=C.OUT_FILES["spare_parts_list"],
        as_of=args.as_of,
    )
    out = SAMPLE_DIR / "velocity.parquet"
    vel.write_parquet(out, compression="zstd")
    print(f"  wrote {out}  ({vel.height:,} rows)  {time.time() - t0:.1f}s")

    # ----- Validation against legacy VELOCITY -----
    print("\nValidating against legacy VELOCITY...")
    master = (
        pl.scan_parquet(C.OUT_FILES["spare_parts_list"])
          .select(["PARTID", "SHIP", "VELOCITY"])
          .rename({"VELOCITY": "VELOCITY_LEGACY"})
          .unique()
          .collect()
    )

    joined = vel.join(master, on=["PARTID", "SHIP"], how="left")
    n_total = joined.height
    n_legacy = joined.filter(pl.col("VELOCITY_LEGACY").is_not_null()).height
    n_legacy_null = n_total - n_legacy
    n_computed = joined.filter(pl.col("VELOCITY_COMPUTED").is_not_null()).height

    n_agree_overall = (
        joined.filter(
            (pl.col("VELOCITY_LEGACY") == pl.col("VELOCITY_COMPUTED"))
            & pl.col("VELOCITY_LEGACY").is_not_null()
        ).height
    )

    cm = confusion_matrix(
        joined.filter(pl.col("VELOCITY_LEGACY").is_not_null()),
        legacy="VELOCITY_LEGACY",
        computed="VELOCITY_COMPUTED",
    )

    # Coverage gain — how many MAST-migrated parts (legacy null) now have a
    # computed velocity?
    coverage_gain = (
        joined.filter(pl.col("VELOCITY_LEGACY").is_null())
              .group_by("VELOCITY_COMPUTED").len()
              .sort("len", descending=True)
    )

    # Mass distribution
    overall_dist = (
        joined.group_by("VELOCITY_COMPUTED").len()
              .sort("len", descending=True)
    )

    # Write report
    lines = []
    lines.append("Velocity recomputation — validation report")
    lines.append("=" * 60)
    lines.append(f"Total (PARTID, SHIP) rows :          {n_total:,}")
    lines.append(f"Legacy VELOCITY populated :          {n_legacy:,}  "
                 f"({100*n_legacy/n_total:.1f}%)")
    lines.append(f"Legacy VELOCITY null      :          {n_legacy_null:,}  "
                 f"({100*n_legacy_null/n_total:.1f}%)")
    lines.append(f"Computed VELOCITY populated:         {n_computed:,}  "
                 f"({100*n_computed/n_total:.1f}%)")
    lines.append(
        f"Agreement (where legacy is set):     {n_agree_overall:,}  "
        f"({100*n_agree_overall/max(1,n_legacy):.1f}%)"
    )
    lines.append("")
    lines.append("Confusion matrix (rows = legacy, cols = computed):")
    lines.append(cm.to_pandas().to_string(index=False))
    lines.append("")
    lines.append("Coverage gain on MAST-migrated rows (legacy null):")
    lines.append(coverage_gain.to_pandas().to_string(index=False))
    lines.append("")
    lines.append("Computed velocity distribution (whole population):")
    lines.append(overall_dist.to_pandas().to_string(index=False))

    report_path = SAMPLE_DIR / "velocity_validation.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {report_path}")
    print()
    print("\n".join(lines[: 12]))
    print(f"\nDone in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
