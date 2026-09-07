"""
Anonymize all source files in one pass.

Strategy:
  * Polars LazyFrames + streaming sinks — never load a whole CSV in RAM.
  * For each file: scan -> anonymize ID columns -> select kept columns -> sink Parquet.
  * Mappings are computed on-the-fly from a salted hash, so there's no need
    to build a global mapping table first. (We dump per-namespace mappings
    afterwards for audit / sanity, by reading the anonymized output and
    distincting back.)

Run:
    set ANON_SALT=<your secret>          # cmd.exe
    $env:ANON_SALT='<your secret>'        # PowerShell
    python 01_anonymize.py [--only stockmax,fcst_analysis]

Generate a fresh salt:
    python -c "import secrets; print(secrets.token_hex(32))"

The salt MUST stay private. Anyone with the salt + an anonymized file can
re-derive the full mapping by enumerating candidate plaintexts.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import polars as pl

import config as C
from hashing import (
    anonymize_columns_lazy,
    get_salt,
    select_kept_columns,
)


# ---------------------------------------------------------------------------
# Per-source loaders.  CSVs are scanned lazily; xlsx are read eagerly because
# Polars cannot scan xlsx (no row-streaming) and the only large xlsx
# (Demand Plan, 233 MB) is not in the anonymization scope here.
# ---------------------------------------------------------------------------

def scan_csv(path: Path) -> pl.LazyFrame:
    return pl.scan_csv(
        path,
        infer_schema_length=10_000,
        ignore_errors=True,             # tolerate stray quotes / encoding hiccups
        try_parse_dates=False,          # keep dates as strings; we'll fix later
        encoding="utf8-lossy",
    )


def read_xlsx_lazy(path: Path) -> pl.LazyFrame:
    # read_excel is eager, so we materialize then convert back to lazy.
    df = pl.read_excel(path)
    return df.lazy()


# ---------------------------------------------------------------------------
# Generic processor used by every source
# ---------------------------------------------------------------------------

def process_source(
    name: str,
    spec: dict,
    salt: str,
    streaming: bool = True,
) -> dict:
    src = C.SOURCES[name]
    out = C.OUT_FILES[name]
    if not src.exists():
        return {"name": name, "skipped": True, "reason": f"missing source: {src}"}

    t0 = time.time()
    if src.suffix.lower() == ".csv":
        lf = scan_csv(src)
    elif src.suffix.lower() in (".xlsx", ".xls"):
        lf = read_xlsx_lazy(src)
    else:
        return {"name": name, "skipped": True, "reason": f"unsupported: {src.suffix}"}

    available = set(lf.collect_schema().names())

    # Anonymize ID columns
    lf = anonymize_columns_lazy(
        lf,
        spec=spec.get("anonymize", {}),
        prefix_map=C.PREFIX,
        salt=salt,
        width=C.ID_WIDTH,
    )

    # Drop / keep
    lf = select_kept_columns(
        lf,
        keep=spec.get("keep", []),
        anonymize_cols=spec.get("anonymize", {}).keys(),
        keep_prefixes=spec.get("keep_prefixes", ()),
    )

    # Sink (streaming) for CSV; for xlsx we have to collect then write.
    if streaming and src.suffix.lower() == ".csv":
        lf.sink_parquet(out, compression="zstd", row_group_size=200_000)
    else:
        lf.collect(streaming=False).write_parquet(out, compression="zstd")

    rows = pl.scan_parquet(out).select(pl.len()).collect().item()
    return {
        "name": name,
        "src":  str(src),
        "out":  str(out),
        "rows": rows,
        "kept_cols": len(pl.scan_parquet(out).collect_schema().names()),
        "elapsed_s": round(time.time() - t0, 1),
        "missing_anon_cols": [
            c for c in spec.get("anonymize", {}) if c not in available
        ],
        "missing_keep_cols": [
            c for c in spec.get("keep", []) if c not in available
        ],
    }


# ---------------------------------------------------------------------------
# Mapping audit — dumps a (raw_count -> surrogate_count) sanity check per
# namespace.  We don't write the actual mapping (that would defeat the
# purpose); we just confirm cardinality stayed sane.
# ---------------------------------------------------------------------------

def audit_cardinality():
    rows = []
    targets = [
        (C.OUT_FILES["spare_part_log"],   ["BRAND", "SHIP", "PARTID", "PARTTYPEID"]),
        (C.OUT_FILES["spare_parts_list"], ["BRAND", "SHIP", "PARTID", "PARTTYPEID",
                                           "PRIMARYVENDOR", "DEPTID"]),
        (C.OUT_FILES["fcst_analysis"],    ["STOCK_ITEM_NUMBER"]),
        (C.OUT_FILES["stockmax"],         ["brand", "Ship", "Partid", "Parttypeid"]),
    ]
    for path, cols in targets:
        if not path.exists():
            continue
        lf = pl.scan_parquet(path)
        present = [c for c in cols if c in lf.collect_schema().names()]
        for c in present:
            n = lf.select(pl.col(c).n_unique()).collect().item()
            rows.append({"file": path.name, "column": c, "n_unique": n})
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SPECS = {
    "spare_part_log":   C.SPARE_PART_LOG,
    "spare_parts_list": C.SPARE_PARTS_LIST,
    "fcst_analysis":    C.FCST_ANALYSIS,
    "stockmax":         C.STOCKMAX,
}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--only",
        help="comma-separated list of source keys to process (default: all)",
    )
    p.add_argument(
        "--no-streaming", action="store_true",
        help="disable streaming sinks (debug only, will use lots of RAM)",
    )
    args = p.parse_args(argv)

    salt = get_salt()
    targets = list(SPECS.keys())
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        targets = [t for t in targets if t in wanted]
        unknown = wanted - set(SPECS)
        if unknown:
            print(f"WARNING: unknown source keys ignored: {sorted(unknown)}", file=sys.stderr)

    print(f"Salt fingerprint (first 8 hex of HMAC over namespace 'check'): "
          f"{__import__('hashlib').blake2b(b'check', key=salt.encode(), digest_size=4).hexdigest()}")
    print(f"Output dir: {C.ANON_DIR}")
    print()

    results = []
    for name in targets:
        print(f"[{name}] processing...", flush=True)
        try:
            r = process_source(name, SPECS[name], salt,
                               streaming=not args.no_streaming)
        except Exception as e:
            r = {"name": name, "error": repr(e)}
        results.append(r)
        print(f"[{name}] -> {r}\n", flush=True)

    print("\n=== Cardinality audit (anonymized output) ===")
    for row in audit_cardinality():
        print(f"  {row['file']:<28} {row['column']:<20} unique={row['n_unique']:>10,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
