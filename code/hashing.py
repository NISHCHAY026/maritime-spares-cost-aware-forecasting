"""
Salted, namespaced surrogate-ID generation.

Design goals:
  * Deterministic — same (raw_value, namespace, salt) always yields the same surrogate.
  * Re-identification-resistant without the salt — a public release of the
    anonymized data does not allow a reader to invert the mapping by trying
    candidate vessel names, because the salt is not in the release.
  * Cross-file consistent — namespace fixes the domain, so e.g. the same
    PARTID in the log and in the master gets the same SKU surrogate.
  * No collisions in practice — 8 hex chars is 16^8 > 4e9 codes per
    namespace (>> the ~1e6 expected uniques).
  * Reversible only with the saved mapping table (which stays private).

The salt is read once from CLI / env and never written to the anonymized
output. The mapping tables ARE saved (privately) so we can re-run with new
data and stay consistent, and so we can sanity-check downstream joins.
"""

from __future__ import annotations

import hashlib
import os
from typing import Iterable

import polars as pl


def _hex_id(raw: str, namespace: str, salt: str, width: int) -> str:
    """
    Return a deterministic zero-padded numeric surrogate.

    Implementation: HMAC-SHA256 of `namespace || 0x1F || raw` keyed by salt,
    truncated to `width` hex chars then converted to a decimal string with
    leading zeros. Truncating SHA-256 to 8 hex chars keeps collision risk
    negligible for our cardinalities (birthday bound on 4e9 codes >> 1e6 ids).
    """
    if raw is None:
        return ""
    msg = f"{namespace}\x1f{raw}".encode("utf-8")
    digest = hashlib.blake2b(msg, key=salt.encode("utf-8"), digest_size=8).hexdigest()
    # Convert truncated hex -> decimal -> zero-padded fixed width.
    n = int(digest[: 2 * width], 16) % (10 ** width)
    return str(n).zfill(width)


def make_surrogate(raw: str, namespace: str, prefix: str, salt: str, width: int) -> str:
    if raw is None or raw == "":
        return ""
    return f"{prefix}{_hex_id(str(raw), namespace, salt, width)}"


def get_salt() -> str:
    """
    Resolution order:
      1. ANON_SALT environment variable.
      2. Local file `output/.salt` (gitignored; written on first run).
      3. Hard fail.
    Never echo the salt to stdout.
    """
    env = os.environ.get("ANON_SALT")
    if env:
        return env.strip()

    # Fall back to the on-disk salt file so re-runs don't need the env var.
    try:
        from config import OUTPUT_DIR  # local import; avoids circular import in tests
    except ImportError:
        OUTPUT_DIR = None

    if OUTPUT_DIR is not None:
        salt_path = OUTPUT_DIR / ".salt"
        if salt_path.exists():
            s = salt_path.read_text(encoding="utf-8").strip()
            if s:
                return s

    raise RuntimeError(
        "No salt available. Set ANON_SALT in the environment, or write a salt to "
        f"{OUTPUT_DIR / '.salt' if OUTPUT_DIR else 'output/.salt'}. "
        "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
    )


# ---------------------------------------------------------------------------
# Polars helpers — applied lazily so we never materialize the whole frame.
# ---------------------------------------------------------------------------

def anonymize_columns_lazy(
    lf: pl.LazyFrame,
    spec: dict[str, str],
    prefix_map: dict[str, str],
    salt: str,
    width: int,
) -> pl.LazyFrame:
    """
    Apply a salted-hash surrogate to each (column -> namespace) in `spec`.
    Columns missing from the frame are silently skipped (logged by caller).
    """
    cols = lf.collect_schema().names()
    exprs = []
    for col, ns in spec.items():
        if col not in cols:
            continue
        prefix = prefix_map[ns]

        def f(s: pl.Series, ns_=ns, prefix_=prefix) -> pl.Series:
            return s.cast(pl.Utf8, strict=False).map_elements(
                lambda x: make_surrogate(x, ns_, prefix_, salt, width)
                          if x is not None else None,
                return_dtype=pl.Utf8,
            )

        exprs.append(pl.col(col).map_batches(f, return_dtype=pl.Utf8).alias(col))
    if exprs:
        lf = lf.with_columns(exprs)
    return lf


def select_kept_columns(
    lf: pl.LazyFrame,
    keep: Iterable[str],
    anonymize_cols: Iterable[str],
    keep_prefixes: Iterable[str] = (),
) -> pl.LazyFrame:
    """
    Drop everything that's not explicitly KEEP'd or ANONYMIZE'd.
    Pure subtraction — preserves source ordering of survivors.
    """
    available = lf.collect_schema().names()
    keep_set = set(keep) | set(anonymize_cols)
    if keep_prefixes:
        for c in available:
            if any(c.startswith(p) for p in keep_prefixes):
                keep_set.add(c)
    survivors = [c for c in available if c in keep_set]
    return lf.select(survivors)
