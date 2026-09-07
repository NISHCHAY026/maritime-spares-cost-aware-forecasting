"""
Focal-segment data restriction.

Restricts every analytical artefact (panel, sample, per-model forecasts,
sim_results, robustness frames) to the SKUs of the focal operating
segment, identified by an anonymised brand-surrogate set.

Membership rule for an SKU (PARTTYPENO surrogate, used as STOCK_ITEM_NUMBER
in the forecast files): the SKU's parts-master row is associated with at
least one in-segment brand surrogate. We treat segment membership as a
*brand-set* membership rather than a per-(SKU, ship) filter — we want to
keep all forecasts produced for SKUs that exist in the segment fleet.

Usage:
    from segment_filter import SEGMENT_BRANDS, segment_skus

    skus = segment_skus()                       # set of STOCK_ITEM_NUMBER (= PARTTYPENO)
    df_seg = df.filter(pl.col("STOCK_ITEM_NUMBER").is_in(list(skus)))
"""

from __future__ import annotations

from functools import lru_cache

import polars as pl

import config as C


# Anonymised brand surrogates for the focal operating segment
# (verified in the anonymized parts master; see commit b8af466).
SEGMENT_BRANDS = frozenset({
    "Brand_31801758",
    "Brand_99675942",
    "Brand_54359150",
})


@lru_cache(maxsize=1)
def segment_skus() -> frozenset:
    """SKUs (PARTTYPENO surrogates) appearing in the SEGMENT fleet."""
    master = pl.scan_parquet(C.OUT_FILES["spare_parts_list"]).select([
        "BRAND", "PARTTYPENO",
    ]).filter(pl.col("BRAND").is_in(list(SEGMENT_BRANDS))).collect()
    return frozenset(master["PARTTYPENO"].unique().to_list())


@lru_cache(maxsize=1)
def segment_partid() -> frozenset:
    """Per-vessel PARTIDs in SEGMENT (for transactional log filters)."""
    master = pl.scan_parquet(C.OUT_FILES["spare_parts_list"]).select([
        "BRAND", "PARTID",
    ]).filter(pl.col("BRAND").is_in(list(SEGMENT_BRANDS))).collect()
    return frozenset(master["PARTID"].unique().to_list())


@lru_cache(maxsize=1)
def segment_ships() -> frozenset:
    master = pl.scan_parquet(C.OUT_FILES["spare_parts_list"]).select([
        "BRAND", "SHIP",
    ]).filter(pl.col("BRAND").is_in(list(SEGMENT_BRANDS))).collect()
    return frozenset(s for s in master["SHIP"].unique().to_list() if s is not None)
