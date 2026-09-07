"""
Velocity (A/B/C/D) classification at (PARTID, SHIP) granularity.

Implements the eight rules supplied by the data owner:

  1. Source: Spare Part Log, Used transactions only, last 5 years.
  2. Aggregation: count Used txns per (PARTID, vessel).
  3. Per vessel: order parts by txn count desc, then cumulative sum.
  4. Thresholds: 50% / 80% / 100% of cumulative txns per vessel.
  5. A: cum <= 50%, B: cum in (50%, 80%], C: cum > 80%.
  6. If an active (replacement) part has a predecessor, take
     max(predecessor's AMOS velocity, computed velocity of new part).
  7. D := obsolete OR scrapped OR has a forward replacement (predecessor
     in a replacement pair).
  8. A -> B if no Used txn in the last 4 quarters.

Order of operations: 1-5 -> 8 -> 6 -> 7. Rule 7 is the final arbiter
(items that meet 7 are D regardless of earlier rules).

Velocity ordering (most-frequent first): A > B > C > D.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import polars as pl


VEL_RANK = {"A": 4, "B": 3, "C": 2, "D": 1, None: 0}
VEL_FROM_RANK = {4: "A", 3: "B", 2: "C", 1: "D", 0: None}


def _max_velocity(a: str | None, b: str | None) -> str | None:
    """Return the higher (= more frequent) of two velocities, A > B > C > D."""
    return VEL_FROM_RANK[max(VEL_RANK.get(a, 0), VEL_RANK.get(b, 0))]


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _parse_transdate(col: str = "TRANSDATE") -> pl.Expr:
    """
    Parse the log's TRANSDATE strings to Date.  Format we've seen is
    'YYYY-MM-DD HH:MM:SS.fff'. Polars' strptime tolerates trailing time.
    """
    return pl.col(col).str.strptime(pl.Date, "%Y-%m-%d %H:%M:%S%.f", strict=False)


def _resolve_as_of(log: pl.LazyFrame, as_of: Optional[str]) -> date:
    """
    If caller supplied an as_of, parse it. Otherwise use the latest TRANSDATE
    in the log that is <= today (the log has some 2090-* future-dated rows).
    """
    if as_of is not None:
        return datetime.strptime(as_of, "%Y-%m-%d").date()
    today = date.today()
    latest = (
        log.with_columns(_parse_transdate().alias("_d"))
           .filter(pl.col("_d") <= today)
           .select(pl.col("_d").max())
           .collect().item()
    )
    return latest if latest is not None else today


# ---------------------------------------------------------------------------
# Rules 1-5: per-vessel cumulative ABC
# ---------------------------------------------------------------------------

def _base_velocity(
    log: pl.LazyFrame,
    cutoff_history: date,
    as_of: date,
) -> pl.DataFrame:
    """
    Rules 1-5. Returns DataFrame with columns
        PARTID, SHIP, TXN_COUNT, CUM_PCT, VEL_BASE
    """
    used = (
        log.with_columns(_parse_transdate().alias("_d"))
           .filter(
               (pl.col("TRANSACTIONCODE") == "Used")
               & (pl.col("_d") >= cutoff_history)
               & (pl.col("_d") <= as_of)
           )
           .group_by(["PARTID", "SHIP"])
           .agg(pl.len().alias("TXN_COUNT"))
           .filter(pl.col("TXN_COUNT") > 0)
    )
    used = used.with_columns(
        pl.col("TXN_COUNT").sum().over("SHIP").alias("SHIP_TOTAL")
    )

    # Per-vessel ranking: sort desc by count, then cumulative sum.
    df = used.collect().sort(["SHIP", "TXN_COUNT"], descending=[False, True])
    df = df.with_columns(
        pl.col("TXN_COUNT").cum_sum().over("SHIP").alias("CUM_TXN"),
    )
    df = df.with_columns(
        (pl.col("CUM_TXN") / pl.col("SHIP_TOTAL")).alias("CUM_PCT"),
    )
    df = df.with_columns(
        pl.when(pl.col("CUM_PCT") <= 0.5).then(pl.lit("A"))
          .when(pl.col("CUM_PCT") <= 0.8).then(pl.lit("B"))
          .otherwise(pl.lit("C"))
          .alias("VEL_BASE")
    )
    return df


# ---------------------------------------------------------------------------
# Rule 8: A -> B if no Used txn in last 4 quarters
# ---------------------------------------------------------------------------

def _apply_recent_downgrade(
    df: pl.DataFrame,
    log: pl.LazyFrame,
    as_of: date,
    quarters: int = 4,
) -> pl.DataFrame:
    cutoff = as_of - timedelta(days=int(quarters * 365.25 / 4))
    recent = (
        log.with_columns(_parse_transdate().alias("_d"))
           .filter(
               (pl.col("TRANSACTIONCODE") == "Used")
               & (pl.col("_d") >= cutoff)
               & (pl.col("_d") <= as_of)
           )
           .select(["PARTID", "SHIP"])
           .unique()
           .with_columns(pl.lit(True).alias("HAS_RECENT_USE"))
           .collect()
    )
    df = df.join(recent, on=["PARTID", "SHIP"], how="left").with_columns(
        pl.col("HAS_RECENT_USE").fill_null(False)
    )
    df = df.with_columns(
        pl.when((pl.col("VEL_BASE") == "A") & (~pl.col("HAS_RECENT_USE")))
          .then(pl.lit("B"))
          .otherwise(pl.col("VEL_BASE"))
          .alias("VELOCITY_AFTER_R8")
    )
    return df


# ---------------------------------------------------------------------------
# Rule 6 (inheritance) and Rule 7 (D)
# ---------------------------------------------------------------------------

def _apply_d_and_inheritance(
    df: pl.DataFrame,
    master: pl.LazyFrame,
) -> pl.DataFrame:
    """
    Rule 7: D for any (PARTID, SHIP) whose master row has STATUS='Obsolete'
            or REPLACEDBY_PARTTYPENO non-null. ('Scrapped' is handled the
            same way if such a STATUS appears.)
    Rule 6: For "active replacement" parts (those whose PARTTYPENO appears
            in another part's REPLACEDBY_PARTTYPENO), inherit max of
            (computed velocity, predecessor's AMOS legacy VELOCITY).
            Computed at PARTTYPENO level, then broadcast to all
            (PARTID, SHIP) instances of that PARTTYPENO.
    """
    m = master.select([
        "PARTID", "SHIP", "PARTTYPENO",
        "VELOCITY", "STATUS", "REPLACEDBY_PARTTYPENO",
    ]).collect()

    # ----- Rule 7 set -----
    obsolete_status = {"Obsolete", "Scrapped", "Inactive"}
    is_d_row = (
        m.with_columns(
            (
                (pl.col("STATUS").is_in(list(obsolete_status)))
                | (pl.col("REPLACEDBY_PARTTYPENO").is_not_null()
                   & (pl.col("REPLACEDBY_PARTTYPENO") != ""))
            ).alias("IS_D")
        )
        .select(["PARTID", "SHIP", "IS_D"])
    )

    # ----- Rule 6: build PARTTYPENO -> predecessor's AMOS legacy velocity -----
    # An "active replacement" part is one whose PARTTYPENO appears in another
    # row's REPLACEDBY_PARTTYPENO. Its predecessor is that other row's
    # PARTTYPENO; we look up the predecessor's legacy VELOCITY and broadcast.
    pred_links = (
        m.filter(
            pl.col("REPLACEDBY_PARTTYPENO").is_not_null()
            & (pl.col("REPLACEDBY_PARTTYPENO") != "")
        )
        .select([
            pl.col("PARTTYPENO").alias("OLD_PARTTYPENO"),
            pl.col("REPLACEDBY_PARTTYPENO").alias("NEW_PARTTYPENO"),
            pl.col("VELOCITY").alias("OLD_AMOS_VELOCITY"),
        ])
        .group_by("NEW_PARTTYPENO")
        .agg(
            # If multiple predecessors, take the most-frequent predecessor velocity.
            pl.col("OLD_AMOS_VELOCITY").drop_nulls().mode().first()
              .alias("PRED_VELOCITY"),
        )
    )
    # Map PARTID -> PARTTYPENO so we can join the predecessor velocity onto df
    partid_to_typeno = m.select(["PARTID", "SHIP", "PARTTYPENO"]).unique()

    df = (
        df.join(partid_to_typeno, on=["PARTID", "SHIP"], how="left")
          .join(pred_links, left_on="PARTTYPENO", right_on="NEW_PARTTYPENO", how="left")
          .join(is_d_row, on=["PARTID", "SHIP"], how="left")
          .with_columns(
              pl.col("IS_D").fill_null(False),
          )
    )

    # Apply Rule 6: inherit max via type-safe when/then chain
    def _rank_expr(col: str) -> pl.Expr:
        return (
            pl.when(pl.col(col) == "A").then(4)
              .when(pl.col(col) == "B").then(3)
              .when(pl.col(col) == "C").then(2)
              .when(pl.col(col) == "D").then(1)
              .otherwise(0)
        )

    rank_new  = _rank_expr("VELOCITY_AFTER_R8")
    rank_pred = _rank_expr("PRED_VELOCITY")
    merged    = pl.max_horizontal([rank_new, rank_pred])
    inherited = (
        pl.when(merged == 4).then(pl.lit("A"))
          .when(merged == 3).then(pl.lit("B"))
          .when(merged == 2).then(pl.lit("C"))
          .when(merged == 1).then(pl.lit("D"))
          .otherwise(pl.lit(None, dtype=pl.Utf8))
    )

    df = df.with_columns(
        pl.when(pl.col("PRED_VELOCITY").is_not_null())
          .then(inherited)
          .otherwise(pl.col("VELOCITY_AFTER_R8"))
          .alias("VELOCITY_AFTER_R6")
    )

    # Apply Rule 7 last
    df = df.with_columns(
        pl.when(pl.col("IS_D"))
          .then(pl.lit("D"))
          .otherwise(pl.col("VELOCITY_AFTER_R6"))
          .alias("VELOCITY_COMPUTED")
    )

    return df


# ---------------------------------------------------------------------------
# Zero-count parts (in master but no Used txns in window): default to C.
# ---------------------------------------------------------------------------

def _add_zero_count_parts(
    df: pl.DataFrame,
    master: pl.LazyFrame,
) -> pl.DataFrame:
    all_pairs = (
        master.select(["PARTID", "SHIP"]).unique().collect()
    )
    seen = df.select(["PARTID", "SHIP"]).unique()
    missing = all_pairs.join(seen, on=["PARTID", "SHIP"], how="anti")
    missing = missing.with_columns([
        pl.lit(0, dtype=pl.UInt32).alias("TXN_COUNT"),
        pl.lit(0).alias("SHIP_TOTAL"),
        pl.lit(0).alias("CUM_TXN"),
        pl.lit(1.0).alias("CUM_PCT"),
        pl.lit("C").alias("VEL_BASE"),
        pl.lit(False).alias("HAS_RECENT_USE"),
        pl.lit("C").alias("VELOCITY_AFTER_R8"),
    ])
    cols = [c for c in df.columns if c in missing.columns]
    return pl.concat([df.select(cols), missing.select(cols)], how="vertical_relaxed")


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def compute_velocity(
    log_path: Path | str,
    master_path: Path | str,
    as_of: Optional[str] = None,
    history_years: int = 5,
    recent_quarters: int = 4,
) -> pl.DataFrame:
    """
    Returns one row per (PARTID, SHIP) with VELOCITY_COMPUTED in {A,B,C,D}.
    """
    log = pl.scan_parquet(log_path)
    master = pl.scan_parquet(master_path)

    as_of_dt = _resolve_as_of(log, as_of)
    cutoff_history = as_of_dt - timedelta(days=int(history_years * 365.25))

    df = _base_velocity(log, cutoff_history, as_of_dt)
    df = _add_zero_count_parts(df, master)
    df = _apply_recent_downgrade(df, log, as_of_dt, quarters=recent_quarters)
    df = _apply_d_and_inheritance(df, master)

    # Tidy
    return df.select([
        "PARTID", "SHIP", "PARTTYPENO",
        "TXN_COUNT", "CUM_PCT", "HAS_RECENT_USE",
        "VEL_BASE", "VELOCITY_AFTER_R8",
        "PRED_VELOCITY", "IS_D",
        "VELOCITY_COMPUTED",
    ]).rename({
        "VEL_BASE": "VELOCITY_RULES_1_5",
        "VELOCITY_AFTER_R8": "VELOCITY_AFTER_RULE_8",
        "PRED_VELOCITY": "PREDECESSOR_AMOS_VELOCITY",
    })
