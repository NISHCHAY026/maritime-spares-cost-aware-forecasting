"""
Central configuration for the anonymization pipeline.

Defines:
  - Source file paths
  - Per-file column policy: KEEP / DROP / ANONYMIZE
  - Mapping namespaces (so the same raw ID hashes to the same surrogate everywhere)
  - Output paths
  - SEGMENT_MODE switch: when env var SEGMENT_ONLY=1, all analytical scripts read /
    write to output/sample/segment/ instead of output/sample/, restricting
    analysis to the focal operating segment.

Edit only this file when source filenames or column lists change.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Paths resolve through, in order: an environment variable, an optional
# gitignored config_local.py, then a default relative to this repository. The
# released code therefore carries no absolute path from the author's machine.
#
#   SPARES_SOURCE_DIR    directory holding the raw source files (never released)
#   SPARES_PROJECT_DIR   repository root, where output/ is written
try:
    import config_local as _local
except ImportError:
    _local = None

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent


def _resolve(env_name, local_attr, default):
    v = os.environ.get(env_name)
    if v:
        return Path(v)
    if _local is not None and getattr(_local, local_attr, None):
        return Path(getattr(_local, local_attr))
    return default


SOURCE_DIR = _resolve("SPARES_SOURCE_DIR", "SOURCE_DIR", _ROOT / "data" / "source")
PROJECT_DIR = _resolve("SPARES_PROJECT_DIR", "PROJECT_DIR", _ROOT)
OUTPUT_DIR = PROJECT_DIR / "output"
ANON_DIR = OUTPUT_DIR / "anonymized"
MAP_DIR = OUTPUT_DIR / "mappings"

for d in (OUTPUT_DIR, ANON_DIR, MAP_DIR):
    d.mkdir(parents=True, exist_ok=True)

# SEGMENT-mode toggle. When SEGMENT_ONLY env var is set, analytical scripts use
# the segment/ subdirectory under output/sample/. This keeps the original
# full-fleet outputs intact for comparison if needed.
SEGMENT_MODE = os.environ.get("SEGMENT_ONLY", "").lower() in ("1", "true", "yes")
SAMPLE_DIR = OUTPUT_DIR / "sample" / ("segment" if SEGMENT_MODE else "")
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
ROBUSTNESS_DIR = SAMPLE_DIR / "robustness"
ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "spare_part_log":   SOURCE_DIR / "Spare part log.csv",
    "spare_parts_list": SOURCE_DIR / "Spare Parts List.csv",
    "fcst_analysis":    SOURCE_DIR / "fcst analysis.csv",
    "stockmax":         SOURCE_DIR / "Stockmax data .xlsx",
    "demand_plan":      SOURCE_DIR / "2026.03 Demand Plan_All Fleet.xlsx",
}

# ---------------------------------------------------------------------------
# Mapping namespaces
#   Every raw value is hashed within a namespace so that, e.g., a vessel and
#   a vendor with the same string don't collide. Namespaces are also what
#   keep BRAND, SHIP, SKU, VENDOR consistent ACROSS files.
# ---------------------------------------------------------------------------

NS_BRAND  = "brand"
NS_SHIP   = "ship"
NS_SKU    = "sku"          # PARTID, PARTID_DB
NS_TYPE   = "parttype"     # PARTTYPEID, PARTTYPENO
NS_VENDOR = "vendor"
NS_DEPT   = "dept"
NS_COMP   = "component"    # COMPNO, component item/type numbers

# ---------------------------------------------------------------------------
# Surrogate ID prefixes (purely cosmetic — analysts read these directly)
# ---------------------------------------------------------------------------

PREFIX = {
    NS_BRAND:  "Brand_",
    NS_SHIP:   "Vessel_",
    NS_SKU:    "SKU_",
    NS_TYPE:   "PT_",
    NS_VENDOR: "Vendor_",
    NS_DEPT:   "Dept_",
    NS_COMP:   "Comp_",
}

# ---------------------------------------------------------------------------
# Column policy per source file.
#
#   KEEP:       passthrough unchanged
#   ANONYMIZE:  {column_name: namespace}  -> remap via salted hash to surrogate
#   DROP:       removed entirely (free-text PII / commercially sensitive labels)
#
# Anything in the source not listed here is DROPPED by default to be safe.
# ---------------------------------------------------------------------------

# --- Spare part log (transactional consumption events) ---
SPARE_PART_LOG = {
    "anonymize": {
        "BRAND":          NS_BRAND,
        "SHIP":           NS_SHIP,
        "PARTID":         NS_SKU,
        "PARTID_DB":      NS_SKU,
        "PARTTYPEID":     NS_TYPE,
        "PARTTYPENO":     NS_TYPE,
        "COMPNO":         NS_COMP,
    },
    "keep": [
        "LOGID",
        "QUANTITY",
        "TRANSACTIONCODE",
        "TRANSDATE",
        "PRICE",
        "CURRENCYCODE",
        "JOBDESCCODE",
    ],
    "drop": [
        "LOGGEDBY", "PARTNAME", "MAKERREF", "LOCATIONNAME",
        "WORKORDERID", "WORKORDERID_DB", "REFERENCENO", "LOGCOMMENT",
        "LOCATION_PATH", "ORDERID", "ORDERLINEID",
    ],
}

# --- Spare parts master list ---
SPARE_PARTS_LIST = {
    "anonymize": {
        "BRAND":              NS_BRAND,
        "SHIP":               NS_SHIP,
        "PARTID":             NS_SKU,
        "PARTID_DB":          NS_SKU,
        "PARTTYPEID":         NS_TYPE,
        "PARTTYPENO":         NS_TYPE,
        "PRIMARYVENDOR":      NS_VENDOR,
        "LASTPURCH_VENDOR":   NS_VENDOR,
        "LASTPURCH_VENDORID": NS_VENDOR,
        "DEPTID":             NS_DEPT,
        "REPLACEDBY_PARTTYPENO": NS_TYPE,
        "LASTPURCH_COMPNO":   NS_COMP,
    },
    "keep": [
        "DEPTNAME", "STOCKUNIT",
        "INSTOCK", "STOCKMIN", "REORDERLEVEL", "STOCKMAX", "ONORDER",
        "LASTPURCH_CREATEDDATE", "LASTPURCH_APPROVEDDATE",
        "LASTPURCH_ORDEREDDATE", "LASTPURCH_RECEIVEDDATE",
        "LASTPURCH_ESTIMATED_DELIVERY_DATE",
        "LASTPURCH_FORMTYPE", "LASTPURCH_FORMSTATUS",
        "LASTPURCH_PRICEPERUNIT_USD", "LASTPURCH_PRICEPERUNIT_USD_BRAND",
        "STOCKCLASS", "CRITICALITY", "STOCKEDITEM",
        "AVGPRICE", "AVERAGEUSAGEPERYEAR",
        "AVERAGE_LEAD_TIME",
        "VELOCITY", "WANTED", "STATUS", "STATUS_SPARETYPE",
        "PERISHABLE", "EXPIRYDATE", "REORDERQUANTITY",
    ],
    "drop": [
        "PARTNAME", "MAKER", "MAKERREF", "USEAMOSNUMBER",
        "LASTPURCH_FORMNO", "LASTPURCH_TITLE",
        "LASTPURCH_FUNCNO", "LASTPURCH_FUNCDESCR",
        "LASTPURCH_COMPNAME", "LASTPURCH_COMPTYPE",
        "LASTPURCH_BUDGETCODE", "LASTPURCH_BUDGETCODEID",
        "REPLACEDBY_PARTNAME", "BUDGET_CODE",
    ],
}

# --- Forecast analysis (wide format: many *_QTR01..28 columns) ---
# Strategy: regex-keep the metric columns (they're not PII), anonymize the IDs,
# drop the rest.
FCST_ANALYSIS = {
    "anonymize": {
        "STOCK_ITEM_NUMBER": NS_TYPE,
    },
    "keep": [
        "FORECAST_ID",
        "CATEGORY",
        "SEGMENTATION_ABC",
        "SEGMENTATION",
        "ABC",
        "AVERAGE_PRICE_CURR",
        "SBA_ALPHA", "SBA_ALPHA_ACTUAL",
        "SES_ALPHA", "SES_ALPHA_ACTUAL",
        "INSERTED_ON",
    ],
    # All numeric *_QTR01..28 metric columns are auto-kept via prefix list
    "keep_prefixes": [
        "FY_QTR", "ACTUALS_",
        "SBA_INTERVAL_", "SBA_PERIODICITY_", "SBA_LEVEL_",
        "SBA_FORECAST_", "SBA_ABS_ERRORS_", "SBA_ABS_PCT_ERRORS_",
        "SBA_FORECAST_BIAS", "SBA_ERROR_SQ_", "SBA_MSE", "SBA_RMSE",
        "SES_FORECAST_", "SES_ABS_ERRORS_", "SES_ABS_PCT_ERRORS_",
        "SES_FORECAST_BIAS", "SES_ERROR_SQ_", "SES_MSE", "SES_RMSE",
        "SMA_FORECAST_", "SMA_ABS_ERRORS_", "SMA_ABS_PCT_ERRORS_",
        "SMA_FORECAST_BIAS", "SMA_ERROR_SQ_", "SMA_MSE", "SMA_RMSE",
    ],
    "drop": [],
}

# --- Stockmax data (deployed policy decisions per SKU x ship x region) ---
STOCKMAX = {
    "anonymize": {
        "brand":      NS_BRAND,
        "Ship":       NS_SHIP,
        "Partid":     NS_SKU,
        "Parttypeid": NS_TYPE,
        "Parttypeno": NS_TYPE,
    },
    "keep": [
        "Strategy", "Velocity",
        "In Stock", "In Stock Value (USD)", "On Order", "Purchase Price",
        "Criticality", "Max Stockgrade", "Stock Grade", "Stock Class",
        "Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8","Q9","Q10",
        "Q11","Q12","Q13","Q14","Q15","Q16","Q17","Q18","Q19","Q20",
        "Quarterly Forecast", "RMSE", "Quarterly Forecast Old",
        "Lead Time Avg", "Lead Time Stdv",
        "BrandAvgLeadTime", "Location Lead Time",
        "Safety Stock", "Lead Time Stock", "Economic Order Quantity",
        "Old Max", "Old Min", "Old Reorder",
        "New Max", "New Min", "New Reorder", "New Reorder Quantity",
        "SOQ Test", "Lead Time Test",
        "Change of Max % Test", "%Change in Forecast Minus Max Test",
        "ReviewedAndUpdatedManually", "M/M Has Been Changed",
        "High Max/Low Forcast", "Low Max/High Forecast",
        "High Max (Discontinued)",
        "Stocking Floor", "Stocking Ceiling",
        "AR slicer 2024", "AR slicer 2025",
        "Region",
    ],
    "drop": [
        "Partname", "Maker", "makerref",
    ],
}

# ---------------------------------------------------------------------------
# Output filenames
# ---------------------------------------------------------------------------

OUT_FILES = {
    "spare_part_log":   ANON_DIR / "consumption_log.parquet",
    "spare_parts_list": ANON_DIR / "parts_master.parquet",
    "fcst_analysis":    ANON_DIR / "forecast_analysis.parquet",
    "stockmax":         ANON_DIR / "stockmax.parquet",
}

# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

# Surrogate IDs use 8-digit zero-padded numbers, which gives ~10^8 namespace
# capacity per dimension — comfortably above the 10^5..10^6 unique values
# expected in any single namespace. Bump if needed.
ID_WIDTH = 8
