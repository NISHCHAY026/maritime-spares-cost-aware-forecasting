# Anonymization: Method, Evidence, Disclosure Inventory, and Residual Risks

*Companion to `README.md`. Prepared for internal publication review.*

---

## 1. Method

- **Keyed, salted surrogates.** Every identifying raw value is mapped to an
  opaque surrogate via a **salted BLAKE2b keyed hash**. The mapping is
  deterministic, so the same raw value yields the same surrogate consistently
  across every file, enabling analysis, but the surrogate carries no embedded
  meaning (e.g. `Brand_########`, `Vessel_########`, `SKU_########`).
- **Namespacing.** Hashing is namespaced per field type (brand, vessel, SKU,
  part-type, vendor, department, component), so a vessel and a vendor that
  happen to share a raw string never collide to the same surrogate.
- **The salt is secret.** It is stored only on the local machine, is excluded
  from version control by `.gitignore`, and is **never shared or published**.
  Without the salt *and* the original raw data, the surrogates cannot be
  reversed in practice.

## 2. Field-by-field policy

The pipeline applies an explicit, auditable policy to every source column
(machine-readable in `code/config.py`). Anything not explicitly listed is
**dropped by default**.

**Replaced with a surrogate (identity removed):** brand, vessel/ship, part IDs
(`PARTID`, `PARTID_DB`), part-type numbers (`PARTTYPEID`, `PARTTYPENO`,
`STOCK_ITEM_NUMBER`, `REPLACEDBY_PARTTYPENO`), vendors (primary and
last-purchase), department ID, and component numbers.

**Dropped entirely (free text / sensitive labels):** part names, maker and
maker references, internal asset/AMOS numbers, location names and paths,
work-order and order IDs, reference numbers, free-text comments, "logged-by"
user, purchase-form titles/numbers, function and component names, and budget
codes.

**Kept (non-identifying, needed for the analysis):** transaction dates,
quantities, prices and currency, lead times, demand-pattern segmentation, ABC
class, criticality, velocity class, stock levels and (s, S) policy parameters,
region, and forecast-accuracy metrics.

## 3. Identity-scrub evidence

- The operator's group-level acronym and the names of its three constituent
  brands *(known to the reviewer; deliberately not written here)* were removed
  from **all** paper text, the LaTeX source, code comments and docstrings, and
  generated output summaries.
- **Verification:** a full-text search of the paper sources for those terms
  returns **zero matches** (checked 2026-05-26). Internal code identifiers and
  output folders that previously embedded the group acronym were also renamed
  to neutral terms.
- A single code comment had previously annotated one brand surrogate with its
  real name, the only place a surrogate-to-identity mapping could have leaked.
  **That annotation has been removed.**

## 4. Disclosure inventory: shared vs withheld

| Artifact | Disposition |
|---|---|
| The paper (PDF / LaTeX / Markdown), anonymized | **Public**, subject to this approval |
| Analysis & pipeline code | **Optional public** (see Residual Risk #2) |
| Raw source data (consumption log, parts master, forecast table, policy table) | **Never shared** |
| Secret salt | **Never shared; never committed** |
| Surrogate to raw-value mappings | **Never shared** |
| Anonymized dataset | **Requested: open public access**, downloadable without per-request approval (see Residual Risk #6). Fallback if not approved: on request, company-controlled. |

## 5. Residual risks, for the reviewer's judgment

These are disclosed deliberately so the decision is informed; none is hidden.

1. **Contextual re-identification (most important).** The paper describes the
   data as coming from "a multi-brand cruise group (three brands, 40 vessels,
   ~220k SKUs)." The operator is never named, but a reader familiar with the
   industry could infer it from this description. *Options:* generalize these
   descriptors further (e.g. drop the vessel/brand counts), or accept the
   residual inference as a disclosure decision.

2. **Version-control history.** If the analysis code is published from its
   existing local Git repository, **earlier commits still contain the brand
   names** (the working tree is clean, but history is not). Before any public
   code release, the history must be purged (e.g. `git filter-repo`) or a
   clean, history-free export shared instead.

3. **Author affiliation.** The paper currently lists the author by name with
   **no affiliation**. Adding a company affiliation would link the company to
   the (un-named) operator and would likely re-identify it. This is intentional
   and should be preserved unless the company decides otherwise.

4. **Retained quantitative values.** Prices, lead times, quantities, and policy
   parameters are kept (anonymized only as to *identity*). Aggregate statistics
   derived from them appear in the paper. The company should confirm that none
   of these aggregates is commercially sensitive in its own right.

5. **"Available on request" data.** The company should define exactly what is
   released on request, to whom, and under what terms. The secret salt and the
   raw data are **never** part of any such release.

6. **Open release of the anonymized dataset (this request).** Publishing the
   dataset for unrestricted public download removes the company's ability to
   vet recipients (screening, NDA) and exposes **row-level real quantitative
   values** (prices, lead times, quantities, dates) even though identities
   are replaced by surrogates. Combined with Residual Risk #1 (contextual
   re-identification), an informed party could attach genuine operational and
   commercial figures to the (inferable) operator. This is materially broader
   than releasing the paper alone or the "on request" posture. If the
   reproducibility benefit does not outweigh it, alternatives include: (a)
   on-request access with requester screening / NDA; (b) releasing only
   aggregated or derived data rather than row-level records; or (c) open
   release of a further-reduced subset.

---

*This document describes the anonymization as implemented at the time of
review. Any later change to the paper, the data shared, or the code released
should be re-checked against Sections 3 to 5 before distribution.*
