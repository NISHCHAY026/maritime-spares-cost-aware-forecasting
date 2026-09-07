# Data terms

The dataset in this directory is anonymised operational data from a commercial
fleet operator. It is published with that operator's approval, for public
download without per-request approval.

It is **not** covered by the MIT licence in the repository root, which applies to
the code only.

## Use

The data is provided for research, teaching and replication. Cite the paper when
you use it.

## Please do not

- Attempt to re-identify the operator, its vessels, its brands, its vendors or
  any individual SKU. The surrogate keys are salted keyed hashes and the salt is
  not published; attempting to invert them is outside the terms of this release.
- Present the figures as the operator's audited financials. Unit prices, lead
  times and usage estimates are as recorded in the source systems, and simulated
  cost is the output of a model that is not validated against realised stockouts.

## Formal licence

A formal open-data licence has not been attached, because the approval that
authorised this release specified public availability but did not settle
downstream redistribution terms. If you need a specific licence for your use,
open an issue and it can be raised with the operator.

## Accuracy and warranty

The data is provided as-is, with no warranty. Known limitations are listed in
Section 7 of the paper; the anonymisation and its residual risks are described in
`../ANONYMIZATION.md`.
