# Track 12 follow-up: out-of-sample test of the post-settlement funding effect (pre-registration)

Written and posted on 2026-11-16 (AEDT) **before** this effect was computed on any data after the training panel. The scored hours were downloaded earlier for unrelated strategy checks. The exploratory finding comes from `docs/TRACK12_EXPLORATORY.md` (item 3).

## Exploratory finding being tested

On the training panel, after settlements whose |settled funding| was in the top decile (≥ 0.0106501 % per 8 h), the next hour's return moved in the direction of the funding sign: signed return −sign(f) × r = −10.29 bp (SE 3.62, day clusters, p = 0.005), versus −0.83 bp for the other 90 %. It was one of eight exploratory tests.

## Test (run once)

- **Unit:** one symbol × one settlement at 00:00, 08:00 or 16:00 UTC with non-zero settled funding f.
- **Outcome:** y = −sign(f) × r, where r = log(close) of the bar opening at the settlement minus log(close) of the bar before. That is the return from s:00 to s+1:00, right after settlement.
- **Group:** |f| ≥ **0.000106501**, the training-panel 90th percentile, fixed and not recomputed on new data.
- **Hypothesis (one-sided, direction from training):** mean y in this group is **negative**.
- **Inference:** mean with a cluster-robust standard error by UTC day (C/(C−1) scaling), t distribution with C − 1 degrees of freedom; one-sided p = P(T ≤ t). **Replicates if p < 0.05.** The other group's mean is reported without a test.
- **Window:** settlements from **2026-09-12 16:00 to 2026-11-13 16:00 UTC** (187 settlements × 20 symbols).
- **Inputs (CSV exports of the downloaded panels; SHA-256 checked by the script):**
  - `data/csv_export/agent__data__warmup_20260912_20261019__panel_1h.parquet.csv.gz` `931051a9c01e220c64d2e3220f07d22e3e5ca159f2570cbc4d1cfe4d4f10fc92` (from parquet `28137cce…`)
  - `data/csv_export/agent__data__forward_20261019_20261114__panel_1h.parquet.csv.gz` `79da4346c5fa930ab2817e4b3ff7009e0ea1d6f99226fd38c1f9469e6ea3993e` (from parquet `c66dd238…`)
- **Code:** `agent/track12_oos_funding.py`; it refuses a second run once `agent/runs/track12_oos_funding.json` exists.

**Power, stated in advance.** If the top group keeps about a tenth of observations (≈ 370), the standard error scales to about 11 bp. At the training effect of −10 bp, the test rejects with only about 20–30 % probability. A rejection is informative; a non-rejection is weak evidence against. Funding regimes also shift over time, so the group's size in the new window is not known in advance.

## Result (run once, 2026-11-16 20:35 AEDT)

Output: `agent/runs/track12_oos_funding.json`; both input hashes matched. 187 settlements, 3,654 symbol-settlements with non-zero funding.

| group | n | day clusters | mean signed return (bp) | SE (bp) | one-sided p (mean < 0) |
|---|---:|---:|---:|---:|---:|
| **|f| ≥ 0.000106501 (primary)** | **140** | 53 | **+5.66** | 10.14 | **0.71** |
| other | 3,514 | 63 | −0.92 | 3.36 | — |

**Does not replicate.** The mean has the opposite sign and p = 0.71. Funding was calmer in the new window: only 3.8 % of observations reached the fixed training threshold (versus 10 % in training), so the primary group has 140 observations and a standard error of 10 bp. Power was below even the 20–30 % stated in advance. The interval (about −14 to +26 bp) still includes the training −10 bp, so the test is inconclusive, but it gives no support. The post-settlement effect stays an unconfirmed exploratory observation and is not pursued as a signal.
