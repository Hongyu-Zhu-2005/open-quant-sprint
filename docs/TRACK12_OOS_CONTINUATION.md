# Track 12 follow-up: out-of-sample test of same-hour continuation (pre-registration)

Written and posted on 2026-11-16 (AEDT) **before** the same-hour continuation signal was computed on any data after the training panel. The scored hours were downloaded earlier for unrelated strategy checks (`docs/FORWARD_EVALUATION.md`, three-way study), but this signal has never been evaluated on them.

## Why

In the pre-registered Track 12 Part 2 run (PR #29), same-hour continuation was the only return test that rejected: k = 40, mean IC +0.0077, 95 % CI [+0.0029, +0.0128], Holm p = 0.009, control −0.0002. The halves disagreed (k = 10–40 significant before the 2025-11-06 midpoint, nothing after). A one-shot test on unseen hours decides whether the full-panel result carries forward.

## Phase A (run now, once)

- **Hypothesis (one-sided, direction fixed from the training panel):** mean hourly cross-sectional Spearman IC between the same-hour signal (k = 40) and the next-hour return is **positive**.
- **Signal and target:** exactly as in Part 2: signal_t = mean_{j=1..40} r_{t+1−24j}, target r_{t+1}, r = log(close).diff() per symbol on the full hourly grid, Spearman IC by hand with average ranks (`src.contrib.hongyu.same_hour_signal`, `hourly_spearman_ic`, commit `684650d`).
- **Scored window:** rows whose target bar opens from **2026-09-12 15:00 to 2026-11-15 23:00 UTC** (1,545 hours). Signal history before the window comes from the training panel, so no warm-up hours are lost.
- **Inputs (SHA-256 checked by the script; the run stops on any mismatch):**
  - `data/train_1h.parquet` `d68e6159bbdfb2e35978b67186d63b45d85d6ba50a6d1c484d5dd388fe573d71`
  - `agent/data/warmup_20260912_20261019/panel_1h.parquet` `28137cce6cd92fd742d121e5b525eaeed299bc2ad067dc18a7bcf828899d180b`
  - `agent/data/forward_20261019_20261114/panel_1h.parquet` `c66dd238c8a4dc47996db30b95540a28eb73db28f384792142afa8fcbc788479`
  - `agent/data/prospective_20261116/2026-11-14.parquet` `2e1bcb0d306ead13980a8bb20e13b4a360520d26bd76891277c64ccffde2b614`
  - `agent/data/prospective_20261116/2026-11-15.parquet` `46694731fdc7fef8d0c3415759ac8456d1e1b825b1ada9aff4f628fb39aff8a4`
- **Inference:** 120-hour moving-block bootstrap of the mean IC, 10,000 resamples, seed 20261116 + k. One-sided p = (1 + #{m_b − m ≥ m}) / (1 + B); two-sided 95 % percentile interval reported.
- **Decision rule:** "replicates" if one-sided p < 0.05 for k = 40. Nothing else is tested.
- **Reported without inference:** k = 5, 10, 20 and the 8-hour-earlier control for every k.
- **Code:** `agent/track12_oos_continuation.py`. It refuses to run a second time once `agent/runs/track12_oos_continuation_phaseA.json` exists.

**Power, stated in advance.** Scaling the full-panel standard error (≈ 0.0025 over 13,910 hours) to 1,545 hours gives ≈ 0.0075. If the true IC is the full-panel +0.0077, Phase A rejects with only about 25–30 % probability; if it is the first-half +0.015, about 60 %. So a rejection is informative, while a non-rejection is weak evidence against the effect. The point estimate and interval are reported either way.

## Phase B (scored once after 2027-01-11 00:00 UTC)

Same test, same code and parameters, on target bars from 2026-11-16 00:00 to 2027-01-10 23:00 UTC (1,344 hours), using the daily files already being sealed under `agent/data/prospective_20261116/`. Phase B is reported on its own and pooled with Phase A (2,889 hours, power about 40 % at IC +0.0077). No change to the signal, k or the decision rule after Phase A.

The Phase B code was written and committed on 2026-11-16, before any Phase B hour existed: `.venv/bin/python -B -m agent.track12_oos_continuation --phase B`. It refuses to run before 2027-01-11 00:00 UTC, checks each sealed daily file against its sidecar SHA-256, and writes `agent/runs/track12_oos_continuation_phaseB.json` once.

## Phase A result (run once, 2026-11-16 20:13 AEDT)

Output: `agent/runs/track12_oos_continuation_phaseA.json`; all five input hashes matched.

| k | same-hour IC [95 % CI] | one-sided p (IC > 0) | control −8 h IC [95 % CI] |
|---:|---|---:|---|
| **40 (primary)** | **−0.0117 [−0.0237, −0.0045]** | **0.970** | −0.0001 [−0.0097, +0.0137] |
| 5 | −0.0049 [−0.0175, +0.0100] | 0.809 | −0.0008 [−0.0081, +0.0108] |
| 10 | −0.0074 [−0.0197, +0.0051] | 0.884 | −0.0057 [−0.0136, +0.0103] |
| 20 | −0.0059 [−0.0208, +0.0041] | 0.715 | −0.0036 [−0.0135, +0.0104] |

**Does not replicate.** The pre-registered one-sided test fails (p = 0.97). Beyond the decision rule, the point estimate has the opposite sign and its two-sided interval excludes both zero and the training estimate +0.0077; every k is negative. Together with the second-half reversal in Part 2, the first-half continuation effect did not persist and is not a tradable signal. Phase B is still scored once after 2027-01-11, unchanged.
