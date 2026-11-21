# Track 12 exploratory follow-ups (items 2–10)

**Exploratory.** These analyses were designed after the pre-registered Track 12 Part 2 run (PR #29) and use the same training panel. They are descriptive; p-values are not corrected for the search that produced them. Code: `agent/track12_exploratory.py` (run on `data/train_1h.parquet`, via a CSV export that reproduces the Part 2 numbers exactly). Output: `agent/runs/track12_exploratory.json`.

## Item 2: hour of day in New York time

**The activity peak moves with US daylight saving.** Splitting the UTC profile by regime, both absolute index returns and volume peak at **14 UTC** under EDT (US equity open 13:30 UTC) and at **15 UTC** under EST (open 14:30 UTC):

| regime | hours | 12 | 13 | 14 | 15 | 16 | 17 UTC |
|---|---:|---:|---:|---:|---:|---:|---:|
| EDT: mean \|index return\| (bp) | 10,231 | 46.4 | 61.4 | **69.7** | 59.7 | 50.0 | 56.2 |
| EST: mean \|index return\| (bp) | 4,640 | 49.3 | 68.1 | 84.8 | **98.7** | 79.2 | 73.6 |
| EDT: log volume − day mean | | 0.205 | 0.597 | **0.697** | 0.516 | 0.354 | 0.207 |
| EST: log volume − day mean | | −0.050 | 0.225 | 0.770 | **0.780** | 0.594 | 0.371 |

The one-hour shift is direct evidence that the intraday activity peak is tied to the US equity session (the first full hour after the 09:30 New York open). The EST sample is smaller and mostly winter, so levels differ; only the peak position is compared.

In New York time the peak-to-trough contrast is slightly larger (|return| 1.96× vs 1.85×; volume 1.07 vs 1.03 log points), but the overall Wald statistics are slightly **lower** (|return| W = 316 vs 327; volume W = 2,750 vs 3,043, ISO-week clusters). The rest of the profile, such as the UTC midnight daily close and Asian and European hours, stays anchored to UTC. The crypto clock is a mix of UTC-anchored and New York-anchored activity. Returns show no hour-of-day pattern in New York time either (p = 0.49, day clusters).

## Item 3: funding settlements conditional on extreme funding

Unit: one symbol × one settlement (00/08/16 UTC), 36,030 observations. The signed return is −sign(funding) × log return, so a positive value means the price moved against the side that pays. "Top decile" means |funding| ≥ 0.0107 % per 8 h. That is only just above Binance's 0.01 % default, so "extreme" here is mild; a stricter cut was not tried. Standard errors are clustered by UTC day.

| conditioning rate | window | top decile mean (bp) | SE | p | other 90 % mean (bp) | top vs rest p |
|---|---|---:|---:|---:|---:|---:|
| settled rate (fixed at s:00) | hour before settlement | −4.67 | 4.05 | 0.25 | −1.39 | 0.44 |
| settled rate (fixed at s:00) | **hour after settlement** | **−10.29** | 3.62 | **0.005** | −0.83 | **0.016** |
| previous settlement's rate | hour before settlement | +0.68 | 4.26 | 0.87 | +1.69 | 0.82 |
| previous settlement's rate | hour after settlement | −5.10 | 2.91 | 0.08 | +0.31 | 0.10 |

- No pre-settlement unwinding: the hour before settlement shows nothing.
- **After settlements with high |funding|, the price moved in the direction of the funding sign by about 10 bp in the next hour.** For example, after a large positive rate (longs paid), the price rose. One reading is funding harvesters closing after collecting: shorts buy back after a positive-rate settlement. The settled rate is known when the post-settlement bar opens, so this conditioning is causal for that bar.
- Caveats: this is one of eight tests run here, found after looking (with Holm over the four top-decile tests it would still be p ≈ 0.02). It weakens to about 5 bp (p = 0.08) when conditioning on the previous settlement's rate. About 10 bp is roughly one 5 + 5 bps round trip. It is a hypothesis, not a result.
- **Out-of-sample check:** the exact test was pre-registered and scored once on 2026-09-12 to 2026-11-13 settlements ([record](TRACK12_OOS_FUNDING.md)). It did not replicate (+5.66 bp, SE 10.1, n = 140, one-sided p = 0.71).

## Item 4: equivalence bounds for hour-of-day means

Using the day-clustered covariance of the 24 hourly index means and two one-sided tests (each hour's 90 % interval; intersection–union, so no multiplicity adjustment is needed):

- every hour's **deviation from the average hour** lies within **±9.8 bp** with 95 % confidence, below a 10 bp round trip;
- every hour's raw mean lies within **±10.7 bp**; the largest observed hourly mean is 5.9 bp in absolute value.

So the null is not only "not significant": an hour-of-day timing edge large enough to pay a round trip is ruled out on this panel at the 5 % level.

## Item 5: why same-hour continuation faded (k = 40)

| quarter | mean IC | 95 % CI (5-day blocks) | p | hours |
|---|---:|---|---:|---:|
| 2025 Q1 | −0.0037 | [−0.0224, +0.0103] | 0.68 | 1,200 |
| **2025 Q2** | **+0.0254** | [+0.0140, +0.0398] | < 0.001 | 2,184 |
| 2025 Q3 | +0.0142 | [−0.0016, +0.0249] | 0.048 | 2,208 |
| 2025 Q4 | +0.0056 | [−0.0048, +0.0157] | 0.29 | 2,208 |
| 2026 Q1 | −0.0038 | [−0.0122, +0.0058] | 0.42 | 2,160 |
| 2026 Q2 | +0.0075 | [−0.0024, +0.0180] | 0.16 | 2,184 |
| 2026 Q3 | +0.0026 | [−0.0094, +0.0169] | 0.69 | 1,766 |

- The effect is an episode, not a structural feature. It came mainly from Q2–Q3 2025 (trailing-60-day IC peaked near +0.037 in July 2025), declined gradually and was around zero from 2026 Q1. It was negative in the out-of-sample window (Figure 2).
- It is broad-based, not driven by outliers or a single coin. First-half mean IC is +0.0149 with all days and +0.0150 without the 1 % most volatile days. Leaving any one symbol out keeps it between +0.0124 (without XRP) and +0.0161 (without LINK).

## Item 6: more powerful hour-of-day tests

- **Smooth profile:** index return on two daily harmonics (four sine and cosine terms), day-clustered F test: **p = 0.99**. Per symbol, 0 of 20 reach raw p < 0.05 (smallest p = 0.31).
- **Single extreme hour:** the max |t| over the 24 hourly deviations is 2.06 (23 UTC); simulated under the null with the estimated correlation, **p = 0.60**.

Tests aimed at smooth curves or one outlying hour find nothing either, so the primary null is not a power artefact of the 24-group Wald test.

## Item 7: hour of day on market-relative returns

Each symbol's return minus the hour's cross-sectional mean, 24-hour Wald test, day clusters, Holm over 20: **5 symbols at raw p < 0.05, 2 after Holm**.

- **INJ** (Holm p < 0.001): strongest at 23 UTC, −9.6 bp relative (t = −4.7). It survives winsorising the 0.1 % tails (p = 3e-5), so it is not one crash bar. It is stronger in the first half (p = 0.0006) than the second (p = 0.06).
- **XRP** (Holm p = 0.026): +6.9 bp at 08 UTC and −6.4 bp at 20 UTC (|t| = 4.1). Only in the first half (p < 0.001; second half p = 0.61).

Relative effects exist for a few coins but sit below a 10 bp round trip and weaken in the second half, the same fading as continuation. They are not pursued; an out-of-sample test would need its own pre-registration.

## Item 8: figures

- `docs/figures/track12_fig1_hour_profile.png`: index mean return by UTC hour with 95 % cluster-robust intervals and the ±10 bp round-trip band.
- `docs/figures/track12_fig2_continuation_ic.png`: trailing-60-day mean IC of same-hour continuation (k = 40), training panel and out-of-sample window.

## Item 9: weekday and weekend activity

Code: `agent/track12_weekday_activity.py`; output: `agent/runs/track12_weekday_activity.json`. ISO-week clusters throughout.

- **Weekends are quiet.** Daily index realised volatility by weekday: Mon 3.95 %, Tue 3.50, Wed 3.46, Thu 3.47, Fri 3.79, **Sat 2.53**, Sun 3.03 (Wald p < 1e-15). Weekend volume is **54 %** of the weekday level and weekend realised volatility is **77 %** (both p ≈ 0).
- **The 13–15 UTC peak shrinks on weekends, when US equities are closed, but does not vanish.** Excess over the rest of the day at 13–15 UTC: log volume **+0.81 on weekdays vs +0.36 on weekends**; |index return| **+25.8 bp vs +8.9 bp**. The hourly profiles differ significantly (volume p = 6e-21; |return| p = 1e-7). The peak stays at 14 UTC on weekends for volume, and moves to 13 UTC for |return|.
- Reading: with item 2 (the peak moves from 14 to 15 UTC when the US leaves daylight saving), most of the afternoon activity peak is tied to the US equity session. A smaller part stays on weekends, presumably the crypto market's own Europe/US afternoon. Returns by weekday showed nothing in Part 2 (p = 0.29).

## Item 10: seasonally adjusted volatility forecasts (negative result)

Code: `agent/seasonal_volatility.py`; output: `agent/runs/seasonal_volatility.json`. Next-hour variance per symbol, all causal: **base** = mean r² over the last 168 hours; **hour** = deseasonalised 168-hour variance × the target hour's factor; **hour+wkd** = separate weekday and weekend factors. Factors are trailing means (or medians) of pooled r²/variance per cell. Evaluated on 280,971 symbol-hours with QLIKE and MSE; Diebold–Mariano with day clusters. Four variants were run and all are reported; no further variants were tried.

| factors | sample | QLIKE gain hour+wkd vs base (t) | MSE (t) | obs where seasonal is better | symbols improved |
|---|---|---:|---:|---:|---:|
| mean | all | −0.118 (−3.4) | worse (−5.9) | 55.5 % | 1 / 20 |
| mean | excl. 2025-10-10 | −0.091 (−4.3) | worse (−5.8) | 55.5 % | 1 / 20 |
| median | all | −0.023 (−1.2) | worse (−2.8) | 54.4 % | 4 / 20 |
| median | excl. 2025-10-10 | −0.010 (−0.7) | worse (−2.7) | 54.4 % | 4 / 20 |

- Seasonal adjustment **fixes calibration by hour**: realised / forecast variance ranges 0.64–1.82 across hours for the base forecast and 0.87–1.06 for the mean-factor seasonal forecast. It also wins in the majority of individual hours.
- It **does not improve average forecast loss**: QLIKE and MSE are never better and often significantly worse. Hourly variance is dominated by jumps whose timing is unrelated to the clock. Lowering forecasts in quiet hours makes the rare quiet-hour jump very costly under QLIKE. The 2025-10-10 liquidation cascade alone accounts for much of the gap with mean factors.
- Use: the intraday profile is useful for **expected activity** (for example, scheduling execution in liquid hours), not for **risk or position scaling**. No feature is added to `src/contrib`.
