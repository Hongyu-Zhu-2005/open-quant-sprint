# Track 12 exploratory record

This branch is the fixed, reproducible record for the exploratory work reported
after the pre-registered Track 12 Part 2 run in PR #29. None of these analyses
changes the pre-registered tests. Unless an out-of-sample protocol explicitly
says otherwise, results are descriptive and uncorrected for the search that
produced them.

The complete narrative, including negative results, is in
[`docs/TRACK12_EXPLORATORY.md`](../docs/TRACK12_EXPLORATORY.md). The original
eight exploratory checks referred to in the funding note are all present in the
two files below; later work added the weekday/weekend and volatility-forecast
analyses.

| Analysis | Code | Machine-readable output |
| --- | --- | --- |
| New York/UTC clock and four extreme-funding windows | `track12_exploratory.py` | `runs/track12_exploratory.json` |
| Equivalence bounds, continuation stability, Fourier/max-t tests, market-relative returns | `track12_exploratory2.py` | `runs/track12_exploratory2.json` |
| Weekday/weekend activity | `track12_weekday_activity.py` | `runs/track12_weekday_activity.json` |
| Four seasonal-volatility forecast variants, including negative results | `seasonal_volatility.py` | `runs/seasonal_volatility.json` |
| Continuation out-of-sample Phase A | `track12_oos_continuation.py` | `runs/track12_oos_continuation_phaseA.json` |
| Funding out-of-sample test | `track12_oos_funding.py` | `runs/track12_oos_funding.json` |

The two out-of-sample protocols and timing records are in
[`docs/TRACK12_OOS_CONTINUATION.md`](../docs/TRACK12_OOS_CONTINUATION.md) and
[`docs/TRACK12_OOS_FUNDING.md`](../docs/TRACK12_OOS_FUNDING.md).

All training-panel commands use the public `data/train_1h.parquet`. The OOS
scripts verify fixed input hashes before running; the later data files are not
committed.
