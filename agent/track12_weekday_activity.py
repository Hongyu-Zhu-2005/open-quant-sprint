"""Track 12 follow-up (exploratory): weekday and weekend activity.

1. Day of week for daily activity: realised index volatility and log volume,
   ISO-week clusters.
2. Does the 13-15 UTC activity peak disappear on weekends, when US equities
   are closed? Hourly profiles of volume and |index return| on weekdays vs
   Saturday/Sunday, with a cluster-robust test that the profiles are equal.

    python -m agent.track12_weekday_activity <panel.csv.gz|panel.parquet>
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

from src.contrib.hongyu import seasonal_wald

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _wide(df: pd.DataFrame, column: str) -> pd.DataFrame:
    data = df.assign(timestamp=pd.to_datetime(df["timestamp"], utc=True))
    wide = data.pivot(index="timestamp", columns="symbol", values=column).sort_index()
    grid = pd.date_range(wide.index.min(), wide.index.max(), freq="h")
    return wide.reindex(grid).astype(float)


def _iso_week(index: pd.DatetimeIndex) -> np.ndarray:
    iso = index.isocalendar()
    return (iso["year"].astype(int) * 100 + iso["week"].astype(int)).to_numpy()


def profile_difference_test(x: np.ndarray, hour: np.ndarray, weekend: np.ndarray, cluster: np.ndarray) -> dict:
    """Test that the hourly profile (deviation from its own mean) is the same on weekdays and weekends.

    d_h = (m_wd,h - mean_h m_wd) - (m_we,h - mean_h m_we); W = d' inv(V_d) d on 23 contrasts,
    with V from cluster-summed influence terms of all 48 cell means; small-sample F.
    """
    keep = ~np.isnan(x)
    x, hour, weekend, cluster = x[keep], hour[keep], weekend[keep], cluster[keep]
    cell = hour + 24 * weekend.astype(int)
    counts = np.bincount(cell, minlength=48).astype(float)
    means = np.bincount(cell, weights=x, minlength=48) / counts
    _, c = np.unique(cluster, return_inverse=True)
    s = np.zeros((c.max() + 1, 48))
    np.add.at(s, (c, cell), (x - means[cell]) / counts[cell])
    n_c = s.shape[0]
    v = n_c / (n_c - 1) * s.T @ s
    center = np.eye(24) - np.ones((24, 24)) / 24
    a = np.hstack([center, -center])            # 24 x 48: centred weekday minus centred weekend
    r = np.hstack([np.eye(23), -np.ones((23, 1))]) @ a   # drop one redundant contrast
    d = r @ means
    wald = float(d @ np.linalg.solve(r @ v @ r.T, d))
    q = 23
    f_stat = wald * (n_c - q) / (q * (n_c - 1))
    wd, we = means[:24], means[24:]
    return {"wald": wald, "p": float(f_dist.sf(f_stat, q, n_c - q)), "clusters": int(n_c),
            "weekday_profile": wd.tolist(), "weekend_profile": we.tolist(),
            "weekday_peak_hour": int(wd.argmax()), "weekend_peak_hour": int(we.argmax())}


def main(argv: list[str]) -> None:
    path = argv[1]
    df = pd.read_csv(path) if path.endswith((".csv", ".csv.gz")) else pd.read_parquet(path)
    close = _wide(df, "close")
    ret = np.log(close.where(close > 0)).diff()
    idx = ret.mean(axis=1)
    qv = _wide(df, "quote_volume").sum(axis=1, min_count=1)
    log_qv = np.log(qv.where(qv > 0))
    t = ret.index
    day = t.floor("D")

    # 1. Daily activity by weekday (complete days only).
    full = idx.groupby(day).count() == 24
    rv = np.sqrt((idx**2).groupby(day).sum())[full]
    vol = np.log(qv.groupby(day).sum())[full]
    wk = _iso_week(rv.index)
    dow = rv.index.dayofweek.to_numpy()
    daily = {}
    for name, series in (("realised index volatility (daily, %)", rv * 100), ("log daily quote volume", vol)):
        means, w, p = seasonal_wald(series.to_numpy(), dow, wk)
        daily[name] = {"by_weekday": dict(zip(DAYS, map(float, means))), "wald": w, "p": p}
    wd_mask, we_mask = dow < 5, dow >= 5
    daily["weekend_to_weekday_ratio"] = {
        "realised_volatility": float(rv[we_mask].mean() / rv[wd_mask].mean()),
        "volume": float(np.exp(vol[we_mask].mean() - vol[wd_mask].mean())),
    }

    # 2. Hourly profile, weekday vs weekend.
    hours = t.hour.to_numpy()
    weekend = np.asarray(t.dayofweek >= 5)
    week = _iso_week(t)
    vol_dm = (log_qv - log_qv.groupby(day).transform("mean")).to_numpy()
    abs_bp = (idx.abs() * 1e4).to_numpy()
    profiles = {
        "log volume minus day mean": profile_difference_test(vol_dm, hours, weekend, week),
        "|index return| (bp)": profile_difference_test(abs_bp, hours, weekend, week),
    }
    for res in profiles.values():
        wd, we = np.array(res["weekday_profile"]), np.array(res["weekend_profile"])
        res["us_open_excess_13_15_minus_rest"] = {
            "weekday": float(wd[13:16].mean() - np.delete(wd, [13, 14, 15]).mean()),
            "weekend": float(we[13:16].mean() - np.delete(we, [13, 14, 15]).mean()),
        }
    print(json.dumps({"daily": daily, "hourly_weekday_vs_weekend": profiles}, indent=2, default=float))


if __name__ == "__main__":
    main(sys.argv)
