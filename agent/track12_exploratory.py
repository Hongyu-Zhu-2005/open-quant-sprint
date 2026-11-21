"""Track 12 follow-up items 2 and 3 (exploratory, after the pre-registered run).

Item 2: hour of day in New York time, and the UTC activity peak split by US
daylight-saving regime. Item 3: funding-settlement hours conditional on
extreme funding. Nothing here was pre-registered; results are descriptive.

    python -m agent.track12_exploratory <panel.parquet|panel.csv.gz>
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

from src.contrib.hongyu import seasonal_wald

NY = "America/New_York"


def _wide(df: pd.DataFrame, column: str) -> pd.DataFrame:
    data = df.assign(timestamp=pd.to_datetime(df["timestamp"], utc=True))
    wide = data.pivot(index="timestamp", columns="symbol", values=column).sort_index()
    grid = pd.date_range(wide.index.min(), wide.index.max(), freq="h")
    return wide.reindex(grid).astype(float)


def _iso_week(index: pd.DatetimeIndex) -> np.ndarray:
    iso = index.isocalendar()
    return (iso["year"].astype(int) * 100 + iso["week"].astype(int)).to_numpy()


def cluster_mean_test(y: np.ndarray, cluster: np.ndarray) -> dict:
    """Mean of y with a cluster-robust standard error and a t(C - 1) p-value."""
    keep = ~np.isnan(y)
    y, cluster = y[keep], cluster[keep]
    n = len(y)
    mean = y.mean()
    _, c_idx = np.unique(cluster, return_inverse=True)
    sums = np.bincount(c_idx, weights=(y - mean) / n)
    c = len(sums)
    se = float(np.sqrt(c / (c - 1) * np.sum(sums**2)))
    t_stat = mean / se if se > 0 else np.nan
    return {"n": int(n), "clusters": int(c), "mean_bp": float(mean * 1e4),
            "se_bp": se * 1e4, "t": float(t_stat),
            "p": float(2 * t_dist.sf(abs(t_stat), c - 1))}


def _profile_stats(series: pd.Series, groups: np.ndarray) -> dict:
    prof = series.groupby(groups).mean()
    return {"peak_hour": int(prof.idxmax()), "trough_hour": int(prof.idxmin()),
            "peak": float(prof.max()), "trough": float(prof.min()),
            "profile": {int(k): float(v) for k, v in prof.items()}}


def new_york_time(df: pd.DataFrame) -> dict:
    close = _wide(df, "close")
    ret = np.log(close.where(close > 0)).diff()
    idx_ret = ret.mean(axis=1)
    total_volume = _wide(df, "quote_volume").sum(axis=1, min_count=1)
    log_vol = np.log(total_volume.where(total_volume > 0))
    utc = ret.index
    ny = utc.tz_convert(NY)
    week = _iso_week(utc)
    utc_day, ny_day = utc.floor("D").asi8, ny.normalize().asi8
    vol_utc = log_vol - log_vol.groupby(utc_day).transform("mean")
    vol_ny = log_vol - log_vol.groupby(ny_day).transform("mean")

    out = {}
    for label, hours, vol in (("UTC", utc.hour.to_numpy(), vol_utc), ("New York", ny.hour.to_numpy(), vol_ny)):
        abs_ret = idx_ret.abs()
        out[label] = {
            "abs_return_W": seasonal_wald(abs_ret.to_numpy(), hours, week)[1],
            "volume_W": seasonal_wald(vol.to_numpy(), hours, week)[1],
            "return_p": seasonal_wald(idx_ret.to_numpy(), hours, utc_day)[2],
            "abs_return_peak_trough_ratio": float(abs_ret.groupby(hours).mean().max() / abs_ret.groupby(hours).mean().min()),
            "volume_peak_minus_trough_log": float(vol.groupby(hours).mean().max() - vol.groupby(hours).mean().min()),
            "abs_return": _profile_stats(abs_ret * 1e4, hours),
            "volume": _profile_stats(vol, hours),
        }

    # UTC profile split by US daylight-saving regime: the US equity open is 13:30 UTC in EDT, 14:30 UTC in EST.
    dst = np.array([bool(t.dst()) for t in ny])
    regimes = {}
    for name, mask in (("EDT (open 13:30 UTC)", dst), ("EST (open 14:30 UTC)", ~dst)):
        h = utc.hour.to_numpy()[mask]
        a = idx_ret.abs().to_numpy()[mask] * 1e4
        v = vol_utc.to_numpy()[mask]
        pa = pd.Series(a).groupby(h).mean()
        pv = pd.Series(v).groupby(h).mean()
        regimes[name] = {"hours": int(mask.sum()),
                         "abs_return_peak_utc_hour": int(pa.idxmax()),
                         "volume_peak_utc_hour": int(pv.idxmax()),
                         "abs_return_bp_12_17_utc": {int(k): round(float(pa[k]), 1) for k in range(12, 18)},
                         "volume_12_17_utc": {int(k): round(float(pv[k]), 3) for k in range(12, 18)}}
    out["dst_split_utc"] = regimes
    return out


def extreme_funding(df: pd.DataFrame) -> dict:
    close = _wide(df, "close")
    ret = np.log(close.where(close > 0)).diff()
    funding = _wide(df, "funding_rate")
    hours = ret.index.hour
    settle_rows = np.flatnonzero(np.isin(hours, (0, 8, 16)))
    settle_rows = settle_rows[(settle_rows >= 1) & (settle_rows < len(ret))]

    f = funding.iloc[settle_rows].to_numpy()                 # rate settled at s:00
    f_prev = funding.iloc[np.clip(settle_rows - 8, 0, None)].to_numpy()  # previous settlement
    pre = ret.iloc[settle_rows - 1].to_numpy()               # bar (s-1):00-s:00, before settlement
    post = ret.iloc[settle_rows].to_numpy()                  # bar s:00-(s+1):00, after settlement
    day = np.repeat(ret.index[settle_rows].floor("D").asi8[:, None], f.shape[1], axis=1)

    results = {}
    for name, cond in (("settled rate", f), ("previous settlement's rate (known in advance)", f_prev)):
        ok = np.isfinite(cond) & (cond != 0) & np.isfinite(pre) & np.isfinite(post)
        threshold = np.quantile(np.abs(cond[ok]), 0.90)
        top = ok & (np.abs(cond) >= threshold)
        rest = ok & ~top
        sign = -np.sign(cond)  # positive when the paying side's unwinding pushes price this way
        block = {"top_decile_abs_rate_threshold_pct": float(threshold * 100)}
        for when, r in (("hour before settlement", pre), ("hour after settlement", post)):
            y = sign * r
            block[when] = {
                "top_decile": cluster_mean_test(y[top], day[top]),
                "other_90pct": cluster_mean_test(y[rest], day[rest]),
                "top_vs_rest_p": seasonal_wald(np.concatenate([y[top], y[rest]]),
                                               np.r_[np.ones(top.sum()), np.zeros(rest.sum())],
                                               np.concatenate([day[top], day[rest]]))[2],
            }
        results[name] = block
    return results


def main(argv: list[str]) -> None:
    path = argv[1]
    df = pd.read_csv(path) if path.endswith((".csv", ".csv.gz")) else pd.read_parquet(path)
    print(json.dumps({"new_york_time": new_york_time(df), "extreme_funding": extreme_funding(df)},
                     indent=2, default=float))


if __name__ == "__main__":
    main(sys.argv)
