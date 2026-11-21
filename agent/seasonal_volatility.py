"""Seasonally adjusted hourly volatility forecasts (exploratory, causal).

Next-hour variance forecasts for each symbol, all using data up to hour t only:
  base:      mean of r^2 over the last 168 hours.
  hour:      deseasonalised 168-hour variance x hour-of-day factor of hour t+1.
  hour+wkd:  as "hour", with separate factors for weekday and weekend hours.
Seasonal factors are trailing means (last 28 days of each cell; last 8 weekend
days for weekend cells) of cross-sectionally pooled r^2 / base variance.

Forecasts are compared on QLIKE (log s2 + r2 / s2) and MSE of r2, with a
Diebold-Mariano test whose standard error is clustered by UTC day.

    python -m agent.seasonal_volatility <panel.csv.gz|panel.parquet>
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

WINDOW = 168


def _wide(df: pd.DataFrame, column: str) -> pd.DataFrame:
    data = df.assign(timestamp=pd.to_datetime(df["timestamp"], utc=True))
    wide = data.pivot(index="timestamp", columns="symbol", values=column).sort_index()
    grid = pd.date_range(wide.index.min(), wide.index.max(), freq="h")
    return wide.reindex(grid).astype(float)


def seasonal_factors(z2: pd.Series, cell: np.ndarray, days: dict[int, int], stat: str = "mean") -> pd.Series:
    """Trailing mean of z2 within each cell; value at t uses occurrences up to and including t.

    `days[k]` is how many past occurrences of cell k to average.
    """
    out = pd.Series(np.nan, index=z2.index)
    for k in np.unique(cell):
        idx = np.flatnonzero(cell == k)
        s = z2.iloc[idx]
        n = days[k]
        roll = s.rolling(n, min_periods=max(3, n // 2))
        out.iloc[idx] = (roll.median() if stat == "median" else roll.mean()).to_numpy()
    return out


def forecasts(ret: pd.DataFrame, stat: str = "mean") -> dict[str, pd.DataFrame]:
    r2 = ret**2
    base = r2.rolling(WINDOW, min_periods=WINDOW).mean()            # uses r up to t
    z2 = (r2 / base.shift(1)).mean(axis=1)                          # pooled, normalised by prior variance
    t = ret.index
    hour = t.hour.to_numpy()
    weekend = np.asarray(t.dayofweek >= 5)
    cell24 = hour
    cell48 = hour + 24 * weekend
    f24 = seasonal_factors(z2, cell24, {k: 28 for k in range(24)}, stat)
    f48 = seasonal_factors(z2, cell48, {k: (20 if k < 24 else 8) for k in range(48)}, stat)

    out = {"base": base.shift(1)}  # forecast for hour t+1 made at t, aligned to the target row
    for name, f, cell in (("hour", f24, cell24), ("hour+wkd", f48, cell48)):
        # Latest factor of each cell known at t, then look up the target hour's cell.
        fac_by_cell = {}
        for k in np.unique(cell):
            col = pd.Series(np.where(cell == k, f.to_numpy(), np.nan), index=t).ffill()
            fac_by_cell[k] = col
        # factor applied to row t (as an observation): its own cell's factor known at t
        own = pd.Series(np.nan, index=t)
        for k, col in fac_by_cell.items():
            own[cell == k] = col[cell == k]
        deseason = r2.div(own, axis=0).rolling(WINDOW, min_periods=WINDOW).mean()
        # target row t+1: factor of cell(t+1) as known at t
        target_fac = pd.Series(np.nan, index=t)
        for k, col in fac_by_cell.items():
            target_fac[cell == k] = col.shift(1)[cell == k]
        out[name] = deseason.shift(1).mul(target_fac, axis=0)
    return out


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, cluster: np.ndarray) -> dict:
    """Mean of loss_b - loss_a (positive = a is better), day-clustered t test."""
    d = loss_b - loss_a
    keep = np.isfinite(d)
    d, cluster = d[keep], cluster[keep]
    n = len(d)
    mean = d.mean()
    _, c = np.unique(cluster, return_inverse=True)
    sums = np.bincount(c, weights=(d - mean) / n)
    n_c = len(sums)
    se = np.sqrt(n_c / (n_c - 1) * np.sum(sums**2))
    tt = mean / se
    return {"mean_gain": float(mean), "t": float(tt), "p": float(2 * t_dist.sf(abs(tt), n_c - 1)), "n": int(n)}


def evaluate(ret: pd.DataFrame, stat: str, drop_days: tuple[str, ...] = ()) -> dict:
    fc = forecasts(ret, stat)
    r2 = (ret**2).to_numpy()
    valid = np.isfinite(r2) & (r2 > 0)
    for f in fc.values():
        valid &= np.isfinite(f.to_numpy()) & (f.to_numpy() > 0)
    days = ret.index.floor("D")
    if drop_days:
        valid &= ~np.asarray(days.isin(pd.to_datetime(list(drop_days), utc=True)))[:, None]
    day = np.repeat(days.asi8[:, None], ret.shape[1], axis=1)
    losses = {}
    for name, f in fc.items():
        s2 = np.where(valid, f.to_numpy(), 1.0)
        losses[name] = {"qlike": np.where(valid, np.log(s2) + np.where(valid, r2, 0) / s2, np.nan),
                        "mse": np.where(valid, (np.where(valid, r2, 0) - s2) ** 2, np.nan)}
    out = {"observations": int(valid.sum())}
    for metric in ("qlike", "mse"):
        out[metric] = {name: float(np.nanmean(v[metric])) for name, v in losses.items()}
        for name in ("hour", "hour+wkd"):
            out[f"{metric}_dm_{name}_vs_base"] = dm_test(losses[name][metric].ravel(), losses["base"][metric].ravel(), day.ravel())
    gain = losses["base"]["qlike"] - losses["hour+wkd"]["qlike"]
    out["qlike_median_obs_gain_hourwkd"] = float(np.nanmedian(gain))
    out["share_obs_hourwkd_better"] = float(np.nanmean(gain[np.isfinite(gain)] > 0))
    out["symbols_improved_qlike_hourwkd"] = int((np.nanmean(gain, axis=0) > 0).sum())
    hours = np.repeat(ret.index.hour.to_numpy()[:, None], ret.shape[1], axis=1)
    out["realised_over_forecast_by_hour"] = {
        name: [float(np.nanmean(np.where(valid & (hours == h), r2, np.nan)) / np.nanmean(np.where(valid & (hours == h), f.to_numpy(), np.nan))) for h in range(24)]
        for name, f in fc.items()}
    return out


def main(argv: list[str]) -> None:
    path = argv[1]
    df = pd.read_csv(path) if path.endswith((".csv", ".csv.gz")) else pd.read_parquet(path)
    close = _wide(df, "close")
    ret = np.log(close.where(close > 0)).diff()
    crash = ("2025-10-10",)
    result = {f"{stat}_factors{'_excl_2025-10-10' if drop else ''}": evaluate(ret, stat, crash if drop else ())
              for stat in ("mean", "median") for drop in (False, True)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv)
