"""Track 12 follow-up items 4-7 (exploratory, training panel only).

4: equivalence bounds (TOST, intersection-union) for hour-of-day means.
5: why same-hour continuation faded: quarterly IC, volatile days, leave-one-symbol-out.
6: more powerful hour-of-day tests: low-order Fourier Wald and a max-|t| test.
7: hour of day on market-relative returns, per symbol, Holm over 20.

    python -m agent.track12_exploratory2 <panel.csv.gz|panel.parquet>
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from scipy.stats import norm

from src.contrib.hongyu import (
    block_bootstrap_mean,
    holm,
    hourly_spearman_ic,
    same_hour_signal,
    seasonal_wald,
)

Z95 = norm.ppf(0.95)


def _wide(df: pd.DataFrame, column: str) -> pd.DataFrame:
    data = df.assign(timestamp=pd.to_datetime(df["timestamp"], utc=True))
    wide = data.pivot(index="timestamp", columns="symbol", values=column).sort_index()
    grid = pd.date_range(wide.index.min(), wide.index.max(), freq="h")
    return wide.reindex(grid).astype(float)


def group_means_cov(x: np.ndarray, group: np.ndarray, cluster: np.ndarray):
    """Group means and their cluster-robust covariance, as in seasonal_wald."""
    keep = ~np.isnan(x)
    x, group, cluster = x[keep], group[keep], cluster[keep]
    _, g = np.unique(group, return_inverse=True)
    _, c = np.unique(cluster, return_inverse=True)
    n_g = np.bincount(g).astype(float)
    m = np.bincount(g, weights=x) / n_g
    s = np.zeros((c.max() + 1, len(m)))
    np.add.at(s, (c, g), (x - m[g]) / n_g[g])
    n_c = s.shape[0]
    return m, n_c / (n_c - 1) * s.T @ s, n_c


def cluster_ols_wald(y: np.ndarray, X: np.ndarray, cluster: np.ndarray, test_cols: list[int]):
    """OLS with CR1 cluster-robust covariance; small-sample F test of the given coefficients."""
    keep = ~np.isnan(y)
    y, X, cluster = y[keep], X[keep], cluster[keep]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    bread = np.linalg.inv(X.T @ X)
    _, c = np.unique(cluster, return_inverse=True)
    scores = np.zeros((c.max() + 1, X.shape[1]))
    np.add.at(scores, c, X * e[:, None])
    n_c = scores.shape[0]
    vcov = n_c / (n_c - 1) * bread @ scores.T @ scores @ bread
    b = beta[test_cols]
    v = vcov[np.ix_(test_cols, test_cols)]
    wald = float(b @ np.linalg.solve(v, b))
    q = len(test_cols)
    f_stat = wald * (n_c - q) / (q * (n_c - 1))
    return {"wald": wald, "p": float(f_dist.sf(f_stat, q, n_c - q)), "coef_bp": (b * 1e4).tolist()}


def fourier_design(hours: np.ndarray, harmonics: int = 2) -> np.ndarray:
    cols = [np.ones(len(hours))]
    for k in range(1, harmonics + 1):
        cols += [np.sin(2 * np.pi * k * hours / 24), np.cos(2 * np.pi * k * hours / 24)]
    return np.column_stack(cols)


def max_t_test(m: np.ndarray, cov: np.ndarray, n_sim: int = 200_000, seed: int = 4) -> dict:
    """Max |t| over hourly deviations from the average hour, simulated under the null."""
    g = len(m)
    c = np.eye(g) - np.ones((g, g)) / g
    d = c @ m
    sig = c @ cov @ c
    sd = np.sqrt(np.diag(sig))
    t = d / sd
    corr = sig / np.outer(sd, sd)
    rng = np.random.default_rng(seed)
    w, vecs = np.linalg.eigh(corr)
    root = vecs * np.sqrt(np.clip(w, 0, None))
    sims = rng.standard_normal((n_sim, g)) @ root.T
    max_null = np.abs(sims).max(axis=1)
    stat = float(np.abs(t).max())
    return {"max_abs_t": stat, "hour": int(np.abs(t).argmax()),
            "p": float((1 + np.count_nonzero(max_null >= stat)) / (1 + n_sim))}


def item4_equivalence(idx_ret: pd.Series, hours: np.ndarray, day: np.ndarray) -> dict:
    m, cov, n_c = group_means_cov(idx_ret.to_numpy(), hours, day)
    se = np.sqrt(np.diag(cov))
    # Intersection-union: "every hour's mean lies within +-D" holds at 5 % if each 90 % CI lies within +-D.
    bound_raw = float(np.max(np.abs(m) + Z95 * se))
    g = len(m)
    c = np.eye(g) - np.ones((g, g)) / g
    d = c @ m
    sd = np.sqrt(np.diag(c @ cov @ c))
    bound_dev = float(np.max(np.abs(d) + Z95 * sd))
    return {"clusters": n_c,
            "bound_raw_mean_bp": bound_raw * 1e4,
            "bound_deviation_from_average_hour_bp": bound_dev * 1e4,
            "largest_abs_hour_mean_bp": float(np.abs(m).max() * 1e4),
            "note": "Each hour's mean (or deviation) is within +-bound with 95 % confidence (TOST via 90 % CIs, intersection-union)."}


def item5_continuation(returns: pd.DataFrame, seed: int = 5) -> dict:
    target = returns.shift(-1)
    ic = hourly_spearman_ic(same_hour_signal(returns, 40), target).iloc[:-1]
    out = {}
    q = ic.groupby(ic.index.to_period("Q").astype(str))
    quarterly = {}
    for name, series in q:
        s = series.dropna()
        if len(s) >= 240:
            mean, lo, hi, p = block_bootstrap_mean(s.to_numpy(), 120, n_boot=4000, seed=seed)
            quarterly[name] = {"mean_ic": mean, "lo": lo, "hi": hi, "p": p, "hours": int(len(s))}
    out["quarterly_k40"] = quarterly
    rolling = ic.rolling(24 * 90, min_periods=24 * 60).mean()
    out["rolling90d_k40_month_end"] = {str(k.date()): float(v) for k, v in rolling.resample("ME").last().dropna().items()}

    half = len(ic) // 2
    first = ic.iloc[:half]
    # Drop the most volatile 1 % of days (by daily mean absolute index return), first half.
    idx_abs = returns.mean(axis=1).abs()
    daily_vol = idx_abs.groupby(idx_abs.index.floor("D")).mean()
    cut = daily_vol.quantile(0.99)
    calm_days = daily_vol.index[daily_vol < cut]
    first_calm = first[first.index.floor("D").isin(calm_days)]
    out["first_half_all"] = float(first.mean())
    out["first_half_without_top1pct_volatile_days"] = float(first_calm.mean())
    # Leave one symbol out (first half).
    loo = {}
    for sym in returns.columns:
        sub = returns.drop(columns=sym)
        ic_s = hourly_spearman_ic(same_hour_signal(sub, 40), sub.shift(-1)).iloc[:-1].iloc[:half]
        loo[sym] = float(ic_s.mean())
    out["first_half_leave_one_symbol_out"] = loo
    return out


def item6_power(idx_ret: pd.Series, returns: pd.DataFrame, hours: np.ndarray, day: np.ndarray) -> dict:
    X = fourier_design(hours, 2)
    index_fourier = cluster_ols_wald(idx_ret.to_numpy(), X, day, [1, 2, 3, 4])
    m, cov, _ = group_means_cov(idx_ret.to_numpy(), hours, day)
    index_maxt = max_t_test(m, cov)
    sym_p = np.array([cluster_ols_wald(returns[s].to_numpy(), X, day, [1, 2, 3, 4])["p"] for s in returns.columns])
    return {"index_fourier_2_harmonics": index_fourier, "index_max_abs_t": index_maxt,
            "symbols_fourier_raw_p_below_5pct": int((sym_p < 0.05).sum()),
            "symbols_fourier_reject_after_holm": int((holm(sym_p) < 0.05).sum()),
            "symbols_fourier_min_p": float(sym_p.min())}


def item7_relative(returns: pd.DataFrame, hours: np.ndarray, day: np.ndarray) -> dict:
    rel = returns.sub(returns.mean(axis=1), axis=0)
    rows = []
    for s in rel.columns:
        _, w, p = seasonal_wald(rel[s].to_numpy(), hours, day)
        rows.append((s, w, p))
    p = np.array([r[2] for r in rows])
    adj = holm(p)
    return {"per_symbol": {s: {"wald": w, "p": pp, "p_holm": float(a)} for (s, w, pp), a in zip(rows, adj)},
            "raw_p_below_5pct": int((p < 0.05).sum()), "reject_after_holm": int((adj < 0.05).sum())}


def main(argv: list[str]) -> None:
    path = argv[1]
    df = pd.read_csv(path) if path.endswith((".csv", ".csv.gz")) else pd.read_parquet(path)
    close = _wide(df, "close")
    returns = np.log(close.where(close > 0)).diff()
    idx_ret = returns.mean(axis=1)
    hours = returns.index.hour.to_numpy()
    day = returns.index.floor("D").asi8
    result = {
        "item4_equivalence": item4_equivalence(idx_ret, hours, day),
        "item5_continuation": item5_continuation(returns),
        "item6_power": item6_power(idx_ret, returns, hours, day),
        "item7_relative": item7_relative(returns, hours, day),
    }
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main(sys.argv)
