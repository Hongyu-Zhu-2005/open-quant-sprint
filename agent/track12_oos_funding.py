"""Track 12 follow-up: one-shot out-of-sample test of the post-settlement funding effect.

Pre-registered in docs/TRACK12_OOS_FUNDING.md before this effect was computed
on any data after the training panel. Checks input hashes and refuses a
second run once the result file exists.

    python -m agent.track12_oos_funding [result_path]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / "agent/runs/track12_oos_funding.json"
INPUTS = {
    "data/csv_export/agent__data__warmup_20260912_20261019__panel_1h.parquet.csv.gz":
        "931051a9c01e220c64d2e3220f07d22e3e5ca159f2570cbc4d1cfe4d4f10fc92",
    "data/csv_export/agent__data__forward_20261019_20261114__panel_1h.parquet.csv.gz":
        "79da4346c5fa930ab2817e4b3ff7009e0ea1d6f99226fd38c1f9469e6ea3993e",
}
THRESHOLD = 0.000106501          # training-panel 90th percentile of |settled funding|, fixed
FIRST_SETTLEMENT = pd.Timestamp("2026-09-12 16:00", tz="UTC")
LAST_SETTLEMENT = pd.Timestamp("2026-11-13 16:00", tz="UTC")
SETTLEMENT_HOURS = (0, 8, 16)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cluster_mean(y: np.ndarray, cluster: np.ndarray) -> dict:
    """Mean with a cluster-robust SE (C / (C - 1) scaling) and t(C - 1) p-values."""
    n = len(y)
    mean = float(y.mean())
    _, c_idx = np.unique(cluster, return_inverse=True)
    sums = np.bincount(c_idx, weights=(y - mean) / n)
    c = len(sums)
    se = float(np.sqrt(c / (c - 1) * np.sum(sums**2)))
    t_stat = mean / se
    return {"n": int(n), "clusters": int(c), "mean_bp": mean * 1e4, "se_bp": se * 1e4,
            "t": float(t_stat), "p_one_sided_negative": float(t_dist.cdf(t_stat, c - 1)),
            "p_two_sided": float(2 * t_dist.sf(abs(t_stat), c - 1))}


def score(panel: pd.DataFrame) -> dict:
    data = panel.assign(timestamp=pd.to_datetime(panel["timestamp"], utc=True))
    close = data.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    funding = data.pivot(index="timestamp", columns="symbol", values="funding_rate").sort_index()
    grid = pd.date_range(close.index.min(), close.index.max(), freq="h")
    if len(grid) != len(close) or close.isna().any().any():
        raise ValueError("panel must be a complete hourly grid without missing closes")
    ret = np.log(close).diff()  # bar s: log(close_s / close_{s-1}), i.e. s:00 -> s+1:00
    settle = ret.index[(ret.index.hour.isin(SETTLEMENT_HOURS))
                       & (ret.index >= FIRST_SETTLEMENT) & (ret.index <= LAST_SETTLEMENT)]
    f = funding.loc[settle].to_numpy()
    r = ret.loc[settle].to_numpy()
    day = np.repeat(settle.floor("D").asi8[:, None], f.shape[1], axis=1)
    ok = np.isfinite(f) & (f != 0) & np.isfinite(r)
    y = -np.sign(f) * r
    top = ok & (np.abs(f) >= THRESHOLD)
    rest = ok & ~top
    primary = cluster_mean(y[top], day[top])
    return {
        "settlements": [str(settle[0]), str(settle[-1]), int(len(settle))],
        "threshold_abs_rate": THRESHOLD,
        "observations_nonzero_funding": int(ok.sum()),
        "primary_top_group": {**primary,
                              "replicates_at_5pct_one_sided": bool(primary["p_one_sided_negative"] < 0.05)},
        "rest_group": cluster_mean(y[rest], day[rest]),
        "top_share": float(top.sum() / ok.sum()),
    }


def main(argv: list[str]) -> None:
    result_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_RESULT
    data_root = Path(argv[2]) if len(argv) > 2 else ROOT
    if result_path.exists():
        sys.exit(f"{result_path} exists; the pre-registered test runs once.")
    hashes = {}
    for name, expected in INPUTS.items():
        actual = _sha256(data_root / name)
        if actual != expected:
            sys.exit(f"hash mismatch for {name}: {actual}")
        hashes[name] = actual
    panel = pd.concat([pd.read_csv(data_root / name) for name in INPUTS], ignore_index=True)
    result = {"protocol": "docs/TRACK12_OOS_FUNDING.md", "inputs_sha256": hashes, **score(panel)}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv)
