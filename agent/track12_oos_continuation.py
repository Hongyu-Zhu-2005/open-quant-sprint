"""Track 12 follow-up: one-shot out-of-sample test of same-hour continuation.

Pre-registered in docs/TRACK12_OOS_CONTINUATION.md before any of the scored
data was used for this signal. Phase A scores 2026-09-12 15:00 to 2026-11-15
23:00 UTC. Phase B scores 2026-11-16 00:00 to 2027-01-10 23:00 UTC, alone and
pooled with Phase A, and refuses to run before 2027-01-11 00:00 UTC. Every
input hash is checked (Phase B: each sealed daily file against its sidecar),
and each phase refuses to run a second time once its result file exists.

Run from the repository root:
    .venv/bin/python -B -m agent.track12_oos_continuation            # Phase A (done)
    .venv/bin/python -B -m agent.track12_oos_continuation --phase B  # after 2027-01-11
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.contrib.hongyu import hourly_spearman_ic, same_hour_signal

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "agent/runs/track12_oos_continuation_phaseA.json"
RESULT_B = ROOT / "agent/runs/track12_oos_continuation_phaseB.json"
PROSPECTIVE = ROOT / "agent/data/prospective_20261116"
B_START = pd.Timestamp("2026-11-16 00:00", tz="UTC")   # first scored return, Phase B
B_END = pd.Timestamp("2027-01-10 23:00", tz="UTC")     # last scored return, Phase B
B_EARLIEST_RUN = pd.Timestamp("2027-01-11 00:00", tz="UTC")

# File -> SHA-256 fixed in the pre-registration.
INPUTS = {
    "data/train_1h.parquet":
        "d68e6159bbdfb2e35978b67186d63b45d85d6ba50a6d1c484d5dd388fe573d71",
    "agent/data/warmup_20260912_20261019/panel_1h.parquet":
        "28137cce6cd92fd742d121e5b525eaeed299bc2ad067dc18a7bcf828899d180b",
    "agent/data/forward_20261019_20261114/panel_1h.parquet":
        "c66dd238c8a4dc47996db30b95540a28eb73db28f384792142afa8fcbc788479",
    "agent/data/prospective_20261116/2026-11-14.parquet":
        "2e1bcb0d306ead13980a8bb20e13b4a360520d26bd76891277c64ccffde2b614",
    "agent/data/prospective_20261116/2026-11-15.parquet":
        "46694731fdc7fef8d0c3415759ac8456d1e1b825b1ada9aff4f628fb39aff8a4",
}
OOS_START = pd.Timestamp("2026-09-12 15:00", tz="UTC")  # first scored return (bar open)
OOS_END = pd.Timestamp("2026-11-15 23:00", tz="UTC")    # last scored return (bar open)
PRIMARY_K = 40
DESCRIPTIVE_K = (5, 10, 20)
CONTROL_OFFSET = 8
BLOCK = 120
N_BOOT = 10_000
SEED = 20261116


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap(ic: np.ndarray, seed: int) -> dict:
    """Moving-block bootstrap of the mean IC, same scheme as Part 2.

    One-sided p for H0: mean <= 0 uses the resamples centred on the observed
    mean: p = (1 + #{m_b - m >= m}) / (1 + B). The interval is two-sided 95 %.
    """
    x = ic[~np.isnan(ic)]
    n = len(x)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / BLOCK))
    prefix = np.concatenate([[0.0], np.cumsum(x)])
    starts = rng.integers(0, n - BLOCK + 1, size=(N_BOOT, n_blocks))
    sums = prefix[starts + BLOCK] - prefix[starts]
    tail = n - (n_blocks - 1) * BLOCK
    sums[:, -1] = prefix[starts[:, -1] + tail] - prefix[starts[:, -1]]
    boot = sums.sum(axis=1) / n
    mean = float(x.mean())
    lo, hi = np.quantile(boot, [0.025, 0.975])
    p_one = (1 + np.count_nonzero(boot - mean >= mean)) / (1 + N_BOOT)
    return {"mean_ic": mean, "ci95": [float(lo), float(hi)],
            "p_one_sided": float(p_one), "n_hours": int(n)}


def score(panel: pd.DataFrame, start: pd.Timestamp = OOS_START, end: pd.Timestamp = OOS_END) -> dict:
    """Score the pre-registered test on target bars from start to end (inclusive)."""
    data = panel.assign(timestamp=pd.to_datetime(panel["timestamp"], utc=True))
    close = data.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    grid = pd.date_range(close.index.min(), close.index.max(), freq="h")
    if len(grid) != len(close) or close.isna().any().any():
        raise ValueError("panel must be a complete hourly grid without missing closes")
    returns = np.log(close).diff()
    target = returns.shift(-1)
    # Row t predicts the return of bar t + 1; score rows whose target bar is in the window.
    target_time = returns.index + pd.Timedelta(hours=1)
    rows = (target_time >= start) & (target_time <= end)
    if not rows.any():
        raise ValueError("no scored hours in the panel")

    out = {}
    for k in (PRIMARY_K, *DESCRIPTIVE_K):
        same = hourly_spearman_ic(same_hour_signal(returns, k), target)[rows].to_numpy()
        ctrl = hourly_spearman_ic(same_hour_signal(returns, k, CONTROL_OFFSET), target)[rows].to_numpy()
        out[f"k{k}"] = {"same_hour": bootstrap(same, SEED + k),
                        "control_minus8h": bootstrap(ctrl, SEED + k + CONTROL_OFFSET)}
    primary = out[f"k{PRIMARY_K}"]["same_hour"]
    return {
        "window": [str(start), str(end)],
        "primary": {"k": PRIMARY_K, **primary,
                    "replicates_at_5pct_one_sided": bool(primary["p_one_sided"] < 0.05)},
        "all": out,
    }


def _phase_a() -> None:
    if RESULT.exists():
        sys.exit(f"{RESULT} exists; the pre-registered test runs once.")
    hashes = {}
    for name, expected in INPUTS.items():
        actual = _sha256(ROOT / name)
        if actual != expected:
            sys.exit(f"hash mismatch for {name}: {actual}")
        hashes[name] = actual
    panel = pd.concat([pd.read_parquet(ROOT / name) for name in INPUTS], ignore_index=True)
    panel = panel.drop_duplicates(["timestamp", "symbol"], keep="first")
    result = {"protocol": "docs/TRACK12_OOS_CONTINUATION.md", "inputs_sha256": hashes,
              **score(panel)}
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["primary"], indent=2))


def _phase_b() -> None:
    if RESULT_B.exists():
        sys.exit(f"{RESULT_B} exists; the pre-registered test runs once.")
    now = pd.Timestamp.now(tz="UTC")
    if now < B_EARLIEST_RUN:
        sys.exit(f"Phase B may run only after {B_EARLIEST_RUN}; it is {now}.")
    hashes = {}
    for name, expected in INPUTS.items():          # Phase A inputs, incl. Nov 14-15
        actual = _sha256(ROOT / name)
        if actual != expected:
            sys.exit(f"hash mismatch for {name}: {actual}")
        hashes[name] = actual
    paths = [ROOT / name for name in INPUTS]
    for day in pd.date_range("2026-11-16", "2027-01-10", freq="D"):
        path = PROSPECTIVE / f"{day.date()}.parquet"
        sidecar = PROSPECTIVE / f"{day.date()}.json"
        if not path.exists() or not sidecar.exists():
            sys.exit(f"missing sealed day {day.date()}")
        expected = json.loads(sidecar.read_text())["sha256"]
        actual = _sha256(path)
        if actual != expected:
            sys.exit(f"hash mismatch for {path.name}: {actual} (sidecar {expected})")
        hashes[str(path.relative_to(ROOT))] = actual
        paths.append(path)
    panel = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    panel = panel.drop_duplicates(["timestamp", "symbol"], keep="first")
    phase_b = score(panel, B_START, B_END)
    pooled = score(panel, OOS_START, B_END)
    result = {"protocol": "docs/TRACK12_OOS_CONTINUATION.md (Phase B)", "inputs_sha256": hashes,
              "phase_b": phase_b, "pooled_a_and_b": pooled}
    RESULT_B.parent.mkdir(parents=True, exist_ok=True)
    RESULT_B.write_text(json.dumps(result, indent=2))
    print(json.dumps({"phase_b": phase_b["primary"], "pooled": pooled["primary"]}, indent=2))


def main() -> None:
    if "--phase" in sys.argv and sys.argv[sys.argv.index("--phase") + 1].upper() == "B":
        _phase_b()
    else:
        _phase_a()


if __name__ == "__main__":
    main()
