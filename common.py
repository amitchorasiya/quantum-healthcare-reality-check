"""Shared utilities for the Quantum Healthcare Reality Check benchmarks.

Everything here is deterministic (fixed seeds) and dependency-light so that
`python experiments/expN/run.py` reproduces the committed results exactly.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

# Repo root = directory containing this file.
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
DATA = ROOT / "data"

# One global seed drives every experiment so results are byte-reproducible.
GLOBAL_SEED = 1729


def set_seed(seed: int = GLOBAL_SEED) -> None:
    """Seed numpy and python-hash randomness."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@contextmanager
def timer():
    """Context manager yielding a callable that returns elapsed seconds."""
    start = time.perf_counter()
    elapsed = {"value": None}

    def read():
        return elapsed["value"] if elapsed["value"] is not None else time.perf_counter() - start

    try:
        yield read
    finally:
        elapsed["value"] = time.perf_counter() - start


def save_results(name: str, payload: dict) -> Path:
    """Write an experiment result dict to results/<name>.json (pretty, sorted)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    return path


def load_results(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def mean_ci(values, z: float = 1.96):
    """Return (mean, half-width of 95% CI) for a list of scalars."""
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return float(arr.mean()) if arr.size else 0.0, 0.0
    mean = float(arr.mean())
    sem = float(arr.std(ddof=1) / np.sqrt(arr.size))
    return mean, z * sem
