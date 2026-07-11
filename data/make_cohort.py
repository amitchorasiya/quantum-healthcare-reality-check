"""Generate a synthetic 'rare-disease-like' patient cohort.

We deliberately build a SMALL-DATA, nonlinear classification problem — the
regime where quantum kernels are theorized to help. Features mimic clinical
markers (labs, age, a couple of correlated biomarkers); the label depends on a
nonlinear interaction of a few of them plus noise. No real patient data, no
HIPAA exposure. Deterministic given the seed.

Run:  python data/make_cohort.py
Emits: data/cohort.csv  (features x1..x8, plus binary 'diagnosis')
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, set_seed  # noqa: E402

N_FEATURES = 8
N_TOTAL = 1200  # generate a pool; experiments subsample small N from it


def generate(n: int = N_TOTAL, n_features: int = N_FEATURES, seed: int = 0) -> pd.DataFrame:
    set_seed(seed)
    rng = np.random.default_rng(seed)

    # Correlated latent clinical factors -> observed features (standardized).
    cov = np.eye(n_features)
    for i in range(n_features - 1):
        cov[i, i + 1] = cov[i + 1, i] = 0.35  # neighbouring markers correlate
    X = rng.multivariate_normal(mean=np.zeros(n_features), cov=cov, size=n)

    # Nonlinear diagnostic signal: interaction + periodic term across 3 markers.
    logit = (
        1.6 * X[:, 0] * X[:, 1]           # biomarker interaction
        - 1.2 * np.cos(1.5 * X[:, 2])     # nonlinear threshold effect
        + 0.8 * X[:, 3]                   # mild linear contributor
    )
    logit += rng.normal(0, 0.8, size=n)   # irreducible clinical noise
    # ~30% prevalence (rare-ish but learnable) via an intercept shift.
    logit -= np.quantile(logit, 0.70)
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < prob).astype(int)

    cols = {f"x{i+1}": X[:, i] for i in range(n_features)}
    cols["diagnosis"] = y
    return pd.DataFrame(cols)


def main():
    df = generate(seed=0)
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "cohort.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}  shape={df.shape}  prevalence={df['diagnosis'].mean():.3f}")


if __name__ == "__main__":
    main()
