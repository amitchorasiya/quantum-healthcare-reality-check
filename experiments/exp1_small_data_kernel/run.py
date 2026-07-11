"""Experiment 1 — "Quantum kernels help when data is scarce."

HYPED CLAIM (research-flagged sweet spot): quantum kernel methods should shine
in the small-data regime typical of rare-disease cohorts, where classical
models overfit.

THE TEST: sweep training-set size N from 30 -> 800 on a nonlinear synthetic
cohort. At each N, compare a Qiskit FidelityQuantumKernel + SVM against FULLY
TUNED classical baselines (RBF-SVM via grid search, XGBoost, logistic
regression). Multiple seeds -> 95% CIs. A tie or loss for quantum is a valid,
reportable outcome.

Runs on the local Aer/Statevector simulator. Free, offline, ~8 qubits.

Usage:
    python experiments/exp1_small_data_kernel/run.py            # full sweep
    python experiments/exp1_small_data_kernel/run.py --quick    # fast smoke run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import DATA, mean_ci, save_results, set_seed  # noqa: E402

TRAIN_SIZES = [30, 50, 80, 120, 200, 400, 800]
N_SEEDS = 5
N_QUBITS = 6  # feature dimension used by the quantum feature map
TEST_SIZE = 300


def load_xy():
    df = pd.read_csv(DATA / "cohort.csv")
    y = df["diagnosis"].to_numpy()
    X = df.drop(columns=["diagnosis"]).to_numpy()
    return X, y


def _statevectors(X, feature_map):
    """Map each row of X through the ZZFeatureMap to its statevector.

    The fidelity quantum kernel is K(x,y) = |<psi(x)|psi(y)>|^2. Rather than
    sampling that overlap circuit for every pair (slow), we compute each
    statevector once on the simulator and take inner products — exact and
    ~100x faster, identical math to FidelityQuantumKernel on a noiseless
    backend.
    """
    from qiskit.quantum_info import Statevector

    params = feature_map.parameters
    states = np.empty((len(X), 2 ** feature_map.num_qubits), dtype=complex)
    for i, row in enumerate(X):
        bound = feature_map.assign_parameters(dict(zip(params, row)))
        states[i] = Statevector(bound).data
    return states


def _fidelity_kernel(states_a, states_b):
    overlaps = states_a.conj() @ states_b.T          # <psi_a | psi_b>
    return np.abs(overlaps) ** 2


def eval_quantum(Xtr, ytr, Xte, yte, n_features):
    from qiskit.circuit.library import zz_feature_map

    feature_map = zz_feature_map(feature_dimension=n_features, reps=2, entanglement="linear")
    sv_tr = _statevectors(Xtr, feature_map)
    sv_te = _statevectors(Xte, feature_map)
    k_train = _fidelity_kernel(sv_tr, sv_tr).real
    k_test = _fidelity_kernel(sv_te, sv_tr).real
    clf = SVC(kernel="precomputed", C=1.0)
    clf.fit(k_train, ytr)
    scores = clf.decision_function(k_test)
    return roc_auc_score(yte, scores)


def eval_rbf_svm(Xtr, ytr, Xte, yte):
    """Tuned RBF-SVM — the strongest classical kernel baseline (no strawman)."""
    grid = GridSearchCV(
        SVC(kernel="rbf"),
        {"C": [0.5, 1, 5, 20], "gamma": ["scale", 0.05, 0.1, 0.3]},
        scoring="roc_auc",
        cv=3,
    )
    grid.fit(Xtr, ytr)
    scores = grid.best_estimator_.decision_function(Xte)
    return roc_auc_score(yte, scores)


def eval_xgb(Xtr, ytr, Xte, yte):
    from xgboost import XGBClassifier

    clf = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        verbosity=0,
    )
    clf.fit(Xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])


def eval_logreg(Xtr, ytr, Xte, yte):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])


def run(quick: bool = False):
    set_seed()
    X, y = load_xy()
    n_features = min(N_QUBITS, X.shape[1])

    train_sizes = [30, 80, 200] if quick else TRAIN_SIZES
    n_seeds = 2 if quick else N_SEEDS

    models = {
        "quantum_kernel_svm": lambda a, b, c, d: eval_quantum(a, b, c, d, n_features),
        "rbf_svm_tuned": eval_rbf_svm,
        "xgboost": eval_xgb,
        "logreg": eval_logreg,
    }

    rows = []
    for n_train in train_sizes:
        per_model = {m: [] for m in models}
        for seed in range(n_seeds):
            rng = np.random.default_rng(1000 + seed)
            idx = rng.permutation(len(X))
            tr, te = idx[:n_train], idx[n_train : n_train + TEST_SIZE]

            scaler = StandardScaler().fit(X[tr])
            Xtr_full, Xte_full = scaler.transform(X[tr]), scaler.transform(X[te])
            # Quantum feature map takes the first n_features dims.
            Xtr_q, Xte_q = Xtr_full[:, :n_features], Xte_full[:, :n_features]

            for name, fn in models.items():
                if name == "quantum_kernel_svm":
                    auc = fn(Xtr_q, y[tr], Xte_q, y[te])
                else:
                    auc = fn(Xtr_full, y[tr], Xte_full, y[te])
                per_model[name].append(auc)

        row = {"n_train": n_train}
        for name in models:
            m, ci = mean_ci(per_model[name])
            row[f"{name}_auc"] = round(m, 4)
            row[f"{name}_ci"] = round(ci, 4)
        rows.append(row)
        print(
            f"N={n_train:4d}  "
            + "  ".join(f"{k.replace('_auc','')}={row[k]:.3f}" for k in row if k.endswith("_auc"))
        )

    payload = {
        "experiment": "exp1_small_data_kernel",
        "claim": "Quantum kernels outperform classical models in the small-data regime (rare-disease cohorts).",
        "config": {
            "train_sizes": train_sizes, "n_seeds": n_seeds, "test_size": TEST_SIZE,
            "n_qubits": n_features, "feature_map": "ZZFeatureMap(reps=2, linear)",
            "backend": "statevector simulator",
        },
        "models": list(models.keys()),
        "rows": rows,
    }
    path = save_results("exp1_small_data_kernel", payload)
    print(f"\nsaved -> {path}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    run(**vars(ap.parse_args()))
