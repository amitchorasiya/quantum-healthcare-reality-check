"""Experiment 4 - the fair fight. Did we handicap quantum?

The user pushed back: maybe quantum lost because we set it up wrong. Good
challenge. So here we give the quantum kernel every advantage we gave classical,
and then some:

  1. TUNE the quantum kernel the same way we tune classical. Grid-search the
     feature map (Z vs ZZ), the number of repetitions, the entanglement pattern,
     and the SVM's C. Classical got a grid search; now quantum does too.

  2. Give quantum a HOME-FIELD problem. Quantum kernels are theorized to help
     when the data has structure a classical kernel struggles to express. So we
     build a dataset with a periodic / parity-like rule (the kind of structure
     quantum feature maps are literally designed around) and test there, not
     just on the smooth synthetic cohort.

We compare the best tuned quantum kernel against the best tuned classical model
on BOTH datasets. Whatever happens, we report it.

Runs on the local statevector simulator. Fast, exact.

    python experiments/exp4_fair_fight/run.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import DATA, save_results, set_seed  # noqa: E402

N_TRAIN = 120
N_TEST = 300
N_SEEDS = 5


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def load_smooth_cohort(n_features=6):
    """The original synthetic cohort (smooth nonlinear rule)."""
    import pandas as pd
    df = pd.read_csv(DATA / "cohort.csv")
    y = df["diagnosis"].to_numpy()
    X = df.drop(columns=["diagnosis"]).to_numpy()[:, :n_features]
    return X, y


def make_quantum_favorable(n=1500, n_features=4, seed=0):
    """A dataset with parity/periodic structure.

    The label depends on the SIGN of a product of features and a cosine term.
    Parity-like rules are the textbook case where a quantum feature map's
    entanglement is supposed to help and a plain RBF kernel struggles. If
    quantum ever wins, it should win here.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-np.pi, np.pi, size=(n, n_features))
    # parity of signs on an entangled subset + a periodic twist
    parity = np.sign(np.prod(np.sin(X[:, :3]), axis=1))
    twist = np.cos(X[:, 0] + X[:, 1]) * np.cos(X[:, 2] - X[:, 3] if n_features > 3 else X[:, 2])
    logit = 2.0 * parity + 1.0 * np.sign(twist)
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < prob).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Quantum kernel, tuned
# ---------------------------------------------------------------------------
def statevectors(X, feature_map):
    from qiskit.quantum_info import Statevector
    params = feature_map.parameters
    states = np.empty((len(X), 2 ** feature_map.num_qubits), dtype=complex)
    for i, row in enumerate(X):
        bound = feature_map.assign_parameters(dict(zip(params, row)))
        states[i] = Statevector(bound).data
    return states


def quantum_kernel_matrix(Xa, Xb, n_features, kind, reps, entanglement):
    from qiskit.circuit.library import zz_feature_map, z_feature_map
    if kind == "zz":
        fm = zz_feature_map(feature_dimension=n_features, reps=reps, entanglement=entanglement)
    else:
        fm = z_feature_map(feature_dimension=n_features, reps=reps)
    sa = statevectors(Xa, fm)
    sb = statevectors(Xb, fm)
    return (np.abs(sa.conj() @ sb.T) ** 2).real


def best_quantum_auc(Xtr, ytr, Xte, yte, n_features):
    """Grid-search the quantum kernel choices, pick the best by CV, score on test."""
    best = {"auc": -1, "cfg": None}
    grid = [
        (kind, reps, ent)
        for kind in ("z", "zz")
        for reps in (1, 2, 3)
        for ent in ("linear", "full")
        if not (kind == "z" and ent == "full")  # z-map has no entanglement knob
    ]
    for kind, reps, ent in grid:
        k_tr = quantum_kernel_matrix(Xtr, Xtr, n_features, kind, reps, ent)
        # small CV on the train kernel to choose C without touching test
        for C in (0.5, 1, 5, 20):
            clf = SVC(kernel="precomputed", C=C)
            try:
                cv = cross_val_score(clf, k_tr, ytr, cv=3, scoring="roc_auc").mean()
            except Exception:
                continue
            if cv > best["auc"]:
                best.update(auc=cv, cfg=(kind, reps, ent, C))
    # refit best on full train, score on test
    kind, reps, ent, C = best["cfg"]
    k_tr = quantum_kernel_matrix(Xtr, Xtr, n_features, kind, reps, ent)
    k_te = quantum_kernel_matrix(Xte, Xtr, n_features, kind, reps, ent)
    clf = SVC(kernel="precomputed", C=C).fit(k_tr, ytr)
    auc = roc_auc_score(yte, clf.decision_function(k_te))
    return auc, best["cfg"]


def best_classical_auc(Xtr, ytr, Xte, yte):
    grid = GridSearchCV(
        SVC(kernel="rbf"),
        {"C": [0.5, 1, 5, 20], "gamma": ["scale", 0.05, 0.1, 0.3, 1.0]},
        scoring="roc_auc", cv=3,
    )
    grid.fit(Xtr, ytr)
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                        eval_metric="logloss", verbosity=0).fit(Xtr, ytr)
    rbf_auc = roc_auc_score(yte, grid.best_estimator_.decision_function(Xte))
    xgb_auc = roc_auc_score(yte, xgb.predict_proba(Xte)[:, 1])
    return max(rbf_auc, xgb_auc)


def evaluate(name, X, y, n_features):
    q_scores, c_scores = [], []
    cfgs = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(500 + seed)
        idx = rng.permutation(len(X))
        tr, te = idx[:N_TRAIN], idx[N_TRAIN:N_TRAIN + N_TEST]
        # quantum feature maps expect angles; scale to [0, pi]
        qscaler = MinMaxScaler((0, np.pi)).fit(X[tr])
        Xtr_q, Xte_q = qscaler.transform(X[tr]), qscaler.transform(X[te])
        cscaler = StandardScaler().fit(X[tr])
        Xtr_c, Xte_c = cscaler.transform(X[tr]), cscaler.transform(X[te])

        q_auc, cfg = best_quantum_auc(Xtr_q, y[tr], Xte_q, y[te], n_features)
        c_auc = best_classical_auc(Xtr_c, y[tr], Xte_c, y[te])
        q_scores.append(q_auc)
        c_scores.append(c_auc)
        cfgs.append(cfg)
    row = {
        "dataset": name,
        "quantum_tuned_auc": round(float(np.mean(q_scores)), 4),
        "quantum_auc_std": round(float(np.std(q_scores)), 4),
        "classical_tuned_auc": round(float(np.mean(c_scores)), 4),
        "classical_auc_std": round(float(np.std(c_scores)), 4),
        "quantum_won": bool(np.mean(q_scores) > np.mean(c_scores)),
        "example_best_cfg": str(cfgs[0]),
    }
    print(f"{name:22s}  quantum={row['quantum_tuned_auc']:.3f}  "
          f"classical={row['classical_tuned_auc']:.3f}  "
          f"quantum_won={row['quantum_won']}  cfg={cfgs[0]}")
    return row


def run(quick: bool = False):
    set_seed()
    n_features = 4
    rows = []
    print("== Fair fight: tuned quantum vs tuned classical ==")
    Xs, ys = load_smooth_cohort(n_features)
    rows.append(evaluate("smooth_cohort", Xs, ys, n_features))
    Xf, yf = make_quantum_favorable(n_features=n_features, seed=0)
    rows.append(evaluate("quantum_favorable", Xf, yf, n_features))

    payload = {
        "experiment": "exp4_fair_fight",
        "question": "If we tune the quantum kernel like classical AND give it a home-field problem, does it win?",
        "config": {"n_train": N_TRAIN, "n_test": N_TEST, "n_seeds": N_SEEDS,
                   "n_features": n_features,
                   "quantum_grid": "featuremap in {z,zz} x reps{1,2,3} x entanglement{linear,full} x C{0.5,1,5,20}",
                   "classical_grid": "tuned RBF-SVM + XGBoost, best of the two"},
        "rows": rows,
    }
    save_results("exp4_fair_fight", payload)
    print("\nsaved -> results/exp4_fair_fight.json")
    return payload


if __name__ == "__main__":
    run()
