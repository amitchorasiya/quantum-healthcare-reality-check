"""Experiment 3 — "Quantum generators need far fewer parameters."

HYPED CLAIM (the research's single strongest quantum result): a Quantum Circuit
Born Machine (QCBM) can match or beat a classical generator on distribution
fidelity while using dramatically fewer trainable parameters — the widely-cited
"50 quantum params beat 22,000 classical params" style result.

THE TEST: learn a target distribution over discretized synthetic patient
risk-bucket categories (a real-ish health-data artifact: the joint
distribution of (age_band, risk_tier, adherence_band)). Train:
  - a QCBM (PennyLane, few qubits) and
  - a classical generator (a small explicit categorical / RBM-style model)
across a sweep of parameter budgets. Compare distribution fidelity (KL, TVD)
AT MATCHED and UNMATCHED parameter counts. We report the parameter count each
needs to reach a target fidelity — the like-for-like version of the headline claim.

Runs on PennyLane's default.qubit simulator. Free, offline.

Usage:
    python experiments/exp3_generative_efficiency/run.py [--quick]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import save_results, set_seed  # noqa: E402

N_QUBITS = 4          # models a 2^4 = 16-category joint distribution
TARGET_KL = 0.05      # "good enough" fidelity threshold


def target_distribution(n_qubits: int, seed: int = 0):
    """A structured, multi-modal target over 2^n categories.

    Interpreted as the joint distribution of discretized patient attributes;
    structure (a few dominant modes) mimics real risk-bucket data rather than
    uniform noise.
    """
    rng = np.random.default_rng(seed)
    dim = 2 ** n_qubits
    p = np.zeros(dim)
    # A handful of dominant modes + light background.
    modes = rng.choice(dim, size=4, replace=False)
    p[modes] = rng.uniform(2.0, 6.0, size=4)
    p += rng.uniform(0.05, 0.25, size=dim)
    return p / p.sum()


def kl_div(p, q, eps=1e-10):
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return float(np.sum(p * np.log(p / q)))


def tvd(p, q):
    return float(0.5 * np.sum(np.abs(p - q)))


# ---------------------------------------------------------------------------
# Quantum Circuit Born Machine (QCBM)
# ---------------------------------------------------------------------------
def train_qcbm(target, n_qubits, n_layers, steps, seed):
    import pennylane as qml
    from pennylane import numpy as pnp

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(weights):
        # Hardware-efficient ansatz: RY + RZ rotations, ring of CNOTs per layer.
        for layer in range(n_layers):
            for w in range(n_qubits):
                qml.RY(weights[layer, w, 0], wires=w)
                qml.RZ(weights[layer, w, 1], wires=w)
            for w in range(n_qubits):
                qml.CNOT(wires=[w, (w + 1) % n_qubits])
        return qml.probs(wires=range(n_qubits))

    rng = np.random.default_rng(seed)
    weights = pnp.array(rng.uniform(0, 2 * np.pi, size=(n_layers, n_qubits, 2)), requires_grad=True)
    target_p = pnp.array(target, requires_grad=False)
    opt = qml.AdamOptimizer(stepsize=0.1)

    def cost(w):
        q = circuit(w)
        return pnp.sum(target_p * pnp.log(pnp.clip(target_p, 1e-10, 1) / pnp.clip(q, 1e-10, 1)))

    for _ in range(steps):
        weights = opt.step(cost, weights)

    final_q = np.array(circuit(weights))
    n_params = int(weights.size)
    return final_q, n_params


# ---------------------------------------------------------------------------
# Classical baseline: explicit softmax categorical model trained by gradient
# descent on KL, but with a PARAMETER BUDGET (low-rank logits) so we can sweep
# param count fairly against the QCBM.
# ---------------------------------------------------------------------------
def train_classical(target, n_qubits, rank, steps, seed):
    dim = 2 ** n_qubits
    rng = np.random.default_rng(seed)
    # Low-rank factorization of the logits: params = rank*(dim rows) folded.
    # Simplest faithful budgeted model: U (dim x rank) @ v (rank,) -> logits.
    U = rng.normal(0, 0.1, size=(dim, rank))
    v = rng.normal(0, 0.1, size=(rank,))
    lr = 0.2

    def softmax(z):
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    for _ in range(steps):
        logits = U @ v
        q = softmax(logits)
        # dKL/dlogits = q - target  (for KL(target||q) w.r.t. softmax logits)
        g_logits = q - target
        gU = np.outer(g_logits, v)
        gv = U.T @ g_logits
        U -= lr * gU
        v -= lr * gv

    q = softmax(U @ v)
    n_params = int(U.size + v.size)
    return q, n_params


def run(quick: bool = False):
    set_seed()
    target = target_distribution(N_QUBITS, seed=0)

    steps = 60 if quick else 200
    qcbm_layers = [1, 2, 3] if quick else [1, 2, 3, 4, 6]
    classical_ranks = [1, 2, 4] if quick else [1, 2, 4, 8, 16]

    q_rows = []
    for L in qcbm_layers:
        q, n_params = train_qcbm(target, N_QUBITS, L, steps, seed=0)
        row = {"layers": L, "n_params": n_params, "kl": round(kl_div(target, q), 4),
               "tvd": round(tvd(target, q), 4)}
        q_rows.append(row)
        print(f"QCBM   L={L}  params={n_params:3d}  KL={row['kl']:.3f}  TVD={row['tvd']:.3f}")

    c_rows = []
    for r in classical_ranks:
        q, n_params = train_classical(target, N_QUBITS, r, steps * 3, seed=0)
        row = {"rank": r, "n_params": n_params, "kl": round(kl_div(target, q), 4),
               "tvd": round(tvd(target, q), 4)}
        c_rows.append(row)
        print(f"CLASS  r={r:2d}  params={n_params:3d}  KL={row['kl']:.3f}  TVD={row['tvd']:.3f}")

    def params_to_reach(rows):
        ok = [r for r in rows if r["kl"] <= TARGET_KL]
        return min((r["n_params"] for r in ok), default=None)

    payload = {
        "experiment": "exp3_generative_efficiency",
        "claim": "Quantum generators (QCBM) reach target distribution fidelity with far fewer parameters than classical generators.",
        "config": {"n_qubits": N_QUBITS, "n_categories": 2 ** N_QUBITS, "target_kl": TARGET_KL,
                   "steps": steps, "backend": "pennylane default.qubit",
                   "metric": "KL(target||model), lower=better"},
        "quantum_rows": q_rows,
        "classical_rows": c_rows,
        "params_to_reach_target": {
            "quantum": params_to_reach(q_rows),
            "classical": params_to_reach(c_rows),
        },
    }
    path = save_results("exp3_generative_efficiency", payload)
    print(f"\nsaved -> {path}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    run(**vars(ap.parse_args()))
