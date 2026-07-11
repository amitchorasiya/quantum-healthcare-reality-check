"""Experiment 2 — "QAOA rivals classical solvers on real optimization."

HYPED CLAIM (research-flagged sweet spot): variational quantum optimization
(QAOA) is a near-term route to solving the combinatorial problems that dominate
healthcare operations — here, nurse-shift assignment.

THE TEST: encode a nurse-shift assignment problem as a QUBO. Solve it with
QAOA (Qiskit's sampler-based QAOA) and compare, per problem size, against:
  - brute-force exact optimum (ground truth, small sizes),
  - simulated annealing (dimod), and
  - a random baseline.
We measure the approximation ratio (QAOA cost / optimal cost) and wall-clock.
QAOA failing to match a cheap classical solver is a valid, reportable result.

Runs on the local statevector simulator. Free, offline.

Usage:
    python experiments/exp2_qaoa_optimization/run.py [--quick]
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import save_results, set_seed, timer  # noqa: E402

# Problem sizes = number of (nurse, shift) binary decision vars.
# Capped at 10: statevector QAOA cost grows as 2^n * iterations * restarts, so
# n>10 is prohibitively slow on a laptop simulator (a finding in itself — see
# the README note on quantum simulation scaling).
SIZES = [4, 6, 8, 10]
QAOA_REPS = 2  # p (QAOA depth)


def build_nurse_qubo(n_vars: int, seed: int):
    """A small nurse-shift QUBO.

    Binary x_i = 1 if a nurse is assigned to a shift slot. We want to:
      - cover shifts (reward assignments, linear negative term),
      - avoid double-booking / over-staffing (quadratic penalty on conflicts),
      - respect a soft preference cost per assignment.
    Returns a symmetric QUBO matrix Q so that cost(x) = x^T Q x.
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((n_vars, n_vars))
    # Linear rewards (coverage) on the diagonal — negative = good to turn on.
    coverage = rng.uniform(1.0, 3.0, size=n_vars)
    prefs = rng.uniform(0.0, 1.0, size=n_vars)
    np.fill_diagonal(Q, -(coverage - prefs))
    # Pairwise conflict penalties between overlapping slots.
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if rng.random() < 0.4:  # ~40% of slot-pairs conflict
                pen = rng.uniform(1.5, 3.0)
                Q[i, j] += pen
                Q[j, i] += pen
    return Q


def qubo_cost(Q, x):
    x = np.asarray(x)
    return float(x @ Q @ x)


def brute_force(Q):
    """Return (optimal_cost, optimal_x, worst_cost) by full enumeration."""
    n = Q.shape[0]
    best_cost, best_x, worst_cost = np.inf, None, -np.inf
    for bits in itertools.product([0, 1], repeat=n):
        c = qubo_cost(Q, bits)
        if c < best_cost:
            best_cost, best_x = c, bits
        if c > worst_cost:
            worst_cost = c
    return best_cost, np.array(best_x), worst_cost


def simulated_annealing(Q, seed):
    import dimod
    from dimod import SimulatedAnnealingSampler

    bqm = dimod.BinaryQuadraticModel.from_qubo({(i, j): Q[i, j] for i in range(Q.shape[0]) for j in range(Q.shape[0]) if Q[i, j] != 0})
    sampler = SimulatedAnnealingSampler()
    res = sampler.sample(bqm, num_reads=50)
    best = res.first
    x = np.array([best.sample.get(i, 0) for i in range(Q.shape[0])])
    return qubo_cost(Q, x), x


def random_baseline(Q, seed, tries=200):
    rng = np.random.default_rng(seed)
    best_cost, best_x = np.inf, None
    for _ in range(tries):
        x = rng.integers(0, 2, size=Q.shape[0])
        c = qubo_cost(Q, x)
        if c < best_cost:
            best_cost, best_x = c, x
    return best_cost, best_x


def solve_qaoa(Q, reps: int, n_restarts: int = 3):
    """QAOA via Qiskit primitives on the local sampler simulator."""
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit.primitives import StatevectorSampler

    n = Q.shape[0]
    # Map QUBO (x in {0,1}) -> Ising (z in {-1,+1}) with x = (1 - z)/2.
    # cost = sum_i Q_ii x_i + sum_{i<j} 2 Q_ij x_i x_j  (Q symmetric).
    paulis, coeffs = [], []
    const = 0.0
    h = np.zeros(n)
    J = np.zeros((n, n))
    for i in range(n):
        const += 0.5 * Q[i, i]
        h[i] += -0.5 * Q[i, i]
        for j in range(i + 1, n):
            qij = Q[i, j] + Q[j, i]  # = 2*Q[i,j] since symmetric
            const += 0.25 * qij
            h[i] += -0.25 * qij
            h[j] += -0.25 * qij
            J[i, j] += 0.25 * qij
    for i in range(n):
        if abs(h[i]) > 1e-12:
            z = ["I"] * n
            z[i] = "Z"
            paulis.append("".join(reversed(z)))
            coeffs.append(h[i])
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > 1e-12:
                z = ["I"] * n
                z[i] = z[j] = "Z"
                paulis.append("".join(reversed(z)))
                coeffs.append(J[i, j])

    hamiltonian = SparsePauliOp(paulis, coeffs=coeffs)

    # Give QAOA a real shot: several random restarts, keep the best bitstring
    # actually observed (variational algos are sensitive to initial params).
    rng = np.random.default_rng(0)
    best_cost, best_x = np.inf, None
    for _ in range(n_restarts):
        init = rng.uniform(0, np.pi, size=2 * reps)
        qaoa = QAOA(sampler=StatevectorSampler(), optimizer=COBYLA(maxiter=150),
                    reps=reps, initial_point=init)
        result = qaoa.compute_minimum_eigenvalue(hamiltonian)
        dist = result.eigenstate
        if hasattr(dist, "binary_probabilities"):
            dist = dist.binary_probabilities()
        # Evaluate the top-k most probable bitstrings against the true QUBO.
        top = sorted(dist, key=dist.get, reverse=True)[:8]
        for key in top:
            bits = key.zfill(n) if isinstance(key, str) else format(int(key), f"0{n}b")
            x = np.array([int(b) for b in reversed(bits)])
            c = qubo_cost(Q, x)
            if c < best_cost:
                best_cost, best_x = c, x
    return best_cost, best_x


def run(quick: bool = False):
    set_seed()
    sizes = [4, 6, 8] if quick else SIZES
    reps = 1 if quick else QAOA_REPS

    rows = []
    for n in sizes:
        Q = build_nurse_qubo(n, seed=n)
        opt_cost, _, worst_cost = brute_force(Q)
        sa_cost, _ = simulated_annealing(Q, seed=n)
        rnd_cost, _ = random_baseline(Q, seed=n)
        with timer() as t:
            qaoa_cost, _ = solve_qaoa(Q, reps=reps, n_restarts=3 if n <= 8 else 2)
        qaoa_time = t()

        # Quality normalized against the true cost range: 1.0 = optimal,
        # 0.0 = worst possible assignment. Well-defined for every instance.
        denom = (worst_cost - opt_cost) or 1e-9
        def quality(c):
            return round(float((worst_cost - c) / denom), 4)

        row = {
            "n_vars": n,
            "optimal_cost": round(opt_cost, 4),
            "qaoa_cost": round(qaoa_cost, 4),
            "sa_cost": round(sa_cost, 4),
            "random_cost": round(rnd_cost, 4),
            "qaoa_quality": quality(qaoa_cost),
            "sa_quality": quality(sa_cost),
            "qaoa_hit_optimum": bool(abs(qaoa_cost - opt_cost) < 1e-6),
            "sa_hit_optimum": bool(abs(sa_cost - opt_cost) < 1e-6),
            "qaoa_time_s": round(qaoa_time, 3),
        }
        rows.append(row)
        print(
            f"n={n:2d}  opt={opt_cost:7.3f}  qaoa={qaoa_cost:7.3f}(q={row['qaoa_quality']:.2f})  "
            f"sa={sa_cost:7.3f}(q={row['sa_quality']:.2f})  qaoa_t={qaoa_time:.2f}s"
        )

    payload = {
        "experiment": "exp2_qaoa_optimization",
        "claim": "QAOA rivals classical solvers on healthcare combinatorial optimization (nurse scheduling).",
        "config": {"sizes": sizes, "qaoa_reps": reps, "optimizer": "COBYLA(100)",
                   "backend": "sampler simulator", "quality_metric": "(random-cost)/(random-optimal), 1.0=optimal"},
        "rows": rows,
    }
    path = save_results("exp2_qaoa_optimization", payload)
    print(f"\nsaved -> {path}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    run(**vars(ap.parse_args()))
