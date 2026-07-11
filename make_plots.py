"""Generate colorful result charts from the committed results/*.json files.

Produces PNGs in results/plots/ used by the README and the Confluence pages.
Deterministic — depends only on the saved result JSONs.

Usage:  python make_plots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PLOTS, load_results  # noqa: E402

# A clean, high-contrast palette (quantum = purple, classical = teal/orange).
Q_COLOR = "#7B2FF7"
C_COLORS = {"rbf_svm_tuned": "#00B8A9", "xgboost": "#F6416C", "logreg": "#FFB400"}
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})


def plot_exp1():
    d = load_results("exp1_small_data_kernel")
    rows = d["rows"]
    ns = [r["n_train"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(ns, [r["quantum_kernel_svm_auc"] for r in rows],
                yerr=[r["quantum_kernel_svm_ci"] for r in rows],
                marker="o", lw=2.5, color=Q_COLOR, label="Quantum kernel + SVM", capsize=3)
    labels = {"rbf_svm_tuned": "RBF-SVM (tuned)", "xgboost": "XGBoost", "logreg": "Logistic reg."}
    for key, color in C_COLORS.items():
        ax.errorbar(ns, [r[f"{key}_auc"] for r in rows], yerr=[r[f"{key}_ci"] for r in rows],
                    marker="s", lw=2, color=color, label=labels[key], capsize=3, alpha=0.9)
    ax.axhline(0.5, ls="--", color="gray", lw=1, label="chance (AUC=0.5)")
    ax.set_xscale("log")
    ax.set_xlabel("Training set size (N)")
    ax.set_ylabel("Test AUC")
    ax.set_title("Exp 1 — Small-data classification: does the quantum kernel win?")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS / "exp1_small_data.png")
    plt.close(fig)


def plot_exp2():
    d = load_results("exp2_qaoa_optimization")
    rows = d["rows"]
    ns = [r["n_vars"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    ax1.plot(ns, [r["qaoa_quality"] for r in rows], marker="o", lw=2.5, color=Q_COLOR, label="QAOA")
    ax1.plot(ns, [r["sa_quality"] for r in rows], marker="s", lw=2.5, color="#00B8A9", label="Simulated annealing")
    ax1.axhline(1.0, ls="--", color="gray", lw=1, label="optimal")
    ax1.set_xlabel("Problem size (binary vars)")
    ax1.set_ylabel("Solution quality (1.0 = optimal)")
    ax1.set_title("Solution quality")
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 1.05)

    ax2.plot(ns, [r["qaoa_time_s"] for r in rows], marker="o", lw=2.5, color=Q_COLOR, label="QAOA (simulated)")
    ax2.set_xlabel("Problem size (binary vars)")
    ax2.set_ylabel("Wall-clock time (s)")
    ax2.set_title("Time-to-solution (QAOA on simulator)")
    ax2.legend(fontsize=9)
    fig.suptitle("Exp 2 — QAOA vs classical solver on nurse-shift scheduling", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS / "exp2_qaoa.png", bbox_inches="tight")
    plt.close(fig)


def plot_exp3():
    d = load_results("exp3_generative_efficiency")
    q, c = d["quantum_rows"], d["classical_rows"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([r["n_params"] for r in q], [r["kl"] for r in q],
            marker="o", lw=2.5, color=Q_COLOR, label="Quantum (QCBM)")
    ax.plot([r["n_params"] for r in c], [r["kl"] for r in c],
            marker="s", lw=2.5, color="#00B8A9", label="Classical (low-rank softmax)")
    ax.axhline(d["config"]["target_kl"], ls="--", color="gray", lw=1,
               label=f"target KL={d['config']['target_kl']}")
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("KL divergence from target (lower = better)")
    ax.set_title("Exp 3 — Parameter efficiency: quantum vs classical generator")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS / "exp3_generative.png")
    plt.close(fig)


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    plot_exp1()
    plot_exp2()
    plot_exp3()
    print(f"wrote plots -> {PLOTS}")


if __name__ == "__main__":
    main()
