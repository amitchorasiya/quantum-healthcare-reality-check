"""Reproduce every result, plot, and verdict card from scratch.

    python run_all.py            # full reproduction (a few minutes)
    python run_all.py --quick    # fast smoke run of the whole pipeline

Runs entirely on local simulators. No network or quantum hardware required
(the LLM verdict step falls back to a deterministic analyzer without an API key).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main(quick: bool = False):
    from data import make_cohort
    import make_plots
    from quantum_analyst import analyst
    import importlib

    print("== 1/6 generate cohort ==")
    make_cohort.main()

    for i, mod_path in enumerate([
        "experiments.exp1_small_data_kernel.run",
        "experiments.exp2_qaoa_optimization.run",
        "experiments.exp3_generative_efficiency.run",
    ], start=2):
        print(f"== {i}/6 {mod_path} ==")
        mod = importlib.import_module(mod_path)
        mod.run(quick=quick)

    print("== 5/6 plots ==")
    make_plots.main()

    print("== 6/6 verdict cards ==")
    analyst.main()

    print("\nDONE. See results/ for JSON, results/plots/ for charts, "
          "results/verdict_cards.json for the LLM verdicts.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
