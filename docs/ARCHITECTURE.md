# Architecture

*How the code fits together, for a newcomer.*

## 1. The big picture

The system is a pipeline. Fake data goes in. Three tests run. Charts and an AI verdict come out. One command, `run_all.py`, drives the whole thing.

```
                        run_all.py  (the driver)
                              |
      +-----------------------+------------------------+
      v                       v                        v
  data/               experiments/            quantum_analyst/
  make_cohort  --CSV-->  exp1 kernel   --results-->  Claude reads the
  (fake data)            exp2 QAOA        *.json      numbers, writes
                         exp3 generative              verdict cards
                              |                        |
                              v                        v
                       make_plots.py            verdict_cards.json
                       -> results/plots/        -> README / blog table
```

## 2. The layers

**Data layer** (`data/make_cohort.py`): makes a fake patient group. Eight lab-style features plus a yes/no diagnosis that follows a hidden rule and some noise. Same output every time. No real records.

**Test layer** (`experiments/`): each test is a self-contained `run.py` with a `run(quick=False)` function. It builds its problem, runs the quantum method on a simulator and the tuned classical opponents, and saves the numbers to `results/<name>.json`.

| Folder | Quantum engine | Classical opponent | Output |
|---|---|---|---|
| exp1_small_data_kernel | Qiskit quantum kernel (statevector) | scikit-learn SVM / XGBoost | AUC vs training size |
| exp2_qaoa_optimization | Qiskit QAOA (sampler sim) | dimod annealing, brute force | quality + time vs size |
| exp3_generative_efficiency | PennyLane QCBM | low-rank softmax generator | fidelity vs parameter count |

**AI layer** (`quantum_analyst/analyst.py`): reads each results file, sends it to Claude with a skeptical prompt, and gets back a verdict card. With no API key, a built-in rule reads the same numbers, so the pipeline always finishes offline.

**Show layer** (`make_plots.py` + README/blog): turns the numbers into charts and the plain-English verdict table.

## 3. The calls I made, and why

| Decision | Why |
|---|---|
| Statevector math for the quantum kernel, not the default sampler | The math is exact on a simulator. Computing each state once and taking inner products runs about 100x faster than sampling every pair, and gives the same answer. |
| Qiskit-native QAOA, not OpenQAOA | OpenQAOA pins Python below 3.11. qiskit-algorithms gives the same QAOA and installs clean on modern Python. |
| Fake data, not Synthea | No Java dependency, and I get to set size and signal on purpose. The small-data test needs that. |
| One fixed seed (1729) | Same numbers every run. Anyone can check my work. |
| Offline rule in the AI layer | The pipeline has to finish with zero outside calls. The AI is a bonus, not a hard need. |
| Cap the puzzle at size 10 in test 2 | Simulating quantum circuits costs 2^n in time and memory. Past 10 variables a laptop stalls. I show that limit instead of hiding it. |

## 4. The flow, step by step

1. `run_all.py` calls `make_cohort.main()`. It writes `data/cohort.csv`.
2. It runs each test's `run(quick)`. Each writes `results/<exp>.json`.
3. `make_plots.main()` reads those files. It writes the PNG charts.
4. `analyst.main()` reads those files. It writes `results/verdict_cards.json`.
5. The README and blog embed the charts and the verdict table.

## 5. How to extend it

- **Add a test.** Make `experiments/expN_.../run.py` with a `run(quick=False)` that writes `results/expN.json`. Add it to the lists in `run_all.py`, `make_plots.py`, and `analyst.py`.
- **Use real hardware.** Point the Qiskit or PennyLane backend at IBM Quantum's free tier instead of the local simulator. The test code stays the same.
- **Use a different AI.** Set `QHRC_MODEL`. The analyst just needs a chat-style call.

## 6. What it depends on

- **Quantum:** qiskit, qiskit-machine-learning, qiskit-algorithms, pennylane
- **Classical ML:** scikit-learn, xgboost, dimod
- **AI writer:** anthropic (optional)
- **Data and plots:** numpy, pandas, scipy, matplotlib

All free. All local. On macOS, xgboost needs `brew install libomp`.
