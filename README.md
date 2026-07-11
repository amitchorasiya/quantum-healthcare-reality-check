# Quantum Healthcare Reality Check

**Does quantum computing beat regular computers on healthcare problems? Or is it just talk?**

I ran three tests on a laptop. Then I let an AI read the raw numbers and call it. Here is what the data said.

> **Short version:** On these tests, well-built classical methods matched or beat quantum every time, even after I tuned quantum properly and handed it a problem built to suit it. The honest claim is not "quantum is worse." It is "quantum is not better at any size I can actually run, and the size where it might win doesn't exist in hardware yet." No one has shown a real quantum win on a useful problem. That is exactly why a calm, tested look beats the hype.

---

## The scorecard

| Test | The claim | Verdict | What the numbers said |
|---|---|---|---|
| **1. Small-data quantum kernel** | Quantum wins when data is scarce (rare disease) | 🔴 **Hype** | Quantum sat near a coin flip (AUC 0.52). Tuned classical reached **0.79**. |
| **2. QAOA scheduling** | Quantum optimization rivals classical solvers | 🟡 **Mixed** | Quantum found good schedules but took **~1150 seconds** at size 10. Classical found the perfect answer at once. |
| **3. Quantum generative efficiency** | Quantum needs far fewer knobs to tune | 🔴 **Hype** | Quantum needed **24** parameters. Classical did the job with **17**. |
| **4. The fair fight** | Maybe quantum lost because it was handicapped | 🔴 **Still lost** | Tuned quantum on its own home-field problem: **0.71 vs 0.77** and **0.55 vs 0.66**. Tuning helped, but classical still won. |

*(An AI reads the raw `results/*.json` files and writes each verdict.)*

---

## New to quantum? Read this first

No physics needed. Here is everything you need to follow along.

- **Bit.** A normal computer stores bits. Each one is a 0 or a 1.
- **Qubit (quantum bit).** A quantum computer uses qubits. A qubit can act like a blend of 0 and 1 at the same time, and two qubits can link so one depends on the other. This is what *might* let a quantum computer try many options at once.
- **Simulator.** A normal program that imitates a small quantum computer on a laptop. Exact, free, and what this project runs on. Real quantum hardware exists too, but it is noisy and small today.
- **Circuit.** A short recipe of steps you run on the qubits, then measure the result.
- **The catch.** Real qubits are noisy and lose the answer as the recipe gets longer. No one has shown a real quantum computer beat a good normal computer on a useful task yet.

The three tools I tested, in plain words:

- **Quantum kernel.** A way to measure how alike two data points are, computed with a quantum circuit, then handed to a normal classifier. Used in Test 1.
- **QAOA (Quantum Approximate Optimization Algorithm).** A quantum method that hunts for good answers to yes/no puzzles, like a schedule. Used in Test 2.
- **QCBM (Quantum Circuit Born Machine).** A quantum circuit that learns to produce samples matching a target pattern. A quantum "generator." Used in Test 3.

A few names from the classical (normal-computer) side, for completeness:

- **SVM (Support Vector Machine)** and **RBF-SVM** — standard classifiers.
- **XGBoost** — a strong, popular model built from decision trees.
- **AUC** — a score from 0.5 (random guess) to 1.0 (perfect) for telling two groups apart.

---

## Why I built this

People talk about quantum computing and healthcare like the future is already here. Faster drug discovery. Smarter patient risk models. Better hospital and clinic planning. It is a great story.

So I tested it.

The research world is blunt about this. No one has shown a real quantum win on a useful problem. Quantum models are hard to train. And a lot of the "quantum beat classical" papers used weak classical opponents or ran on perfect simulators.

So I skipped the hype and did the boring, useful thing. I picked the three claims people repeat most. I gave the classical side a real, tuned fight. And I let the numbers talk.

Each test aims at a spot where quantum is *supposed* to shine. So a tie or a loss is a real result worth sharing.

---

## The three tests

### 1 · Small data, the "rare disease" case
`experiments/exp1_small_data_kernel/`

Rare diseases come with tiny datasets. People say quantum "similarity" helps here. I put a **Qiskit quantum kernel** and an SVM against a tuned **RBF-SVM**, **XGBoost**, and logistic regression. I grew the training set from 30 patients to 800.

![Test 1](results/plots/exp1_small_data.png)

> The quantum kernel hugged the coin-flip line at every size. This is a known trap called kernel concentration. An off-the-shelf quantum "similarity score" is not a better one.

### 2 · Scheduling nurses
`experiments/exp2_qaoa_optimization/`

Hospitals run on hard puzzles. Who works which shift. Where to place a clinic. I turned nurse scheduling into a math problem and solved it with **QAOA** (Quantum Approximate Optimization Algorithm), then with brute force and simulated annealing.

![Test 2](results/plots/exp2_qaoa.png)

> QAOA found good schedules. But the time to simulate it blew up as the puzzle grew. Classical annealing found the perfect answer on every run, at once.

### 3 · Fewer knobs to tune
`experiments/exp3_generative_efficiency/`

Here is the crown-jewel claim, backed by real papers. A quantum generator (a **QCBM**, Quantum Circuit Born Machine) learns a pattern with far fewer settings than a classical one. I rebuilt the test with a tuned classical baseline and swept the parameter count.

![Test 3](results/plots/exp3_generative.png)

> Both models learned the pattern. The classical one got there with 17 parameters. The quantum one needed 24. The famous "quantum uses way fewer knobs" line did not hold up once the classical side was allowed to be lean too.

### 4 · The fair fight
`experiments/exp4_fair_fight/`

The obvious objection: maybe quantum lost because I tuned classical and left quantum on defaults. So I tuned the quantum kernel too (grid-searched the feature map, depth, and entanglement) and built it a home-field problem, a parity dataset with the exact structure quantum feature maps are meant to exploit.

> Tuning helped a lot. Quantum on the healthcare data jumped from 0.52 to **0.71**. But tuned classical still hit **0.77**, and on quantum's own home-field problem it was **0.55 vs 0.66**. I tried to make quantum win and could not. Which sharpens the claim: quantum is not better at any size I can run, not that it is worse.

---

## The AI layer

`quantum_analyst/` feeds each test's raw numbers to Claude. The prompt tells it to stay skeptical and cite the data, not cheer for quantum. Out comes a plain verdict card: the claim, the finding, the call, the catch. That card writes the article for me. With no API key, a built-in rule reads the same numbers so the whole thing still runs offline.

---

## Run it yourself (free, offline, a few minutes)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # macOS xgboost: brew install libomp
python run_all.py                              # the three tests
python experiments/exp4_fair_fight/run.py      # the tuned rematch

# optional: real AI verdicts instead of the offline rule
export ANTHROPIC_API_KEY=sk-...
python quantum_analyst/analyst.py
```

Everything runs on a simulator on your own machine. No quantum hardware. No patient data. The cohort is fake. Fixed seeds mean you get the same numbers I did.

## What is in the box
```
data/               fake patient generator (no real records)
experiments/        exp1 kernel . exp2 QAOA . exp3 generative . exp4 fair fight
quantum_analyst/    the AI verdict writer (Claude)
results/            saved numbers + plots/ + verdict_cards.json
make_plots.py       charts   .   run_all.py   one-command run
```

## The fine print
- **Simulators only.** No noise, no errors. Real hardware is harder, not easier.
- **Small sizes.** Few qubits, small datasets. The story can change at scale.
- **A snapshot.** This is where near-term quantum stands right now. The field moves fast.
- The goal is a clear read, not a put-down. Knowing where quantum is *not* ready is how you spot the day it becomes ready. That day is coming.

## Sources

Full, verified reference list with per-claim notes: **[REFERENCES.md](REFERENCES.md)**.

The short version. Our own numbers come from `results/` and reproduce with
`python run_all.py`. The background reading:

- **Why our quantum kernel sat at chance (Test 1):** exponential concentration in
  quantum kernels, Thanasilp et al. 2022 ([arXiv:2208.11060](https://arxiv.org/abs/2208.11060)).
- **Why variational models are hard to train (Test 2):** barren plateaus, McClean
  et al. 2018 ([arXiv:1803.11173](https://arxiv.org/abs/1803.11173)).
- **The parameter-efficiency claim we tested (Test 3):** BO-QGAN, Smith and Guven
  2025 ([arXiv:2506.01177](https://arxiv.org/abs/2506.01177)) and QCA-MolGAN, Thomas
  et al. 2025 ([arXiv:2509.05051](https://arxiv.org/abs/2509.05051)). Both report a
  win on *molecule generation*, a harder task than ours; our small tabular test did
  not reproduce it.
- **State of the field:** Cordier et al. 2021 ([arXiv:2112.00760](https://arxiv.org/abs/2112.00760)),
  Cerezo et al. 2022 ([arXiv:2303.09491](https://arxiv.org/abs/2303.09491)), and two
  drug-discovery reviews ([arXiv:2408.13479](https://arxiv.org/abs/2408.13479),
  [arXiv:2409.15645](https://arxiv.org/abs/2409.15645)).

We claim no quantum advantage. Where a paper reports one and our test disagrees,
[REFERENCES.md](REFERENCES.md) explains why (usually a different, harder task).

*MIT-licensed. An independent project. Not tied to or backed by any company.*
