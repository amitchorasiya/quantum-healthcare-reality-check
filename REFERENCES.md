# References and Data Sources

Every claim in this project traces to one of two things: a number our own code
produced (see `results/`), or a peer-reviewed paper listed below. Each entry has
a verified title, authors, year, and link, plus a note on how it connects to our
tests. Where the literature and our result disagree, we say so.

## How to read our claims

- **"Our result" claims** come straight from `results/*.json`, reproducible with
  `python run_all.py`. Numbers like "quantum kernel AUC 0.52" or "hardware
  fidelity 0.91" are measured, not quoted.
- **"Background" claims** (for example, "no operational quantum advantage exists
  yet") come from the review papers below, not from us.
- We do **not** claim a quantum advantage anywhere. Our headline finding is that
  tuned classical methods matched or beat quantum on these small tests.

## The state of the field

1. **Biology and medicine in the landscape of quantum advantages** — Cordier,
   Sawaya, Guerreschi, McWeeney (2021; J. R. Soc. Interface 2022).
   <https://arxiv.org/abs/2112.00760>
   A framework for where quantum could help in biology and medicine. Grounds our
   point that a quantum advantage means reducing a real resource (time, space, or
   data), and that most such advantages are still open questions.

2. **Challenges and Opportunities in Quantum Machine Learning** — Cerezo, Verdon,
   Huang, Cincio, Coles (Nature Computational Science 2022; arXiv 2023).
   <https://arxiv.org/abs/2303.09491>
   States plainly that trainability of quantum ML models is an open problem.
   Grounds our "noise and trainability, not the math, are the wall" theme.

3. **Quantum Machine Learning in Drug Discovery: Applications in Academia and
   Pharmaceutical Industries** — Smaldone, Shee, Kyro, et al. (2024).
   <https://arxiv.org/abs/2409.15645>
   A broad review of quantum ML for drug discovery. We cite it for context on the
   field, not for any specific benchmark number.

4. **Quantum-machine-assisted Drug Discovery** — Zhou, Chen, Cheng, et al.
   (NPJ Drug Discovery 2026; arXiv Aug 2024). <https://arxiv.org/abs/2408.13479>
   Reviews quantum computing across the drug-development cycle (molecular
   simulation, drug-target interaction, trial optimization). Context for why
   people expect quantum to matter in healthcare.

## Why our quantum kernel sat at a coin flip (Test 1)

5. **Exponential concentration in quantum kernel methods** — Thanasilp, Wang,
   Cerezo, Holmes (2022, rev. 2024). <https://arxiv.org/abs/2208.11060>
   Shows quantum kernel values concentrate exponentially toward a fixed value as
   qubits grow, which leaves a model whose predictions barely depend on the input.
   This is the direct explanation for our Test 1 result: the off-the-shelf
   `ZZFeatureMap` kernel hugged the AUC 0.5 line while tuned classical reached 0.79.

## Why variational quantum models are hard to train (Test 2 context)

6. **Barren plateaus in quantum neural network training landscapes** — McClean,
   Boixo, Smelyanskiy, Babbush, Neven (Nature Communications 2018).
   <https://arxiv.org/abs/1803.11173>
   Gradients vanish exponentially with qubit count for random circuits. Background
   for why QAOA needed several restarts and still had no edge over classical
   annealing in Test 2.

## The "quantum generators need fewer parameters" claim (Test 3)

We tested this claim directly and it did **not** reproduce on our small task
(quantum needed 24 parameters, classical 17). The papers that report an advantage
do so on molecule generation, a different and harder task than ours. We cite them
so readers can see the claim we were testing, not as support for our own numbers.

7. **Bridging Quantum and Classical Computing in Drug Design: Architecture
   Principles for Improved Molecule Generation** — Smith, Guven (2025).
   <https://arxiv.org/abs/2506.01177>
   Reports a hybrid quantum GAN (BO-QGAN) with a 2.27x higher Drug Candidate Score
   than prior quantum-hybrid benchmarks, "while reducing parameter count by more
   than 60%." This is the strongest parameter-efficiency claim in the space. Note:
   it is on molecule generation, not our tabular task, and uses recreated
   baselines rather than a head-to-head classical model.

8. **QCA-MolGAN: Quantum Circuit Associative Molecular GAN with Multi-Agent
   Reinforcement Learning** — Thomas, Chen, Okadome Valencia, Jose, Wu (2025;
   IEEE Quantum AI). <https://arxiv.org/abs/2509.05051>
   Uses a Quantum Circuit Born Machine as a learnable prior, the same building
   block as our Test 3 generator. Context for the QCBM design, not a benchmark
   we rely on.

## Quantum kernels for drug screening (related to Test 1)

9. **Q2SAR: A Quantum Multiple Kernel Learning Approach for Drug Discovery** —
   Giraldo, Ruiz, Caruso, Mancilla, Bellomo (2025).
   <https://arxiv.org/abs/2506.14920>
   Reports a quantum-multiple-kernel SVM beating a classical gradient-boosting
   model on DYRK1A kinase-inhibitor AUC. We flag the caveat that this is a single
   dataset with a small test set, and our own kernel test on different data went
   the other way. Included so readers can compare, not as proof either direction.

## Tools we used

- **Qiskit** and **Qiskit Machine Learning** (IBM) — quantum circuits, the
  `ZZFeatureMap` kernel. <https://github.com/qiskit-community/qiskit-machine-learning>
- **Qiskit Algorithms** — QAOA. <https://github.com/qiskit-community/qiskit-algorithms>
- **PennyLane** (Xanadu) — the Quantum Circuit Born Machine.
  <https://pennylane.ai>
- **scikit-learn**, **XGBoost**, **dimod** — the tuned classical baselines.
- **IBM Quantum** — the real-hardware run on `ibm_fez` (see the hardware repo).
  <https://quantum.cloud.ibm.com>

## Data

The patient cohort is fully synthetic, generated by `data/make_cohort.py` with a
fixed seed. No real patient records are used, so there is no privacy risk. The
generator, the seed, and the resulting CSV are all in this repo, so anyone can
reproduce the exact dataset.

## What we deliberately do not claim

- We do not claim quantum beats classical on any task. It did not, in our tests.
- We do not claim these small results generalize to large problems. They may not.
- We do not claim the simulator reflects real hardware. It does not; the hardware
  repo measures exactly that gap.
- Results run on simulators and one small real-hardware pass. Both are stated
  wherever a number appears.
