# Design Document

*Written for someone brand new to quantum computing. No physics needed.*

## 1. What are we building?

A small experiment in code. We take three popular claims about quantum computing in healthcare. We test each one against tuned regular-computer methods on the same problem. Then an AI writes up the verdict.

Think product review, not sales brochure.

## 2. Quantum computing in sixty seconds

- A normal computer stores **bits**. Each one is a 0 or a 1.
- A quantum computer uses **qubits**. A qubit can be a blend of 0 and 1 at once. Qubits can also link up, so one depends on another.
- In theory, this lets a quantum computer explore many options at once. That *might* make some problems faster.
- In practice, right now: real quantum computers are small, noisy, and error-prone. No one has shown one beating a good classical computer on a useful task. This is the most important fact in the whole project.

We do not use a real quantum computer. We use a **simulator**. That is a normal program that copies the math of a small quantum computer on a laptop. It is standard practice and it is free.

## 3. Why these three tests?

Each one aims at a spot people call quantum's home turf. If quantum helps healthcare soon, it shows up here first. So this is the best shot it gets.

| # | The claim | The quantum tool | The classical opponent |
|---|---|---|---|
| 1 | "With very little data, like a rare disease, quantum similarity helps." | Quantum kernel + SVM | Tuned SVM, XGBoost, logistic regression |
| 2 | "Quantum solves scheduling puzzles as well as classical solvers." | QAOA | Simulated annealing, brute force |
| 3 | "Quantum generators learn with far fewer knobs to tune." | Quantum Circuit Born Machine | Small classical generator |

## 4. The rules I set

1. **No weak opponents.** The classical side is tuned. It would be easy, and cheap, to hobble it so quantum wins. I did the reverse.
2. **The numbers talk.** A tie or a loss for quantum is a real result, not a failure.
3. **Anyone can rerun it.** Fixed seeds, saved results, one command rebuilds everything.
4. **Free and local.** Simulators only. No cloud account. No quantum hardware. No real patient data.
5. **The AI stays skeptical.** The prompt tells it to cite the numbers, not cheer for quantum.

## 5. The data, and why it is safe

We make a **fake** patient group. Made-up people with realistic lab values and a "diagnosis" that follows a hidden rule plus some noise. Because it is fake, there is no privacy risk. And we can set the difficulty and the size on purpose, which the small-data test needs.

## 6. What "success" means

Not "quantum wins." Success is a clear answer to each claim, explained so a newcomer gets *why*. The likely result, "classical still wins today, and here is where and why," is more useful than hype. That is the whole point.

## 7. Word list

| Term | Plain meaning |
|---|---|
| Qubit | A quantum bit. Can be a blend of 0 and 1. |
| Simulator | Software pretending to be a quantum computer, on your laptop. |
| Quantum kernel | A way to measure how alike two data points are, using a quantum circuit, then handed to a normal classifier. |
| QAOA | A quantum method that hunts for good answers to yes/no puzzles. |
| QCBM | A quantum circuit that learns to make samples matching a target pattern. |
| AUC | A score from 0.5 (random guess) to 1.0 (perfect) for how well a model separates two groups. |
| KL divergence | How far a learned pattern sits from the target. Lower is better. |
