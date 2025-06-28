# Preventing Regression in Machine Learning: Forgetting-Penalized and Pareto-Aware Training

## Background

A well-known problem in machine learning is **regression**: as models update, they sometimes "forget" how to correctly handle examples they previously got right. This is especially frustrating in production or user-facing systems, where a model suddenly failing on known-good cases can be more disruptive than missing new ones.

Catastrophic forgetting is well-studied in **continual learning** ([French, 1999](https://pubmed.ncbi.nlm.nih.gov/10322466/)), and rehearsal/buffer methods are common. But for standard supervised learning, less attention has been paid to *actively penalizing regression* during ordinary training.

---

## Approaches Compared

We compare three strategies:

**1. Standard Training (Baseline)**  
The usual approach—minimize training loss with no explicit mechanism to prevent forgetting.

**2. Forgetting-Penalized Training**  
Inspired by continual learning (e.g., [Kirkpatrick et al., 2017, EWC](https://www.pnas.org/doi/10.1073/pnas.1611835114)), we add a penalty term whenever an example previously classified correctly becomes incorrect. This discourages the model from "unlearning" prior knowledge, but does not completely prevent all changes.

**3. Soft Pareto-Penalized Training**  
Building on the idea of Pareto improvements and recent work in multi-objective optimization (e.g., [Lin et al., 2019](https://arxiv.org/abs/1912.12854) and [Navon et al., 2020](https://arxiv.org/abs/2010.04104)), we penalize *any* increase in per-example loss, not just flips from correct to incorrect. This enforces a softer but broader form of "do no harm" across all examples, not just those at the margin.

---

## Experiment

On the Adult income dataset, we ran all three methods with identical architectures, tuning penalties and including a warmup phase so that penalties only activate after the model has stabilized.

---

## Results

| Method            | Total Forgetting | Final Train Acc | Final Val Acc |
|-------------------|------------------|-----------------|---------------|
| Baseline          | 5668             | 0.794           | 0.788         |
| Forgetting Pen.   | 122              | 0.759           | 0.760         |
| Soft Pareto       | 290              | 0.786           | 0.783         |

- Both penalized approaches **dramatically reduced forgetting**—by an order of magnitude or more—compared to baseline.
- **Soft Pareto** achieved a strong balance: low forgetting with almost no loss in accuracy.
- **Forgetting-penalized** (hard) kept forgetting even lower, but at a greater cost to overall accuracy.
- Standard training had the highest accuracy, but at the expense of frequent forgetting/regression.

---

## Contribution

While regularization and continual learning are established topics, our work demonstrates that **lightweight, penalty-based methods** for "locking in" learned cases during standard training can sharply reduce regression *without major tradeoffs in accuracy*. The soft Pareto loss is a practical, easily-implemented variant that achieves a good balance between progress and stability.

---

## When Is This Useful?

- **Production/mission-critical ML**: Where regression on known-good cases is unacceptable.
- **Human-facing models**: Where users notice and care when predictions flip on previously solved cases.
- **Medical, fraud, or compliance applications**: Where “do no harm” is a central requirement.
- **Curriculum learning and staged training**: Where it is important to consolidate learning on early/easy cases as new data is introduced.

---

## Summary

If you care about avoiding regression on learned examples, simple penalty terms (either for forgetting or for loss increases) can be effective, easy to add to existing training loops, and provide a practical "Pareto-improvement bias" in ordinary supervised learning.
