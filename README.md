# Preventing Regression in Machine Learning: Forgetting-Penalized and Pareto-Aware Training

## Background

A well-known problem in machine learning is **regression**: as models update, they sometimes "forget" how to correctly handle examples they previously got right. This is especially frustrating in production or user-facing systems, where a model suddenly failing on known-good cases can be more disruptive than missing new ones.

Catastrophic forgetting is well-studied in **continual learning** (French, 1999), and rehearsal/buffer methods are common. But for standard supervised learning, less attention has been paid to *actively penalizing regression* during ordinary training.

---

## Approaches Compared

We compare three strategies:

**1. Standard Training (Baseline)**  
The usual approach—minimize training loss with no explicit mechanism to prevent forgetting.

**2. Forgetting-Penalized Training**  
Inspired by continual learning methods like Elastic Weight Consolidation (Kirkpatrick et al., 2017), this adds a penalty whenever an example previously classified correctly becomes incorrect. It discourages "unlearning," but does not eliminate all changes.

**3. Soft Pareto-Penalized Training**  
Drawing on Pareto-improvement ideas and recent multi-task optimization research (Lin et al., 2019; Navon et al., 2021), this method penalizes *any* increase in per-example loss—not just flips from correct to incorrect. It enforces a softer, broader "do no harm" principle across all training examples.

---

## Experiment

On the Adult income dataset, we trained all three methods with identical neural network architectures. Penalties were introduced after a warmup period, allowing the model to stabilize before beginning to penalize regressions.

---

## Results

| Method           | Total Forgetting | Final Train Acc | Final Val Acc |
|------------------|------------------|-----------------|---------------|
| **Baseline**     | 5668             | 0.794           | 0.788         |
| **Forgetting Pen.** | 122          | 0.759           | 0.760         |
| **Soft Pareto**  | 290              | 0.786           | 0.783         |

- Both penalized methods reduced forgetting by an order of magnitude **compared to baseline**.
- **Soft Pareto** provided a strong trade-off: low forgetting with minimal accuracy loss.
- **Forgetting-Penalized** achieved the lowest forgetting, but at a more significant cost to accuracy.
- **Baseline training** delivered the highest accuracy—but experienced frequent regression.

---

## Contribution

While regularization and continual learning are well-established, our work shows that **simple, lightweight penalty-based mechanisms**—added to ordinary training—can greatly reduce regression *without substantial accuracy loss*. The **Soft Pareto loss** is especially practical, implementing a “do no harm” bias that’s easy to integrate.

---

## Where It Matters

- **Production-grade systems** where regression on known-good cases is unacceptable.
- **Human-facing models** where consistency matters to user trust.
- **High-stakes domains** like medical, fraud detection, or compliance.
- **Curriculum or staged learning setups**, where early learning shouldn't be overwritten by later stages.

---

## Summary

If maintaining correctness on previously learned examples matters—even under normal supervised training—then adding **penalty terms** for forgetting or loss regression is effective, easy to implement, and provides a natural “Pareto bias” in practice.
