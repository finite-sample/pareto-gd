# (Don't) Forget About It: Forgetting-Penalized Supervised Learning

## TL;DR

Projected gradient descent achieves the best accuracy-NFR trade-off across 32 OpenML-CC18 datasets and 2 model types (MLP, Logistic Regression). Using the combined score (Accuracy − 2×NFR), projected_gd scores 0.749 vs 0.723 for BCWI and 0.526 for baseline.

---

## Results

### Combined Score (Accuracy − λ×NFR)

| Method | λ=1 | λ=2 | λ=5 |
|--------|-----|-----|-----|
| projected_gd | 0.770 | **0.749** | 0.686 |
| bcwi | 0.749 | 0.723 | 0.644 |
| selective_distill | 0.694 | 0.599 | 0.313 |
| fixed_anchor | 0.687 | 0.590 | 0.299 |
| confidence_drop | 0.671 | 0.556 | 0.213 |
| baseline | 0.655 | 0.526 | 0.138 |

### Pareto Win Rate (64 dataset-model pairs)

| Method | Wins | Rate |
|--------|------|------|
| projected_gd | 47/64 | 73% |
| bcwi | 29/64 | 45% |
| fixed_anchor | 29/64 | 45% |
| selective_distill | 28/64 | 44% |
| confidence_drop | 21/64 | 33% |
| baseline | 20/64 | 31% |

### Key Findings

1. **Constraint-based methods vastly outperform penalty-based methods**
   - projected_gd and bcwi achieve NFR ~0.02 while maintaining accuracy
   - Penalty methods plateau at NFR ~0.10-0.13

2. **Projected GD dominates the Pareto frontier** (73% win rate)
   - Explores full feasible region via iterative projection
   - BCWI is restricted to 1D line segment between incumbent and candidate

3. **LogReg validates theory** (Theorem 4: NFR monotonicity)
   - NFR monotonicity holds exactly for linear models, enabling reliable binary search
   - LogReg baseline NFR = 0.210 vs projected_gd NFR = 0.024

---

## Background

A well-known problem in machine learning is **model regression**: as models update, they sometimes "forget" how to correctly handle examples they previously got right. This is especially frustrating in production or user-facing systems, where a model suddenly failing on known-good cases can be more disruptive than missing new ones.

Catastrophic forgetting is well-studied in **continual learning** (French, 1999), and rehearsal/buffer methods are common. But for standard supervised learning, less attention has been paid to *actively preventing regression* during ordinary training.

We compare **six methods** spanning penalty-based training, constrained optimization, and post-hoc interpolation.

---

## Approaches Compared

### Penalty-Based Methods

**1. Baseline (Standard ERM)**
Standard empirical risk minimization with no explicit mechanism to prevent forgetting.

**2. Confidence Drop Penalty**
Penalizes any per-example loss increase vs the previous epoch. Implements a "do no harm" principle but uses epoch-local comparisons, limiting its effectiveness.

**3. Fixed Anchor Penalty**
Uses incumbent model's loss as anchor on a held-out set. Penalizes when candidate loss exceeds incumbent loss: `max(0, ℓ_candidate(x) - ℓ_incumbent(x))`.

**4. Selective Distillation**
Distills candidate model to match incumbent's predictions (soft targets) on an anchor set where the incumbent was correct.

### Constraint-Based Methods

**5. Projected Gradient Descent**
Train with ERM, then project back to the NFR ≤ ε feasible region after each epoch. Projection via binary search for interpolation weight with incumbent.

**6. Backwards Compatible Weight Interpolation (BCWI)**
Post-hoc approach: train candidate freely via ERM, then find interpolation weight α such that `θ = α·θ_incumbent + (1-α)·θ_candidate` achieves target NFR.

---

## Repository Structure

```
pareto-gd/
├── README.md
├── ms/                     # Manuscript
│   ├── forget.tex
│   ├── forget.bib
│   └── forget.pdf
├── scripts/                # Python scripts and notebooks
│   ├── run_constrained.py          # Main benchmark runner
│   ├── datasets.py                 # Dataset loading (OpenML)
│   ├── models.py                   # MLP and training utilities
│   ├── training.py                 # All 6 training methods
│   ├── metrics.py                  # NFR, PFR, accuracy metrics
│   ├── analyze_results.py          # Results analysis
│   ├── forget-smooth.ipynb
│   ├── pareto_gd_basic.ipynb
│   ├── pareto_gd_penalized.ipynb
│   └── penalized-sgd-soft-pareto.ipynb
├── tabs/                   # Output tables (CSV)
└── figs/                   # Output figures (PDF, PNG)
```

---

## Quick Start

### Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib torch scipy openml
```

### Run a Quick Test

```bash
python3 scripts/run_constrained.py --datasets diabetes --n-splits 1 --model mlp
```

### Run Full CC18 Benchmark

```bash
python3 scripts/run_constrained.py --cc18 --n-splits 5 --model mlp
python3 scripts/run_constrained.py --cc18 --n-splits 5 --model logreg
```

This runs all 6 methods across 32 OpenML-CC18 datasets with 5 random splits each.

### Analyze Existing Results

```bash
python3 scripts/run_constrained.py --analyze-only --input tabs/cc18_results/results.csv
```

Outputs:
- `tabs/cc18_results/results.csv` - Raw results (method, dataset, split, model_type, metrics)
- `tabs/cc18_results/summary.csv` - Aggregated statistics per method/dataset
- `tabs/cc18_results/rankings.csv` - Method rankings per dataset
- `figs/cc18_results/all_datasets_pareto.pdf` - Combined Pareto frontiers

---

## Where It Matters

- **Production-grade systems** where regression on known-good cases is unacceptable.
- **Human-facing models** where consistency matters to user trust.
- **High-stakes domains** like medical, fraud detection, or compliance.
- **Model update pipelines** where backwards compatibility is required.
