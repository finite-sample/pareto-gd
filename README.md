# (Don't) Forget About It: Forgetting-Penalized Supervised Learning

## TL;DR

Constraint-based methods (projected GD, BCWI) vastly outperform penalty-based methods for preventing model regression. Mean NFR ~0.008 vs ~0.03 across 5 datasets.

---

## Results

| Method | Mean NFR |
|--------|----------|
| projected_gd | 0.008 |
| bcwi | 0.009 |
| fixed_anchor | 0.029 |
| selective_distill | 0.029 |
| confidence_drop | 0.032 |
| baseline | 0.038 |

### Key Findings

1. **Constraint-based methods vastly outperform penalty-based methods**
   - projected_gd and bcwi achieve NFR < 0.01 while maintaining accuracy
   - Penalty methods plateau at NFR ~0.015-0.03

2. **Training-time constraints beat post-hoc interpolation** (3/5 datasets)
   - projected_gd explores full feasible region via iterative projection
   - bcwi is restricted to 1D line segment between incumbent and candidate

3. **All methods beat baseline** - baseline always ranks last (rank 6)

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
python3 scripts/run_constrained.py --datasets diabetes --n-splits 1
```

### Run Full Benchmark

```bash
python3 scripts/run_constrained.py --n-splits 10
```

This runs all 6 methods across 5 datasets (adult, bank, credit, diabetes, spambase) with 10 random splits each.

### Analyze Existing Results

```bash
python3 scripts/run_constrained.py --analyze-only --input tabs/results.csv
```

Outputs:
- `tabs/results.csv` - Raw results (method, dataset, split, metrics)
- `tabs/summary.csv` - Aggregated statistics per method/dataset
- `tabs/rankings.csv` - Method rankings per dataset
- `figs/all_datasets_pareto.pdf` - Combined Pareto frontiers

---

## Where It Matters

- **Production-grade systems** where regression on known-good cases is unacceptable.
- **Human-facing models** where consistency matters to user trust.
- **High-stakes domains** like medical, fraud detection, or compliance.
- **Model update pipelines** where backwards compatibility is required.
