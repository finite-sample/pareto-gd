# (Don't) Forget About It: Forgetting-Penalized Supervised Learning

## Summary

**Problem**: When ML models are retrained, they sometimes start misclassifying examples they previously got right. This "model regression" is costly in production systems where consistency matters.

**Solutions**: We compare six methods for preventing regression:
- *Penalty-based*: add loss terms that discourage forgetting (confidence drop, fixed anchor, selective distillation)
- *Constraint-based*: enforce hard limits on regression rate during or after training (projected gradient descent, BCWI)
- *Baseline*: standard training with no regression prevention

**Finding**: Constraint-based methods achieve larger NFR reductions than penalty-based methods. Projected GD appears on the Pareto frontier in 96% of datasets (78/81), achieving +80% NFR improvement over baseline with the best average rank (2.49). Benchmark: 81 OpenML-CC18 datasets × 2 models (MLP, logreg) × 10 splits = 54,120 training runs.

---

## Results (81 datasets × 2 models)

### Benchmark Summary

| Method | Frontier | Free Wins | ΔNFR | ΔAcc | Avg Rank | #1 Wins | Hypervolume |
|--------|----------|-----------|------|------|----------|---------|-------------|
| Projected GD | 78/81 | 20 | +80% | -2.2% | 2.49 | 42 | 0.7835 |
| BCWI | 46/81 | 5 | +73% | -2.9% | 3.83 | 10 | 0.7681 |
| Fixed Anchor | 45/81 | 18 | +30% | -0.8% | 2.81 | 16 | 0.7550 |
| Selective Distill | 33/81 | 37 | +24% | -0.1% | 3.12 | 11 | 0.7527 |
| Confidence Drop | 18/81 | 37 | +8% | -0.1% | 3.96 | 1 | 0.7398 |
| Baseline (ERM) | 10/81 | 0 | +0% | +0.0% | 4.78 | 1 | 0.7360 |

### Pareto Frontier Membership (81 datasets)

| Method | Appearances | Rate |
|--------|-------------|------|
| Projected GD | 78/81 | 96% |
| BCWI | 46/81 | 57% |
| Fixed Anchor | 45/81 | 56% |
| Selective Distill | 33/81 | 41% |
| Confidence Drop | 18/81 | 22% |
| Baseline (ERM) | 10/81 | 12% |

**Key takeaways:**
- **Projected GD** dominates: 96% Pareto frontier, 42 #1 wins, best avg rank (2.49)
- **Constraint-based methods** achieve +73-80% NFR improvement over baseline
- **Penalty methods** achieve more "free wins" (37 each for Selective Distill and Confidence Drop)
- **Baseline** appears on frontier in only 12% of datasets, confirming regression is worth preventing

---

## Methods

### Penalty-Based

**1. Baseline (Standard ERM)**
Standard empirical risk minimization with no explicit mechanism to prevent forgetting.

**2. Confidence Drop Penalty**
Penalizes any per-example loss increase vs the previous epoch. Implements a "do no harm" principle but uses epoch-local comparisons, limiting its effectiveness.

**3. Fixed Anchor Penalty**
Uses incumbent model's loss as anchor on a held-out set. Penalizes when candidate loss exceeds incumbent loss: `max(0, ℓ_candidate(x) - ℓ_incumbent(x))`.

**4. Selective Distillation**
Distills candidate model to match incumbent's predictions (soft targets) on an anchor set where the incumbent was correct.

### Constraint-Based

**5. Projected Gradient Descent**
Train with ERM, then project back to the NFR ≤ ε feasible region after each epoch. Projection via binary search for interpolation weight with incumbent.

**6. Backwards Compatible Weight Interpolation (BCWI)**
Post-hoc approach: train candidate freely via ERM, then find interpolation weight α such that `θ = α·θ_incumbent + (1-α)·θ_candidate` achieves target NFR.

---

## Why Regression Matters

Model regression has real dollar costs. Consider a customer service operation:

**Setup**:
- 100,000 requests processed annually
- ML model triages requests, with human review for errors
- Cost per error: $50 (specialist time + correction overhead)
- Incumbent model accuracy: 85%

**The Problem**: You train a new model with 87% accuracy—but it misclassifies 5% of examples the incumbent got right (NFR = 5%).

| Model | Accuracy | NFR | Errors | Regression Cost | Total Annual Cost |
|-------|----------|-----|--------|-----------------|-------------------|
| Incumbent | 85.0% | 0% | 15,000 | $0 | **$750,000** |
| Candidate (raw) | 87.0% | 5% | 13,000 + 4,250 | $212,500 | **$862,500** |
| Candidate + Projected GD | 86.5% | 0.5% | 13,500 + 425 | $21,250 | **$696,250** |

**Key insight**: The raw candidate is *worse* despite higher accuracy. The 4,250 regressions (5% of 85,000 incumbent-correct examples) cost $212,500—exceeding the $100,000 saved from 2,000 fewer baseline errors.

**With Projected GD**: Trading 0.5% accuracy for 0.5% NFR saves $166,250 annually. The constraint-based approach finds the optimal trade-off.

### Break-Even Analysis

At what NFR does accuracy improvement become worthless?

```
Break-even NFR = (accuracy_gain × total_examples) / incumbent_correct
              = (0.02 × 100,000) / 85,000
              = 2.35%
```

Any NFR above 2.35% means the candidate costs more than it saves—regardless of its higher accuracy.

---

## Quick Start

### Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib torch scipy openml
```

### Run a Quick Test

```bash
python3 scripts/run_experiments.py --datasets diabetes --n-splits 1 --model mlp
```

### Run Full CC18 Benchmark

```bash
python3 scripts/run_experiments.py --cc18-all --n-splits 10 --model mlp --outdir tabs/
```

This runs all 6 methods across 81 OpenML-CC18 datasets with 10 random splits each.

### Analyze Results

```bash
python3 scripts/analyze.py tabs/results.csv
```

Outputs:
- `tabs/results.csv` - Raw results (method, dataset, split, model_type, metrics)
- `tabs/summary.csv` - Aggregated statistics per method/dataset
- `tabs/summary.tex` - LaTeX summary table
- `tabs/pareto_frontier.tex` - LaTeX Pareto frontier table
- `tabs/rankings.tex` - LaTeX rankings table
- `figs/all_datasets_pareto.pdf` - Combined Pareto frontiers
- `figs/figure1_representative.pdf` - Representative 2x2 figure

---

## Repository Structure

```
pareto-gd/
├── README.md
├── ms/                     # Manuscript
│   ├── forget.tex
│   ├── forget.bib
│   └── forget.pdf
├── scripts/                # Python scripts
│   ├── run_experiments.py          # Main benchmark runner
│   ├── analyze.py                  # Results analysis and table generation
│   ├── datasets.py                 # Dataset loading (OpenML)
│   ├── models.py                   # MLP and training utilities
│   ├── training.py                 # Binary classification methods
│   ├── training_multiclass.py      # Multiclass methods
│   └── metrics.py                  # NFR, PFR, accuracy metrics
├── tabs/                   # Output tables (CSV, LaTeX)
└── figs/                   # Output figures (PDF)
```
