# Soft NFR: Convex Relaxation for Post-Hoc Model Selection

## Problem Setup

Given:
- **Incumbent model** θ_inc (deployed model we want to avoid regressing)
- **Candidate model** θ_cand (newly trained model with potentially better accuracy)
- **Checkpoints** θ_1, ..., θ_k (intermediate models saved during training)

We want to find a model that maximizes accuracy while limiting negative flips on examples the incumbent classified correctly.

## Hard NFR vs Soft NFR

### Hard NFR (BCWI's approach)
Count discrete prediction flips from correct → wrong:

```
NFR(θ) = |{x : inc_correct(x) ∧ ¬θ_correct(x)}| / |{x : inc_correct(x)}|
```

Properties:
- Discrete, non-differentiable
- Requires grid search over interpolation parameter α
- Cannot use gradient-based optimization

### Soft NFR (This work)
Continuous relaxation using loss differences:

```
soft_NFR(θ) = mean(max(0, ℓ_θ(x) - ℓ_inc(x))) for x where incumbent is correct
```

where ℓ(x) is the binary cross-entropy loss.

Properties:
- Continuous and differentiable
- Enables gradient-based optimization
- For logistic regression / MLPs: forms a convex program over weight simplex

## Optimization Problem

### 1D Case (fair BCWI comparison)
Given incumbent and single candidate, optimize over line segment:

```
θ(α) = α·θ_inc + (1-α)·θ_cand, α ∈ [0,1]

max  Accuracy(θ(α))
s.t. soft_NFR(θ(α)) ≤ ε
     α ∈ [0,1]
```

### kD Case (checkpoint simplex)
Given k checkpoints {θ_1, ..., θ_k}, optimize over simplex:

```
θ(w) = Σᵢ wᵢ·θᵢ, where w ∈ Δᵏ (probability simplex)

max  Accuracy(θ(w))
s.t. soft_NFR(θ(w)) ≤ ε
     Σᵢ wᵢ = 1, wᵢ ≥ 0
```

## Convexity Analysis

For fixed data (x, y) and fixed incumbent loss ℓ_inc(x):

1. **Linear interpolation**: θ(w) = Σᵢ wᵢ·θᵢ is linear in w
2. **Logits**: For MLPs, logits are linear in final layer weights (and approximately linear for small weight changes)
3. **BCE loss**: Convex in logits
4. **max(0, ·)**: Preserves convexity
5. **mean**: Preserves convexity

Therefore, soft_NFR(θ(w)) is approximately convex in w for small deviations from incumbent.

## Connection to Fixed Anchor Penalty

The soft NFR loss is identical to the "Fixed Anchor Penalty" used during training:

```python
# Fixed Anchor (training): penalize loss increases on anchor set
penalty = mean(max(0, model_loss - incumbent_loss))

# Soft NFR (post-hoc): same formula, but used for model selection
soft_nfr = mean(max(0, model_loss - incumbent_loss))
```

The key difference:
- **Fixed Anchor**: Used as regularizer during SGD training
- **Soft NFR**: Used as constraint for post-hoc optimization over pre-trained checkpoints

## Methods Compared

1. **BCWI** (baseline): Hard NFR, grid search, 1D (incumbent ↔ candidate line)
2. **Soft-NFR-1D**: Soft NFR, grid search, 1D — fair comparison to BCWI
3. **Soft-NFR-kD**: Soft NFR, gradient descent, k-dim simplex with checkpoints

## Expected Benefits

1. **Soft NFR correlates with hard NFR** but is smoother, enabling better optimization
2. **Gradient-based search** may find better solutions than grid search
3. **Checkpoint simplex** expands the search space beyond 1D line

## Evaluation

Sweep over α (or simplex weights) and record:
- Accuracy
- Hard NFR (discrete flip count)
- Soft NFR (loss-based relaxation)

Plot Pareto frontiers: Accuracy vs Hard NFR for all methods.

Metrics:
- Area under Pareto frontier
- Win rate at specific NFR thresholds (0%, 0.5%, 1%, 2%, 3%)
- Correlation between soft NFR and hard NFR

## Results

### Benchmark Setup

- **CC18 Benchmark**: 72 OpenML datasets (binary + multi-class converted to binary), 10 splits each
- **Large Dataset Validation**: 5 additional datasets (Higgs, Click_prediction, ACSIncome, covertype, airlines), subsampled to 100k, 5 splits each
- **Evaluation**: Select operating point on validation set (NFR ≤ 5%), report test accuracy

### Key Finding: KD Performance Depends on Dataset Size

| Dataset Size | # Datasets | BCWI Acc | KD Acc | KD Win Rate |
|-------------|------------|----------|--------|-------------|
| ≤1k         | 18         | 85.48%   | 85.03% | 20.6%       |
| 1k-5k       | 30         | 92.84%   | 92.65% | 19.3%       |
| 5k-10k      | 8          | 90.80%   | 90.93% | 22.5%       |
| 10k-50k     | 10         | 91.90%   | 91.96% | 41.0%       |
| >50k (CC18) | 6          | 88.15%   | 88.19% | 53.3%       |
| **100k (validation)** | **5** | **72.47%** | **74.22%** | **92.0%** |

### Interpretation

**Why KD underperforms on small datasets:**
- The 1D incumbent→candidate line is a strong regularizer (1 degree of freedom)
- KD's 6-checkpoint simplex has more capacity but overfits validation on small data
- With limited validation samples, the extra flexibility finds spuriously good points

**Why KD excels on large datasets:**
- More validation data → less overfitting from extra flexibility
- The optimal trade-off often lies outside the 1D line
- Checkpoints explore different regions of weight space that endpoint interpolation misses

### Soft NFR as a Proxy

**Correlation with Hard NFR: Pearson r = 0.70**

This validates soft NFR as a useful differentiable proxy for the discrete hard NFR metric.

### Practical Recommendations

| Dataset Size | Recommendation |
|-------------|----------------|
| n < 5k      | Use BCWI (simpler, more robust) |
| 5k < n < 10k | Either method works |
| n > 10k     | Consider KD, or run both and pick via nested CV |

### Detailed Results (100k Validation Datasets)

| Dataset | BCWI | KD | KD Wins |
|---------|------|-----|---------|
| Higgs | 69.34% | **71.76%** | 5/5 |
| Click_prediction | 83.15% | **83.31%** | 5/5 |
| ACSIncome | 69.30% | **71.37%** | 5/5 |
| covertype | 80.01% | **81.74%** | 4/5 |
| airlines | 60.57% | **62.94%** | 4/5 |

**Average improvement from KD on large data: +1.75% accuracy, -4.8% NFR**

## Running the Benchmark

```bash
# Full CC18 benchmark (72 datasets, 10 splits)
python3 explorations/soft_nfr/benchmark.py --cc18 --n-splits 10

# Large dataset validation (subsampled to 100k)
python3 explorations/soft_nfr/benchmark.py \
  --dataset-ids 45570,1218,46801,44121,1169 \
  --max-samples 100000 \
  --n-splits 5 \
  --outdir explorations/soft_nfr/results_large

# Analyze results
python3 explorations/soft_nfr/analyze.py
```
