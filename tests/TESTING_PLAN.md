# Module Isolation Testing Plan

## Goal

Identify **exactly which module** is responsible for the ~1% TRE improvement (instead of the expected 50–65%). Each test is **self-contained** — it does NOT depend on any upstream module's output. Tests are ordered from "needs nothing" to "needs everything".

---

## Test Execution Order

```
Test 1 (Evaluation)      — needs: keypoints only
Test 2 (SVF Fitter)      — needs: keypoints + torch
Test 3 (Features)        — needs: volumes + GPU (DINOv3)
Test 4 (Matching)        — needs: features from Test 3 + keypoints
Test 5 (End-to-End)      — needs: everything (synthetic warp, controlled)
```

---

## Test 1: Evaluation Sanity (`test_A_evaluation.py`)

**Module under test:** `eval/metrics.py`, `transform/warp.py`, `transform/mind_convex_adam.py`  
**Depends on:** Dataset loading only (no features, no matching)

### What it checks
1. TRE computation is correct (coordinate conventions, axis order)
2. MIND-SSC ConvexAdam baseline produces **large** TRE reduction
3. Identity displacement gives exactly the initial TRE

### Pass criteria
| Check | Pass | Fail |
|-------|------|------|
| Identity displacement TRE = initial TRE | exact match | any difference |
| MIND baseline TRE reduction | > 40% on most pairs | < 10% |
| Per-axis displacement correlation | > 0.5 on all axes | near 0 |

### Visualization
- Bar chart: initial vs final TRE per pair
- Scatter plot: ideal vs actual displacement per axis

---

## Test 2: SVF Fitter on Ground-Truth Correspondences (`test_B_fitter_gt.py`)

**Module under test:** `transform/fitter.py`, `transform/svf.py`, `transform/integrate.py`  
**Depends on:** Keypoints only (no features, no matching)

### What it checks
1. When given **perfect correspondences** (GT keypoints), can the fitter produce a displacement that drastically reduces TRE?
2. This completely isolates the fitter from feature quality and matching quality.

### How it works
- Load GT keypoint pairs `(fixed_kp, moving_kp)` from train set
- Feed them directly as correspondences to `DiffeomorphicFitter.fit()`
- Evaluate TRE using the resulting displacement

### Pass criteria
| Check | Pass | Fail |
|-------|------|------|
| TRE with GT correspondences | < 5mm | > 10mm |
| Displacement not near zero | max > 5 voxels | max < 1 |
| Per-axis correlation | > 0.8 | < 0.3 |
| Jacobian folding | < 1% | > 5% |

### Visualization
- Convergence plot (loss vs iteration per level)
- Before/after displacement overlay on mid-axial slice
- Per-axis ideal-vs-actual scatter

---

## Test 3: Feature Quality / Discriminability (`test_C_features.py`)

**Module under test:** `features/triplanar_fuser.py`, `features/*_extractor.py`  
**Depends on:** GPU, DINOv3/MATCHA weights (no matching, no fitting)

### What it checks
1. **GT similarity separation**: cosine similarity between features at known-corresponding points vs random pairs
2. Feature PCA visualization: do features look anatomically meaningful?

### How it works
- Load one train pair (with GT keypoints)
- Extract tri-planar features for both volumes
- Sample features at GT keypoint locations
- Compute:
  - `sim_positive[k] = cos(feat_fixed[fkp[k]], feat_moving[mkp[k]])`
  - `sim_negative[k] = cos(feat_fixed[fkp[k]], feat_moving[random])`
- Compare distributions

### Pass criteria
| Check | Pass | Fail |
|-------|------|------|
| Mean positive similarity | > 0.4 | < 0.2 |
| Mean negative similarity | < 0.2 | > 0.3 |
| Separation (pos - neg) | > 0.2 | < 0.05 |
| PCA visualization | anatomical structure visible | noise |

### Visualization
- Histogram: positive vs negative similarity distributions
- 3 mid-slice PCA visualizations (RGB from top-3 PCs)
- Similarity heatmap at a few keypoints

---

## Test 4: Matching Quality (`test_D_matching.py`)

**Module under test:** `matching/gwot3d.py`, `matching/sampling.py`, `matching/filters.py`  
**Depends on:** Features from Test 3 + GT keypoints

### What it checks
1. Given features at GT keypoint locations, how well do NN / OT / GWOT recover the known correspondences?
2. This tells you whether matching is helping or hurting beyond raw feature quality.

### How it works
- Use GT keypoints as the point sets (fixed_kp, moving_kp)
- Extract features at both sets of keypoints
- Run NN matching, OT matching, GWOT matching
- Evaluate: what % of GT pairs are matched within 5mm / 10mm / 20mm (PCK metric)

### Pass criteria
| Check | Pass | Fail |
|-------|------|------|
| NN PCK@10mm | > 30% | < 10% |
| GWOT PCK@10mm | > NN | worse than NN |
| # of matches after filtering | > 50% of input | < 10% |

### Visualization
- PCK curve (% correct vs threshold in mm) for NN, OT, GWOT
- Match error histogram
- 3D scatter plot of correct vs incorrect matches

---

## Test 5: End-to-End with Synthetic Warp (`test_E_synthetic.py`)

**Module under test:** Full pipeline  
**Depends on:** Everything

### What it checks
1. Take one volume, apply a **known** smooth deformation → create a synthetic "moving" image
2. Run the full pipeline to recover the deformation
3. If this fails → code bug. If this passes but real data fails → domain shift.

### How it works
- Load one FBCT volume
- Create a synthetic SVF, integrate to displacement
- Warp the volume to create "synthetic moving"
- Create synthetic keypoints by warping real keypoints
- Run the full sparse pipeline on (fixed=original, moving=warped)
- Compare recovered displacement to the known ground truth

### Pass criteria
| Check | Pass | Fail |
|-------|------|------|
| Synthetic TRE recovery | > 70% reduction | < 30% |
| Displacement direction correlation | > 0.7 per axis | < 0.3 |

### Visualization
- Side-by-side: original, synthetic moving, recovered
- Displacement field comparison (GT vs predicted)

---

## How to Read Results

After running all tests, use this decision tree:

```
Test 1 (Eval) fails?
  → Fix coordinate conventions / evaluation code first
  → Nothing downstream is reliable

Test 1 passes, Test 2 (Fitter) fails?
  → Fitter is broken (optimization, scaling, regularization)
  → Features & matching are irrelevant until fitter works

Test 2 passes, Test 3 (Features) fails?
  → Features don't transfer to CT/CBCT
  → Matching can't save bad features
  → Consider: MIND features, higher resolution, or dense alignment

Test 3 passes, Test 4 (Matching) fails?
  → Matching implementation/hyperparameters are the bottleneck
  → Debug GWOT: check transport matrix, confidence filtering

Test 4 passes, Test 5 (Synthetic) fails?
  → Integration bug in the full pipeline
  → Check coordinate flows, displacement composition

All pass, but real data still fails?
  → Domain shift (FBCT↔CBCT gap too large for current features)
  → Try: modality-specific features, intensity refinement, or denser sampling
```

---

## Running the Tests

```bash
# Run all diagnostic tests on a single GPU node:
salloc -A mhf_apu -p apu --gres=gpu:1 --time=04:00:00

# Test 1 — no GPU needed for evaluation, GPU needed for MIND
python -m pipeline.tests.test_A_evaluation --pair 0

# Test 2 — CPU-only (fitter uses torch but CPU is fine for 1 pair)
python -m pipeline.tests.test_B_fitter_gt --pair 0

# Test 3 — needs GPU for feature extraction
python -m pipeline.tests.test_C_features --pair 0

# Test 4 — needs GPU for feature extraction, CPU for matching
python -m pipeline.tests.test_D_matching --pair 0

# Test 5 — needs GPU for full pipeline
python -m pipeline.tests.test_E_synthetic --pair 0

# Or run all at once:
for t in A B C D E; do
    echo "===== TEST $t ====="
    python -m pipeline.tests.test_${t}_* --pair 0
done
```


----
# Start with Test A (fastest, no features needed):
python -m pipeline.tests.test_A_evaluation --pair 0

# Then Test B (no GPU needed for CPU mode):
python -m pipeline.tests.test_B_fitter_gt --pair 0 --device cuda

# Then Test C (needs GPU for DINOv3):
python -m pipeline.tests.test_C_features --pair 0

# Then Test D (reuses cached features):
python -m pipeline.tests.test_D_matching --pair 0

# Finally Test E (synthetic, needs GPU):
python -m pipeline.tests.test_E_synthetic --pair 0

