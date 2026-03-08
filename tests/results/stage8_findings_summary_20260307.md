# Stage 8 Findings Summary

Latest successful suite:
- `pipeline/tests/results/stage8_suite/logs/stage8_suite_20260307_191642.log`

Earlier failed suite:
- `pipeline/tests/results/stage8_suite/logs/stage8_suite_20260307_191600.log`

## Critical Issues

1. The first unattended suite run failed because the process could not see a HIP/CUDA GPU while `DEVICE=cuda` was requested.
- Evidence: every test in `stage8_suite_20260307_191600.log` failed with `RuntimeError: No HIP GPUs are available`.
- Interpretation: this was an environment/runtime availability issue, not a code logic issue in the tests themselves.
- Action taken: `pipeline/tests/run_stage8_suite.sh` now performs a GPU preflight check and automatically switches to `DEVICE=cpu` if no GPU is visible, instead of crashing the whole suite immediately.

2. `GWOT` is consistently harmful on realistic matching problems.
- Evidence:
  - `dinov3`: real distractor `NN=78.5%`, `GWOT=32.5%`
  - `matcha`: real distractor `NN=84.2%`, `GWOT=65.5%`
  - `mind`: real distractor `NN=75.0%`, `GWOT=18.0%`
- Interpretation: the main sparse-pipeline failure is not generic matching failure; it is specifically that the current `gwot` path degrades realistic correspondence quality compared with plain `nn`.
- Consequence: `gwot` should not be the default matcher for this pipeline in its current form.

3. `MIND` is not suitable as a sparse correspondence descriptor on real CT/CBCT, even though the dense MIND baseline is correct.
- Evidence:
  - `test_9`: `mind` real feature retrieval `nn@1=1.0%`, separation `0.024`
  - `test_10`: `mind` real GT-aligned `NN=21.4%`, real distractor `GWOT=18.0%`
  - `test_11`: our `mind_convex_adam` matches the original implementation exactly on both real and synthetic cases
- Interpretation: there is no implementation bug in the dense MIND baseline; the failure is specifically the sparse descriptor usage on real data.

4. The fitter is not the primary bug, but it is highly sensitive to match corruption and poor spatial coverage.
- Evidence from `test_12`:
  - perfect all GT: `TRE=0.000`
  - perfect 2k subset: `TRE=4.401`
  - noisy sigma 2: `TRE=6.083`
  - noisy sigma 5: `TRE=9.572`
  - noisy sigma 10: `TRE=15.557`
  - outlier 10%: `TRE=6.959`
  - outlier 30%: `TRE=13.571`
  - outlier 50%: `TRE=21.230`
  - clustered local region only: `TRE=10.594`
- Interpretation: once correspondence noise/outlier rate gets high enough, or the support becomes too local, the fitter overfits bad supervision and degrades TRE.
- Consequence: the sparse stage must aggressively reject bad matches and enforce broad spatial coverage before fitting.

## Module-by-Module Conclusions

### Test 8: Coordinate and Sampling Audit

Result: passed

What it proves:
- `voxel_to_feature_coords()` is numerically consistent
- descriptor sampling/interpolation is correct
- there is no evidence of a coordinate conversion or trilinear sampling bug

Key numbers:
- synthetic interpolation error: mean `0.000000`, max `0.000001`
- identity retrieval:
  - `dinov3`: `100.0%`
  - `matcha`: `100.0%`
  - `mind`: `91.7%`

Conclusion:
- coordinate mapping and descriptor sampling are not the bottleneck.

### Test 9: Backend Feature Audit

Result: passed

What it proves:
- backend feature quality differs sharply by model
- the real-data problem is not uniform across all backends

Key numbers:
- `dinov3`
  - real: `nn@1=49.6%`, separation `0.210`
  - synthetic: `nn@1=86.2%`, separation `0.306`
  - interpretation: usable feature stage
- `matcha`
  - real: `nn@1=74.0%`, separation `0.317`
  - synthetic: `nn@1=99.4%`, separation `0.487`
  - interpretation: strongest feature backend tested
- `mind`
  - real: `nn@1=1.0%`, separation `0.024`
  - synthetic: `nn@1=85.2%`, separation `0.222`
  - interpretation: strong domain-transfer collapse on real data

Conclusion:
- feature extraction is not globally broken.
- `matcha` is the best sparse feature backend.
- `mind` should not be used as the sparse feature backend for real CT/CBCT.

### Test 10: Backend Matching Audit

Result: passed

What it proves:
- the realistic sparse bottleneck is the matcher, specifically `gwot`

Key numbers:
- `dinov3`
  - real GT-aligned: `NN=66.2%`, `OT=98.3%`, `GWOT=98.3%`
  - real distractor: `NN=78.5%`, `GWOT=32.5%`
- `matcha`
  - real GT-aligned: `NN=80.7%`, `OT=96.6%`, `GWOT=95.6%`
  - real distractor: `NN=84.2%`, `GWOT=65.5%`
- `mind`
  - real GT-aligned: `NN=21.4%`, `OT=31.1%`, `GWOT=29.9%`
  - real distractor: `NN=75.0%`, `GWOT=18.0%`

Interpretation:
- `OT/GWOT` can look excellent on GT-indexed or synthetic setups while still underperforming badly on realistic distractor pools.
- `NN` is the safest sparse matcher right now.
- `matcha + nn` is the strongest sparse combination observed so far.

Conclusion:
- the sparse pipeline should prefer `nn`, not `gwot`, on real runs.

### Test 11: MIND Baseline Audit

Result: passed

What it proves:
- `pipeline/transform/mind_convex_adam.py` matches the original DINO-Reg implementation exactly

Key numbers:
- real case:
  - ours `TRE=9.772`
  - original `TRE=9.772`
  - field diff max `0.000000`
  - field diff mean `0.000000`
- synthetic case:
  - ours `TRE=1.294`
  - original `TRE=1.294`
  - field diff max `0.000000`
  - field diff mean `0.000000`

Conclusion:
- there is no remaining parity bug in the MIND baseline implementation.
- if baseline performance is modest on this real pair, that is dataset/pair behavior, not a porting error.

### Test 12: Fitter Stress Audit

Result: passed

What it proves:
- the fitter works well with correct and moderately noisy correspondences
- it fails when fed heavy noise, many outliers, or poor spatial coverage

Operational thresholds suggested by observed behavior:
- good:
  - perfect all GT
  - perfect 2k subset
  - noisy sigma 2
  - outlier 10%
- borderline:
  - noisy sigma 5
- bad:
  - noisy sigma 10
  - outlier 30%
  - outlier 50%
  - clustered local region only

Conclusion:
- the fitter is not the root bug.
- the sparse pipeline must control correspondence quality and coverage before fitting.

## Final Diagnosis

The root cause is now clear:

1. There is no coordinate/sampling bug.
2. There is no MIND baseline implementation bug.
3. The fitter is correct but sensitive to corrupted supervision.
4. The current `gwot` path is the main harmful component for realistic sparse matching.
5. The best sparse backend tested is `matcha`, and the best sparse matcher tested is `nn`.
6. `mind` should be reserved for dense baseline registration, not sparse point correspondence on real data.

## Best Current Pipeline Recommendation

If the goal is the best current sparse pipeline:
- feature backend: `matcha`
- matcher: `nn`
- keep the sparse trust gate / fallback behavior enabled
- do not default to `gwot`

If the goal is the most reliable pipeline regardless of sparse novelty:
- use dense MIND ConvexAdam baseline as the safe fallback
- only allow sparse correction if the match set passes strong trust checks

## Practical Next Actions

1. Make `matcha + nn` the main sparse path for real runs.
2. Keep `gwot` behind an explicit experimental flag, not as default.
3. Tighten sparse acceptance criteria using the fitter stress results:
- reject high-noise / low-coverage / clustered-only match sets
- reject overly local support
- reject match sets whose effective coverage is too small even if PCK on a small subset looks good
4. Preserve MIND baseline fallback exactly as-is; its implementation is already verified correct.
