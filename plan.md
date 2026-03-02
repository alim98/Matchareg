Excellent direction. This is the strongest **training-free** and still **paperable** path.

## Proposed paper idea (refined)

**Title concept (working):**
**Training-Free 3D Multimodal Registration via Foundation Features + Gromov-Wasserstein Correspondence + Diffeomorphic Fitting**

### Core idea

Replace DINO-Reg’s simple feature matching stage with a much stronger correspondence pipeline:

1. **Foundation features**

   * **Primary:** MATCHA descriptors (best correspondence prior)
   * **Fallback / simpler:** DINOv3 dense features

2. **Matcher**

   * **GWOT (Gromov-Wasserstein Optimal Transport)** with a spatial smoothness prior (instead of nearest-neighbor / raw cosine matching)

3. **Transformation**

   * **3D diffeomorphic fitting** (SVF + scaling-and-squaring style integration), still optimization-based and training-free

This keeps the method **training-free for registration**, while making the largest upgrade where DINO-Reg is weakest: **feature-to-correspondence quality**. DINO-Reg itself is already a training-free correspondence + diffeomorphic optimization framework, so this is a clean and defensible extension.

---

## Why this is the best option (paperability + likely improvement)

### 1) It upgrades the right bottleneck

DINO-Reg already shows that training-free registration can work well by using foundation features + correspondence + diffeomorphic optimization. Your improvement should target the **correspondence stage**, not just “swap encoder and hope.”

### 2) GWOT is a real algorithmic jump, not a cosmetic swap

GWOT explicitly adds **spatial smoothness** into the matching objective (via Gromov-Wasserstein structure), which is exactly what noisy dense correspondences need before fitting a smooth deformation field. The GWOT paper shows this improves correspondence consistency and can be competitive with heavier feature ensembles while being more efficient. ([arXiv][1])

### 3) MATCHA is a stronger correspondence descriptor than plain DINOv2

MATCHA is designed specifically for robust correspondences (geometric/semantic/temporal) by fusing diffusion features and DINOv2. It reports strong matching performance with a unified descriptor. ([CVF Open Access][2])

### 4) It remains training-free in the registration sense

No registration network is trained on medical pairs. You use **pretrained features** + **optimization**, which preserves the key DINO-Reg appeal. DINO-Reg also targets the same training-free setting.

---

## Important caveat (must be stated in paper)

**Inferred, not experimentally confirmedv3 are 2D natural-image foundation models. Adapting them to 3D CT/CBCT is a research hypothesis (likely promising), but not guaranteed.

**Could not verify with peer-reviewed sources** that the exact combination
**MATCHA/DINOv3 + GWOT + 3D diffeomorphic fitting** has already been published for medical image registration.
(That is good for novelty, but do a final systematic search before submission.)

---

# Full implementation blueprint (team handoff)

## A) Datasets (what to download exactly)

### Primary benchmark (recommended): **ThoraxCBCT (OncoReg / Learn2Reg)**

This is the best fit because:

* It is a **3D FBCT↔CBCT** registration task (same modality family, but hard in practice)
* It is the dataset used in the OncoReg setting and used as validation ground before hidden evaluation
* It has **public training/validation images + trunk masks + keypoints**, and hidden/private evaluation setup exists (good for a strong paper story) ([arXiv][3])

#### What is in ThoraxCBCT (from OncoReg paper)

* **19 cases total**
* **2 pairs per case** (1 FBCT + 2 CBCT per patient)
* Split:

  * **11 train**
  * **3 val**
  * **5 test**
* Preprocessed to:

  * **390 × 280 × 300**
  * **1.0 mm isotropic**
* Includes:

  * **manual pre-alignment**
  * **trunk masks**
  * **Fӧrstner keypoints**
  * labels/annotations for evaluation (availability depends on split/challenge mode) ([arXiv][3])

#### Download target

From the OncoReg / Learn2Reg challenge resources, download:

* ThoraxCBCT **training images**
* ThoraxCBCT **validation images**
* **trunk masks**
* **provided keypoints**
* challenge auxiliary files / evaluation scripts from OncoReg / L2R GitHub repos (they are explicitly referenced by organizers) ([arXiv][3])

---

### Secondary benchmark (optional but good): **Learn2Reg Abdomen MRI-CT**

DINO-Reg reports results on this dataset, so it is useful for direct comparison against the paper baseline. DINO-Reg describes it as 3D abdominal MRI-CT with paired training/val/test and many unlabeled scans.

A Learn2Reg dataset page also lists the abdomen MRI/CT collection (122 scans) in the challenge dataset catalog. ([learn2reg.grand-challenge.org][4])dary (not primary)

* More multimodal (MRI↔CT), so harder feature-domain transfer
* Good for generalization proof, but ThoraxCBCT is the cleaner first target

---

## B) What exactly to compare against

### Baseline to reproduce

* **DINO-Reg** (paper baseline)

  * foundation features (DINOv2-based correspondences)
  * training-free matching
  * diffeomorphic optimization (3D smooth transform)

### Your method variants (must run all ablations)

1. **DINOv2 + NN matching + diffeo fit** (DINO-Reg style)
2. **DINOv3 + NN matching + diffeo fit**
3. **MATCHA + NN matching + diffeo fit**
4. **DINOv3 + GWOT + diffeoGWOT + diffeo fit**  ← main method
5. **MATCHA + GWOT + non-diffeomorphic fit** (to show diffeo matters)
6. **MATCHA + OT (no GW term) + diffeo fit** (to isolate GW effect)

This ablation grid makes the paper strong:

* feature effect
* matcher effect
* transform model effect

---

# C) Method (precise, high-level but implementation-ready)

## Stage 0: Preprocessing (no learning)

### Inputs

* Moving volume (I_m) (FBCT)
* Fixed volume (I_f) (CBCT)
* Trunk masks (M_m, M_f)
* Optional provided keypoints (for initialization / validation)

### Preprocessing rules

* Use challenge-preprocessed NIfTI volumes **as-is** (already cropped/resampled/pre-aligned in ThoraxCBCT). Do not add another heavy resample unless necessary. ([arXiv][3])
* Intensity normalization (robust, modality-safe):

  * Clip per-volume to percentiles (e.g., 0.5–99.5)
  * Z-score inside trunk mask
* Keep mask for all later steps (feature extraction, matching, fitting)

---

## Stage 1: 3D foundation feature extraction (training-free)

This is the key engineering step.

### Option A (main): MATCHA features

MATCHA is a 2D correspondence feature model combining diffusion features + DINOv2 with a dynamic fusion module; it outputs a unified descriptor for correspondence tasks. ([CVF Open Access][2])

### Option B (fallback): DINOv3 features

DINOv3 is a strong foundation vision model with improved dense features and official model access / code paths. ([arXiv][5])

---

### 3D adaptation strategy (recommended)

Because MATCHA/DINOv3 are 2D, do **tri-planar feature extraction**:

For each voxel neighborhood:

* Extract slices in:

  * **axial**
  * **coronal**
  * **sagittal**
* Convert each slice to pseudo-RGB:

  * simplest: repeat channel 3x
  * better: stack adjacent slices ([s-1, s, s+1]) as 3 channels

Then:

* Run MATCHA or DINOv3 on each slice
* Get patch/grid features
* Upsample back to slice resolution
* Fuse tri-planar features per voxel:

  * concat + PCA (or concat + L2 norm)
  * final voxel descriptor (f(x)\in\mathbb{R}^d)

### Why tri-planar (practical)

* Avoids 3D model retraining
* Preserves some 3D context
* Easy to parallelize on GPU

**Inferred, not experimentally confirmed.**
Tri-planar fusion is a practical adaptation choice; exact best fusion (concat vs PCA vs learned-free weighting) must be ablated.

---

## Stage 2: Sparse 3D candidate point sampling (for scalable matching)

Do **not** run full dense voxel-to-voxel GWOT (too expensive).

### Sample points inside trunk mask

For each image:

* Uniform + feature-saliency sampling inside mask
* Example:

  * 8k–20k points per volume at coarse stage
  * denser later for refinement
* Optional:

  * include provided Fӧrstner keypoints explicitly (high-value anchors) on ThoraxCBCT ([arXiv][3])

### Descriptor at sampled points

* (f_i = f(x_i)) for moving points
* (g_j = g(y_j)) for fixed points

---

## Stage 3: GWOT-based 3D correspondence (main novelty)

This is the paper’s core.

GWOT replaces nearest-neighbor matching with an OT objective that combines:

* **feature similarity**
* **spatial consistency (GW term)**
* optional **mass relaxation** (unbalanced OT)
* optional **coarse spatial prior**

The GWOT paper formulates feature + spatial-structure matching and shows the spatial smoothness effect is key. ([arXiv][1])

---

### 3D GWOT objective (recommended)

Let:

* (X={x_i}_{i=1}^N) sampled points in moving
* (Y={y_j}_{j=1}^M) sampled points in fixed
* (P\in\mathbb{R}_+^{N\times M}) transport/matching matrix

#### 1) Feature cost

[
C^{feat}_{ij} = 1 - \cos(f_i, g_j)
]

#### 2) Spatial structure costs (within each image)

Build local distance graphs:

* (D^X_{ii'} = |x_i-x_{i'}|) if (|x_i-x_{i'}|<r), else 0
* (D^Y_{jj'} = |y_j-y_{j'}|) if (|y_j-y_{j'}|<r), else 0

This mirrors the GWOT local-smoothness idea (local neighborhood geometry should be preserved), but extended to 3D coordinates. GWOT’s 2D formulation explicitly uses a local radius-based spatial prior; this extension is natural and paperable. ([arXiv][1])

#### 3) Optional coarse spatial prior (important for medical)

Add:
[
C^{prior}_{ij} = \frac{|A x_i - y_j|^2}{\sigma^2}
]
where (A) is identity or coarse affine/prealignment transform (challenge data already has manual pre-alignment). This prevents absurd matches across anatomy. ThoraxCBCT is pre-aligned already, so this prior is especially effective. ([arXiv][3])

#### 4) Final objective

Use fused/unbalanced GWOT:
[
\min_P ;\langle C^{feat} + \lambda_p C^{prior}, P\rangle

* \lambda_{gw},\mathcal{L}_{GW}(D^X,D^Y,P)
* \lambda_{mass},\text{UBOT regularization}
* \varepsilon,\text{Entropy}(P)
  ]

- (\lambda_{gw}): spatial smoothness strength
- (\lambda_p): anatomical plausibility prior
- (\lambda_{mass}): allows unmatched mass (important for CBCT artifacts / partial visibility)
- (\varepsilon): entropic regularization for stable optimization

---

### Solving GWOT (practical)

GWOT uses an efficient optimization scheme (projected gradient + OT solves) and emphasizes sparse/local structure for efficiency. ([bmva-archive.org.uk][6])

#### Recommended implementation strategy

* Use **POT** (Python Optimal Transport) for OT primitives
* Implement GW gradient on sparse neighborhood graphs
* Multi-stage optimization:

  1. coarse points (8k)
  2. medium points (15k)
  3. fine points (25k)

#### Output

From (P):

* Extract correspondences by mutual-high-probability pairs
* Keep confidence score (w_{ij}=P_{ij})
* Filter:

  * mutual consistency
  * mask consistency
  * local cycle consistency (optional)

This gives a robust sparse set of weighted 3D correspondences.

---

## Stage 4: 3D diffeomorphic fitting (no training)

Fit a smooth invertible transform to the correspondences.

### Parameterization

Use a **stationary velocity field (SVF)** (v(x)) on a 3D grid or B-spline lattice, then exponentiate to get a diffeomorphism:
[
\phi = \exp(v)
]

This is the same family of diffeomorphic fitting principle used in modern registration pipelines and aligns with DINO-Reg’s optimization-based philosophy.

### Objective for fitting

[
\min_v \sum_k w_k |\phi(x_k)-y_k|^2

* \lambda_s |L v|^2
* \lambda_j ,\mathcal{P}_{jac}(\phi)
  ]

Where:

* ( (x_k,y_k,w_k) ) are GWOT matches
* (L): smoothness operator (bending/diffusion)
* (\mathcal{P}_{jac}): penalty for non-positive Jacobian / implausible deformation

### Optimization schedule (important)

Multi-resolution:

1. **Coarse SVF grid** (e.g., 8–12 mm control spacing)
2. **Medium**
3. **Fine** (2–4 mm)

At each level:

* fit SVF to correspondences
* integaring)
* warp moving points
* optionally re-run local GWOT refinement in warped space

This alternating scheme is strong and still training-free.

---

## Stage 5: Optional refinement (still training-free)

### Descriptor refinement (recommended before intensity)

Run a final local correspondence update after coarse warping:

* small search radius
* same GWOT, but local only
* refit SVF

### Intensity refinement (optional)

Because FBCT↔CBCT is difficult due to artifacts, intensity loss can help but also destabilize.
If added, use only as **final tiny refinement**:

* local NCC inside trunk mask
* strong regularization
* small step count

If this hurts robustness, remove it and keep the paper purely feature-correspondence driven.

---

# D) Exact engineering modules to assign to a team

## Module 1 — Data I/O

**Owner:** 1 engineer

Responsibilities:

* Read NIfTI images/masks
* Handle challenge splits
* Normalize intensities
* Cache cropped tensors
* Export warps / metrics inputs

Deliverables:

* `dataset_thoraxcbct.py`
* `dataset_abdomenmrct.py`
* cached preprocessed volumes

---

## Module 2 — Foundation feature extraction

**Owner:** 1–2 engineers

Responsibilities:

* Integrate MATCHA repo (primary) and DINOv3 (fallback)
* Tri-planar slicing and batching
* Pseudo-RGB conversion
* Patch feature upsampling to voxel grid
* Feature caching (critical)

Notes:

* MATCHA official repo exists (CVPR 2025 highlight implementation). ([GitHub][7])
* DINOv3 official repo provides model loading paths / weights access instructions. ([GitHub][8])

Deliverables:

* `features/matcha_2d.py`
* `features/dinov3_2d.py`
* `features/triplanar_fuser.py`
* disk cache of voxel descriptors

---

## Module 3 — GWOT matcher (main novelty)

**Owner:** strongest algorithm engineer

Responsibilities:

* Sparse point sampler in masks
* Feature cost matrix construction
* 3D local graph construction
* GWOT optimizer (fused + unbalanced + entropic)
* Confidence extraction and filtering

Deliverables:

* `matching/gwot3d.py`
* `matching/sampling.py`
* `matching/filters.py`

---

## Module 4 — Diffeomorphic fitter

**Owner:** registration/math engineer

Responsibilities:

* SVF parameterization
* scaling-and-squaring integrator
* warp points / warp volumes
* Jacobian determinant checks
* multiresolution schedule

Deliverables:

* `transform/svf.py`
* `transform/integrate.py`
* `transform/warp.py`

---

## Module 5 — Evaluation and ablations

**Owner:** 1 engineer

Responsibilities:

* Local metrics (TRE, Dice if labels available)
* Challenge packaging / export
* Ablation runner
* Runtime + memory profiling (important for paper)

Deliverables:

* `eval/metrics.py`
* `scripts/run_ablation.py`
* `scripts/export_challenge_submission.py`

---

# E) Suggested hyperparameter ranges (starting points)

These are sensible initial ranges for the first sweep.

## Feature extraction

* Patch stride:

  * MATCHA: use native output stride, then upsample
  * DINOv3: ViT-B/16 or ConvNeXt-Tiny first
* Tri-planar fusion:

  * concat + L2 norm (baseline)
  * concat + PCA(256) (faster GWOT)

## Sampling

* Coarse stage: 8k points
* Mid: 15k points
* Fine: 25k points
* Enforce mask coverage (stratified over z)

## GWOT

* Local radius (r): 8–15 mm
* (\lambda_{gw}): 0.1–2.0
* (\lambda_p): 0.05–1.0
* Entropy (\varepsilon): 0.01–0.1
* Unbalanced mass reg: 0.1–10

## Diffeomorphic fitting

* Smoothness (\lambda_s): strong at coarse, lower at fine
* Jacobian penalty (\lambda_j): medium (enough to avoid foldings)
* 3–5 outer alternations (GWOT ↔ fit)

---

# F) What makes it paperable (contribution framing)

Your paper should claim **three** contributions:

1. **Training-free 3D registration upgrade**

   * A stronger correspondence engine for DINO-Reg-style methods using foundation descriptors + GWOT

2. **3D GWOT matching for medical registration**

   * Extends spatially-regularized OT matching from 2D correspondence to sparse 3D anatomical matching

3. **Foundation-feature adaptation for volumetric registration**

   * Tri-planar MATCHA/DINOv3 descriptors for 3D CT/CBCT without registration training

That is a solid, non-trivial contribution set.

---

# G) Critical risks (and how to de-risk)

## Risk 1: MATCHA domain shift (natural RGB → CT/CBCT)

**Inferred, not experimentally confirmed.**
MATCHA may not transfer as well as hoped.

### Mitigation

* Keep DINOv3 fallback path ready
* Add simple feature normalization + mask-aware sampling
* Benchmark both early (week 1)

---

## Risk 2: GWOT scaling in 3D

Full dense 3D OT is too expensive.

### Mitigation

* Sparse sampled points only
* Local graph GW term (radius-based)
* Multi-stage coarse-to-fine matching

---

## Risk 3: Bad correspondences can break diffeomorphic fitting

### Mitigation

* Confidence-weighted fitting
* Mutual filtering
* Robust loss (Huber) on correspondence residuals
* Strong coarse smoothness prior

---

# H) Minimal first experiment (fastest path to proof)

Before full pipeline, run this:

### Pilot (2–3 days)

* Dataset: ThoraxCBCT train/val only
* Features: DINOv3 tri-planar
* Matching:

  * NN baseline
  * GWOT (sparse 3D)
* Transform:

  * simple smooth displacement fit (not yet full diffeo)
* Metric:

  * TRE on provided keypoints

If GWOT improves TRE over NN with same features, you already validated the main hypothesis.

Then add:

* full SVF diffeo fitting
* MATCHA features
* challenge packaging

---

# I) Final recommendation (what to build first)

## Main path (recommended)

**DINOv3 features + 3D GWOT + SVF diffeomorphic fitting** first

Why first?

* Easier integration than MATCHA
* Stronger dense features than older DINO family
* Lower engineering risk
* Fastest path to a reliable prototype ([arXiv][5])

## Paper-best path (after prototype works)

Swap feature backend to:
**MATCHA features + 3D GWOT + SVF diffeomorphic fitting**

This is the highest-upside version for the final paper if transfer works. MATCHA is explicitly built for correspondence and is likely to improve match quality. ([CVF Open Access][2])

---

If you want, next I can write the **exact method section (paper-style English)** with equations and notation (ready for Overleaf), plus a **repo skeleton** and **experiment table template**.

[1]: https://arxiv.org/html/2602.03105v1 "https://arxiv.org/html/2602.03105v1"
[2]: https://openaccess.thecvf.com/content/CVPR2025/papers/Xue_MATCHA_Towards_Matching_Anything_CVPR_2025_paper.pdf "https://openaccess.thecvf.com/content/CVPR2025/papers/Xue_MATCHA_Towards_Matching_Anything_CVPR_2025_paper.pdf"
[3]: https://arxiv.org/pdf/2503.23179 "https://arxiv.org/pdf/2503.23179"
[4]: https://learn2reg.grand-challenge.org/Learn2Reg2021/?utm_source=chatgpt.com "Learn2reg 2021 - Grand Challenge"
[5]: https://arxiv.org/abs/2508.10104 "https://arxiv.org/abs/2508.10104"
[6]: https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_721/paper.pdf "https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_721/paper.pdf"
[7]: https://github.com/feixue94/matcha "https://github.com/feixue94/matcha"
[8]: https://github.com/facebookresearch/dinov3 "https://github.com/facebookresearch/dinov3"
