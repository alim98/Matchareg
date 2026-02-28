#!/usr/bin/env python3
"""
End-to-end registration pipeline runner.

Implements the full pipeline from plan.md:
  Stage 0: Preprocessing
  Stage 1: Tri-planar foundation feature extraction (DINOv3 / MATCHA)
  Stage 2: Sparse point sampling with Förstner keypoint inclusion
  Stage 3: Multi-stage GWOT / NN matching (coarse→medium→fine)
  Stage 4: Multi-resolution diffeomorphic fitting with GWOT↔fit alternation
  Stage 5: Optional intensity refinement (local NCC)
  Stage 6: Evaluation (TRE, Jacobian stats)

Usage:
    python -m pipeline.scripts.run_pipeline --pair 0 --feature dinov3 --matcher gwot
    python -m pipeline.scripts.run_pipeline --split val --feature dinov3 --matcher nn
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import PipelineConfig
from pipeline.data.dataset_thoraxcbct import ThoraxCBCTDataset
from pipeline.data.preprocessing import (
    robust_intensity_normalize,
    generate_trunk_mask,
)
from pipeline.features.triplanar_fuser import TriplanarFuser, save_features, load_features
from pipeline.matching.sampling import (
    sample_points_in_mask,
    sample_descriptors_at_points,
    include_keypoints,
)
from pipeline.matching.gwot3d import match
from pipeline.matching.filters import filter_matches
from pipeline.transform.fitter import DiffeomorphicFitter
from pipeline.transform.warp import warp_points
from pipeline.eval.metrics import compute_tre, compute_jacobian_stats, print_results
from pipeline.viz.visualizer import PipelineVisualizer

import functools

log_file = PROJECT_ROOT / "pipeline_run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode='w')
    ]
)
logger = logging.getLogger(__name__)

def log_and_validate(func):
    """Decorator to comprehensively log function execution, validate arrays, and track time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"==> Starting [ {func.__name__} ]")
        t_start = time.time()
        
        # Log input shapes if they are arrays/tensors
        for i, a in enumerate(args):
            if hasattr(a, 'shape'):
                logger.info(f"    arg{i}.shape: {a.shape}, min: {a.min() if hasattr(a, 'min') else 'N/A'}, max: {a.max() if hasattr(a, 'max') else 'N/A'}")
                if hasattr(a, 'isnan') and hasattr(a, 'any') and a.isnan().any():
                     logger.warning(f"    WARNING: arg{i} contains NaNs!")
                elif hasattr(np, 'isnan') and isinstance(a, np.ndarray) and np.isnan(a).any():
                     logger.warning(f"    WARNING: arg{i} contains NaNs!")
                     
        for k, v in kwargs.items():
            if hasattr(v, 'shape'):
                logger.info(f"    kwarg {k}.shape: {v.shape}")
                
        # Run function
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"CRASH in [ {func.__name__} ]: {e}")
            raise
            
        t_end = time.time()
        logger.info(f"<== Finished [ {func.__name__} ] in {t_end - t_start:.2f}s")
        
        # Log output features
        if hasattr(result, 'shape'):
            logger.info(f"    Output shape: {result.shape}")
        elif isinstance(result, tuple):
            logger.info(f"    Output tuple shapes: {[r.shape if hasattr(r, 'shape') else type(r) for r in result]}")
            
        return result
    return wrapper

logger.info(f"Logs are being saved to {log_file}")

@log_and_validate
def create_feature_extractor(config: PipelineConfig):
    """Create the appropriate feature extractor based on config."""
    if config.features.backend == "dinov3":
        from pipeline.features.dinov3_extractor import DINOv3Extractor
        return DINOv3Extractor(
            repo_path=str(config.paths.dinov3_repo),
            weights_path=str(config.paths.dinov3_weights),
            device=config.device,
        )
    elif config.features.backend == "matcha":
        from pipeline.features.matcha_extractor import MATCHAExtractor
        return MATCHAExtractor(
            repo_path=str(config.paths.matcha_repo),
            weights_path=str(config.paths.matcha_weights),
            device=config.device,
        )
    else:
        raise ValueError(f"Unknown backend: {config.features.backend}")


@log_and_validate
def voxel_to_feature_coords(points: np.ndarray, original_shape: tuple,
                             feature_shape: tuple, downsample: int) -> np.ndarray:
    """
    Convert voxel coordinates to feature grid coordinates.

    WHY THIS IS CORRECT EVEN WHEN SLICES ARE RESIZED:
    TriplanarFuser may resize slices (e.g. 140px → 448px) before patch
    embedding. The output patch resolution is new_h // patch_size (e.g. 28).
    fH is read from axial_pf.shape[2] — the ACTUAL output, not a formula.

    So fH/dH = (new_h/patch_size) / dH = scale/patch_size, and:
        feat_y = y_down * fH/dH = y_down * scale / patch_size   ← correct

    The resize factor is implicitly absorbed into fH. No explicit scale
    factor tracking is needed.

    For coronal and sagittal planes: axes are rearranged in fuse_triplanar
    so that each axis of the unified (fD, fH, fW) grid corresponds to
    (D, H, W) of the original volume before trilinear resampling. The same
    (fD/dD, fH/dH, fW/dW) ratios therefore apply uniformly to all planes.

    Args:
        points: (N, 3) in voxel coords (z, y, x)
        original_shape: (D, H, W) of original volume
        feature_shape: (fD, fH, fW) of feature grid
        downsample: volume downsample factor

    Returns:
        feat_points: (N, 3) in feature grid coords
    """
    D, H, W = original_shape
    fD, fH, fW = feature_shape

    feat_points = points.copy()
    # First account for downsampling
    feat_points[:, 0] = feat_points[:, 0] / downsample
    feat_points[:, 1] = feat_points[:, 1] / downsample
    feat_points[:, 2] = feat_points[:, 2] / downsample

    # Then scale to feature grid.
    # fH/dH captures resize + patch pooling in one ratio (see docstring).
    dD = D // downsample
    dH = H // downsample
    dW = W // downsample

    feat_points[:, 0] = feat_points[:, 0] * fD / dD
    feat_points[:, 1] = feat_points[:, 1] * fH / dH
    feat_points[:, 2] = feat_points[:, 2] * fW / dW

    # Clip to valid range
    feat_points[:, 0] = np.clip(feat_points[:, 0], 0, fD - 1)
    feat_points[:, 1] = np.clip(feat_points[:, 1], 0, fH - 1)
    feat_points[:, 2] = np.clip(feat_points[:, 2], 0, fW - 1)

    return feat_points



@log_and_validate
def get_features(
    config: PipelineConfig,
    volume: np.ndarray,
    case_id: str,
    fuser: TriplanarFuser,
    mask: np.ndarray = None,
):
    """Extract or load cached tri-planar features for a volume.

    Args:
        mask: pre-computed trunk mask (uint8, same shape as volume).
              If None, generates one via thresholding (fallback only).
    """
    cache_path = config.paths.feature_cache_dir / f"{case_id}_{config.features.backend}.npz"

    if config.features.use_cache and cache_path.exists():
        logger.info(f"Loading cached features: {cache_path}")
        return load_features(cache_path)

    # Use provided mask or fall back to thresholding
    if mask is None:
        logger.info("  No dataset mask available — generating via thresholding (fallback)")
        mask = generate_trunk_mask(volume)

    vol_norm = robust_intensity_normalize(volume, mask=mask)

    # Extract tri-planar features (returns tuple)
    result = fuser.fuse_triplanar(vol_norm)

    # Cache
    if config.features.use_cache:
        save_features(result, cache_path)

    return result


@log_and_validate
def sample_and_describe(
    feats: np.ndarray,
    feat_shape: tuple,
    orig_shape: tuple,
    mask: np.ndarray,
    n_points: int,
    downsample: int,
    keypoints: np.ndarray = None,
    rng: np.random.RandomState = None,
    z_stratified: bool = True,
    n_keypoint_anchors: int = 0,
) -> tuple:
    """
    Sample points inside mask and extract descriptors.

    Args:
        keypoints: Foerstner keypoints to optionally inject as anchors.
        n_keypoint_anchors: max keypoints to add (0 = disabled).
            ThoraxCBCT has ~30k keypoints per volume; injecting all of them
            crashes OT/GWOT and biases NN. Cap at a small number (e.g. 500)
            or keep disabled (default).

    Returns:
        (voxel_pts, descriptors_normalized)
    """
    voxel_pts = sample_points_in_mask(mask, n_points, z_stratified=z_stratified, rng=rng)

    # Inject a capped subsample of Foerstner keypoints as anchor points
    if keypoints is not None and n_keypoint_anchors > 0:
        if rng is None:
            rng = np.random.RandomState(42)
        if len(keypoints) > n_keypoint_anchors:
            idx = rng.choice(len(keypoints), n_keypoint_anchors, replace=False)
            keypoints_sub = keypoints[idx]
        else:
            keypoints_sub = keypoints
        voxel_pts = include_keypoints(voxel_pts, keypoints_sub, mask=mask)
        logger.info(f"    Added {len(keypoints_sub)} keypoint anchors → {len(voxel_pts)} total points")

    # Convert to feature grid coords
    feat_pts = voxel_to_feature_coords(voxel_pts, orig_shape, feat_shape, downsample)

    # Sample descriptors
    desc = sample_descriptors_at_points(feats, feat_pts)

    # L2 normalize
    norms = np.linalg.norm(desc, axis=1, keepdims=True) + 1e-8
    desc = desc / norms

    return voxel_pts, desc


@log_and_validate
def run_matching_stage(
    config: PipelineConfig,
    fixed_feats: np.ndarray,
    moving_feats: np.ndarray,
    fixed_feat_shape: tuple,
    moving_feat_shape: tuple,
    fixed_orig_shape: tuple,
    moving_orig_shape: tuple,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    n_points: int,
    downsample: int,
    displacement: torch.Tensor = None,
    fixed_keypoints: np.ndarray = None,
    moving_keypoints: np.ndarray = None,
    stage_name: str = "coarse",
) -> dict:
    """
    Run one matching stage: sample points, extract descriptors, match, filter.

    If displacement is provided, warps moving points before matching
    (GWOT↔fit alternation).

    Returns:
        filtered match result dict or None if too few matches
    """
    logger.info(f"  [{stage_name}] Sampling {n_points} points...")
    rng = np.random.RandomState(42)

    # Keypoint injection: only if explicitly enabled and capped.
    # Default is disabled (n_keypoint_anchors=0) because:
    # 1) ThoraxCBCT has ~30k keypoints per volume — injecting all crashes OT/GWOT
    # 2) Keypoints won't exist at test time — using them in matching is cheating
    kp_cap = config.sampling.n_keypoint_anchors if config.sampling.include_keypoints else 0

    # Sample and describe fixed points
    fixed_voxel_pts, fixed_desc = sample_and_describe(
        fixed_feats, fixed_feat_shape, fixed_orig_shape,
        fixed_mask, n_points, downsample,
        keypoints=fixed_keypoints, rng=rng,
        z_stratified=config.sampling.z_stratified,
        n_keypoint_anchors=kp_cap,
    )

    # Sample and describe moving points
    rng_m = np.random.RandomState(123)
    moving_voxel_pts, moving_desc = sample_and_describe(
        moving_feats, moving_feat_shape, moving_orig_shape,
        moving_mask, n_points, downsample,
        keypoints=moving_keypoints, rng=rng_m,
        z_stratified=config.sampling.z_stratified,
        n_keypoint_anchors=kp_cap,
    )

    # If we have a current displacement, warp moving points to align
    # with fixed space for better matching (GWOT↔fit alternation)
    if displacement is not None:
        warped_moving_pts = warp_points(moving_voxel_pts, displacement)
        logger.info(f"  [{stage_name}] Warped moving points with current displacement")
    else:
        warped_moving_pts = moving_voxel_pts

    # Diagnostic: cosine similarity
    n_sample = min(500, len(fixed_desc))
    cos_sample = fixed_desc[:n_sample] @ moving_desc[:n_sample].T
    logger.info(f"  [{stage_name}] Cosine sim (sample): "
                f"mean={cos_sample.mean():.3f}, std={cos_sample.std():.3f}, "
                f"max={cos_sample.max():.3f}")

    # Match — use WARPED moving points for spatial priors (closer to fixed)
    logger.info(f"  [{stage_name}] Matching ({config.matcher.method})...")
    match_kwargs = {}
    if config.matcher.method == "gwot":
        match_kwargs = {
            "lambda_gw":    config.gwot.lambda_gw,
            "lambda_prior": config.gwot.lambda_prior,
            "epsilon":      config.gwot.epsilon,
            "lambda_mass":  config.gwot.lambda_mass,
            "local_radius": config.gwot.local_radius,
            "max_iter":     config.gwot.max_iter,
        }
    elif config.matcher.method == "ot":
        match_kwargs = {
            "lambda_prior": config.gwot.lambda_prior,
            "epsilon":      config.gwot.epsilon,
            "lambda_mass":  config.gwot.lambda_mass,
            "max_iter":     config.gwot.max_iter,
        }


    match_result = match(
        moving_desc, fixed_desc, warped_moving_pts, fixed_voxel_pts,
        method=config.matcher.method, **match_kwargs
    )

    # Filter matches. Pass warped_moving_pts as points_src_geom so the
    # displacement filter measures the RESIDUAL after warping (not the
    # original full displacement). The returned matched_points_src always
    # contains original (unwarped) moving coords — correct for the SVF fitter.
    filtered = filter_matches(
        match_result, moving_voxel_pts, fixed_voxel_pts,
        confidence_threshold=config.gwot.confidence_threshold,
        mask_src=moving_mask,
        mask_tgt=fixed_mask,
        max_displacement=150.0,  # reject physically implausible matches
        points_src_geom=warped_moving_pts,  # geometry used during matching
    )
    logger.info(f"  [{stage_name}] Matches: {filtered['n_matches']} / {filtered['n_before_filter']}")

    if filtered["n_matches"] < 10:
        logger.warning(f"  [{stage_name}] Too few matches ({filtered['n_matches']})")
        return None

    return filtered


@log_and_validate
def run_pair(
    config: PipelineConfig,
    dataset: ThoraxCBCTDataset,
    pair_idx: int,
    method: str = "mind",
    fuser: "TriplanarFuser" = None,
    downsample: int = 2,
    visualize: bool = False,
) -> dict:
    """
    Run registration on a single pair.

    Methods:
        'mind': MIND-SSC ConvexAdam baseline (full resolution, no feature extraction)
        'sparse': Foundation features + sparse matching + SVF fitting (plan.md pipeline)
    """
    t0 = time.time()

    # Load data
    logger.info(f"\n{'='*60}")
    pair_info = dataset.get_pair_info(pair_idx)
    logger.info(f"Processing pair {pair_idx}: {pair_info['moving_id']} → {pair_info['fixed_id']}")

    data = dataset[pair_idx]
    fixed_img = data["fixed_img"]
    moving_img = data["moving_img"]

    # Initial TRE (before registration)
    initial_tre = {}
    if data["moving_keypoints"] is not None and data["fixed_keypoints"] is not None:
        initial_tre = compute_tre(data["moving_keypoints"], data["fixed_keypoints"])
        logger.info(f"Initial TRE: {initial_tre['mean_tre']:.3f} mm")

    # ThoraxCBCT does not provide pre-segmented trunk masks — generate them
    # via Hounsfield-unit thresholding + morphology.
    fixed_mask  = generate_trunk_mask(fixed_img)
    moving_mask = generate_trunk_mask(moving_img)


    D, H, W = fixed_img.shape
    total_matches = -1
    match_result = None

    viz = PipelineVisualizer(config.paths.output_dir, pair_idx, enabled=visualize)
    viz.input(fixed_img, moving_img, data["fixed_keypoints"], data["moving_keypoints"])

    if method == "mind":
        # =============================================================
        # MIND-SSC ConvexAdam — proven baseline (matches original DINO-Reg)
        # Pass images directly from get_fdata() — no transposition needed.
        # Displacement channels = (dim0, dim1, dim2) of the volume.
        # =============================================================
        logger.info(f"\nMethod: MIND-SSC ConvexAdam")
        from pipeline.transform.mind_convex_adam import mind_convex_adam

        displacement = mind_convex_adam(
            fixed_img=fixed_img,
            moving_img=moving_img,
            mind_r=1,
            mind_d=2,
            lambda_weight=0,     # Adam disabled: makes TRE worse in all grid_sp sweep results
            grid_sp=4,           # Best from test_grid_sp.py sweep (9.772mm convex-only)
            disp_hw=4,
            n_iter_adam=0,
            grid_sp_adam=2,
            ic=True,
            device=config.device,
        )

    elif method == "sparse":
        # =============================================================
        # Foundation features (DINOv3/MATCHA) + sparse matching + SVF
        # =============================================================
        assert fuser is not None, "Feature fuser required for sparse method"
        logger.info(f"\nMethod: {config.features.backend} + {config.matcher.method} + SVF")

        # Stage 1: Feature extraction
        logger.info("Stage 1: Feature extraction...")
        fixed_result  = get_features(config, fixed_img,  data["fixed_id"],  fuser, mask=fixed_mask)
        moving_result = get_features(config, moving_img, data["moving_id"], fuser, mask=moving_mask)
        fixed_feats, fixed_feat_shape, fixed_orig_shape = fixed_result
        moving_feats, moving_feat_shape, moving_orig_shape = moving_result

        viz.features(fixed_feats.transpose(1, 2, 3, 0),
                     moving_feats.transpose(1, 2, 3, 0))

        # ============================================================
        # Outer GWOT⇔SVF alternation loop
        # ============================================================
        # Each iteration:
        #   1) Warp moving pts with current displacement (better correspondence search)
        #   2) Sample + match in the (partially) warped space
        #   3) Fit SVF on the new correspondences
        # First iteration: displacement=None (no warping yet)
        # Subsequent: warp moving pts before re-matching
        # filter_matches always returns UNWARPED moving coords in matched_points_src
        # so the fitter always produces the full fixed→moving displacement.
        # ============================================================
        n_outer = config.fitting.n_outer_iters
        logger.info(f"Outer loop: {n_outer} iteration(s) of match+fit")

        fitter = DiffeomorphicFitter(
            volume_shape=(D, H, W),
            grid_spacings=config.fitting.grid_spacings,
            n_iters_per_level=config.fitting.n_iters_per_level,
            lambda_smooth=config.fitting.lambda_smooth,
            lambda_jac=config.fitting.lambda_jac,
            n_squaring_steps=config.fitting.n_squaring_steps,
            lr=config.fitting.lr,
            device=config.device,
        )

        displacement = None
        match_result  = None

        for outer_iter in range(n_outer):
            # Point-count schedule: coarse → medium → fine for NN.
            # OT/GWOT always uses n_points_ot (O(N²) memory constraint).
            if config.matcher.method in ("ot", "gwot"):
                n_pts = config.sampling.n_points_ot
            elif n_outer == 1 or outer_iter == 0:
                n_pts = config.sampling.n_points_coarse
            elif outer_iter < n_outer - 1:
                n_pts = config.sampling.n_points_medium
            else:
                n_pts = config.sampling.n_points_fine

            logger.info(f"\nOuter iter {outer_iter + 1}/{n_outer}: "
                        f"{n_pts} pts, matcher={config.matcher.method}")

            match_result = run_matching_stage(
                config, fixed_feats, moving_feats,
                fixed_feat_shape, moving_feat_shape,
                fixed_orig_shape, moving_orig_shape,
                fixed_mask, moving_mask,
                n_pts, downsample,
                displacement=displacement,   # warp moving pts for iter ≥1
                fixed_keypoints=data["fixed_keypoints"],
                moving_keypoints=data["moving_keypoints"],
                stage_name=f"iter{outer_iter + 1}",
            )

            if match_result is None:
                logger.warning(f"  Matching failed at outer iter {outer_iter + 1}")
                if displacement is None:
                    logger.error("  No displacement to fall back to — aborting")
                    return {"pair_idx": pair_idx, "error": "matching_failed",
                            "runtime_seconds": time.time() - t0}
                logger.warning("  Keeping displacement from previous iteration")
                break

            # Fit SVF: field is fixed→moving.
            # matched_points_tgt = fixed pts (field origin)
            # matched_points_src = original (unwarped) moving pts (field target)
            displacement = fitter.fit(
                matched_src=match_result["matched_points_tgt"],
                matched_tgt=match_result["matched_points_src"],
                weights=match_result["weights"],
            )
            logger.info(f"  Iter {outer_iter + 1} displacement: "
                        f"max={displacement.abs().max().item():.1f}, "
                        f"mean={displacement.abs().mean().item():.1f}")

        total_matches = match_result["n_matches"] if match_result else 0
    else:
        raise ValueError(f"Unknown method: {method}")

    logger.info(f"  Displacement: max={displacement.abs().max().item():.1f}, "
                f"mean={displacement.abs().mean().item():.1f}")
    if match_result is not None:
        viz.matches(match_result, fixed_img, moving_img)
    viz.displacement(displacement, fixed_img)


    # =========================================================
    # DIAGNOSTICS: Use DINO-Reg evaluation convention
    # =========================================================
    # ConvexAdam convention: d(x_fixed) = x_moving - x_fixed
    # Evaluation: warped_fixed = fixed + d(fixed), TRE = ||warped_fixed - moving||
    if data["moving_keypoints"] is not None and data["fixed_keypoints"] is not None:
        from scipy.ndimage import map_coordinates as mc
        mkp = data["moving_keypoints"]  # (N, 3)
        fkp = data["fixed_keypoints"]   # (N, 3)
        disp_np = displacement.cpu().numpy()

        # DINO-Reg convention: sample displacement at FIXED keypoints
        fixed_disp = np.zeros_like(fkp)
        for ax in range(3):
            fixed_disp[:, ax] = mc(
                disp_np[0, ax],
                [fkp[:, 0], fkp[:, 1], fkp[:, 2]],
                order=1, mode='nearest',
            )
        warped_fixed = fkp + fixed_disp
        tre_dinoreg = np.linalg.norm(warped_fixed - mkp, axis=1).mean()

        logger.info(f"\n  === EVALUATION (DINO-Reg convention) ===")
        logger.info(f"  Initial TRE: {np.linalg.norm(fkp - mkp, axis=1).mean():.3f} mm")
        logger.info(f"  DINO-Reg eval (sample@fixed, compare to moving): {tre_dinoreg:.3f} mm")
        
        # Per-axis analysis
        ideal_disp = mkp - fkp  # d should be: moving - fixed
        for ax in range(3):
            ideal = ideal_disp[:, ax]
            actual = fixed_disp[:, ax]
            corr = np.corrcoef(ideal, actual)[0, 1] if ideal.std() > 0 else 0
            logger.info(f"  Axis {ax}: ideal mean={ideal.mean():+.2f} std={ideal.std():.2f}, "
                        f"actual mean={actual.mean():+.2f} std={actual.std():.2f}, corr={corr:+.3f}")

    # =========================================================
    # Stage 5: Optional intensity refinement (local NCC)
    # =========================================================
    if config.fitting.intensity_refine:
        logger.info("\nStage 5: Intensity refinement (local NCC)...")
        try:
            from pipeline.transform.intensity_refine import intensity_refinement

            # Normalize volumes for NCC
            fixed_norm = robust_intensity_normalize(fixed_img, mask=fixed_mask)
            moving_norm = robust_intensity_normalize(moving_img, mask=moving_mask)

            displacement = intensity_refinement(
                fixed_img=fixed_norm,
                moving_img=moving_norm,
                current_displacement=displacement,
                fixed_mask=fixed_mask,
                grid_spacing=config.fitting.grid_spacings[-1],  # finest grid
                lambda_smooth=5.0,  # strong regularization
                lr=1e-3,
                n_iters=50,
                n_squaring_steps=config.fitting.n_squaring_steps,
                win_size=9,
                device=config.device,
            )

            if data["moving_keypoints"] is not None and data["fixed_keypoints"] is not None:
                post_ncc_tre = compute_tre(
                    data["moving_keypoints"], data["fixed_keypoints"], displacement
                )
                logger.info(f"  Post-NCC TRE: {post_ncc_tre['mean_tre']:.3f} mm")
        except Exception as e:
            logger.warning(f"  Intensity refinement failed: {e}")
            logger.warning("  Continuing with correspondence-only displacement")

    # =========================================================
    # Stage 6: Evaluation
    # =========================================================
    logger.info("\nStage 6: Evaluation...")
    results = {
        "pair_idx": pair_idx,
        "fixed_id": data["fixed_id"],
        "moving_id": data["moving_id"],
        "n_matches": total_matches,
        "runtime_seconds": time.time() - t0,
    }

    if data["moving_keypoints"] is not None and data["fixed_keypoints"] is not None:
        # Use DINO-Reg evaluation convention: sample at fixed, compare to moving
        from scipy.ndimage import map_coordinates as mc_eval
        mkp_eval = data["moving_keypoints"]
        fkp_eval = data["fixed_keypoints"]
        disp_eval = displacement.cpu().numpy()
        
        fixed_disp_eval = np.zeros_like(fkp_eval)
        for ax_e in range(3):
            fixed_disp_eval[:, ax_e] = mc_eval(
                disp_eval[0, ax_e],
                [fkp_eval[:, 0], fkp_eval[:, 1], fkp_eval[:, 2]],
                order=1, mode='nearest',
            )
        warped_fixed_eval = fkp_eval + fixed_disp_eval
        errors_eval = np.linalg.norm(warped_fixed_eval - mkp_eval, axis=1)
        
        tre = {
            "mean_tre": float(np.mean(errors_eval)),
            "median_tre": float(np.median(errors_eval)),
            "std_tre": float(np.std(errors_eval)),
            "max_tre": float(np.max(errors_eval)),
            "min_tre": float(np.min(errors_eval)),
            "n_keypoints": len(errors_eval),
        }
        results["initial_tre"] = initial_tre
        results["final_tre"] = tre
        print_results(tre, pair_id=f"{data['moving_id']} → {data['fixed_id']}")

    jac_stats = compute_jacobian_stats(displacement)
    results["jac_stats"] = jac_stats

    viz.output(fixed_img, moving_img, displacement,
               data["fixed_keypoints"], data["moving_keypoints"])

    logger.info(f"Total runtime: {results['runtime_seconds']:.1f}s")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run registration pipeline")
    parser.add_argument("--pair", type=int, default=None, help="Single pair index")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--method", type=str, default="mind",
                        choices=["mind", "sparse"],
                        help="'mind': MIND-SSC ConvexAdam baseline. "
                             "'sparse': foundation features + matching + SVF fitting.")
    parser.add_argument("--feature", type=str, default="dinov3", choices=["dinov3", "matcha"])
    parser.add_argument("--matcher", type=str, default="gwot", choices=["nn", "ot", "gwot"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_points", type=int, default=None,
                        help="Override coarse n_points (default: use config)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--downsample", type=int, default=2, help="Volume downsample factor")
    parser.add_argument("--no-intensity-refine", action="store_true",
                        help="Explicitly disable intensity refinement (default: already off)")
    parser.add_argument("--intensity-refine", action="store_true",
                        help="Enable local-NCC intensity refinement after SVF fitting")
    parser.add_argument("--n-outer-iters", type=int, default=None,
                        help="GWOT⇔fit alternation iterations (default: 1 = single pass)")
    parser.add_argument("--visualize", action="store_true",
                        help="Save diagnostic PNG panels to output/viz/pair_XX/")
    args = parser.parse_args()

    # Config
    config = PipelineConfig()
    config.features.backend = args.feature
    config.matcher.method = args.matcher
    config.device = args.device
    if args.n_points is not None:
        config.sampling.n_points_coarse = args.n_points
    config.features.slice_batch_size = args.batch_size
    # Intensity refinement: --intensity-refine enables, --no-intensity-refine disables
    if getattr(args, "intensity_refine", False):
        config.fitting.intensity_refine = True
    if getattr(args, "no_intensity_refine", False):
        config.fitting.intensity_refine = False
    if getattr(args, "n_outer_iters", None) is not None:
        config.fitting.n_outer_iters = args.n_outer_iters

    # Dataset
    dataset = ThoraxCBCTDataset(config.paths.data_root, split=args.split)
    logger.info(f"Dataset: {dataset}")

    # Feature extractor + fuser (only needed for sparse method)
    fuser = None
    if args.method == "sparse":
        extractor = create_feature_extractor(config)
        fuser = TriplanarFuser(
            extractor,
            batch_size=config.features.slice_batch_size,
            fusion=config.features.fusion_method,
            downsample=args.downsample,
            device=config.device,
        )

    # Run
    if args.pair is not None:
        results = [run_pair(config, dataset, args.pair, args.method, fuser, args.downsample,
                            visualize=args.visualize)]
    else:
        results = []
        for i in range(len(dataset)):
            r = run_pair(config, dataset, i, args.method, fuser, args.downsample,
                         visualize=args.visualize)
            results.append(r)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        if "error" in r:
            print(f"Pair {r['pair_idx']}: ERROR - {r['error']}")
        elif "final_tre" in r:
            init = r["initial_tre"]["mean_tre"]
            final = r["final_tre"]["mean_tre"]
            improvement = (1 - final / init) * 100
            print(f"Pair {r['pair_idx']}: TRE {init:.2f} → {final:.2f} mm "
                  f"({improvement:+.1f}%, {r['n_matches']} matches, "
                  f"{r['runtime_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
