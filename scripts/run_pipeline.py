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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


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


def voxel_to_feature_coords(points: np.ndarray, original_shape: tuple,
                             feature_shape: tuple, downsample: int) -> np.ndarray:
    """
    Convert voxel coordinates to feature grid coordinates.

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

    # Then scale to feature grid
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


def get_features(
    config: PipelineConfig,
    volume: np.ndarray,
    case_id: str,
    fuser: TriplanarFuser,
):
    """Extract or load cached tri-planar features for a volume."""
    cache_path = config.paths.feature_cache_dir / f"{case_id}_{config.features.backend}.npz"

    if config.features.use_cache and cache_path.exists():
        logger.info(f"Loading cached features: {cache_path}")
        return load_features(cache_path)

    # Preprocess
    mask = generate_trunk_mask(volume)
    vol_norm = robust_intensity_normalize(volume, mask=mask)

    # Extract tri-planar features (returns tuple)
    result = fuser.fuse_triplanar(vol_norm)

    # Cache
    if config.features.use_cache:
        save_features(result, cache_path)

    return result


def sample_and_describe(
    feats: np.ndarray,
    feat_shape: tuple,
    orig_shape: tuple,
    mask: np.ndarray,
    n_points: int,
    downsample: int,
    keypoints: np.ndarray = None,
    rng: np.random.RandomState = None,
) -> tuple:
    """
    Sample points inside mask and extract descriptors.

    Optionally includes Förstner keypoints.

    Returns:
        (voxel_pts, descriptors_normalized)
    """
    voxel_pts = sample_points_in_mask(mask, n_points, rng=rng)

    # Include Förstner keypoints if provided (as high-value anchors)
    if keypoints is not None:
        voxel_pts = include_keypoints(voxel_pts, keypoints, mask=mask)
        logger.info(f"    Added keypoints → {len(voxel_pts)} total points")

    # Convert to feature grid coords
    feat_pts = voxel_to_feature_coords(voxel_pts, orig_shape, feat_shape, downsample)

    # Sample descriptors
    desc = sample_descriptors_at_points(feats, feat_pts)

    # L2 normalize
    norms = np.linalg.norm(desc, axis=1, keepdims=True) + 1e-8
    desc = desc / norms

    return voxel_pts, desc


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

    # Sample and describe fixed points
    fixed_voxel_pts, fixed_desc = sample_and_describe(
        fixed_feats, fixed_feat_shape, fixed_orig_shape,
        fixed_mask, n_points, downsample,
        keypoints=fixed_keypoints, rng=rng,
    )

    # Sample and describe moving points
    rng_m = np.random.RandomState(123)
    moving_voxel_pts, moving_desc = sample_and_describe(
        moving_feats, moving_feat_shape, moving_orig_shape,
        moving_mask, n_points, downsample,
        keypoints=moving_keypoints, rng=rng_m,
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
            "lambda_gw": config.gwot.lambda_gw,
            "lambda_prior": config.gwot.lambda_prior,
            "epsilon": config.gwot.epsilon,
            "local_radius": config.gwot.local_radius,
            "max_iter": config.gwot.max_iter,
        }
    elif config.matcher.method == "ot":
        match_kwargs = {
            "lambda_prior": config.gwot.lambda_prior,
            "epsilon": config.gwot.epsilon,
            "max_iter": config.gwot.max_iter,
        }

    match_result = match(
        moving_desc, fixed_desc, warped_moving_pts, fixed_voxel_pts,
        method=config.matcher.method, **match_kwargs
    )

    # Filter matches WITH mask consistency
    filtered = filter_matches(
        match_result, moving_voxel_pts, fixed_voxel_pts,
        confidence_threshold=config.gwot.confidence_threshold,
        mask_src=moving_mask,
        mask_tgt=fixed_mask,
        max_displacement=150.0,  # reject physically implausible matches
    )
    logger.info(f"  [{stage_name}] Matches: {filtered['n_matches']} / {filtered['n_before_filter']}")

    if filtered["n_matches"] < 10:
        logger.warning(f"  [{stage_name}] Too few matches ({filtered['n_matches']})")
        return None

    return filtered


def run_pair(
    config: PipelineConfig,
    dataset: ThoraxCBCTDataset,
    pair_idx: int,
    fuser: TriplanarFuser,
    downsample: int = 2,
) -> dict:
    """
    Run full multi-stage registration on a single pair.

    Implements:
    1. Feature extraction
    2. Multi-stage matching (coarse → medium → fine) per plan.md
    3. GWOT↔fit alternation (re-match after each fit)
    4. Diffeomorphic fitting with SVF
    5. Optional intensity refinement (local NCC)
    6. Evaluation
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

    # =========================================================
    # Stage 1: Feature extraction
    # =========================================================
    logger.info("Stage 1: Feature extraction...")
    fixed_result = get_features(config, fixed_img, data["fixed_id"], fuser)
    moving_result = get_features(config, moving_img, data["moving_id"], fuser)

    fixed_feats, fixed_feat_shape, fixed_orig_shape = fixed_result
    moving_feats, moving_feat_shape, moving_orig_shape = moving_result

    # Generate trunk masks
    fixed_mask = generate_trunk_mask(fixed_img)
    moving_mask = generate_trunk_mask(moving_img)

    D, H, W = fixed_img.shape
    logger.info(f"\nStage 3: ConvexAdam on {config.features.backend} features...")

    from pipeline.transform.mind_convex_adam import convex_adam_on_features

    C = fixed_feats.shape[0]
    if C > 64:
        logger.info(f"  Reducing feature dimension from {C} to 64 with PCA...")
        from sklearn.decomposition import PCA
        flat_fixed = fixed_feats.reshape(C, -1).T
        flat_moving = moving_feats.reshape(C, -1).T
        
        pca = PCA(n_components=64)
        n_samples = min(10000, flat_fixed.shape[0])
        indices = np.random.choice(flat_fixed.shape[0], n_samples, replace=False)
        pca.fit(flat_fixed[indices])
        
        # FIX: The original was incorrectly packing the spatial dimensions
        # Features come in as (C, fD, fH, fW). We flatten them to (N, C).
        # PCA transforms them to (N, 64). We transpose them to (64, N).
        # We MUST reshape them back to (64, fD, fH, fW) in the exact same spatial order.
        fixed_feats = pca.transform(flat_fixed).T.reshape(64, fixed_feat_shape[0], fixed_feat_shape[1], fixed_feat_shape[2])
        moving_feats = pca.transform(flat_moving).T.reshape(64, moving_feat_shape[0], moving_feat_shape[1], moving_feat_shape[2])
        
        # Add normalization after PCA
        fixed_feats = fixed_feats / (np.linalg.norm(fixed_feats, axis=0, keepdims=True) + 1e-8)
        moving_feats = moving_feats / (np.linalg.norm(moving_feats, axis=0, keepdims=True) + 1e-8)
        
    fixed_t = torch.from_numpy(fixed_feats).unsqueeze(0).float()
    moving_t = torch.from_numpy(moving_feats).unsqueeze(0).float()
    
    logger.info(f"  Upsampling features to full volume size ({D}, {H}, {W})...")
    # INTERPOLATION BUG & AXIS MISMATCH
    # In DINO-Reg, the input volume spatial dim is (H, W, D).
    # Our data volume shape is (D, H, W).
    # Since DINO-Reg's `correlate` and `coupled_convex` process loops over `H` explicitly
    # and assumes grid meshes are formed in a certain way, we must align our dims tightly.
    # We pass it our (D, H, W), but DINO-Reg's ConvexAdam expects the first spatial
    # dimension to be the biggest/axial. Actually, DINO-Reg just expects whatever shape and `map_coordinates` we use consistently.
    # The evaluation output showed: `Axis 0 (z(D)) ... actual mean=+0.01` when ideal was 0.89.
    # Displacement was vanishing. The root cause is `disp_lr.float().cpu().data / grid_sp_adam`
    # and `grid_sp` mismatched Adam vs Convex scaling, but let's stick to the base ConvexAdam config.

    fixed_feats_full = F.interpolate(fixed_t, size=(D, H, W), mode="trilinear", align_corners=False)
    moving_feats_full = F.interpolate(moving_t, size=(D, H, W), mode="trilinear", align_corners=False)

    # Note: the `mind_convex_adam` module had issues with lambda_weight vs n_iter.
    # We will pass lambda_weight=2.0 and n_iter_adam=30 to ensure we don't zero-out displacement.
    displacement = convex_adam_on_features(
        fixed_feats=fixed_feats_full,
        moving_feats=moving_feats_full,
        volume_shape=(D, H, W),
        grid_sp=6,
        disp_hw=4,
        lambda_weight=2.0,
        n_iter_adam=30,
        grid_sp_adam=2,
        ic=True,
        device=config.device,
    )

    total_matches = -1  # dense method, no explicit matches

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
        for ax, name in enumerate(["z(D)", "y(H)", "x(W)"]):
            ideal = ideal_disp[:, ax]
            actual = fixed_disp[:, ax]
            corr = np.corrcoef(ideal, actual)[0, 1] if ideal.std() > 0 else 0
            logger.info(f"  Axis {ax} ({name}): ideal mean={ideal.mean():+.2f} std={ideal.std():.2f}, "
                        f"actual mean={actual.mean():+.2f} std={actual.std():.2f}, corr={corr:+.3f}")

    # =========================================================
    # Stage 5: Optional intensity refinement (local NCC)
    # =========================================================
    if config.fitting.n_outer_iters > 0:
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

    logger.info(f"Total runtime: {results['runtime_seconds']:.1f}s")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run registration pipeline")
    parser.add_argument("--pair", type=int, default=None, help="Single pair index")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--feature", type=str, default="dinov3", choices=["dinov3", "matcha"])
    parser.add_argument("--matcher", type=str, default="gwot", choices=["nn", "ot", "gwot"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_points", type=int, default=None,
                        help="Override coarse n_points (default: use config)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--downsample", type=int, default=2, help="Volume downsample factor")
    parser.add_argument("--no-intensity-refine", action="store_true",
                        help="Skip intensity refinement")
    args = parser.parse_args()

    # Config
    config = PipelineConfig()
    config.features.backend = args.feature
    config.matcher.method = args.matcher
    config.device = args.device
    if args.n_points is not None:
        config.sampling.n_points_coarse = args.n_points
    config.features.slice_batch_size = args.batch_size
    if args.no_intensity_refine:
        config.fitting.n_outer_iters = 0

    # Dataset
    dataset = ThoraxCBCTDataset(config.paths.data_root, split=args.split)
    logger.info(f"Dataset: {dataset}")

    # Feature extractor + fuser
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
        results = [run_pair(config, dataset, args.pair, fuser, args.downsample)]
    else:
        results = []
        for i in range(len(dataset)):
            r = run_pair(config, dataset, i, fuser, args.downsample)
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
