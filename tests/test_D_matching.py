#!/usr/bin/env python3
"""
Test D: Matching Quality
=========================

Tests NN / OT / GWOT matching quality using GT keypoints as the point sets.
This tells you whether matching is helping or hurting beyond raw feature quality.

Uses GT keypoints as BOTH the sampled point sets — so we know the ground-truth
pairing and can compute PCK (Percentage of Correct Keypoints) metrics.

Usage:
    python -m pipeline.tests.test_D_matching --pair 0
    python -m pipeline.tests.test_D_matching --pair 0 --feature matcha
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import PipelineConfig
from pipeline.data.dataset_thoraxcbct import ThoraxCBCTDataset
from pipeline.data.preprocessing import generate_trunk_mask, robust_intensity_normalize
from pipeline.features.triplanar_fuser import TriplanarFuser, load_features, save_features
from pipeline.matching.sampling import sample_descriptors_at_points, sample_points_in_mask, include_keypoints
from pipeline.matching.gwot3d import nn_matching, match

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

VIZ_DIR = PROJECT_ROOT / "pipeline" / "tests" / "results" / "test_D"
VIZ_DIR.mkdir(parents=True, exist_ok=True)


def _voxel_to_feature_coords(points, original_shape, feature_shape, downsample):
    """Convert voxel coordinates to feature grid coordinates."""
    D, H, W = original_shape
    fD, fH, fW = feature_shape

    if (D, H, W) == (fD, fH, fW):
        feat_points = points.copy()
        feat_points[:, 0] = np.clip(feat_points[:, 0], 0, fD - 1)
        feat_points[:, 1] = np.clip(feat_points[:, 1], 0, fH - 1)
        feat_points[:, 2] = np.clip(feat_points[:, 2], 0, fW - 1)
        return feat_points

    feat_points = points.copy()
    feat_points[:, 0] = feat_points[:, 0] / downsample * fD / (D // downsample)
    feat_points[:, 1] = feat_points[:, 1] / downsample * fH / (H // downsample)
    feat_points[:, 2] = feat_points[:, 2] / downsample * fW / (W // downsample)
    feat_points[:, 0] = np.clip(feat_points[:, 0], 0, fD - 1)
    feat_points[:, 1] = np.clip(feat_points[:, 1], 0, fH - 1)
    feat_points[:, 2] = np.clip(feat_points[:, 2], 0, fW - 1)
    return feat_points


def _extract_features(config, volume, case_id, fuser, mask=None):
    """Extract or load cached features."""
    cache_path = config.paths.feature_cache_dir / f"{case_id}_{config.features.backend}.npz"
    if config.features.use_cache and cache_path.exists():
        logger.info(f"  Loading cached features: {cache_path}")
        return load_features(cache_path)
    if mask is None:
        mask = generate_trunk_mask(volume)
    if hasattr(fuser, "extract_volume_features"):
        result = fuser.extract_volume_features(volume.astype(np.float32))
    else:
        vol_norm = robust_intensity_normalize(volume, mask=mask)
        result = fuser.fuse_triplanar(vol_norm)
    if config.features.use_cache:
        save_features(result, cache_path)
    return result


def compute_pck(matched_src_pts, matched_tgt_pts, gt_src_pts, gt_tgt_pts,
                src_indices, tgt_indices, thresholds=[2, 5, 10, 20, 50]):
    """
    Compute PCK: Percentage of Correct Keypoints.
    
    For each match (src_idx → tgt_idx), the 'correct' target is gt_tgt_pts[src_idx].
    The match error is ||matched_tgt_pts[matched] - gt_tgt_pts[src_idx]||.
    
    Since we use GT keypoints as point sets, gt_src_pts[k] pairs with gt_tgt_pts[k].
    A match (src=k → tgt=j) is correct if j == k.
    The spatial error is ||gt_tgt_pts[j] - gt_tgt_pts[k]||.
    """
    n_matches = len(src_indices)
    if n_matches == 0:
        return {f"pck@{t}mm": 0.0 for t in thresholds}, np.array([])
    
    # For each matched pair, compute distance between matched target and GT target
    errors = np.linalg.norm(
        gt_tgt_pts[tgt_indices] - gt_tgt_pts[src_indices],  # matched vs correct target
        axis=1
    )
    
    pck = {}
    for t in thresholds:
        pck[f"pck@{t}mm"] = float((errors < t).mean() * 100)
    
    return pck, errors


def compute_candidate_pck(candidate_points, gt_targets, src_indices, tgt_indices,
                          thresholds=[2, 5, 10, 20, 50]):
    """PCK when the target set is a candidate pool rather than GT-index aligned."""
    n_matches = len(src_indices)
    if n_matches == 0:
        return {f"pck@{t}mm": 0.0 for t in thresholds}, np.array([])

    errors = np.linalg.norm(
        candidate_points[tgt_indices] - gt_targets[src_indices],
        axis=1
    )
    pck = {}
    for t in thresholds:
        pck[f"pck@{t}mm"] = float((errors < t).mean() * 100)
    return pck, errors


def test_D1_matching_at_gt_points(config, dataset, pair_idx, fuser, downsample=2):
    """
    D1: Use GT keypoints as the point sets and test matching accuracy.
    
    Compares NN, OT, GWOT using features extracted at GT keypoint locations.
    
    Pass: GWOT PCK@10mm > NN PCK@10mm, and both > 30%
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST D1: Matching at GT keypoints (pair {pair_idx})")
    logger.info(f"{'='*60}")
    
    data = dataset[pair_idx]
    fkp = data["fixed_keypoints"]
    mkp = data["moving_keypoints"]
    
    if fkp is None or mkp is None:
        logger.warning(f"  SKIP: No keypoints")
        return None
    
    # Subsample keypoints (OT/GWOT can handle ~2000 max)
    n_pts = min(2000, len(fkp))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(fkp), n_pts, replace=False)
    fkp_sub = fkp[idx]
    mkp_sub = mkp[idx]
    
    logger.info(f"  Using {n_pts}/{len(fkp)} GT keypoints as point sets")
    
    # Extract features
    logger.info(f"  Extracting features ({config.features.backend})...")
    fixed_mask = generate_trunk_mask(data["fixed_img"])
    moving_mask = generate_trunk_mask(data["moving_img"])
    
    fixed_result = _extract_features(config, data["fixed_img"], data["fixed_id"], fuser, fixed_mask)
    moving_result = _extract_features(config, data["moving_img"], data["moving_id"], fuser, moving_mask)
    
    fixed_feats, fixed_feat_shape, fixed_orig_shape = fixed_result
    moving_feats, moving_feat_shape, moving_orig_shape = moving_result
    
    # Sample descriptors at GT keypoints
    fkp_feat = _voxel_to_feature_coords(fkp_sub, fixed_orig_shape, fixed_feat_shape, downsample)
    mkp_feat = _voxel_to_feature_coords(mkp_sub, moving_orig_shape, moving_feat_shape, downsample)
    
    fixed_desc = sample_descriptors_at_points(fixed_feats, fkp_feat)
    moving_desc = sample_descriptors_at_points(moving_feats, mkp_feat)
    
    # L2 normalize
    fixed_desc = fixed_desc / (np.linalg.norm(fixed_desc, axis=1, keepdims=True) + 1e-8)
    moving_desc = moving_desc / (np.linalg.norm(moving_desc, axis=1, keepdims=True) + 1e-8)
    
    # Run matching with different methods
    # NOTE: In the pipeline, desc_src=moving, desc_tgt=fixed.
    # Here: we use moving as src, fixed as tgt (same convention as pipeline)
    methods_to_test = ["nn", "ot", "gwot"]
    all_results = {}
    
    for method in methods_to_test:
        logger.info(f"\n  --- {method.upper()} matching ---")
        t0 = time.time()
        
        try:
            kwargs = {}
            if method == "nn":
                kwargs = {
                    "max_displacement": config.sampling.max_displacement,
                    "margin_threshold": config.matcher.nn_margin_threshold,
                }
            elif method == "gwot":
                kwargs = {
                    "lambda_gw": config.gwot.lambda_gw,
                    "lambda_prior": config.gwot.lambda_prior,
                    "epsilon": config.gwot.epsilon,
                    "lambda_mass": config.gwot.lambda_mass,
                    "local_radius": config.gwot.local_radius,
                    "max_iter": config.gwot.max_iter,
                }
            elif method == "ot":
                kwargs = {
                    "lambda_prior": config.gwot.lambda_prior,
                    "epsilon": config.gwot.epsilon,
                    "lambda_mass": config.gwot.lambda_mass,
                    "max_iter": config.gwot.max_iter,
                }
            
            result = match(
                moving_desc, fixed_desc,
                mkp_sub, fkp_sub,
                method=method, **kwargs
            )
            elapsed = time.time() - t0
            
            src_idx = result["matches_src_idx"]
            tgt_idx = result["matches_tgt_idx"]
            weights = result["weights"]
            
            n_matches = len(src_idx)
            n_correct_identity = (src_idx == tgt_idx).sum()
            
            # PCK: since src[k] should match tgt[k] (same index = correct pair),
            # the spatial error is ||mkp_sub[tgt_idx] - mkp_sub[src_idx]||
            # But actually what matters is: for source point src_idx[i],
            # the correct target is src_idx[i] (identity mapping).
            # The matched target is tgt_idx[i].
            # Error = ||fkp_sub[tgt_idx[i]] - fkp_sub[src_idx[i]]||
            # (since both point sets are GT keypoints, fkp_sub[k] corresponds to mkp_sub[k])
            pck, errors = compute_pck(
                mkp_sub, fkp_sub,  # matched src/tgt arrays (for indexing)
                mkp_sub, fkp_sub,  # GT arrays
                src_idx, tgt_idx,
            )
            
            logger.info(f"  {method.upper()}: {n_matches} matches, "
                        f"identity matches: {n_correct_identity}/{n_matches} "
                        f"({n_correct_identity/max(n_matches,1)*100:.1f}%), "
                        f"time: {elapsed:.2f}s")
            if len(weights) > 0:
                logger.info(f"  Weights: min={weights.min():.4f}, max={weights.max():.4f}, "
                            f"mean={weights.mean():.4f}")
            for k, v in pck.items():
                logger.info(f"    {k}: {v:.1f}%")
            if len(errors) > 0:
                logger.info(f"  Match error: mean={errors.mean():.1f}mm, "
                            f"median={np.median(errors):.1f}mm, "
                            f"std={errors.std():.1f}mm")
            
            all_results[method] = {
                "n_matches": n_matches,
                "n_correct_identity": int(n_correct_identity),
                "identity_pct": float(n_correct_identity / max(n_matches, 1) * 100),
                "pck": pck,
                "errors": errors,
                "runtime_s": elapsed,
            }
            
        except Exception as e:
            logger.error(f"  {method.upper()} failed: {e}")
            import traceback
            traceback.print_exc()
            all_results[method] = {"error": str(e)}
    
    # Determine pass/fail
    nn_pck10 = all_results.get("nn", {}).get("pck", {}).get("pck@10mm", 0)
    gwot_pck10 = all_results.get("gwot", {}).get("pck", {}).get("pck@10mm", 0)
    
    passed_nn = nn_pck10 > 30
    passed_gwot_better = gwot_pck10 > nn_pck10
    passed_gwot = gwot_pck10 > 50  # GWOT must be able to recover at least 50% accurate correspondences
    passed = passed_gwot and passed_gwot_better
    
    logger.info(f"\n  NN PCK@10mm > 30%?        {'✅' if passed_nn else '❌'} ({nn_pck10:.1f}%)")
    logger.info(f"  GWOT PCK@10mm >= NN?      {'✅' if passed_gwot_better else '❌'} "
                f"(GWOT={gwot_pck10:.1f}%, NN={nn_pck10:.1f}%)")
    
    if not passed_gwot_better:
        logger.warning("  GWOT doesn't improve over NN — issue is in features, not matching")
    
    # Visualization
    _visualize_D1(all_results, n_pts, pair_idx, config)
    
    return {
        "test": "D1_matching_at_gt",
        "pair": pair_idx,
        "n_points": n_pts,
        "backend": config.features.backend,
        "results": {m: {k: v for k, v in r.items() if k != "errors"} 
                    for m, r in all_results.items()},
        "passed": passed,
    }


def test_D2_matching_with_distractors(config, dataset, pair_idx, fuser, downsample=2):
    """
    D2: Use GT fixed keypoints as sources, but match into a candidate pool of
    mostly random moving points plus the true moving correspondences.

    This is much closer to the real sparse pipeline than D1 because the target
    set is no longer GT-index aligned.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST D2: Matching with distractor candidate pool (pair {pair_idx})")
    logger.info(f"{'='*60}")

    data = dataset[pair_idx]
    fkp = data["fixed_keypoints"]
    mkp = data["moving_keypoints"]

    if fkp is None or mkp is None:
        logger.warning("  SKIP: No keypoints")
        return None

    fixed_mask = generate_trunk_mask(data["fixed_img"])
    moving_mask = generate_trunk_mask(data["moving_img"])

    fixed_result = _extract_features(config, data["fixed_img"], data["fixed_id"], fuser, fixed_mask)
    moving_result = _extract_features(config, data["moving_img"], data["moving_id"], fuser, moving_mask)
    fixed_feats, fixed_feat_shape, fixed_orig_shape = fixed_result
    moving_feats, moving_feat_shape, moving_orig_shape = moving_result

    rng = np.random.RandomState(42)
    n_src = min(500, len(fkp))
    src_idx_global = rng.choice(len(fkp), n_src, replace=False)
    fkp_src = fkp[src_idx_global]
    mkp_gt = mkp[src_idx_global]

    moving_random = sample_points_in_mask(moving_mask, 2000, z_stratified=True, rng=np.random.RandomState(123))
    moving_candidates = include_keypoints(moving_random, mkp_gt, mask=moving_mask)

    logger.info(f"  Sources: {len(fkp_src)} GT fixed keypoints")
    logger.info(f"  Target candidates: {len(moving_candidates)} moving points "
                f"({len(moving_candidates) - len(moving_random)} GT correspondences injected)")

    fkp_feat = _voxel_to_feature_coords(fkp_src, fixed_orig_shape, fixed_feat_shape, downsample)
    cand_feat = _voxel_to_feature_coords(moving_candidates, moving_orig_shape, moving_feat_shape, downsample)

    fixed_desc = sample_descriptors_at_points(fixed_feats, fkp_feat)
    moving_desc = sample_descriptors_at_points(moving_feats, cand_feat)

    fixed_desc = fixed_desc / (np.linalg.norm(fixed_desc, axis=1, keepdims=True) + 1e-8)
    moving_desc = moving_desc / (np.linalg.norm(moving_desc, axis=1, keepdims=True) + 1e-8)

    methods_to_test = ["nn", "gwot"]
    all_results = {}

    for method in methods_to_test:
        logger.info(f"\n  --- {method.upper()} matching ---")
        t0 = time.time()

        kwargs = {}
        if method == "nn":
            kwargs = {
                "max_displacement": config.sampling.max_displacement,
                    "margin_threshold": config.matcher.nn_margin_threshold,
            }
        elif method == "gwot":
            kwargs = {
                "lambda_gw": config.gwot.lambda_gw,
                "lambda_prior": config.gwot.lambda_prior,
                "epsilon": config.gwot.epsilon,
                "lambda_mass": config.gwot.lambda_mass,
                "local_radius": config.gwot.local_radius,
                "max_iter": config.gwot.max_iter,
            }

        result = match(
            fixed_desc, moving_desc,
            fkp_src, moving_candidates,
            method=method, **kwargs
        )
        elapsed = time.time() - t0

        src_idx = result["matches_src_idx"]
        tgt_idx = result["matches_tgt_idx"]
        weights = result["weights"]

        pck, errors = compute_candidate_pck(
            moving_candidates, mkp_gt, src_idx, tgt_idx
        )

        logger.info(f"  {method.upper()}: {len(src_idx)} matches, time: {elapsed:.2f}s")
        if len(weights) > 0:
            logger.info(f"  Weights: min={weights.min():.4f}, max={weights.max():.4f}, "
                        f"mean={weights.mean():.4f}")
        for k, v in pck.items():
            logger.info(f"    {k}: {v:.1f}%")
        if len(errors) > 0:
            logger.info(f"  Match error: mean={errors.mean():.1f}mm, "
                        f"median={np.median(errors):.1f}mm, std={errors.std():.1f}mm")

        all_results[method] = {
            "n_matches": int(len(src_idx)),
            "pck": pck,
            "errors": errors,
            "runtime_s": elapsed,
        }

    nn_pck10 = all_results.get("nn", {}).get("pck", {}).get("pck@10mm", 0)
    gwot_pck10 = all_results.get("gwot", {}).get("pck", {}).get("pck@10mm", 0)
    passed = gwot_pck10 > nn_pck10 and gwot_pck10 > 30.0

    logger.info(f"\n  GWOT PCK@10mm > NN? {'✅' if gwot_pck10 > nn_pck10 else '❌'} "
                f"(GWOT={gwot_pck10:.1f}%, NN={nn_pck10:.1f}%)")
    logger.info(f"  GWOT PCK@10mm > 30%? {'✅' if gwot_pck10 > 30.0 else '❌'} "
                f"({gwot_pck10:.1f}%)")

    return {
        "test": "D2_matching_with_distractors",
        "pair": pair_idx,
        "n_points": len(fkp_src),
        "backend": config.features.backend,
        "results": {m: {k: v for k, v in r.items() if k != "errors"}
                    for m, r in all_results.items()},
        "passed": passed,
    }


def _visualize_D1(all_results, n_pts, pair_idx, config):
    """Visualize matching quality comparison."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. PCK curves
    ax = axes[0]
    thresholds = [2, 5, 10, 20, 50]
    colors = {"nn": "blue", "ot": "orange", "gwot": "green"}
    for method, result in all_results.items():
        if "error" in result:
            continue
        pck_values = [result["pck"].get(f"pck@{t}mm", 0) for t in thresholds]
        ax.plot(thresholds, pck_values, 'o-', label=method.upper(), 
                color=colors.get(method, 'gray'), linewidth=2, markersize=6)
    ax.set_xlabel("Threshold (mm)")
    ax.set_ylabel("PCK (%)")
    ax.set_title("Matching Accuracy (PCK)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # 2. Error histograms
    ax = axes[1]
    for method, result in all_results.items():
        if "error" in result or len(result.get("errors", [])) == 0:
            continue
        ax.hist(result["errors"], bins=50, alpha=0.5, 
                label=f"{method.upper()} (μ={result['errors'].mean():.1f}mm)",
                color=colors.get(method, 'gray'), density=True)
    ax.set_xlabel("Match Error (mm)")
    ax.set_ylabel("Density")
    ax.set_title("Match Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Summary bars
    ax = axes[2]
    methods = [m for m in all_results if "error" not in all_results[m]]
    x = np.arange(len(methods))
    
    identity_pcts = [all_results[m]["identity_pct"] for m in methods]
    n_matches_pcts = [all_results[m]["n_matches"] / n_pts * 100 for m in methods]
    
    width = 0.35
    ax.bar(x - width/2, n_matches_pcts, width, label="% points matched", 
           color='steelblue', alpha=0.7)
    ax.bar(x + width/2, identity_pcts, width, label="% identity matches",
           color='seagreen', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in methods])
    ax.set_ylabel("Percentage")
    ax.set_title("Matching Efficiency")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f"Test D1: Matching Quality — {config.features.backend}, Pair {pair_idx} ({n_pts} pts)",
                 fontsize=13)
    plt.tight_layout()
    
    path = VIZ_DIR / f"D1_matching_{config.features.backend}_pair{pair_idx}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  📊 Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Test D: Matching Quality")
    parser.add_argument("--pair", type=int, default=0)
    parser.add_argument("--feature", type=str, default="dinov3", choices=["dinov3", "matcha", "mind"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    
    config = PipelineConfig()
    config.features.backend = args.feature
    config.device = args.device
    
    dataset = ThoraxCBCTDataset(config.paths.data_root, split="train")
    logger.info(f"Dataset: {dataset}")
    
    # Create feature extractor
    logger.info(f"Loading {args.feature} extractor...")
    if args.feature == "dinov3":
        from pipeline.features.dinov3_extractor import DINOv3Extractor
        extractor = DINOv3Extractor(
            repo_path=str(config.paths.dinov3_repo),
            weights_path=str(config.paths.dinov3_weights),
            device=args.device,
        )
    elif args.feature == "matcha":
        from pipeline.features.matcha_extractor import MATCHAExtractor
        extractor = MATCHAExtractor(
            repo_path=str(config.paths.matcha_repo),
            weights_path=str(config.paths.matcha_weights),
            device=args.device,
        )
    elif args.feature == "mind":
        from pipeline.features.mind_extractor import MINDExtractor
        extractor = MINDExtractor(device=args.device)

    if args.feature == "mind":
        fuser = extractor
    else:
        fuser = TriplanarFuser(
            extractor,
            batch_size=args.batch_size,
            fusion=config.features.fusion_method,
            downsample=args.downsample,
            device=args.device,
        )
    
    results = []
    r1 = test_D1_matching_at_gt_points(config, dataset, args.pair, fuser, args.downsample)
    if r1: results.append(r1)
    r2 = test_D2_matching_with_distractors(config, dataset, args.pair, fuser, args.downsample)
    if r2: results.append(r2)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY — TEST D: MATCHING ({args.feature})")
    logger.info(f"{'='*60}")
    for r in results:
        status = "✅" if r["passed"] else "❌"
        logger.info(f"  {status} {r['test']}")
        for method, mresult in r.get("results", {}).items():
            if "error" in mresult:
                logger.info(f"    {method}: ERROR")
            else:
                pck10 = mresult.get("pck", {}).get("pck@10mm", 0)
                if "identity_pct" in mresult:
                    logger.info(f"    {method}: {mresult['n_matches']} matches, "
                                f"PCK@10mm={pck10:.1f}%, "
                                f"identity={mresult['identity_pct']:.1f}%")
                else:
                    logger.info(f"    {method}: {mresult['n_matches']} matches, "
                                f"PCK@10mm={pck10:.1f}%")


if __name__ == "__main__":
    main()
