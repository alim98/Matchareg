#!/usr/bin/env python3
"""
Test C: Feature Quality / Discriminability
============================================

Tests whether DINOv3/MATCHA features extracted via tri-planar fusion
carry any correspondence signal on CT/CBCT data.

This is the SINGLE MOST IMPORTANT diagnostic — if features are not
discriminative, matching and fitting cannot possibly work.

Tests:
  C1. GT similarity separation (positive vs negative pairs)
  C2. PCA visualization of feature volumes

Usage:
    python -m pipeline.tests.test_C_features --pair 0
    python -m pipeline.tests.test_C_features --pair 0 --feature matcha
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
from pipeline.data.preprocessing import robust_intensity_normalize, generate_trunk_mask
from pipeline.features.triplanar_fuser import TriplanarFuser, save_features, load_features
from pipeline.matching.sampling import sample_descriptors_at_points

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

VIZ_DIR = PROJECT_ROOT / "pipeline" / "tests" / "results" / "test_C"
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
    feat_points[:, 0] = feat_points[:, 0] / downsample
    feat_points[:, 1] = feat_points[:, 1] / downsample
    feat_points[:, 2] = feat_points[:, 2] / downsample
    
    dD = D // downsample
    dH = H // downsample
    dW = W // downsample
    
    feat_points[:, 0] = feat_points[:, 0] * fD / dD
    feat_points[:, 1] = feat_points[:, 1] * fH / dH
    feat_points[:, 2] = feat_points[:, 2] * fW / dW
    
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


def test_C1_similarity_separation(config, dataset, pair_idx, fuser, downsample=2):
    """
    C1: GT similarity separation.
    
    Compare cosine similarity for:
      - Positive pairs: (fixed_kp[k], moving_kp[k])  — known correspondences
      - Negative pairs: (fixed_kp[k], moving_kp[random]) — random non-matches
    
    Features are discriminative iff positive >> negative.
    
    Pass: mean_positive > 0.4, mean_negative < 0.35, separation > 0.2,
    and top-1 NN retrieval is meaningfully above chance.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST C1: GT Similarity Separation (pair {pair_idx})")
    logger.info(f"{'='*60}")
    
    data = dataset[pair_idx]
    fkp = data["fixed_keypoints"]
    mkp = data["moving_keypoints"]
    
    if fkp is None or mkp is None:
        logger.warning(f"  SKIP: No keypoints")
        return None
    
    # Extract features
    logger.info(f"  Extracting features ({config.features.backend})...")
    t0 = time.time()
    
    fixed_mask = generate_trunk_mask(data["fixed_img"])
    moving_mask = generate_trunk_mask(data["moving_img"])
    
    fixed_result = _extract_features(config, data["fixed_img"], data["fixed_id"], fuser, fixed_mask)
    moving_result = _extract_features(config, data["moving_img"], data["moving_id"], fuser, moving_mask)
    
    fixed_feats, fixed_feat_shape, fixed_orig_shape = fixed_result
    moving_feats, moving_feat_shape, moving_orig_shape = moving_result
    
    logger.info(f"  Feature extraction: {time.time()-t0:.1f}s")
    logger.info(f"  Fixed feat shape: {fixed_feats.shape}, grid: {fixed_feat_shape}")
    logger.info(f"  Moving feat shape: {moving_feats.shape}, grid: {moving_feat_shape}")
    
    # Convert keypoints to feature coords
    fkp_feat = _voxel_to_feature_coords(fkp, fixed_orig_shape, fixed_feat_shape, downsample)
    mkp_feat = _voxel_to_feature_coords(mkp, moving_orig_shape, moving_feat_shape, downsample)
    
    # Sample descriptors at keypoints
    fixed_desc = sample_descriptors_at_points(fixed_feats, fkp_feat)
    moving_desc = sample_descriptors_at_points(moving_feats, mkp_feat)
    
    # L2 normalize
    fixed_desc = fixed_desc / (np.linalg.norm(fixed_desc, axis=1, keepdims=True) + 1e-8)
    moving_desc = moving_desc / (np.linalg.norm(moving_desc, axis=1, keepdims=True) + 1e-8)
    
    logger.info(f"  Descriptors: fixed {fixed_desc.shape}, moving {moving_desc.shape}")
    logger.info(f"  Descriptor stats: fixed norm mean={np.linalg.norm(fixed_desc, axis=1).mean():.3f}, "
                f"moving norm mean={np.linalg.norm(moving_desc, axis=1).mean():.3f}")
    
    # Positive pairs: (fixed_kp[k], moving_kp[k])
    n_kp = len(fkp)
    positive_sims = np.array([
        np.dot(fixed_desc[k], moving_desc[k]) for k in range(n_kp)
    ])
    
    # Negative pairs: (fixed_kp[k], moving_kp[random])
    rng = np.random.RandomState(42)
    n_neg = min(n_kp, 5000)
    neg_sims = []
    for k in range(n_neg):
        rand_idx = rng.randint(0, n_kp)
        while rand_idx == k:
            rand_idx = rng.randint(0, n_kp)
        neg_sims.append(np.dot(fixed_desc[k], moving_desc[rand_idx]))
    negative_sims = np.array(neg_sims)
    
    # Also compute the full similarity matrix for a subset
    n_sub = min(500, n_kp)
    sim_matrix = fixed_desc[:n_sub] @ moving_desc[:n_sub].T
    
    # Diagonal = positive, off-diagonal = negative
    diag_sims = np.diag(sim_matrix)
    off_diag_mean = (sim_matrix.sum() - diag_sims.sum()) / (n_sub * n_sub - n_sub)
    
    stats = {
        "positive_mean": float(positive_sims.mean()),
        "positive_std": float(positive_sims.std()),
        "positive_median": float(np.median(positive_sims)),
        "negative_mean": float(negative_sims.mean()),
        "negative_std": float(negative_sims.std()),
        "negative_median": float(np.median(negative_sims)),
        "separation": float(positive_sims.mean() - negative_sims.mean()),
        "matrix_diag_mean": float(diag_sims.mean()),
        "matrix_offdiag_mean": float(off_diag_mean),
    }
    
    logger.info(f"\n  === SIMILARITY STATISTICS ===")
    logger.info(f"  Positive pairs: mean={stats['positive_mean']:.4f}, "
                f"std={stats['positive_std']:.4f}, median={stats['positive_median']:.4f}")
    logger.info(f"  Negative pairs: mean={stats['negative_mean']:.4f}, "
                f"std={stats['negative_std']:.4f}, median={stats['negative_median']:.4f}")
    logger.info(f"  Separation: {stats['separation']:.4f}")
    logger.info(f"  Sim matrix diag mean: {stats['matrix_diag_mean']:.4f}")
    logger.info(f"  Sim matrix off-diag mean: {stats['matrix_offdiag_mean']:.4f}")
    
    # How many positive pairs beat the median negative?
    pct_positive_above_neg_median = (positive_sims > stats['negative_median']).mean() * 100
    logger.info(f"  % positive > negative median: {pct_positive_above_neg_median:.1f}%")
    
    # Nearest-neighbor retrieval accuracy
    # For each fixed keypoint, is the correct moving keypoint the nearest neighbor?
    nn_correct = 0
    for k in range(n_sub):
        nn_idx = np.argmax(sim_matrix[k])
        if nn_idx == k:
            nn_correct += 1
    nn_accuracy = nn_correct / n_sub * 100
    logger.info(f"  NN retrieval accuracy (top-1, {n_sub} pts): {nn_accuracy:.1f}%")
    stats["nn_accuracy"] = nn_accuracy
    
    # Pass/fail
    passed_positive = stats['positive_mean'] > 0.4
    passed_negative = stats['negative_mean'] < 0.35
    passed_separation = stats['separation'] > 0.2
    passed_nn = stats['nn_accuracy'] > 10.0
    passed = passed_positive and passed_negative and passed_separation and passed_nn
    
    logger.info(f"\n  Positive mean > 0.4? {'✅' if passed_positive else '❌'} ({stats['positive_mean']:.4f})")
    logger.info(f"  Negative mean < 0.35? {'✅' if passed_negative else '❌'} ({stats['negative_mean']:.4f})")
    logger.info(f"  Separation > 0.2?   {'✅' if passed_separation else '❌'} ({stats['separation']:.4f})")
    logger.info(f"  NN acc > 10%?       {'✅' if passed_nn else '❌'} ({stats['nn_accuracy']:.1f}%)")
    logger.info(f"  {'✅ PASS' if passed else '❌ FAIL'} — "
                f"Features {'are' if passed else 'ARE NOT'} discriminative")
    
    if not passed:
        logger.warning("  🚨 Features don't transfer to CT/CBCT. Matching cannot work.")
        logger.warning("  Consider: MIND features, higher resolution, or dense alignment.")
    
    # Visualization
    _visualize_C1(positive_sims, negative_sims, sim_matrix, stats, pair_idx, config)
    _visualize_C1_pca(fixed_feats, moving_feats, data["fixed_img"], data["moving_img"], 
                      pair_idx, config)
    
    return {
        "test": "C1_similarity_separation",
        "pair": pair_idx,
        "backend": config.features.backend,
        **stats,
        "passed": passed,
    }


def _visualize_C1(pos_sims, neg_sims, sim_matrix, stats, pair_idx, config):
    """Plot similarity distributions."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Histogram of similarities
    ax = axes[0]
    ax.hist(pos_sims, bins=60, alpha=0.6, label=f"Positive (μ={pos_sims.mean():.3f})", 
            color='green', density=True)
    ax.hist(neg_sims, bins=60, alpha=0.6, label=f"Negative (μ={neg_sims.mean():.3f})", 
            color='red', density=True)
    ax.axvline(pos_sims.mean(), color='darkgreen', linestyle='--', linewidth=2)
    ax.axvline(neg_sims.mean(), color='darkred', linestyle='--', linewidth=2)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title(f"GT vs Random Similarity\n(separation={stats['separation']:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Similarity matrix (subset)
    ax = axes[1]
    n_show = min(100, sim_matrix.shape[0])
    im = ax.imshow(sim_matrix[:n_show, :n_show], cmap='viridis', aspect='auto',
                   vmin=-0.2, vmax=0.8)
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Moving keypoint index")
    ax.set_ylabel("Fixed keypoint index")
    ax.set_title(f"Similarity Matrix ({n_show}×{n_show})\nDiagonal should be bright")
    
    # 3. Per-keypoint: positive similarity vs rank of positive among all targets
    ax = axes[2]
    n_sub = min(300, sim_matrix.shape[0])
    ranks = []
    for k in range(n_sub):
        row = sim_matrix[k]
        rank = (row > row[k]).sum()  # how many targets have higher similarity
        ranks.append(rank)
    ranks = np.array(ranks)
    
    ax.hist(ranks, bins=50, color='steelblue', alpha=0.7)
    ax.axvline(0, color='green', linewidth=2, linestyle='--', label='Rank 0 = perfect')
    ax.set_xlabel("Rank of GT match (0 = best)")
    ax.set_ylabel("Count")
    ax.set_title(f"Retrieval Rank Distribution\n(rank 0: {(ranks==0).sum()}/{n_sub} = {(ranks==0).mean()*100:.1f}%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"Test C1: Feature Discriminability — {config.features.backend}, Pair {pair_idx}",
                 fontsize=13)
    plt.tight_layout()
    
    path = VIZ_DIR / f"C1_similarity_{config.features.backend}_pair{pair_idx}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  📊 Saved: {path}")


def _visualize_C1_pca(fixed_feats, moving_feats, fixed_img, moving_img, pair_idx, config):
    """PCA visualization of feature volumes — shows if features are anatomically meaningful."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        logger.warning("  matplotlib/sklearn not available — skipping PCA viz")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for vol_idx, (feats, img, label) in enumerate([
        (fixed_feats, fixed_img, "Fixed"),
        (moving_feats, moving_img, "Moving"),
    ]):
        # feats shape: (C, D, H, W) — take mid slice along each axis
        C, fD, fH, fW = feats.shape
        
        for ax_idx, (axis_name, sl_idx, dims) in enumerate([
            ("Axial (z)", fD // 2, (1, 2, 3)),    # mid-z: features at (z, :, :)
            ("Coronal (y)", fH // 2, (1, 2, 3)),   # mid-y
            ("Sagittal (x)", fW // 2, (1, 2, 3)),  # mid-x
        ]):
            ax = axes[vol_idx, ax_idx]
            
            # Extract 2D feature slice
            if ax_idx == 0:  # axial
                feat_2d = feats[:, sl_idx, :, :]  # (C, H, W)
            elif ax_idx == 1:  # coronal
                feat_2d = feats[:, :, sl_idx, :]  # (C, D, W)
            else:  # sagittal
                feat_2d = feats[:, :, :, sl_idx]  # (C, D, H)
            
            h, w = feat_2d.shape[1], feat_2d.shape[2]
            pixels = feat_2d.reshape(C, -1).T  # (H*W, C)
            
            # PCA to 3 components for RGB
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(pixels)  # (H*W, 3)
            
            # Normalize to [0, 1] for display
            for c in range(3):
                v = pca_result[:, c]
                lo, hi = np.percentile(v, [2, 98])
                pca_result[:, c] = np.clip((v - lo) / (hi - lo + 1e-8), 0, 1)
            
            rgb = pca_result.reshape(h, w, 3)
            ax.imshow(rgb)
            ax.set_title(f"{label} — {axis_name}\n(PCA variance: {pca.explained_variance_ratio_.sum():.2f})")
            ax.axis('off')
    
    fig.suptitle(f"Test C1: Feature PCA — {config.features.backend}, Pair {pair_idx}\n"
                 "Anatomical structures should be visible as distinct colors",
                 fontsize=13)
    plt.tight_layout()
    
    path = VIZ_DIR / f"C1_pca_{config.features.backend}_pair{pair_idx}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  📊 Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Test C: Feature Quality")
    parser.add_argument("--pair", type=int, default=0)
    parser.add_argument("--feature", type=str, default="dinov3", choices=["dinov3", "matcha", "mind"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    
    config = PipelineConfig()
    config.features.backend = args.feature
    config.features.use_cache = False
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
    r1 = test_C1_similarity_separation(config, dataset, args.pair, fuser, args.downsample)
    if r1: results.append(r1)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY — TEST C: FEATURES ({args.feature})")
    logger.info(f"{'='*60}")
    for r in results:
        status = "✅" if r["passed"] else "❌"
        logger.info(f"  {status} {r['test']}: "
                    f"pos={r['positive_mean']:.3f}, neg={r['negative_mean']:.3f}, "
                    f"sep={r['separation']:.3f}, NN acc={r.get('nn_accuracy', 0):.1f}%")
    
    n_pass = sum(1 for r in results if r["passed"])
    if n_pass < len(results):
        logger.error("  ⚠️  FEATURES ARE NOT DISCRIMINATIVE — matching will not work!")
        sys.exit(1)
    else:
        logger.info("  ✅ Features are discriminative — proceed to Test D (matching)")


if __name__ == "__main__":
    main()
