#!/usr/bin/env python3
"""
Diagnostic script: Why does rea pair registration fail?

Compares:
1. Image overlap/content similarity between Fixed (CBCT) and Moving (FBCT)
2. Feature matching quality with NN (no OT overhead)
3. Where keypoints live relative to masks
4. Whether features at known corresponding keypoints actually match
"""
import sys, os, logging, time
import numpy as np
import torch

# Make sure pipeline is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pipeline.config import PipelineConfig
from pipeline.data.dataset_thoraxcbct import ThoraxCBCTDataset
from pipeline.data.preprocessing import robust_intensity_normalize, generate_trunk_mask

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = PipelineConfig()
    dataset = ThoraxCBCTDataset(config.paths.data_root)
    data = dataset[args.pair]

    fixed_img = data["fixed_img"]
    moving_img = data["moving_img"]
    fkp = data["fixed_keypoints"]
    mkp = data["moving_keypoints"]

    D, H, W = fixed_img.shape
    logger.info(f"Volume shape: ({D}, {H}, {W})")
    logger.info(f"Keypoints: {len(fkp)}")

    # ================================================================
    # 1. Content overlap analysis
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 1: Content overlap")
    logger.info("=" * 60)

    fixed_mask = generate_trunk_mask(fixed_img)
    moving_mask = generate_trunk_mask(moving_img)

    overlap = (fixed_mask & moving_mask).sum()
    union = (fixed_mask | moving_mask).sum()
    dice = 2 * overlap / (fixed_mask.sum() + moving_mask.sum() + 1e-8)
    iou = overlap / (union + 1e-8)
    logger.info(f"  Fixed mask voxels:  {fixed_mask.sum():,} ({100*fixed_mask.mean():.1f}%)")
    logger.info(f"  Moving mask voxels: {moving_mask.sum():,} ({100*moving_mask.mean():.1f}%)")
    logger.info(f"  Overlap:            {overlap:,}")
    logger.info(f"  Dice:               {dice:.3f}")
    logger.info(f"  IoU:                {iou:.3f}")

    # Where is the mask content? Check z-range, y-range, x-range
    for name, mask in [("Fixed", fixed_mask), ("Moving", moving_mask)]:
        z_any = np.any(mask, axis=(1, 2))
        y_any = np.any(mask, axis=(0, 2))
        x_any = np.any(mask, axis=(0, 1))
        z_range = (np.argmax(z_any), D - 1 - np.argmax(z_any[::-1]))
        y_range = (np.argmax(y_any), H - 1 - np.argmax(y_any[::-1]))
        x_range = (np.argmax(x_any), W - 1 - np.argmax(x_any[::-1]))
        logger.info(f"  {name} mask extent: z=[{z_range[0]}, {z_range[1]}], "
                    f"y=[{y_range[0]}, {y_range[1]}], x=[{x_range[0]}, {x_range[1]}]")

    # ================================================================
    # 2. Keypoint distribution relative to masks
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 2: Keypoints vs masks")
    logger.info("=" * 60)

    fkp_in_fmask = sum(1 for p in fkp.astype(int)
                       if 0 <= p[0] < D and 0 <= p[1] < H and 0 <= p[2] < W
                       and fixed_mask[p[0], p[1], p[2]])
    fkp_in_mmask = sum(1 for p in fkp.astype(int)
                       if 0 <= p[0] < D and 0 <= p[1] < H and 0 <= p[2] < W
                       and moving_mask[p[0], p[1], p[2]])
    mkp_in_fmask = sum(1 for p in mkp.astype(int)
                       if 0 <= p[0] < D and 0 <= p[1] < H and 0 <= p[2] < W
                       and fixed_mask[p[0], p[1], p[2]])
    mkp_in_mmask = sum(1 for p in mkp.astype(int)
                       if 0 <= p[0] < D and 0 <= p[1] < H and 0 <= p[2] < W
                       and moving_mask[p[0], p[1], p[2]])

    logger.info(f"  Fixed KP in fixed mask:  {fkp_in_fmask}/{len(fkp)} ({100*fkp_in_fmask/len(fkp):.1f}%)")
    logger.info(f"  Fixed KP in moving mask: {fkp_in_mmask}/{len(fkp)} ({100*fkp_in_mmask/len(fkp):.1f}%)")
    logger.info(f"  Moving KP in fixed mask: {mkp_in_fmask}/{len(mkp)} ({100*mkp_in_fmask/len(mkp):.1f}%)")
    logger.info(f"  Moving KP in moving mask: {mkp_in_mmask}/{len(mkp)} ({100*mkp_in_mmask/len(mkp):.1f}%)")

    # Keypoint displacement stats
    kp_disp = mkp - fkp
    logger.info(f"\n  Keypoint displacements (mm):")
    logger.info(f"    Overall: mean={np.linalg.norm(kp_disp, axis=1).mean():.2f}, "
                f"max={np.linalg.norm(kp_disp, axis=1).max():.2f}")
    for ax, name in enumerate(["z", "y", "x"]):
        logger.info(f"    Axis {ax} ({name}): mean={kp_disp[:, ax].mean():+.2f}, "
                    f"std={kp_disp[:, ax].std():.2f}, "
                    f"range=[{kp_disp[:, ax].min():.1f}, {kp_disp[:, ax].max():.1f}]")

    # ================================================================
    # 3. Image similarity in overlap region
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 3: Image similarity")
    logger.info("=" * 60)

    overlap_mask = fixed_mask & moving_mask
    if overlap_mask.sum() > 0:
        f_vals = fixed_img[overlap_mask > 0]
        m_vals = moving_img[overlap_mask > 0]
        corr = np.corrcoef(f_vals, m_vals)[0, 1]
        mae = np.mean(np.abs(f_vals - m_vals))
        logger.info(f"  In overlap region ({overlap_mask.sum():,} voxels):")
        logger.info(f"    Correlation: {corr:.4f}")
        logger.info(f"    MAE:         {mae:.1f} HU")
        logger.info(f"    Fixed  mean={f_vals.mean():.1f}, std={f_vals.std():.1f}")
        logger.info(f"    Moving mean={m_vals.mean():.1f}, std={m_vals.std():.1f}")

    # ================================================================
    # 4. Feature cosine similarity at KNOWN corresponding keypoints
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 4: Feature similarity at known keypoints")
    logger.info("=" * 60)

    from pipeline.features.triplanar_fuser import TriplanarFuser, load_features
    from pipeline.features.dinov3_extractor import DINOv3Extractor

    # Load or extract features
    cache_f = config.paths.feature_cache_dir / f"{data['fixed_id']}_dinov3.npz"
    cache_m = config.paths.feature_cache_dir / f"{data['moving_id']}_dinov3.npz"

    if cache_f.exists() and cache_m.exists():
        logger.info("  Loading cached features...")
        fixed_feats, fixed_feat_shape, fixed_orig_shape = load_features(cache_f)
        moving_feats, moving_feat_shape, moving_orig_shape = load_features(cache_m)
    else:
        logger.info("  Extracting features (this takes ~2 min)...")
        extractor = DINOv3Extractor(
            repo_path=str(config.paths.dinov3_repo),
            weights_path=str(config.paths.dinov3_weights),
            device=args.device,
        )
        fuser = TriplanarFuser(extractor, batch_size=4, device=args.device)
        fixed_feats, fixed_feat_shape, fixed_orig_shape = fuser.fuse_triplanar(
            robust_intensity_normalize(fixed_img, mask=fixed_mask))
        moving_feats, moving_feat_shape, moving_orig_shape = fuser.fuse_triplanar(
            robust_intensity_normalize(moving_img, mask=moving_mask))

    logger.info(f"  Fixed features:  {fixed_feats.shape}")
    logger.info(f"  Moving features: {moving_feats.shape}")

    # Sample features at keypoint locations
    from pipeline.scripts.run_pipeline import voxel_to_feature_coords
    from pipeline.matching.sampling import sample_descriptors_at_points

    downsample = 2

    # Convert keypoints to feature coords
    fkp_feat = voxel_to_feature_coords.__wrapped__(fkp, fixed_orig_shape, fixed_feat_shape, downsample)
    mkp_feat = voxel_to_feature_coords.__wrapped__(mkp, moving_orig_shape, moving_feat_shape, downsample)

    # Sample descriptors at keypoints
    f_desc = sample_descriptors_at_points(fixed_feats, fkp_feat)
    m_desc = sample_descriptors_at_points(moving_feats, mkp_feat)

    # L2 normalize
    f_desc = f_desc / (np.linalg.norm(f_desc, axis=1, keepdims=True) + 1e-8)
    m_desc = m_desc / (np.linalg.norm(m_desc, axis=1, keepdims=True) + 1e-8)

    # Cosine similarity between CORRESPONDING keypoints
    cos_corresponding = np.sum(f_desc * m_desc, axis=1)

    # NN matching on keypoints
    n_test = min(2000, len(f_desc))
    idx = np.random.RandomState(42).choice(len(f_desc), n_test, replace=False)
    f_sub = f_desc[idx]
    m_sub = m_desc[idx]
    sim_matrix = f_sub @ m_sub.T  # (n_test, n_test)

    # For each fixed keypoint, find best match in moving
    best_fwd = np.argmax(sim_matrix, axis=1)
    best_bwd = np.argmax(sim_matrix, axis=0)

    # How many are correct (i.e., best match = corresponding keypoint)?
    correct_fwd = np.sum(best_fwd == np.arange(n_test))
    mutual = np.sum((best_fwd == np.arange(n_test)) & (best_bwd[best_fwd] == np.arange(n_test)))

    # Top-k accuracy
    for k in [1, 5, 10, 50]:
        topk_idx = np.argsort(-sim_matrix, axis=1)[:, :k]
        topk_correct = sum(1 for i in range(n_test) if i in topk_idx[i])
        logger.info(f"  Top-{k} accuracy: {topk_correct}/{n_test} ({100*topk_correct/n_test:.1f}%)")

    logger.info(f"\n  Cosine sim at CORRESPONDING keypoints:")
    logger.info(f"    mean={cos_corresponding.mean():.4f}, std={cos_corresponding.std():.4f}")
    logger.info(f"    min={cos_corresponding.min():.4f}, max={cos_corresponding.max():.4f}")
    logger.info(f"    <0.5: {(cos_corresponding < 0.5).sum()} ({100*(cos_corresponding < 0.5).mean():.1f}%)")
    logger.info(f"    <0.3: {(cos_corresponding < 0.3).sum()} ({100*(cos_corresponding < 0.3).mean():.1f}%)")

    logger.info(f"\n  NN matching on {n_test} keypoints:")
    logger.info(f"    Forward correct: {correct_fwd}/{n_test} ({100*correct_fwd/n_test:.1f}%)")
    logger.info(f"    Mutual correct:  {mutual}/{n_test} ({100*mutual/n_test:.1f}%)")

    # ================================================================
    # 5. NN matching on random sampled points (same as pipeline)
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 5: NN matching on 2000 random points (pipeline-style)")
    logger.info("=" * 60)

    from pipeline.matching.sampling import sample_points_in_mask, sample_descriptors_at_points

    rng = np.random.RandomState(42)
    fixed_pts = sample_points_in_mask(fixed_mask, 2000, z_stratified=True, rng=rng)
    rng_m = np.random.RandomState(123)
    moving_pts = sample_points_in_mask(moving_mask, 2000, z_stratified=True, rng=rng_m)

    fixed_feat_pts = voxel_to_feature_coords.__wrapped__(fixed_pts, fixed_orig_shape, fixed_feat_shape, downsample)
    moving_feat_pts = voxel_to_feature_coords.__wrapped__(moving_pts, moving_orig_shape, moving_feat_shape, downsample)

    fixed_desc_rand = sample_descriptors_at_points(fixed_feats, fixed_feat_pts)
    moving_desc_rand = sample_descriptors_at_points(moving_feats, moving_feat_pts)

    fixed_desc_rand = fixed_desc_rand / (np.linalg.norm(fixed_desc_rand, axis=1, keepdims=True) + 1e-8)
    moving_desc_rand = moving_desc_rand / (np.linalg.norm(moving_desc_rand, axis=1, keepdims=True) + 1e-8)

    sim_rand = fixed_desc_rand @ moving_desc_rand.T
    logger.info(f"  Sim matrix: mean={sim_rand.mean():.4f}, max={sim_rand.max():.4f}")

    # NN matching
    fwd = np.argmax(sim_rand, axis=1)
    bwd = np.argmax(sim_rand, axis=0)
    mutual_mask = bwd[fwd] == np.arange(2000)
    n_mutual = mutual_mask.sum()
    logger.info(f"  Mutual NN matches: {n_mutual}/2000")

    # Distance of mutual matches
    if n_mutual > 0:
        matched_f = fixed_pts[mutual_mask]
        matched_m = moving_pts[fwd[mutual_mask]]
        dists = np.linalg.norm(matched_f - matched_m, axis=1)
        logger.info(f"  Match distances: mean={dists.mean():.1f}, std={dists.std():.1f}, "
                    f"max={dists.max():.1f}")
        logger.info(f"  Matches < 20mm: {(dists < 20).sum()}")
        logger.info(f"  Matches < 50mm: {(dists < 50).sum()}")

    logger.info("\n" + "=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
