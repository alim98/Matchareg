#!/usr/bin/env python3
"""
Test 15: Adam refinement diagnostic — now with proximity fix verification.

Tests:
  A. MIND convex only (baseline: 9.77mm)
  B. MIND convex + MIND Adam WITH proximity=1.0 (FIXED)
  B0. MIND convex + MIND Adam WITHOUT proximity (OLD, broken — should be worse)
  C. MIND convex + MATCHA Adam WITH proximity=1.0 (FIXED)
  D. Feature alignment check
  E. Proximity weight sweep (0.25, 0.5, 1.0, 2.0) for MIND Adam
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def eval_displacement(disp_np, fkp, mkp, label=""):
    from scipy.ndimage import map_coordinates as mc
    fixed_disp = np.zeros_like(fkp)
    for ax in range(3):
        fixed_disp[:, ax] = mc(
            disp_np[0, ax], [fkp[:, 0], fkp[:, 1], fkp[:, 2]],
            order=1, mode='nearest',
        )
    warped = fkp + fixed_disp
    errors = np.linalg.norm(warped - mkp, axis=1)
    tre = errors.mean()

    ideal = mkp - fkp
    logger.info(f"  [{label}] TRE = {tre:.3f} mm")
    for ax in range(3):
        corr = np.corrcoef(ideal[:, ax], fixed_disp[:, ax])[0, 1] if ideal[:, ax].std() > 0 else 0
        logger.info(
            f"    axis {ax}: ideal mean={ideal[:, ax].mean():+.2f} std={ideal[:, ax].std():.2f}, "
            f"actual mean={fixed_disp[:, ax].mean():+.2f} std={fixed_disp[:, ax].std():.2f}, "
            f"corr={corr:+.3f}"
        )
    return tre


def prepare_matcha_pca(config, data, D, H, W):
    from pipeline.features.triplanar_fuser import load_features
    from sklearn.decomposition import PCA
    import gc

    cache_fix = config.paths.feature_cache_dir / f"{data['fixed_id']}_{config.features.backend}.npz"
    cache_mov = config.paths.feature_cache_dir / f"{data['moving_id']}_{config.features.backend}.npz"

    if cache_fix.exists() and cache_mov.exists():
        logger.info("  Loading cached features...")
        fixed_feats, fixed_feat_shape, _ = load_features(cache_fix)
        moving_feats, moving_feat_shape, _ = load_features(cache_mov)
    else:
        raise RuntimeError("Feature cache not found - run feature extraction first")

    n_pca = 16
    C = fixed_feats.shape[0]

    flat_fix = fixed_feats.reshape(C, -1).T
    flat_mov = moving_feats.reshape(C, -1).T

    rng = np.random.RandomState(42)
    n_sub = min(50000, flat_fix.shape[0])
    fit_data = np.vstack([
        flat_fix[rng.choice(len(flat_fix), n_sub, replace=False)],
        flat_mov[rng.choice(len(flat_mov), n_sub, replace=False)],
    ])

    pca = PCA(n_components=n_pca)
    pca.fit(fit_data)
    del fit_data
    logger.info(f"  PCA: {C} -> {n_pca}, variance: {pca.explained_variance_ratio_.sum():.3f}")

    pca_fix = pca.transform(flat_fix).T.reshape(n_pca, *fixed_feat_shape)
    del flat_fix
    pca_mov = pca.transform(flat_mov).T.reshape(n_pca, *moving_feat_shape)
    del flat_mov, fixed_feats, moving_feats
    gc.collect()

    pca_fix_t = torch.from_numpy(pca_fix).unsqueeze(0).float()
    del pca_fix
    pca_mov_t = torch.from_numpy(pca_mov).unsqueeze(0).float()
    del pca_mov

    pca_fix_full = F.interpolate(pca_fix_t, size=(D, H, W), mode='trilinear', align_corners=False)
    del pca_fix_t
    pca_mov_full = F.interpolate(pca_mov_t, size=(D, H, W), mode='trilinear', align_corners=False)
    del pca_mov_t
    gc.collect()

    pca_fix_full /= (pca_fix_full.norm(dim=1, keepdim=True) + 1e-6)
    pca_mov_full /= (pca_mov_full.norm(dim=1, keepdim=True) + 1e-6)

    return pca_fix_full, pca_mov_full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    from pipeline.config import PipelineConfig
    from pipeline.data.dataset_thoraxcbct import ThoraxCBCTDataset
    from pipeline.data.preprocessing import generate_trunk_mask
    from pipeline.transform.mind_convex_adam import mind_convex_adam, convex_adam_on_features

    config = PipelineConfig()
    config.features.backend = "matcha"
    config.device = args.device
    dataset = ThoraxCBCTDataset(config.paths.data_root)

    data = dataset[args.pair]
    fixed_img = data["fixed_img"]
    moving_img = data["moving_img"]
    fkp = data["fixed_keypoints"]
    mkp = data["moving_keypoints"]
    D, H, W = fixed_img.shape

    initial_tre = np.linalg.norm(fkp - mkp, axis=1).mean()
    logger.info(f"Initial TRE: {initial_tre:.3f} mm")
    ideal = mkp - fkp
    for ax in range(3):
        logger.info(f"  ideal axis {ax}: mean={ideal[:, ax].mean():+.2f} std={ideal[:, ax].std():.2f}")

    results = {}

    logger.info("\n" + "=" * 60)
    logger.info("TEST A: MIND convex only (baseline)")
    logger.info("=" * 60)
    disp_mind = mind_convex_adam(
        fixed_img, moving_img, mind_r=1, mind_d=2,
        lambda_weight=0, grid_sp=4, disp_hw=4, n_iter_adam=0,
        grid_sp_adam=2, ic=True, device=args.device,
    )
    results['A'] = eval_displacement(disp_mind.cpu().numpy(), fkp, mkp, "MIND convex")

    logger.info("\n" + "=" * 60)
    logger.info("TEST B0: MIND Adam WITHOUT proximity (OLD broken behavior)")
    logger.info("=" * 60)
    disp_b0 = mind_convex_adam(
        fixed_img, moving_img, mind_r=1, mind_d=2,
        lambda_weight=1.25, grid_sp=4, disp_hw=4, n_iter_adam=300,
        grid_sp_adam=2, ic=True, device=args.device,
        proximity_weight=0.0,
    )
    results['B0'] = eval_displacement(disp_b0.cpu().numpy(), fkp, mkp, "MIND Adam (NO prox)")
    del disp_b0
    torch.cuda.empty_cache()

    logger.info("\n" + "=" * 60)
    logger.info("TEST B: MIND Adam WITH proximity=1.0 (FIXED)")
    logger.info("=" * 60)
    disp_b = mind_convex_adam(
        fixed_img, moving_img, mind_r=1, mind_d=2,
        lambda_weight=1.25, grid_sp=4, disp_hw=4, n_iter_adam=300,
        grid_sp_adam=2, ic=True, device=args.device,
        proximity_weight=1.0,
    )
    results['B'] = eval_displacement(disp_b.cpu().numpy(), fkp, mkp, "MIND Adam (prox=1.0)")
    del disp_b
    torch.cuda.empty_cache()

    logger.info("\n" + "=" * 60)
    logger.info("TEST C: MIND convex + MATCHA Adam WITH proximity=1.0")
    logger.info("=" * 60)
    logger.info("  Preparing MATCHA PCA features...")
    pca_fix, pca_mov = prepare_matcha_pca(config, data, D, H, W)
    disp_c = convex_adam_on_features(
        pca_fix, pca_mov, (D, H, W),
        grid_sp=4, disp_hw=4, lambda_weight=1.25, n_iter_adam=300,
        grid_sp_adam=2, ic=False, device=args.device,
        initial_disp=disp_mind,
        proximity_weight=1.0,
    )
    results['C'] = eval_displacement(disp_c.cpu().numpy(), fkp, mkp, "MIND+MATCHA Adam (prox=1.0)")
    del disp_c, pca_fix, pca_mov
    torch.cuda.empty_cache()

    logger.info("\n" + "=" * 60)
    logger.info("TEST D: Feature alignment check at GT keypoints")
    logger.info("=" * 60)
    pca_fix_d, pca_mov_d = prepare_matcha_pca(config, data, D, H, W)
    pca_fix_np = pca_fix_d.numpy()
    pca_mov_np = pca_mov_d.numpy()
    n_ch = pca_fix_np.shape[1]
    n_test = min(500, len(fkp))
    rng = np.random.RandomState(42)
    test_idx = rng.choice(len(fkp), n_test, replace=False)

    from scipy.ndimage import map_coordinates as mc
    feat_at_fkp = np.zeros((n_test, n_ch))
    feat_at_mkp = np.zeros((n_test, n_ch))
    feat_at_random = np.zeros((n_test, n_ch))
    for c in range(n_ch):
        feat_at_fkp[:, c] = mc(pca_fix_np[0, c], [fkp[test_idx, 0], fkp[test_idx, 1], fkp[test_idx, 2]], order=1, mode='nearest')
        feat_at_mkp[:, c] = mc(pca_mov_np[0, c], [mkp[test_idx, 0], mkp[test_idx, 1], mkp[test_idx, 2]], order=1, mode='nearest')
        rand_z = rng.uniform(0, D - 1, n_test)
        rand_y = rng.uniform(0, H - 1, n_test)
        rand_x = rng.uniform(0, W - 1, n_test)
        feat_at_random[:, c] = mc(pca_mov_np[0, c], [rand_z, rand_y, rand_x], order=1, mode='nearest')

    cos_correct = np.sum(feat_at_fkp * feat_at_mkp, axis=1)
    cos_random = np.sum(feat_at_fkp * feat_at_random, axis=1)

    logger.info(f"  Cosine(fix@fkp, mov@mkp) [correct pairs]: mean={cos_correct.mean():.3f} std={cos_correct.std():.3f}")
    logger.info(f"  Cosine(fix@fkp, mov@random) [random pairs]: mean={cos_random.mean():.3f} std={cos_random.std():.3f}")
    logger.info(f"  Separation: {cos_correct.mean() - cos_random.mean():.3f}")
    if cos_correct.mean() > cos_random.mean() + 0.05:
        logger.info("  PASS: Features can distinguish correct from random correspondences")
    else:
        logger.info("  FAIL: Features cannot distinguish correct from random correspondences")
    del pca_fix_d, pca_mov_d, pca_fix_np, pca_mov_np
    torch.cuda.empty_cache()

    logger.info("\n" + "=" * 60)
    logger.info("TEST E: Proximity weight sweep (MIND Adam)")
    logger.info("=" * 60)
    for pw in [0.25, 0.5, 2.0, 4.0]:
        logger.info(f"\n--- proximity_weight = {pw} ---")
        disp_e = mind_convex_adam(
            fixed_img, moving_img, mind_r=1, mind_d=2,
            lambda_weight=1.25, grid_sp=4, disp_hw=4, n_iter_adam=300,
            grid_sp_adam=2, ic=True, device=args.device,
            proximity_weight=pw,
        )
        results[f'E_pw{pw}'] = eval_displacement(
            disp_e.cpu().numpy(), fkp, mkp, f"MIND Adam (prox={pw})")
        del disp_e
        torch.cuda.empty_cache()

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Initial TRE:                          {initial_tre:.3f} mm")
    logger.info(f"  A.  MIND convex only:                 {results['A']:.3f} mm")
    logger.info(f"  B0. MIND Adam (NO proximity):         {results['B0']:.3f} mm "
                f"{'WORSE' if results['B0'] > results['A'] else 'BETTER'} than A")
    logger.info(f"  B.  MIND Adam (proximity=1.0):        {results['B']:.3f} mm "
                f"{'WORSE' if results['B'] > results['A'] else 'BETTER'} than A")
    logger.info(f"  C.  MIND+MATCHA Adam (proximity=1.0): {results['C']:.3f} mm "
                f"{'WORSE' if results['C'] > results['A'] else 'BETTER'} than A")
    for pw in [0.25, 0.5, 2.0, 4.0]:
        k = f'E_pw{pw}'
        logger.info(f"  E.  MIND Adam (proximity={pw}):       {results[k]:.3f} mm "
                    f"{'WORSE' if results[k] > results['A'] else 'BETTER'} than A")

    best_key = min(results, key=results.get)
    logger.info(f"\n  BEST: {best_key} = {results[best_key]:.3f} mm")


if __name__ == "__main__":
    main()
