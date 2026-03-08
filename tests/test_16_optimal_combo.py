#!/usr/bin/env python3
"""
Test 16: Find the optimal combination now that the proximity fix works.

Sweep: {MIND, MATCHA} features x {0.1, 0.25, 0.5} proximity weight
All starting from MIND convex initialization.
"""
import argparse
import logging
import sys
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

    if not (cache_fix.exists() and cache_mov.exists()):
        raise RuntimeError("Feature cache not found")

    logger.info("  Loading cached features...")
    fixed_feats, fixed_feat_shape, _ = load_features(cache_fix)
    moving_feats, moving_feat_shape, _ = load_features(cache_mov)

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

    results = {}

    logger.info("\n" + "=" * 60)
    logger.info("BASELINE: MIND convex only")
    logger.info("=" * 60)
    disp_mind = mind_convex_adam(
        fixed_img, moving_img, mind_r=1, mind_d=2,
        lambda_weight=0, grid_sp=4, disp_hw=4, n_iter_adam=0,
        grid_sp_adam=2, ic=True, device=args.device,
    )
    results['baseline'] = eval_displacement(disp_mind.cpu().numpy(), fkp, mkp, "MIND convex")

    prox_weights = [0.1, 0.25, 0.5]

    for pw in prox_weights:
        logger.info("\n" + "=" * 60)
        logger.info(f"MIND Adam (prox={pw})")
        logger.info("=" * 60)
        d = mind_convex_adam(
            fixed_img, moving_img, mind_r=1, mind_d=2,
            lambda_weight=1.25, grid_sp=4, disp_hw=4, n_iter_adam=300,
            grid_sp_adam=2, ic=True, device=args.device,
            proximity_weight=pw,
        )
        results[f'mind_pw{pw}'] = eval_displacement(d.cpu().numpy(), fkp, mkp, f"MIND Adam (pw={pw})")
        del d
        torch.cuda.empty_cache()

    logger.info("\n  Preparing MATCHA PCA features (once)...")
    pca_fix, pca_mov = prepare_matcha_pca(config, data, D, H, W)

    for pw in prox_weights:
        logger.info("\n" + "=" * 60)
        logger.info(f"MIND convex + MATCHA Adam (prox={pw})")
        logger.info("=" * 60)
        d = convex_adam_on_features(
            pca_fix, pca_mov, (D, H, W),
            grid_sp=4, disp_hw=4, lambda_weight=1.25, n_iter_adam=300,
            grid_sp_adam=2, ic=False, device=args.device,
            initial_disp=disp_mind,
            proximity_weight=pw,
        )
        results[f'matcha_pw{pw}'] = eval_displacement(d.cpu().numpy(), fkp, mkp, f"MATCHA Adam (pw={pw})")
        del d
        torch.cuda.empty_cache()

    del pca_fix, pca_mov
    torch.cuda.empty_cache()

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Initial TRE:         {initial_tre:.3f} mm")
    logger.info(f"  MIND convex only:    {results['baseline']:.3f} mm")
    for pw in prox_weights:
        mk = f'mind_pw{pw}'
        logger.info(f"  MIND Adam pw={pw}:   {results[mk]:.3f} mm "
                    f"{'BETTER' if results[mk] < results['baseline'] else 'WORSE'}")
    for pw in prox_weights:
        mk = f'matcha_pw{pw}'
        logger.info(f"  MATCHA Adam pw={pw}: {results[mk]:.3f} mm "
                    f"{'BETTER' if results[mk] < results['baseline'] else 'WORSE'}")

    best_key = min(results, key=results.get)
    logger.info(f"\n  BEST: {best_key} = {results[best_key]:.3f} mm")


if __name__ == "__main__":
    main()
