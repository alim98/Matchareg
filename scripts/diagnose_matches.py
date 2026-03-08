#!/usr/bin/env python3
"""
Quick diagnostic: are the matches spatially coherent?

Tests:
1. What fraction of matches point in roughly the same direction?
2. What if we filter to only spatially coherent matches?
3. Does the fitter work with GT correspondences (same setup)?
"""
import sys, os, logging, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pipeline.config import PipelineConfig
from pipeline.data.dataset_thoraxcbct import ThoraxCBCTDataset
from pipeline.data.preprocessing import robust_intensity_normalize, generate_trunk_mask
from pipeline.transform.fitter import DiffeomorphicFitter
from pipeline.eval.metrics import compute_tre

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

    initial_tre = compute_tre(mkp, fkp)
    logger.info(f"Initial TRE: {initial_tre['mean_tre']:.3f} mm")

    # ================================================================
    # TEST 1: Does the fitter work with 2000 GT keypoints?
    # ================================================================
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Fitter with 2000 GT keypoints")
    logger.info("="*60)

    rng = np.random.RandomState(42)
    idx = rng.choice(len(fkp), 2000, replace=False)
    fkp_sub = fkp[idx]
    mkp_sub = mkp[idx]

    fitter = DiffeomorphicFitter(
        volume_shape=(D, H, W),
        grid_spacings=[10.0, 6.0, 3.0],
        n_iters_per_level=200,
        lambda_smooth=1.0,
        lambda_jac=0.1,
        n_squaring_steps=7,
        lr=0.1,
        device=args.device,
    )

    weights = np.ones(2000, dtype=np.float64)
    t0 = time.time()
    displacement_gt = fitter.fit(
        matched_src=fkp_sub,
        matched_tgt=mkp_sub,
        weights=weights,
    )
    elapsed = time.time() - t0

    final_gt = compute_tre(mkp, fkp, displacement=displacement_gt)
    logger.info(f"  GT matches TRE: {initial_tre['mean_tre']:.3f} → {final_gt['mean_tre']:.3f} mm")
    logger.info(f"  Improvement: {(1 - final_gt['mean_tre']/initial_tre['mean_tre'])*100:+.1f}%")
    logger.info(f"  Displacement: max={displacement_gt.abs().max():.1f}, mean={displacement_gt.abs().mean():.1f}")
    logger.info(f"  Runtime: {elapsed:.1f}s")

    # ================================================================
    # TEST 2: Fitter with GT + 50% noise (random wrong matches)
    # ================================================================
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Fitter with 50% wrong matches")
    logger.info("="*60)

    n_good = 1000
    n_bad = 1000
    idx_good = rng.choice(len(fkp), n_good, replace=False)
    idx_bad_src = rng.choice(len(fkp), n_bad, replace=False)
    idx_bad_tgt = rng.choice(len(mkp), n_bad, replace=False)  # random pairing = wrong

    src_mixed = np.vstack([fkp[idx_good], fkp[idx_bad_src]])
    tgt_mixed = np.vstack([mkp[idx_good], mkp[idx_bad_tgt]])  # wrong targets for bad matches
    weights_mixed = np.ones(n_good + n_bad, dtype=np.float64)

    fitter2 = DiffeomorphicFitter(
        volume_shape=(D, H, W),
        grid_spacings=[10.0, 6.0, 3.0],
        n_iters_per_level=200,
        lambda_smooth=1.0,
        lambda_jac=0.1,
        n_squaring_steps=7,
        lr=0.1,
        device=args.device,
    )

    displacement_mixed = fitter2.fit(
        matched_src=src_mixed,
        matched_tgt=tgt_mixed,
        weights=weights_mixed,
    )

    final_mixed = compute_tre(mkp, fkp, displacement=displacement_mixed)
    logger.info(f"  50% wrong TRE: {initial_tre['mean_tre']:.3f} → {final_mixed['mean_tre']:.3f} mm")
    logger.info(f"  Displacement: max={displacement_mixed.abs().max():.1f}, mean={displacement_mixed.abs().mean():.1f}")

    # ================================================================
    # TEST 3: Fitter with 30% correct + 70% noise (realistic)
    # ================================================================
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Fitter with 30% correct + 70% wrong (realistic)")
    logger.info("="*60)

    n_good = 600
    n_bad = 1400
    idx_good = rng.choice(len(fkp), n_good, replace=False)
    idx_bad_src = rng.choice(len(fkp), n_bad, replace=False)
    idx_bad_tgt = rng.choice(len(mkp), n_bad, replace=False)

    src_noisy = np.vstack([fkp[idx_good], fkp[idx_bad_src]])
    tgt_noisy = np.vstack([mkp[idx_good], mkp[idx_bad_tgt]])
    weights_noisy = np.ones(n_good + n_bad, dtype=np.float64)

    fitter3 = DiffeomorphicFitter(
        volume_shape=(D, H, W),
        grid_spacings=[10.0, 6.0, 3.0],
        n_iters_per_level=200,
        lambda_smooth=1.0,
        lambda_jac=0.1,
        n_squaring_steps=7,
        lr=0.1,
        device=args.device,
    )

    displacement_noisy = fitter3.fit(
        matched_src=src_noisy,
        matched_tgt=tgt_noisy,
        weights=weights_noisy,
    )

    final_noisy = compute_tre(mkp, fkp, displacement=displacement_noisy)
    logger.info(f"  30% correct TRE: {initial_tre['mean_tre']:.3f} → {final_noisy['mean_tre']:.3f} mm")
    logger.info(f"  Displacement: max={displacement_noisy.abs().max():.1f}, mean={displacement_noisy.abs().mean():.1f}")

    # ================================================================
    # SUMMARY
    # ================================================================
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"  Initial TRE:        {initial_tre['mean_tre']:.3f} mm")
    logger.info(f"  100% GT (2000):     {final_gt['mean_tre']:.3f} mm")
    logger.info(f"  50% GT + 50% noise: {final_mixed['mean_tre']:.3f} mm")
    logger.info(f"  30% GT + 70% noise: {final_noisy['mean_tre']:.3f} mm")
    logger.info(f"  Pipeline result:    10.082 mm (2000 LAP matches)")


if __name__ == "__main__":
    main()
