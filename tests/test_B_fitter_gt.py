#!/usr/bin/env python3
"""
Test B: SVF Fitter on Ground-Truth Correspondences
====================================================

Feeds the diffeomorphic fitter PERFECT correspondences (GT keypoints)
and checks whether it can recover a displacement that drastically reduces TRE.

This completely isolates the fitter from feature quality and matching quality.
If this fails → the SVF fitting is broken (direction, scaling, optimization).

Usage:
    python -m pipeline.tests.test_B_fitter_gt --pair 0
    python -m pipeline.tests.test_B_fitter_gt --pair 0 --device cpu
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
from pipeline.eval.metrics import compute_tre, compute_jacobian_stats
from pipeline.transform.fitter import DiffeomorphicFitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

VIZ_DIR = PROJECT_ROOT / "pipeline" / "tests" / "results" / "test_B"
VIZ_DIR.mkdir(parents=True, exist_ok=True)


def test_B1_fitter_with_all_gt(dataset, pair_idx, device="cuda"):
    """
    B1: Feed ALL GT keypoints as correspondences and fit SVF.
    
    Pass: TRE < 5mm (from ~25mm initial)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST B1: Fitter with ALL GT keypoints (pair {pair_idx})")
    logger.info(f"{'='*60}")
    
    data = dataset[pair_idx]
    fkp = data["fixed_keypoints"]
    mkp = data["moving_keypoints"]
    fixed_img = data["fixed_img"]
    
    if fkp is None or mkp is None:
        logger.warning(f"  SKIP: No keypoints for pair {pair_idx}")
        return None
    
    D, H, W = fixed_img.shape
    logger.info(f"  Volume shape: ({D}, {H}, {W})")
    logger.info(f"  GT keypoints: {len(fkp)} pairs")
    
    # Initial TRE
    initial = compute_tre(mkp, fkp, displacement=None)
    logger.info(f"  Initial TRE: {initial['mean_tre']:.3f} mm")
    
    # Fit SVF with GT correspondences
    # The fitter expects: matched_src = field origin, matched_tgt = field target
    # Our displacement convention: d(x_fixed) = x_moving - x_fixed
    # So: matched_src = fkp (origin in fixed space), matched_tgt = mkp (target in moving space)
    logger.info(f"  Fitting SVF with GT keypoints...")
    
    fitter = DiffeomorphicFitter(
        volume_shape=(D, H, W),
        grid_spacings=[10.0, 6.0, 3.0],
        n_iters_per_level=200,
        n_squaring_steps=7,
        device=device,
    )
    
    weights = np.ones(len(fkp), dtype=np.float64)
    t0 = time.time()
    displacement = fitter.fit(
        matched_src=fkp,    # fixed keypoints (field origin)
        matched_tgt=mkp,    # moving keypoints (field target)
        weights=weights,
    )
    elapsed = time.time() - t0
    
    logger.info(f"  Fitting completed in {elapsed:.1f}s")
    logger.info(f"  Displacement: max={displacement.abs().max().item():.1f}, "
                f"mean={displacement.abs().mean().item():.3f}")
    
    # Final TRE
    final = compute_tre(mkp, fkp, displacement=displacement)
    improvement = (1 - final['mean_tre'] / initial['mean_tre']) * 100
    
    logger.info(f"  Final TRE: {final['mean_tre']:.3f} mm")
    logger.info(f"  Improvement: {improvement:+.1f}%")
    
    # Jacobian stats
    jac = compute_jacobian_stats(displacement)
    logger.info(f"  Jacobian: mean={jac['jac_mean']:.3f}, "
                f"min={jac['jac_min']:.3f}, %neg={jac['jac_pct_negative']:.2f}%")
    
    # Per-axis analysis
    from scipy.ndimage import map_coordinates as mc
    disp_np = displacement.cpu().numpy()
    fixed_disp = np.zeros_like(fkp)
    for ax in range(3):
        fixed_disp[:, ax] = mc(
            disp_np[0, ax],
            [fkp[:, 0], fkp[:, 1], fkp[:, 2]],
            order=1, mode='nearest',
        )
    
    ideal_disp = mkp - fkp
    axis_names = ["z", "y", "x"]
    correlations = []
    for ax in range(3):
        ideal = ideal_disp[:, ax]
        actual = fixed_disp[:, ax]
        corr = np.corrcoef(ideal, actual)[0, 1] if ideal.std() > 0 else 0
        correlations.append(corr)
        logger.info(f"  Axis {ax} ({axis_names[ax]}): "
                    f"ideal range=[{ideal.min():.1f}, {ideal.max():.1f}], "
                    f"actual range=[{actual.min():.1f}, {actual.max():.1f}], "
                    f"corr={corr:+.3f}")
    
    passed = final['mean_tre'] < 5.0
    logger.info(f"  {'✅ PASS' if passed else '❌ FAIL'} — "
                f"TRE {'<' if passed else '>'} 5mm with GT correspondences")
    
    # Visualization
    _visualize_B1(fkp, mkp, fixed_disp, ideal_disp, initial, final, jac, pair_idx)
    
    return {
        "test": "B1_fitter_all_gt",
        "pair": pair_idx,
        "n_keypoints": len(fkp),
        "initial_tre": initial['mean_tre'],
        "final_tre": final['mean_tre'],
        "improvement_pct": improvement,
        "axis_correlations": correlations,
        "jac_pct_negative": jac['jac_pct_negative'],
        "runtime_s": elapsed,
        "passed": passed,
    }


def test_B2_fitter_with_subset_gt(dataset, pair_idx, n_subset=2000, device="cuda"):
    """
    B2: Feed a SUBSET of GT keypoints (realistic match count) and fit SVF.
    
    Tests whether the fitter works with the number of correspondences
    that GWOT would typically produce.
    
    Pass: TRE < 8mm
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST B2: Fitter with {n_subset} GT keypoints (pair {pair_idx})")
    logger.info(f"{'='*60}")
    
    data = dataset[pair_idx]
    fkp = data["fixed_keypoints"]
    mkp = data["moving_keypoints"]
    fixed_img = data["fixed_img"]
    
    if fkp is None or mkp is None:
        logger.warning(f"  SKIP: No keypoints for pair {pair_idx}")
        return None
    
    D, H, W = fixed_img.shape
    
    # Subsample
    rng = np.random.RandomState(42)
    if len(fkp) > n_subset:
        idx = rng.choice(len(fkp), n_subset, replace=False)
        fkp_sub = fkp[idx]
        mkp_sub = mkp[idx]
    else:
        fkp_sub = fkp
        mkp_sub = mkp
    
    logger.info(f"  Using {len(fkp_sub)}/{len(fkp)} GT keypoints")
    
    # Initial TRE (full set)
    initial = compute_tre(mkp, fkp, displacement=None)
    logger.info(f"  Initial TRE (full): {initial['mean_tre']:.3f} mm")
    
    # Fit SVF
    fitter = DiffeomorphicFitter(
        volume_shape=(D, H, W),
        grid_spacings=[10.0, 6.0, 3.0],
        n_iters_per_level=200,
        n_squaring_steps=7,
        device=device,
    )
    
    weights = np.ones(len(fkp_sub), dtype=np.float64)
    t0 = time.time()
    displacement = fitter.fit(
        matched_src=fkp_sub,
        matched_tgt=mkp_sub,
        weights=weights,
    )
    elapsed = time.time() - t0
    
    # Evaluate TRE on FULL keypoint set (not just subset used for fitting)
    final = compute_tre(mkp, fkp, displacement=displacement)
    improvement = (1 - final['mean_tre'] / initial['mean_tre']) * 100
    
    logger.info(f"  Final TRE (full set): {final['mean_tre']:.3f} mm")
    logger.info(f"  Improvement: {improvement:+.1f}%")
    logger.info(f"  Runtime: {elapsed:.1f}s")
    
    passed = final['mean_tre'] < 8.0
    logger.info(f"  {'✅ PASS' if passed else '❌ FAIL'} — "
                f"TRE {'<' if passed else '>'} 8mm with {n_subset} GT correspondences")
    
    return {
        "test": "B2_fitter_subset_gt",
        "pair": pair_idx,
        "n_subset": len(fkp_sub),
        "initial_tre": initial['mean_tre'],
        "final_tre": final['mean_tre'],
        "improvement_pct": improvement,
        "runtime_s": elapsed,
        "passed": passed,
    }


def test_B3_fitter_with_noisy_gt(dataset, pair_idx, noise_std=10.0, device="cuda"):
    """
    B3: Feed GT keypoints with Gaussian noise added.
    
    Simulates what happens when matching is imperfect — the target
    points are offset from their true positions by noise_std mm.
    
    This tests the fitter's robustness to noisy correspondences.
    
    Pass: TRE still improves significantly despite noise.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST B3: Fitter with noisy GT (σ={noise_std}mm, pair {pair_idx})")
    logger.info(f"{'='*60}")
    
    data = dataset[pair_idx]
    fkp = data["fixed_keypoints"]
    mkp = data["moving_keypoints"]
    fixed_img = data["fixed_img"]
    
    if fkp is None or mkp is None:
        logger.warning(f"  SKIP: No keypoints for pair {pair_idx}")
        return None
    
    D, H, W = fixed_img.shape
    rng = np.random.RandomState(42)
    
    # Subsample + add noise to targets
    n_pts = min(2000, len(fkp))
    idx = rng.choice(len(fkp), n_pts, replace=False)
    fkp_sub = fkp[idx]
    mkp_noisy = mkp[idx] + rng.randn(n_pts, 3) * noise_std
    
    # Clip to volume bounds
    mkp_noisy[:, 0] = np.clip(mkp_noisy[:, 0], 0, D - 1)
    mkp_noisy[:, 1] = np.clip(mkp_noisy[:, 1], 0, H - 1)
    mkp_noisy[:, 2] = np.clip(mkp_noisy[:, 2], 0, W - 1)
    
    initial = compute_tre(mkp, fkp, displacement=None)
    logger.info(f"  Initial TRE: {initial['mean_tre']:.3f} mm")
    logger.info(f"  Using {n_pts} keypoints with noise σ={noise_std}mm")
    
    fitter = DiffeomorphicFitter(
        volume_shape=(D, H, W),
        grid_spacings=[10.0, 6.0, 3.0],
        n_iters_per_level=200,
        n_squaring_steps=7,
        device=device,
    )
    
    weights = np.ones(n_pts, dtype=np.float64)
    displacement = fitter.fit(
        matched_src=fkp_sub,
        matched_tgt=mkp_noisy,
        weights=weights,
    )
    
    final = compute_tre(mkp, fkp, displacement=displacement)
    improvement = (1 - final['mean_tre'] / initial['mean_tre']) * 100
    
    logger.info(f"  Final TRE: {final['mean_tre']:.3f} mm")
    logger.info(f"  Improvement: {improvement:+.1f}%")
    
    passed = improvement > 20.0
    logger.info(f"  {'✅ PASS' if passed else '❌ FAIL'} — "
                f"{'>' if passed else '<'} 20% improvement with noisy correspondences")
    
    return {
        "test": "B3_fitter_noisy_gt",
        "pair": pair_idx,
        "noise_std": noise_std,
        "initial_tre": initial['mean_tre'],
        "final_tre": final['mean_tre'],
        "improvement_pct": improvement,
        "passed": passed,
    }


def _visualize_B1(fkp, mkp, actual_disp, ideal_disp, initial, final, jac, pair_idx):
    """Generate diagnostic plots for B1."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping visualization")
        return
    
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    
    # Per-axis scatter
    axis_names = ["z", "y", "x"]
    for ax_idx in range(3):
        ax = axes[ax_idx]
        ideal = ideal_disp[:, ax_idx]
        actual = actual_disp[:, ax_idx]
        corr = np.corrcoef(ideal, actual)[0, 1] if ideal.std() > 0 else 0
        
        ax.scatter(ideal, actual, alpha=0.2, s=5, c='steelblue')
        lim = max(abs(ideal).max(), abs(actual).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1)
        ax.set_xlabel(f"Ideal ({axis_names[ax_idx]})")
        ax.set_ylabel(f"Predicted ({axis_names[ax_idx]})")
        ax.set_title(f"Axis {ax_idx}: r={corr:.3f}")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    # Per-keypoint TRE histogram
    ax = axes[3]
    initial_errors = np.linalg.norm(fkp - mkp, axis=1)
    warped = fkp + actual_disp
    final_errors = np.linalg.norm(warped - mkp, axis=1)
    
    ax.hist(initial_errors, bins=50, alpha=0.5, label=f"Initial ({initial_errors.mean():.1f}mm)", color='red')
    ax.hist(final_errors, bins=50, alpha=0.5, label=f"Final ({final_errors.mean():.1f}mm)", color='green')
    ax.set_xlabel("TRE (mm)")
    ax.set_ylabel("Count")
    ax.set_title("Per-keypoint TRE distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"Test B1: SVF Fitter with GT Keypoints — Pair {pair_idx}\n"
                 f"TRE: {initial['mean_tre']:.2f} → {final['mean_tre']:.2f} mm "
                 f"({(1-final['mean_tre']/initial['mean_tre'])*100:+.1f}%), "
                 f"Jac %neg={jac['jac_pct_negative']:.2f}%",
                 fontsize=12)
    plt.tight_layout()
    
    path = VIZ_DIR / f"B1_fitter_gt_pair{pair_idx}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  📊 Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Test B: SVF Fitter on GT Correspondences")
    parser.add_argument("--pair", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    config = PipelineConfig()
    dataset = ThoraxCBCTDataset(config.paths.data_root, split="train")
    logger.info(f"Dataset: {dataset}")
    
    results = []
    
    r1 = test_B1_fitter_with_all_gt(dataset, args.pair, device=args.device)
    if r1: results.append(r1)
    
    r2 = test_B2_fitter_with_subset_gt(dataset, args.pair, n_subset=2000, device=args.device)
    if r2: results.append(r2)
    
    r3 = test_B3_fitter_with_noisy_gt(dataset, args.pair, noise_std=10.0, device=args.device)
    if r3: results.append(r3)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY — TEST B: SVF FITTER")
    logger.info(f"{'='*60}")
    for r in results:
        status = "✅" if r["passed"] else "❌"
        logger.info(f"  {status} {r['test']}: "
                    f"TRE {r['initial_tre']:.2f} → {r['final_tre']:.2f} mm "
                    f"({r['improvement_pct']:+.1f}%)")
    
    n_pass = sum(1 for r in results if r["passed"])
    if n_pass < len(results):
        logger.error("  ⚠️  FITTER IS BROKEN — fix before testing features/matching!")
        sys.exit(1)
    else:
        logger.info("  ✅ Fitter works correctly — proceed to Test C (features)")


if __name__ == "__main__":
    main()
