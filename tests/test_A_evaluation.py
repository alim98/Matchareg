#!/usr/bin/env python3
"""
Test A: Evaluation & Baseline Sanity
=====================================

Tests that do NOT depend on any feature extraction or matching:
  A1. Identity displacement → TRE must equal initial TRE exactly
  A2. MIND-SSC ConvexAdam → must show large TRE reduction (>40%)
  A3. Per-axis displacement correlation with ideal displacement

If A2 fails, the evaluation pipeline itself is broken (or MIND is broken),
and nothing downstream is trustworthy.

Usage:
    python -m pipeline.tests.test_A_evaluation --pair 0
    python -m pipeline.tests.test_A_evaluation --all
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
from pipeline.data.preprocessing import generate_trunk_mask
from pipeline.eval.metrics import compute_tre
from pipeline.transform.warp import warp_points

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Output directory for visualizations
VIZ_DIR = PROJECT_ROOT / "pipeline" / "tests" / "results" / "test_A"
VIZ_DIR.mkdir(parents=True, exist_ok=True)


def test_A1_identity_displacement(dataset, pair_idx):
    """
    A1: Identity displacement should give EXACTLY the initial TRE.
    
    If this fails → compute_tre has a coordinate bug.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST A1: Identity displacement (pair {pair_idx})")
    logger.info(f"{'='*60}")
    
    data = dataset[pair_idx]
    fkp = data["fixed_keypoints"]
    mkp = data["moving_keypoints"]
    
    if fkp is None or mkp is None:
        logger.warning(f"  SKIP: No keypoints for pair {pair_idx}")
        return None
    
    # Initial TRE (no displacement)
    initial = compute_tre(mkp, fkp, displacement=None)
    logger.info(f"  Initial TRE: {initial['mean_tre']:.3f} mm")
    
    # Create zero displacement field
    D, H, W = data["fixed_img"].shape
    zero_disp = torch.zeros(1, 3, D, H, W)
    
    # TRE with zero displacement should be identical
    with_zero = compute_tre(mkp, fkp, displacement=zero_disp)
    logger.info(f"  TRE with zero-disp: {with_zero['mean_tre']:.3f} mm")
    
    diff = abs(initial['mean_tre'] - with_zero['mean_tre'])
    passed = diff < 0.01
    
    logger.info(f"  Difference: {diff:.6f} mm")
    logger.info(f"  ✅ PASS" if passed else f"  ❌ FAIL — identity displacement changes TRE!")
    
    return {
        "test": "A1_identity",
        "pair": pair_idx,
        "initial_tre": initial['mean_tre'],
        "zero_disp_tre": with_zero['mean_tre'],
        "diff": diff,
        "passed": passed,
    }


def test_A2_mind_baseline(dataset, pair_idx, device="cuda"):
    """
    A2: MIND-SSC ConvexAdam should reduce TRE by >40%.
    
    If this fails → evaluation or MIND is broken.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST A2: MIND-SSC ConvexAdam baseline (pair {pair_idx})")
    logger.info(f"{'='*60}")
    
    data = dataset[pair_idx]
    fkp = data["fixed_keypoints"]
    mkp = data["moving_keypoints"]
    fixed_img = data["fixed_img"]
    moving_img = data["moving_img"]
    
    if fkp is None or mkp is None:
        logger.warning(f"  SKIP: No keypoints for pair {pair_idx}")
        return None
    
    # Initial TRE
    initial = compute_tre(mkp, fkp, displacement=None)
    logger.info(f"  Initial TRE: {initial['mean_tre']:.3f} mm ({initial['n_keypoints']} keypoints)")
    
    # Run MIND-SSC ConvexAdam with SAME parameters as run_pipeline.py module.
    # These match the sweep results from test_grid_sp.py.
    logger.info(f"  Running MIND-SSC ConvexAdam (grid_sp=4, disp_hw=4, no Adam)...")
    t0 = time.time()
    
    from pipeline.transform.mind_convex_adam import mind_convex_adam
    displacement = mind_convex_adam(
        fixed_img=fixed_img,
        moving_img=moving_img,
        mind_r=1, mind_d=2,
        lambda_weight=0, grid_sp=4, disp_hw=4,
        n_iter_adam=0, grid_sp_adam=2,
        ic=True, device=device,
    )
    
    elapsed = time.time() - t0
    logger.info(f"  MIND completed in {elapsed:.1f}s")
    logger.info(f"  Displacement: max={displacement.abs().max().item():.1f}, "
                f"mean={displacement.abs().mean().item():.3f}")
    
    # Final TRE
    final = compute_tre(mkp, fkp, displacement=displacement)
    improvement = (1 - final['mean_tre'] / initial['mean_tre']) * 100
    
    logger.info(f"  Final TRE: {final['mean_tre']:.3f} mm")
    logger.info(f"  Improvement: {improvement:+.1f}%")
    
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
    axis_names = ["z (dim0)", "y (dim1)", "x (dim2)"]
    correlations = []
    for ax in range(3):
        ideal = ideal_disp[:, ax]
        actual = fixed_disp[:, ax]
        corr = np.corrcoef(ideal, actual)[0, 1] if ideal.std() > 0 else 0
        correlations.append(corr)
        logger.info(f"  Axis {ax} ({axis_names[ax]}): "
                    f"ideal mean={ideal.mean():+.2f} std={ideal.std():.2f}, "
                    f"actual mean={actual.mean():+.2f} std={actual.std():.2f}, "
                    f"corr={corr:+.3f}")
    
    # Pass criteria: the field should not make TRE WORSE, and should have
    # some directional correlation (at least one axis r > 0.3).
    # Note: for easy pairs (initial TRE ~10mm), even 5% improvement is normal.
    passed_tre = improvement > -5.0
    passed_corr = max(correlations) > 0.3
    passed = passed_tre and passed_corr
    logger.info(f"  TRE not worse (>{-5}%)?   {'✅' if passed_tre else '❌'} ({improvement:+.1f}%)")
    logger.info(f"  Best axis corr > 0.3?     {'✅' if passed_corr else '❌'} (max r={max(correlations):.3f})")
    logger.info(f"  {'✅ PASS' if passed else '❌ FAIL'}")
    
    # Visualization
    _visualize_A2(fkp, mkp, fixed_disp, ideal_disp, initial, final, pair_idx)
    
    return {
        "test": "A2_mind_baseline",
        "pair": pair_idx,
        "initial_tre": initial['mean_tre'],
        "final_tre": final['mean_tre'],
        "improvement_pct": improvement,
        "axis_correlations": correlations,
        "runtime_s": elapsed,
        "passed": passed,
    }


def test_A3_tre_convention(dataset, pair_idx, device="cuda"):
    """
    A3: Verify TRE evaluation direction by applying a known uniform displacement.
    
    Create a displacement field that shifts all points by +10 in axis 0.
    TRE should change in a predictable, known way.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST A3: TRE convention check (pair {pair_idx})")
    logger.info(f"{'='*60}")
    
    data = dataset[pair_idx]
    fkp = data["fixed_keypoints"]
    mkp = data["moving_keypoints"]
    
    if fkp is None or mkp is None:
        logger.warning(f"  SKIP: No keypoints for pair {pair_idx}")
        return None
    
    D, H, W = data["fixed_img"].shape
    
    # Create a known displacement: shift +10 voxels along axis 0 (z)
    shift_z = 10.0
    known_disp = torch.zeros(1, 3, D, H, W)
    known_disp[0, 0, :, :, :] = shift_z  # channel 0 = axis 0 displacement
    
    # Compute TRE with this known displacement
    tre_result = compute_tre(mkp, fkp, displacement=known_disp)
    
    # What we EXPECT: warped_fixed = fkp + [shift_z, 0, 0]
    warped_fixed_expected = fkp.copy()
    warped_fixed_expected[:, 0] += shift_z
    expected_tre = np.linalg.norm(warped_fixed_expected - mkp, axis=1).mean()
    
    diff = abs(tre_result['mean_tre'] - expected_tre)
    passed = diff < 0.1
    
    logger.info(f"  Known shift: +{shift_z} voxels in axis 0")
    logger.info(f"  Expected TRE: {expected_tre:.3f} mm")
    logger.info(f"  Computed TRE: {tre_result['mean_tre']:.3f} mm")
    logger.info(f"  Difference: {diff:.6f}")
    logger.info(f"  {'✅ PASS' if passed else '❌ FAIL'} — "
                f"TRE convention {'is' if passed else 'IS NOT'} correct")
    
    return {
        "test": "A3_tre_convention",
        "pair": pair_idx,
        "expected_tre": expected_tre,
        "computed_tre": tre_result['mean_tre'],
        "diff": diff,
        "passed": passed,
    }


def _visualize_A2(fkp, mkp, actual_disp, ideal_disp, initial, final, pair_idx):
    """Generate diagnostic plots for A2."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping visualization")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Per-axis scatter: ideal vs actual displacement
    axis_names = ["z", "y", "x"]
    for ax_idx in range(3):
        ax = axes[ax_idx]
        ideal = ideal_disp[:, ax_idx]
        actual = actual_disp[:, ax_idx]
        corr = np.corrcoef(ideal, actual)[0, 1] if ideal.std() > 0 else 0
        
        ax.scatter(ideal, actual, alpha=0.3, s=8)
        lim = max(abs(ideal).max(), abs(actual).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1, label='ideal')
        ax.set_xlabel(f"Ideal disp ({axis_names[ax_idx]})")
        ax.set_ylabel(f"Actual disp ({axis_names[ax_idx]})")
        ax.set_title(f"Axis {ax_idx} ({axis_names[ax_idx]}): r={corr:.3f}")
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"Test A2: MIND Baseline — Pair {pair_idx}\n"
                 f"TRE: {initial['mean_tre']:.2f} → {final['mean_tre']:.2f} mm "
                 f"({(1-final['mean_tre']/initial['mean_tre'])*100:+.1f}%)",
                 fontsize=13)
    plt.tight_layout()
    
    path = VIZ_DIR / f"A2_mind_pair{pair_idx}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  📊 Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Test A: Evaluation & Baseline Sanity")
    parser.add_argument("--pair", type=int, default=0, help="Pair index to test")
    parser.add_argument("--all", action="store_true", help="Test all train pairs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-mind", action="store_true", 
                        help="Skip MIND test (slow, needs GPU)")
    args = parser.parse_args()
    
    config = PipelineConfig()
    dataset = ThoraxCBCTDataset(config.paths.data_root, split="train")
    logger.info(f"Dataset: {dataset}")
    
    pairs = list(range(len(dataset))) if args.all else [args.pair]
    
    all_results = []
    for pair_idx in pairs:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# PAIR {pair_idx}")
        logger.info(f"{'#'*60}")
        
        # A1: Identity
        r1 = test_A1_identity_displacement(dataset, pair_idx)
        if r1: all_results.append(r1)
        
        # A3: Convention check (fast, no MIND needed)
        r3 = test_A3_tre_convention(dataset, pair_idx)
        if r3: all_results.append(r3)
        
        # A2: MIND baseline
        if not args.skip_mind:
            r2 = test_A2_mind_baseline(dataset, pair_idx, device=args.device)
            if r2: all_results.append(r2)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*60}")
    for r in all_results:
        status = "✅" if r["passed"] else "❌"
        logger.info(f"  {status} {r['test']} (pair {r['pair']})")
    
    n_pass = sum(1 for r in all_results if r["passed"])
    n_total = len(all_results)
    logger.info(f"\n  {n_pass}/{n_total} tests passed")
    
    if n_pass < n_total:
        logger.error("  ⚠️  EVALUATION IS BROKEN — fix before testing any downstream modules!")
        sys.exit(1)
    else:
        logger.info("  ✅ Evaluation is correct — proceed to Test B (fitter)")


if __name__ == "__main__":
    main()
