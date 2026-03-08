#!/usr/bin/env python3
"""
Test E: End-to-End with Synthetic Warp
========================================

Takes one volume, applies a KNOWN smooth deformation to create a synthetic 
"moving" image, then runs the full pipeline to recover it.

If this fails → code bug (coordinates, displacement composition, etc.)
If this passes but real data fails → domain shift / feature non-transfer.

Usage:
    python -m pipeline.tests.test_E_synthetic --pair 0
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
from pipeline.eval.metrics import compute_tre, compute_jacobian_stats
from pipeline.transform.integrate import scaling_and_squaring
from pipeline.transform.warp import warp_volume, warp_points
from pipeline.transform.svf import SVFField

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

VIZ_DIR = PROJECT_ROOT / "pipeline" / "tests" / "results" / "test_E"
VIZ_DIR.mkdir(parents=True, exist_ok=True)


def create_synthetic_deformation(volume_shape, max_displacement=15.0, 
                                  smoothness=20.0, device="cuda", seed=42):
    """
    Create a smooth synthetic deformation field.
    
    Uses a coarse random velocity field + heavy smoothing to create a
    physically plausible deformation (no folding).
    
    Args:
        volume_shape: (D, H, W)
        max_displacement: approximate max displacement in voxels
        smoothness: Gaussian smoothing sigma (larger = smoother)
        
    Returns:
        displacement_fwd: (1, 3, D, H, W) synthetic mapping from Fixed to Moving
        displacement_inv: (1, 3, D, H, W) synthetic mapping from Moving to Fixed
        velocity: (1, 3, D, H, W) the velocity field used
    """
    D, H, W = volume_shape
    torch.manual_seed(seed)
    
    # Create a coarse velocity field and upsample
    coarse_size = 8
    v_coarse = torch.randn(1, 3, coarse_size, coarse_size, coarse_size, device=device) * 3.0
    
    # Upsample to full resolution
    v = torch.nn.functional.interpolate(
        v_coarse, size=(D, H, W), mode='trilinear', align_corners=True
    )
    
    # Apply Gaussian smoothing for physical plausibility
    from scipy.ndimage import gaussian_filter
    v_np = v.cpu().numpy()
    for c in range(3):
        v_np[0, c] = gaussian_filter(v_np[0, c], sigma=smoothness)
    
    # Scale to desired max displacement
    v_torch = torch.from_numpy(v_np).float().to(device)
    current_max = v_torch.abs().max().item()
    if current_max > 0:
        v_torch = v_torch * (max_displacement / current_max)
    
    # Integrate velocity to get inverse and forward deformations (diffeomorphisms!)
    displacement_fwd = scaling_and_squaring(v_torch, n_steps=7)
    displacement_inv = scaling_and_squaring(-v_torch, n_steps=7)
    
    logger.info(f"  Synthetic deformation: max_disp={displacement_fwd.abs().max().item():.1f}, "
                f"mean_disp={displacement_fwd.abs().mean().item():.3f}")
    
    # Verify no folding
    from pipeline.transform.warp import compute_jacobian_determinant
    jac = compute_jacobian_determinant(displacement_fwd)
    pct_neg = (jac <= 0).float().mean().item() * 100
    logger.info(f"  Jacobian folding: {pct_neg:.2f}%")
    
    return displacement_fwd, displacement_inv, v_torch


def test_E1_synthetic_warp(config, dataset, pair_idx, device="cuda"):
    """
    E1: Full end-to-end test with synthetic deformation.
    
    1. Load one volume
    2. Apply known deformation → create "synthetic moving"
    3. Create synthetic keypoints by warping real keypoints
    4. Run MIND-SSC ConvexAdam to recover the deformation
    5. Compare recovered displacement to known GT
    
    Pass: >70% TRE recovery
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST E1: Synthetic warp end-to-end (pair {pair_idx})")
    logger.info(f"{'='*60}")
    
    data = dataset[pair_idx]
    fixed_img = data["fixed_img"]
    fkp = data["fixed_keypoints"]
    
    if fkp is None:
        logger.warning(f"  SKIP: No keypoints")
        return None
    
    D, H, W = fixed_img.shape
    logger.info(f"  Volume: ({D}, {H}, {W})")
    logger.info(f"  Keypoints: {len(fkp)}")
    
    # 1. Create synthetic deformation
    logger.info(f"\n  Step 1: Creating synthetic deformation...")
    gt_displacement, gt_displacement_inv, gt_velocity = create_synthetic_deformation(
        (D, H, W), max_displacement=15.0, smoothness=20.0, device=device
    )
    
    # 2. Warp volume to create synthetic moving image
    logger.info(f"  Step 2: Warping volume...")
    fixed_tensor = torch.from_numpy(fixed_img).float().unsqueeze(0).unsqueeze(0).to(device)
    # moving(x) = fixed(x + disp_inv(x)) — image warping uses the inverse lookup
    moving_tensor = warp_volume(fixed_tensor, gt_displacement_inv)
    synthetic_moving = moving_tensor.squeeze().cpu().numpy()
    
    # 3. Warp keypoints to create synthetic moving keypoints
    logger.info(f"  Step 3: Warping keypoints...")
    # Keypoint advection uses the forward displacement (Fixed -> Moving)
    synthetic_mkp = warp_points(fkp, gt_displacement)
    
    # Compute "initial" TRE (fixed vs synthetic_moving keypoints)
    initial_errors = np.linalg.norm(fkp - synthetic_mkp, axis=1)
    initial_tre = initial_errors.mean()
    logger.info(f"  Synthetic initial TRE: {initial_tre:.3f} mm")
    logger.info(f"  (This should equal ~mean displacement magnitude)")
    
    # 4. Run MIND-SSC ConvexAdam to recover deformation
    logger.info(f"\n  Step 4: Running MIND-SSC ConvexAdam...")
    from pipeline.transform.mind_convex_adam import mind_convex_adam
    
    t0 = time.time()
    recovered_displacement = mind_convex_adam(
        fixed_img=fixed_img,
        moving_img=synthetic_moving,
        mind_r=1, mind_d=2,
        lambda_weight=0, grid_sp=4, disp_hw=4,
        n_iter_adam=0, grid_sp_adam=2,
        ic=True, device=device,
    )
    elapsed = time.time() - t0
    logger.info(f"  MIND completed in {elapsed:.1f}s")
    
    # 5. Evaluate recovery
    logger.info(f"\n  Step 5: Evaluating recovery...")
    
    # TRE with recovered displacement (same-modality, should work well)
    final_tre_result = compute_tre(synthetic_mkp, fkp, displacement=recovered_displacement)
    final_tre = final_tre_result['mean_tre']
    recovery_pct = (1 - final_tre / initial_tre) * 100
    
    logger.info(f"  Initial TRE: {initial_tre:.3f} mm")
    logger.info(f"  Final TRE: {final_tre:.3f} mm")
    logger.info(f"  Recovery: {recovery_pct:.1f}%")
    
    # Per-axis displacement correlation
    from scipy.ndimage import map_coordinates as mc
    recovered_np = recovered_displacement.cpu().numpy()
    gt_np = gt_displacement.cpu().numpy()
    
    recovered_at_kp = np.zeros_like(fkp)
    gt_at_kp = np.zeros_like(fkp)
    for ax in range(3):
        recovered_at_kp[:, ax] = mc(recovered_np[0, ax], 
                                     [fkp[:, 0], fkp[:, 1], fkp[:, 2]],
                                     order=1, mode='nearest')
        gt_at_kp[:, ax] = mc(gt_np[0, ax],
                              [fkp[:, 0], fkp[:, 1], fkp[:, 2]],
                              order=1, mode='nearest')
    
    axis_names = ["z", "y", "x"]
    correlations = []
    for ax in range(3):
        corr = np.corrcoef(gt_at_kp[:, ax], recovered_at_kp[:, ax])[0, 1]
        correlations.append(corr)
        logger.info(f"  Axis {ax} ({axis_names[ax]}): "
                    f"GT range=[{gt_at_kp[:, ax].min():.1f}, {gt_at_kp[:, ax].max():.1f}], "
                    f"recovered range=[{recovered_at_kp[:, ax].min():.1f}, {recovered_at_kp[:, ax].max():.1f}], "
                    f"corr={corr:+.3f}")
    
    passed = recovery_pct > 70.0
    logger.info(f"\n  {'✅ PASS' if passed else '❌ FAIL'} — "
                f"{'>' if passed else '<'} 70% recovery on synthetic warp")
    
    if not passed:
        logger.warning("  ⚠️  pipeline cannot recover even SAME-MODALITY deformations!")
        logger.warning("  This means there's a fundamental code bug.")
    
    # Visualization
    _visualize_E1(fixed_img, synthetic_moving, gt_displacement, recovered_displacement,
                  fkp, synthetic_mkp, gt_at_kp, recovered_at_kp, 
                  initial_tre, final_tre, recovery_pct, pair_idx)
    
    return {
        "test": "E1_synthetic_warp",
        "pair": pair_idx,
        "initial_tre": initial_tre,
        "final_tre": final_tre,
        "recovery_pct": recovery_pct,
        "axis_correlations": correlations,
        "runtime_s": elapsed,
        "passed": passed,
    }


def _visualize_E1(fixed_img, moving_img, gt_disp, recovered_disp,
                  fkp, mkp, gt_at_kp, recovered_at_kp,
                  initial_tre, final_tre, recovery_pct, pair_idx):
    """Visualize synthetic warp recovery."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    D = fixed_img.shape[0]
    mid_z = D // 2
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    
    # Row 1: Images
    axes[0, 0].imshow(fixed_img[mid_z], cmap='gray')
    axes[0, 0].set_title("Original (fixed)")
    
    axes[0, 1].imshow(moving_img[mid_z], cmap='gray')
    axes[0, 1].set_title("Synthetic Moving")
    
    # Difference images
    diff_before = np.abs(fixed_img[mid_z] - moving_img[mid_z])
    axes[0, 2].imshow(diff_before, cmap='hot')
    axes[0, 2].set_title("Difference (before)")
    
    # GT displacement magnitude at mid slice
    gt_np = gt_disp.cpu().numpy()
    gt_mag = np.sqrt(gt_np[0, 0, mid_z]**2 + gt_np[0, 1, mid_z]**2 + gt_np[0, 2, mid_z]**2)
    axes[0, 3].imshow(gt_mag, cmap='jet')
    axes[0, 3].set_title("GT Displacement Magnitude")
    
    # Row 2: Per-axis displacement comparison
    axis_names = ["z", "y", "x"]
    for ax_idx in range(3):
        ax = axes[1, ax_idx]
        gt = gt_at_kp[:, ax_idx]
        rec = recovered_at_kp[:, ax_idx]
        corr = np.corrcoef(gt, rec)[0, 1] if gt.std() > 0 else 0
        
        ax.scatter(gt, rec, alpha=0.2, s=5, c='steelblue')
        lim = max(abs(gt).max(), abs(rec).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1)
        ax.set_xlabel(f"GT disp ({axis_names[ax_idx]})")
        ax.set_ylabel(f"Recovered disp ({axis_names[ax_idx]})")
        ax.set_title(f"Axis {ax_idx}: r={corr:.3f}")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    # TRE histogram
    ax = axes[1, 3]
    initial_errors = np.linalg.norm(fkp - mkp, axis=1)
    warped = fkp + recovered_at_kp
    final_errors = np.linalg.norm(warped - mkp, axis=1)
    
    ax.hist(initial_errors, bins=40, alpha=0.5, label=f"Before ({initial_tre:.1f}mm)", color='red')
    ax.hist(final_errors, bins=40, alpha=0.5, label=f"After ({final_tre:.1f}mm)", color='green')
    ax.set_xlabel("Error (mm)")
    ax.set_title("Per-keypoint error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    for ax in axes.ravel():
        ax.axis('off') if len(ax.images) > 0 else None
    
    fig.suptitle(f"Test E1: Synthetic Warp Recovery — Pair {pair_idx}\n"
                 f"TRE: {initial_tre:.2f} → {final_tre:.2f} mm "
                 f"(recovery: {recovery_pct:.1f}%)",
                 fontsize=14)
    plt.tight_layout()
    
    path = VIZ_DIR / f"E1_synthetic_pair{pair_idx}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  📊 Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Test E: Synthetic Warp End-to-End")
    parser.add_argument("--pair", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    config = PipelineConfig()
    config.device = args.device
    dataset = ThoraxCBCTDataset(config.paths.data_root, split="train")
    logger.info(f"Dataset: {dataset}")
    
    results = []
    r1 = test_E1_synthetic_warp(config, dataset, args.pair, device=args.device)
    if r1: results.append(r1)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY — TEST E: SYNTHETIC WARP")
    logger.info(f"{'='*60}")
    for r in results:
        status = "✅" if r["passed"] else "❌"
        logger.info(f"  {status} {r['test']}: "
                    f"TRE {r['initial_tre']:.2f} → {r['final_tre']:.2f} mm "
                    f"(recovery: {r['recovery_pct']:.1f}%)")
    
    n_pass = sum(1 for r in results if r["passed"])
    if n_pass < len(results):
        logger.error("  ⚠️  PIPELINE HAS A CODE BUG — cannot recover even synthetic deformations!")
        sys.exit(1)
    else:
        logger.info("  ✅ Pipeline mechanics are correct")
        logger.info("  If real data still fails → domain shift / feature transfer issue")


if __name__ == "__main__":
    main()
