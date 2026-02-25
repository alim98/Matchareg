"""
Evaluation metrics for 3D registration.

TRE (Target Registration Error), Jacobian determinant statistics.
"""
import numpy as np
import torch
from typing import Dict, Optional

from ..transform.warp import compute_jacobian_determinant


def compute_tre(
    keypoints_src: np.ndarray,
    keypoints_tgt: np.ndarray,
    displacement: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute Target Registration Error (TRE) on keypoint pairs.

    TRE = ||φ(x_k) - y_k|| for each keypoint pair.

    Args:
        keypoints_src: (K, 3) source keypoints in voxel coords
        keypoints_tgt: (K, 3) target keypoints in voxel coords
        displacement: optional (1, 3, D, H, W) displacement field.
                      If None, computes initial (unregistered) TRE.

    Returns:
        Dict with mean_tre, median_tre, std_tre, max_tre, all in mm
        (assumes 1mm isotropic voxels for ThoraxCBCT).
    """
    if displacement is not None:
        from ..transform.warp import warp_points
        warped_src = warp_points(keypoints_src, displacement)
    else:
        warped_src = keypoints_src.copy()

    # Euclidean distance per keypoint
    errors = np.linalg.norm(warped_src - keypoints_tgt, axis=1)

    return {
        "mean_tre": float(np.mean(errors)),
        "median_tre": float(np.median(errors)),
        "std_tre": float(np.std(errors)),
        "max_tre": float(np.max(errors)),
        "min_tre": float(np.min(errors)),
        "n_keypoints": len(errors),
        "tre_per_keypoint": errors,
    }


def compute_jacobian_stats(
    displacement: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute Jacobian determinant statistics.

    Args:
        displacement: (1, 3, D, H, W) displacement field

    Returns:
        Dict with mean, std, min, max, pct_negative
    """
    jac_det = compute_jacobian_determinant(displacement)
    jac_np = jac_det.cpu().numpy().ravel()

    return {
        "jac_mean": float(np.mean(jac_np)),
        "jac_std": float(np.std(jac_np)),
        "jac_min": float(np.min(jac_np)),
        "jac_max": float(np.max(jac_np)),
        "jac_pct_negative": float(np.mean(jac_np <= 0) * 100),
    }


def print_results(
    tre_results: Dict,
    jac_results: Optional[Dict] = None,
    pair_id: str = "",
):
    """Pretty-print evaluation results."""
    print(f"\n{'='*50}")
    if pair_id:
        print(f"Pair: {pair_id}")
    print(f"{'='*50}")
    print(f"TRE (mm):")
    print(f"  Mean:   {tre_results['mean_tre']:.3f}")
    print(f"  Median: {tre_results['median_tre']:.3f}")
    print(f"  Std:    {tre_results['std_tre']:.3f}")
    print(f"  Min:    {tre_results['min_tre']:.3f}")
    print(f"  Max:    {tre_results['max_tre']:.3f}")
    print(f"  N keypoints: {tre_results['n_keypoints']}")

    if jac_results:
        print(f"\nJacobian determinant:")
        print(f"  Mean:  {jac_results['jac_mean']:.4f}")
        print(f"  Std:   {jac_results['jac_std']:.4f}")
        print(f"  Min:   {jac_results['jac_min']:.4f}")
        print(f"  Max:   {jac_results['jac_max']:.4f}")
        print(f"  % ≤ 0: {jac_results['jac_pct_negative']:.2f}%")
    print(f"{'='*50}\n")
