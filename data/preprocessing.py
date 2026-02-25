"""
Preprocessing utilities for 3D medical volumes.

Handles intensity normalization, trunk mask generation,
and slice-to-pseudo-RGB conversion for foundation model input.
"""
import numpy as np
from typing import Tuple, Optional

try:
    from scipy import ndimage
except ImportError:
    raise ImportError("scipy is required: pip install scipy")


def robust_intensity_normalize(
    volume: np.ndarray,
    mask: Optional[np.ndarray] = None,
    low_pct: float = 0.5,
    high_pct: float = 99.5,
) -> np.ndarray:
    """
    Robust intensity normalization: percentile clipping + z-score.

    Args:
        volume: 3D array (D, H, W)
        mask: optional binary mask; stats computed inside mask only
        low_pct: lower percentile for clipping
        high_pct: upper percentile for clipping

    Returns:
        Normalized volume (float32)
    """
    vol = volume.copy().astype(np.float32)

    if mask is not None:
        values = vol[mask > 0]
    else:
        values = vol.ravel()

    lo = np.percentile(values, low_pct)
    hi = np.percentile(values, high_pct)
    vol = np.clip(vol, lo, hi)

    if mask is not None:
        mu = np.mean(vol[mask > 0])
        sigma = np.std(vol[mask > 0]) + 1e-8
    else:
        mu = np.mean(vol)
        sigma = np.std(vol) + 1e-8

    vol = (vol - mu) / sigma
    return vol


def generate_trunk_mask(
    volume: np.ndarray,
    threshold_pct: float = 10.0,
    min_component_size: int = 10000,
) -> np.ndarray:
    """
    Generate a simple trunk/body mask via thresholding + morphology.

    Uses Otsu-like percentile thresholding on the intensity histogram,
    followed by binary closing and largest-component selection.

    Args:
        volume: 3D array (D, H, W)
        threshold_pct: percentile threshold (values above this are body)
        min_component_size: minimum connected component size to keep

    Returns:
        Binary mask (uint8), same shape as volume.
    """
    threshold = np.percentile(volume, threshold_pct)
    mask = (volume > threshold).astype(np.uint8)

    # Binary closing to fill small holes
    struct = ndimage.generate_binary_structure(3, 2)
    mask = ndimage.binary_closing(mask, structure=struct, iterations=3).astype(np.uint8)

    # Keep largest connected component
    labeled, n_labels = ndimage.label(mask)
    if n_labels > 1:
        component_sizes = np.bincount(labeled.ravel())
        component_sizes[0] = 0  # ignore background
        largest = np.argmax(component_sizes)
        mask = (labeled == largest).astype(np.uint8)

    # Final fill
    mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
    return mask


def volume_slice_to_pseudo_rgb(
    volume: np.ndarray,
    slice_idx: int,
    axis: int = 0,
) -> np.ndarray:
    """
    Convert a single axial/coronal/sagittal slice to pseudo-RGB
    by stacking adjacent slices as 3 channels: [s-1, s, s+1].

    Args:
        volume: 3D array (D, H, W), normalized
        slice_idx: index of the central slice along the given axis
        axis: 0=axial, 1=coronal, 2=sagittal

    Returns:
        Pseudo-RGB image of shape (3, H', W') in float32,
        where H', W' are the two remaining spatial dims.
    """
    n_slices = volume.shape[axis]

    # Clamp indices
    idx_prev = max(0, slice_idx - 1)
    idx_curr = slice_idx
    idx_next = min(n_slices - 1, slice_idx + 1)

    slices = []
    for idx in [idx_prev, idx_curr, idx_next]:
        s = np.take(volume, idx, axis=axis)
        slices.append(s)

    # Stack as (3, H', W')
    rgb = np.stack(slices, axis=0).astype(np.float32)
    return rgb


def extract_all_pseudo_rgb_slices(
    volume: np.ndarray,
    axis: int = 0,
) -> np.ndarray:
    """
    Extract pseudo-RGB slices for all positions along an axis.

    Args:
        volume: 3D array (D, H, W), normalized
        axis: 0=axial, 1=coronal, 2=sagittal

    Returns:
        Array of shape (N_slices, 3, H', W')
    """
    n_slices = volume.shape[axis]
    all_slices = []
    for i in range(n_slices):
        rgb = volume_slice_to_pseudo_rgb(volume, i, axis=axis)
        all_slices.append(rgb)
    return np.stack(all_slices, axis=0)


def rescale_to_model_range(
    slices: np.ndarray,
    target_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    target_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Rescale z-scored pseudo-RGB slices to ImageNet normalization range.

    Input slices should be already z-scored (mean≈0, std≈1).
    This maps them to approximately match ImageNet statistics expected
    by foundation models (DINOv3, MATCHA).

    Args:
        slices: array of shape (..., 3, H, W), z-scored
        target_mean: per-channel target means (ImageNet defaults)
        target_std: per-channel target stds (ImageNet defaults)

    Returns:
        Rescaled array with same shape.
    """
    out = slices.copy()
    for c in range(3):
        # Map from N(0,1) to N(target_mean, target_std)
        out[..., c, :, :] = out[..., c, :, :] * target_std[c] + target_mean[c]
    return out
