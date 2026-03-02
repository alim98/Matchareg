"""
Preprocessing utilities for 3D medical volumes.

Handles intensity normalization, trunk mask generation,
and slice-to-pseudo-RGB conversion for foundation model input.
"""
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

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
    method: str = "ct_air",
    percentile: float = 30.0,
    min_component_size: int = 10000,
) -> np.ndarray:
    """
    Generate a simple trunk/body mask via CT border-air detection.
    """
    D, H, W = volume.shape
    
    # 1. Threshold air explicitly (CT values < -850 HU)
    # Thorax CBCT air is usually ~ -1000
    air_mask = volume < -850
    
    # 2. Identify the outside air (connected component touching the border)
    markers = np.zeros_like(volume, dtype=int)
    markers[0, :, :] = 1
    markers[-1, :, :] = 1
    markers[:, 0, :] = 1
    markers[:, -1, :] = 1
    markers[:, :, 0] = 1
    markers[:, :, -1] = 1
    markers = markers & air_mask
    
    labeled_air, num_features = ndimage.label(air_mask)
    outside_air_mask = np.zeros_like(air_mask, dtype=bool)
    if num_features > 0:
        border_labels = np.unique(labeled_air[markers == 1])
        border_labels = border_labels[border_labels > 0]
        outside_air_mask = np.isin(labeled_air, border_labels)

    # 3. Body is everything that is NOT outside air
    body_mask = ~outside_air_mask
    
    # 4. Fill holes slice by slice (lungs)
    for z in range(D):
        body_mask[z] = ndimage.binary_fill_holes(body_mask[z])
    
    # 5. Keep largest connected component
    labeled, n_labels = ndimage.label(body_mask)
    if n_labels > 1:
        component_sizes = np.bincount(labeled.ravel())
        component_sizes[0] = 0  # ignore background
        largest = np.argmax(component_sizes)
        mask = (labeled == largest).astype(np.uint8)
    else:
        mask = body_mask.astype(np.uint8)

    coverage = mask.mean()
    logger.info(f"Trunk mask computed (CT-air border): coverage={coverage:.3f}")

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


def crop_to_mask(volume: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[slice, slice, slice]]:
    """
    Crop the volume and mask to the bounding box of the non-zero mask entries.
    Returns:
        cropped_volume, cropped_mask, bbox_slices
    """
    non_zeros = np.nonzero(mask)
    if len(non_zeros[0]) == 0:
        # Fallback if mask is somehow completely empty
        return volume, mask, (slice(None), slice(None), slice(None))

    z_min, z_max = np.min(non_zeros[0]), np.max(non_zeros[0])
    y_min, y_max = np.min(non_zeros[1]), np.max(non_zeros[1])
    x_min, x_max = np.min(non_zeros[2]), np.max(non_zeros[2])

    bbox_slices = (
        slice(z_min, z_max + 1),
        slice(y_min, y_max + 1),
        slice(x_min, x_max + 1)
    )

    cropped_volume = volume[bbox_slices]
    cropped_mask = mask[bbox_slices]

    return cropped_volume, cropped_mask, bbox_slices


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
    clip_sigma: float = 3.0,
    target_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    target_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Convert z-scored pseudo-RGB slices to ImageNet-normalized range.

    ViT backbones (DINOv3, MATCHA) expect:  (x_pixel - mean) / std
    where x_pixel ∈ [0, 1] and mean/std are the ImageNet statistics.

    Pipeline:
      1. Clip z-score to [-clip_sigma, +clip_sigma]  (captures 99.7% of data)
      2. Scale to [0, 1]
      3. Apply per-channel (x - mean) / std

    Args:
        slices: array of shape (..., 3, H, W), z-scored (mean≈0, std≈1)
        clip_sigma: z-score clipping range (default 3.0)
        target_mean: per-channel ImageNet means
        target_std: per-channel ImageNet stds

    Returns:
        Rescaled array with same shape, ready for ViT input.
    """
    # Step 1+2: clip and scale to [0, 1]
    out = np.clip(slices, -clip_sigma, clip_sigma)
    out = (out + clip_sigma) / (2.0 * clip_sigma)  # → [0, 1]

    # Step 3: ImageNet normalization per channel
    out = out.copy()
    for c in range(3):
        out[..., c, :, :] = (out[..., c, :, :] - target_mean[c]) / target_std[c]

    return out.astype(np.float32)

