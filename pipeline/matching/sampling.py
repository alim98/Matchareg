"""
Sparse point sampling inside segmentation masks.

Provides stratified sampling for GWOT matching with optional
Förstner keypoint anchoring.
"""
import numpy as np
from typing import Optional, Tuple


def sample_points_in_mask(
    mask: np.ndarray,
    n_points: int,
    z_stratified: bool = True,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Uniformly sample points inside a binary mask.

    Args:
        mask: 3D binary mask (D, H, W)
        n_points: number of points to sample
        z_stratified: if True, stratify samples across z-slices
        rng: random state for reproducibility

    Returns:
        points: (n_points, 3) array of (z, y, x) voxel coordinates
    """
    if rng is None:
        rng = np.random.RandomState(42)

    if z_stratified:
        return _sample_z_stratified(mask, n_points, rng)
    else:
        return _sample_uniform(mask, n_points, rng)


def _sample_uniform(
    mask: np.ndarray, n_points: int, rng: np.random.RandomState
) -> np.ndarray:
    """Uniform random sampling of foreground voxels."""
    coords = np.argwhere(mask > 0)  # (N_fg, 3)
    n_fg = len(coords)
    if n_fg <= n_points:
        return coords.astype(np.float64)

    indices = rng.choice(n_fg, n_points, replace=False)
    return coords[indices].astype(np.float64)


def _sample_z_stratified(
    mask: np.ndarray, n_points: int, rng: np.random.RandomState
) -> np.ndarray:
    """Stratified sampling: equal points per z-slice (proportional to mask area)."""
    D = mask.shape[0]

    # Count foreground voxels per slice
    slice_counts = np.array([np.sum(mask[z] > 0) for z in range(D)])
    total_fg = slice_counts.sum()

    if total_fg == 0:
        raise ValueError("Mask is empty — cannot sample points.")

    # Allocate points per slice proportional to mask area
    slice_fracs = slice_counts / total_fg
    points_per_slice = np.round(slice_fracs * n_points).astype(int)

    # Adjust to hit exactly n_points
    diff = n_points - points_per_slice.sum()
    if diff > 0:
        # Add to slices with most foreground
        top_slices = np.argsort(-slice_counts)
        for i in range(diff):
            points_per_slice[top_slices[i % D]] += 1
    elif diff < 0:
        # Remove from slices with most allocated
        top_slices = np.argsort(-points_per_slice)
        for i in range(-diff):
            if points_per_slice[top_slices[i % D]] > 0:
                points_per_slice[top_slices[i % D]] -= 1

    # Sample from each slice
    all_points = []
    for z in range(D):
        n_z = points_per_slice[z]
        if n_z == 0:
            continue
        yx_coords = np.argwhere(mask[z] > 0)  # (n_fg_z, 2)
        if len(yx_coords) == 0:
            continue
        if len(yx_coords) <= n_z:
            chosen = yx_coords
        else:
            idx = rng.choice(len(yx_coords), n_z, replace=False)
            chosen = yx_coords[idx]
        z_col = np.full((len(chosen), 1), z)
        all_points.append(np.hstack([z_col, chosen]))

    points = np.vstack(all_points).astype(np.float64)
    return points


def sample_descriptors_at_points(
    feature_volume: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """
    Sample descriptors from a 3D feature volume at given points
    using trilinear interpolation.

    Args:
        feature_volume: (feat_dim, D, H, W)
        points: (N, 3) — (z, y, x) voxel coordinates (can be fractional)

    Returns:
        descriptors: (N, feat_dim)
    """
    import torch
    import torch.nn.functional as F

    feat_dim = feature_volume.shape[0]
    D, H, W = feature_volume.shape[1], feature_volume.shape[2], feature_volume.shape[3]
    N = points.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to torch tensor with batch and spatial dims
    # feature_volume: (C, D, H, W) -> (1, C, D, H, W)
    vol_t = torch.from_numpy(feature_volume).unsqueeze(0).float().to(device)

    # grid_sample expects coordinates normalized to [-1, 1], with order (x, y, z)
    # Our points are (z, y, x) in pixels
    pts_t = torch.from_numpy(points).float().to(device)
    x = pts_t[:, 2]
    y = pts_t[:, 1]
    z = pts_t[:, 0]

    # Normalize to [-1, 1] range: 2 * (coord / (size - 1)) - 1
    # where size > 1 is checked to avoid div by zero (already guaranteed here)
    nx = 2.0 * x / max(W - 1, 1) - 1.0
    ny = 2.0 * y / max(H - 1, 1) - 1.0
    nz = 2.0 * z / max(D - 1, 1) - 1.0

    # Grid shape: (N, 1, 1, 3) -> stack as (x, y, z) inside the grid tensor
    grid = torch.stack([nx, ny, nz], dim=-1).view(1, N, 1, 1, 3)

    # Sample descriptor
    # Note: align_corners=True matches scipy's map_coordinates behavior for pixel grids
    sampled = F.grid_sample(
        vol_t,
        grid,
        mode="bilinear",        # In 3D this performs trilinear interpolation
        padding_mode="border",  # Equivalent to mode='nearest' used in map_coordinates clip
        align_corners=True
    )
    
    # sampled is (1, C, N, 1, 1) -> view to (N, C)
    descriptors = sampled.view(feat_dim, N).transpose(0, 1).cpu().numpy()

    return descriptors


def include_keypoints(
    sampled_points: np.ndarray,
    keypoints: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Add provided keypoints to the sampled point set, deduplicating by
    rounded voxel coordinate.

    Args:
        sampled_points: (N, 3) existing sampled points
        keypoints: (K, 3) keypoints to add
        mask: optional mask to filter keypoints inside mask

    Returns:
        combined: (M, 3) combined point set with duplicates removed
    """
    if mask is not None:
        # Filter keypoints to those inside mask
        kp_int = np.round(keypoints).astype(int)
        valid = (
            (kp_int[:, 0] >= 0) & (kp_int[:, 0] < mask.shape[0]) &
            (kp_int[:, 1] >= 0) & (kp_int[:, 1] < mask.shape[1]) &
            (kp_int[:, 2] >= 0) & (kp_int[:, 2] < mask.shape[2])
        )
        kp_int = kp_int[valid]
        keypoints = keypoints[valid]
        inside = mask[kp_int[:, 0], kp_int[:, 1], kp_int[:, 2]] > 0
        keypoints = keypoints[inside]

    combined = np.vstack([sampled_points, keypoints])

    # Deduplicate by rounded integer coordinate
    # np.unique returns sorted unique rows on the rounded version;
    # we use those indices to select from the original float array.
    _, unique_idx = np.unique(
        np.round(combined).astype(int), axis=0, return_index=True
    )
    return combined[unique_idx]

