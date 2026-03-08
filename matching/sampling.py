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
    """Globally anchored lattice sampling with random fill."""
    coords = np.argwhere(mask > 0)
    total_fg = len(coords)
    if total_fg == 0:
        raise ValueError("Mask is empty — cannot sample points.")
    if total_fg <= n_points:
        return coords.astype(np.float64)

    step = max(1, int(round((total_fg / max(n_points, 1)) ** (1.0 / 3.0))))
    best_points = None
    best_gap = None

    z_offsets = sorted({0, step // 2})
    y_offsets = sorted({0, step // 2})
    x_offsets = sorted({0, step // 2})

    for z_off in z_offsets:
        for y_off in y_offsets:
            for x_off in x_offsets:
                submask = mask[z_off::step, y_off::step, x_off::step]
                pts = np.argwhere(submask > 0)
                if len(pts) == 0:
                    continue
                pts = pts.astype(np.int64, copy=False)
                pts[:, 0] = pts[:, 0] * step + z_off
                pts[:, 1] = pts[:, 1] * step + y_off
                pts[:, 2] = pts[:, 2] * step + x_off
                gap = abs(len(pts) - n_points)
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    best_points = pts

    if best_points is None or len(best_points) == 0:
        idx = rng.choice(total_fg, n_points, replace=False)
        return coords[idx].astype(np.float64)

    if len(best_points) > n_points:
        idx = np.linspace(0, len(best_points) - 1, n_points, dtype=int)
        return best_points[idx].astype(np.float64)

    if len(best_points) < n_points:
        chosen = best_points
        chosen_set = {tuple(p) for p in chosen.tolist()}
        remaining_mask = np.array([tuple(p) not in chosen_set for p in coords], dtype=bool)
        remaining = coords[remaining_mask]
        n_extra = min(n_points - len(chosen), len(remaining))
        if n_extra > 0:
            extra_idx = rng.choice(len(remaining), n_extra, replace=False)
            chosen = np.vstack([chosen, remaining[extra_idx]])
        return chosen.astype(np.float64)

    return best_points.astype(np.float64)


def _sample_slice_with_coverage(
    yx_coords: np.ndarray,
    n_points: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Prefer a regular lattice over pure random picks for better coverage."""
    if len(yx_coords) <= n_points:
        return yx_coords

    y_min, x_min = yx_coords.min(axis=0)
    y_max, x_max = yx_coords.max(axis=0)
    area = max(len(yx_coords), 1)
    step = max(1, int(np.sqrt(area / max(n_points, 1))))

    y_mod = (step // 2)
    x_mod = (step // 2)
    lattice_mask = (
        ((yx_coords[:, 0] - y_min) % step == y_mod % step) &
        ((yx_coords[:, 1] - x_min) % step == x_mod % step)
    )
    lattice = yx_coords[lattice_mask]

    if len(lattice) >= n_points:
        idx = np.linspace(0, len(lattice) - 1, n_points, dtype=int)
        return lattice[idx]

    remaining_mask = ~lattice_mask
    remaining = yx_coords[remaining_mask]
    n_extra = n_points - len(lattice)
    extra_idx = rng.choice(len(remaining), n_extra, replace=False)
    return np.vstack([lattice, remaining[extra_idx]])


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

