"""
Correspondence filtering utilities.

Post-processing of raw matches: confidence thresholding, mutual consistency,
and mask boundary exclusion.
"""
import numpy as np
from typing import Dict, Optional


def filter_matches(
    match_result: Dict,
    points_src: np.ndarray,
    points_tgt: np.ndarray,
    confidence_threshold: float = 0.01,
    mask_src: Optional[np.ndarray] = None,
    mask_tgt: Optional[np.ndarray] = None,
    max_displacement: Optional[float] = None,
    points_src_geom: Optional[np.ndarray] = None,
) -> Dict:
    """
    Filter matches by confidence, mask validity, and displacement.

    Args:
        match_result: dict with matches_src_idx, matches_tgt_idx, weights
        points_src: (N, 3) all source points — used for the RETURNED coordinates
        points_tgt: (M, 3) all target points
        confidence_threshold: minimum weight to keep
        mask_src: optional source mask to verify match points are inside
        mask_tgt: optional target mask
        max_displacement: maximum allowed displacement (mm)
        points_src_geom: (N, 3) source points used ONLY for the spatial displacement
            filter (e.g. warped moving pts). Defaults to points_src when None.
            Use this when points_src are in original space but the displacement
            filter should operate on the warped/matched geometry.

    Returns:
        Filtered dict with matched_points_src, matched_points_tgt, weights
    """
    src_idx = match_result["matches_src_idx"]
    tgt_idx = match_result["matches_tgt_idx"]
    weights = match_result["weights"]

    # Source pts used for spatial/mask filtering (may differ from returned coords)
    pts_src_for_filter = points_src_geom if points_src_geom is not None else points_src

    # Confidence filter
    valid = weights >= confidence_threshold

    # Displacement filter — uses geom coords (warped), not the original output coords
    if max_displacement is not None:
        matched_src_geom = pts_src_for_filter[src_idx]
        matched_tgt = points_tgt[tgt_idx]
        displacements = np.linalg.norm(matched_src_geom - matched_tgt, axis=1)
        valid &= displacements < max_displacement

    # Mask filter — checked in the respective image's own coordinate space
    if mask_src is not None:
        pts = np.round(pts_src_for_filter[src_idx]).astype(int)
        for i in range(len(pts)):
            z, y, x = pts[i]
            if (z < 0 or z >= mask_src.shape[0] or
                y < 0 or y >= mask_src.shape[1] or
                x < 0 or x >= mask_src.shape[2] or
                mask_src[z, y, x] == 0):
                valid[i] = False

    if mask_tgt is not None:
        pts = np.round(points_tgt[tgt_idx]).astype(int)
        for i in range(len(pts)):
            z, y, x = pts[i]
            if (z < 0 or z >= mask_tgt.shape[0] or
                y < 0 or y >= mask_tgt.shape[1] or
                x < 0 or x >= mask_tgt.shape[2] or
                mask_tgt[z, y, x] == 0):
                valid[i] = False

    return {
        "matched_points_src": points_src[src_idx[valid]],   # original (unwarped) coords
        "matched_points_tgt": points_tgt[tgt_idx[valid]],
        "weights": weights[valid],
        "n_matches": int(valid.sum()),
        "n_before_filter": len(src_idx),
    }
