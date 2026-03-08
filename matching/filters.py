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

    # --------------------------------------------------------------------------
    # FAST SPATIAL RANSAC / LOCAL CONSENSUS FILTER
    # --------------------------------------------------------------------------
    # Many matches are completely random outliers (e.g. 70%). These random matches
    # point in entirely random directions. True matches are locally coherent.
    # For each match, we check its K nearest spatial neighbors. If its displacement
    # vector is wildly different from the median displacement of its neighbors, it's an outlier.

    # Only run consensus filter if we have enough points to build a meaningful neighborhood
    MIN_POINTS_FOR_CONSENSUS = 100
    if valid.sum() > MIN_POINTS_FOR_CONSENSUS:
        from scipy.spatial import cKDTree
        import logging
        logger = logging.getLogger(__name__)

        # Get the current valid geometry points and their displacements
        # Use geometry points to measure distance since they represent the "current" state
        valid_src_geom = pts_src_for_filter[src_idx[valid]]
        valid_tgt = points_tgt[tgt_idx[valid]]
        
        # Calculate displacement vectors for all valid matches
        disp_vectors = valid_tgt - valid_src_geom  # (V, 3) where V is sum(valid)
        
        # Build KD-tree on the source geometry to find local spatial neighborhoods
        tree = cKDTree(valid_src_geom)
        
        # Find 15 nearest neighbors for each point (including itself)
        k_neighbors = min(15, len(valid_src_geom))
        _, neighbor_indices = tree.query(valid_src_geom, k=k_neighbors)  # (V, K)
        
        # For each point, get the displacement vectors of its K neighbors
        # neighbor_indices has shape (V, K) -> gather from disp_vectors (V, 3) 
        # Result: (V, K, 3)
        neighbor_disps = disp_vectors[neighbor_indices]
        
        # Calculate the median displacement vector in each neighborhood
        median_neighbor_disps = np.median(neighbor_disps, axis=1)  # (V, 3)
        
        # Calculate deviation of each point's displacement from its neighborhood's median
        deviations = np.linalg.norm(disp_vectors - median_neighbor_disps, axis=1)  # (V,)
        
        # A match is locally consistent if its deviation is small (e.g., < 15mm)
        CONSENSUS_THRESHOLD = 15.0  # mm
        consensus_valid = deviations < CONSENSUS_THRESHOLD
        
        logger.info(f"  Local consensus filter: kept {consensus_valid.sum()}/{len(valid_src_geom)} "
                    f"matches (threshold={CONSENSUS_THRESHOLD}mm deviation)")
        
        # Update the master valid mask
        # We need to map the boolean array of size V back to the original array of size N
        valid_indices = np.where(valid)[0]
        consensus_invalid_indices = valid_indices[~consensus_valid]
        valid[consensus_invalid_indices] = False

    return {
        "matched_points_src": points_src[src_idx[valid]],   # original (unwarped) coords
        "matched_points_tgt": points_tgt[tgt_idx[valid]],
        "weights": weights[valid],
        "n_matches": int(valid.sum()),
        "n_before_filter": len(src_idx),
    }
