"""
GWOT (Gromov-Wasserstein Optimal Transport) 3D correspondence matcher.

Implements feature + spatial structure matching for sparse 3D point sets.
Supports: nearest-neighbor baseline, standard OT, and full GWOT.
"""
import logging
from typing import Dict, Literal, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

# Try to import POT (Python Optimal Transport)
try:
    import ot
    HAS_POT = True
except ImportError:
    HAS_POT = False
    logger.warning("POT not installed. Only 'nn' matcher available. Install: pip install pot")


def compute_feature_cost(
    desc_src: np.ndarray,
    desc_tgt: np.ndarray,
) -> np.ndarray:
    """
    Compute feature cost matrix: C_feat[i,j] = 1 - cos(f_i, g_j).

    Args:
        desc_src: (N, D) source descriptors (L2-normalized)
        desc_tgt: (M, D) target descriptors (L2-normalized)

    Returns:
        C_feat: (N, M) cost matrix
    """
    # Cosine similarity → cosine distance
    cos_sim = desc_src @ desc_tgt.T  # (N, M)
    return 1.0 - cos_sim


def compute_spatial_prior(
    points_src: np.ndarray,
    points_tgt: np.ndarray,
    sigma: float = 50.0,
) -> np.ndarray:
    """
    Compute spatial prior cost: C_prior[i,j] = |x_i - y_j|^2 / σ^2.

    Encourages matches that are spatially plausible (nearby after pre-alignment).

    Args:
        points_src: (N, 3) source coordinates
        points_tgt: (M, 3) target coordinates
        sigma: spatial scale parameter (mm)

    Returns:
        C_prior: (N, M) spatial prior cost
    """
    dist_sq = cdist(points_src, points_tgt, metric="sqeuclidean")
    return dist_sq / (sigma ** 2)


def build_local_distance_matrix(
    points: np.ndarray,
    radius: float,
) -> np.ndarray:
    """
    Build local distance matrix with truncation at radius.
    D[i,j] = |p_i - p_j| if < radius, else 0.

    Used for Gromov-Wasserstein structural matching.

    Args:
        points: (N, 3) point coordinates
        radius: truncation radius (mm)

    Returns:
        D: (N, N) truncated distance matrix
    """
    D = cdist(points, points, metric="euclidean")
    D[D > radius] = 0.0
    return D


def nn_matching(
    desc_src: np.ndarray,
    desc_tgt: np.ndarray,
    points_src: np.ndarray,
    points_tgt: np.ndarray,
    sim_threshold: float = 0.3,
    max_displacement: float = 150.0,
) -> Dict:
    """
    Nearest-neighbor matching with mutual consistency and similarity threshold.

    For each source point, find the closest target by cosine similarity.
    Uses mutual consistency (forward+backward) + minimum similarity threshold.

    Args:
        sim_threshold: minimum cosine similarity to keep
        max_displacement: maximum spatial distance in voxels

    Returns dict with:
        matches_src_idx: (K,) indices into source
        matches_tgt_idx: (K,) indices into target
        weights: (K,) confidence weights (cosine similarity)
    """
    cos_sim = desc_src @ desc_tgt.T  # (N, M)
    N, M = cos_sim.shape

    # Forward matches: src → tgt
    fwd_idx = np.argmax(cos_sim, axis=1)  # (N,)
    fwd_sim = cos_sim[np.arange(N), fwd_idx]

    # Backward matches: tgt → src
    bwd_idx = np.argmax(cos_sim, axis=0)  # (M,)

    # Mutual consistency filter
    src_indices = np.arange(N)
    mutual = bwd_idx[fwd_idx] == src_indices

    # Similarity threshold
    sim_ok = fwd_sim >= sim_threshold

    # Spatial displacement limit
    if max_displacement is not None and max_displacement > 0:
        disp = np.linalg.norm(
            points_src[src_indices] - points_tgt[fwd_idx], axis=1
        )
        spatial_ok = disp < max_displacement
    else:
        spatial_ok = np.ones(N, dtype=bool)

    # Combine: mutual + sim threshold + spatial
    valid = mutual & sim_ok & spatial_ok

    logger.info(f"  NN matching: {valid.sum()}/{N} matches "
                f"(mutual: {mutual.sum()}, "
                f"sim>{sim_threshold}: {sim_ok.sum()}, "
                f"spatial<{max_displacement}: {spatial_ok.sum()})")

    return {
        "matches_src_idx": src_indices[valid],
        "matches_tgt_idx": fwd_idx[valid],
        "weights": fwd_sim[valid],
    }


def ot_matching(
    desc_src: np.ndarray,
    desc_tgt: np.ndarray,
    points_src: np.ndarray,
    points_tgt: np.ndarray,
    lambda_prior: float = 0.2,
    epsilon: float = 0.05,
    lambda_mass: float = 1.0,
    max_iter: int = 100,
) -> Dict:
    """
    Standard (non-Gromov) Optimal Transport matching.

    Uses Sinkhorn with feature cost + spatial prior.

    Returns dict with matches and transport matrix.
    """
    if not HAS_POT:
        raise ImportError("POT is required for OT matching: pip install pot")

    N, M = len(desc_src), len(desc_tgt)

    # Cost matrix
    C_feat = compute_feature_cost(desc_src, desc_tgt)
    C_prior = compute_spatial_prior(points_src, points_tgt)
    C = C_feat + lambda_prior * C_prior

    # Normalize cost
    C = C / C.max()

    # Uniform marginals
    a = np.ones(N) / N
    b = np.ones(M) / M

    # Sinkhorn
    P = ot.sinkhorn(
        a, b, C, reg=epsilon, numItermax=max_iter, warn=False
    )

    return _extract_matches_from_transport(P, N, M)


def gwot_matching(
    desc_src: np.ndarray,
    desc_tgt: np.ndarray,
    points_src: np.ndarray,
    points_tgt: np.ndarray,
    lambda_gw: float = 0.5,
    lambda_prior: float = 0.2,
    epsilon: float = 0.05,
    lambda_mass: float = 1.0,
    local_radius: float = 12.0,
    max_iter: int = 100,
) -> Dict:
    """
    Full GWOT (Gromov-Wasserstein Optimal Transport) matching.

    Combines:
    - Feature similarity (cosine distance)
    - Spatial structure preservation (GW term with local distance graphs)
    - Anatomical plausibility (spatial prior)
    - Entropic regularization

    Args:
        desc_src: (N, D) source descriptors
        desc_tgt: (M, D) target descriptors
        points_src: (N, 3) source coordinates (mm)
        points_tgt: (M, 3) target coordinates (mm)
        lambda_gw: weight for GW structural term
        lambda_prior: weight for spatial correspondence prior
        epsilon: entropic regularization
        lambda_mass: unbalanced mass regularization (not used in balanced version)
        local_radius: radius for local distance graphs (mm)
        max_iter: max iterations

    Returns dict with:
        matches_src_idx, matches_tgt_idx, weights, transport_matrix
    """
    if not HAS_POT:
        raise ImportError("POT is required for GWOT matching: pip install pot")

    N, M = len(desc_src), len(desc_tgt)

    # 1) Feature cost
    C_feat = compute_feature_cost(desc_src, desc_tgt)
    C_prior = compute_spatial_prior(points_src, points_tgt)
    C = C_feat + lambda_prior * C_prior
    C = C / (C.max() + 1e-8)

    # 2) Spatial structure matrices (local distance graphs)
    logger.info(f"Building local distance graphs (radius={local_radius}mm)...")
    D_src = build_local_distance_matrix(points_src, local_radius)
    D_tgt = build_local_distance_matrix(points_tgt, local_radius)

    # Normalize
    D_src = D_src / (D_src.max() + 1e-8)
    D_tgt = D_tgt / (D_tgt.max() + 1e-8)

    # 3) Uniform marginals
    a = np.ones(N) / N
    b = np.ones(M) / M

    # 4) Fused Gromov-Wasserstein
    logger.info(f"Running fused GW-OT (N={N}, M={M}, λ_gw={lambda_gw}, ε={epsilon})...")
    alpha = lambda_gw / (1.0 + lambda_gw)  # POT's alpha is weight of GW vs linear

    P = ot.gromov.fused_gromov_wasserstein(
        M=C,
        C1=D_src,
        C2=D_tgt,
        p=a,
        q=b,
        loss_fun="square_loss",
        alpha=alpha,
        armijo=False,
        log=False,
        numItermax=max_iter,
    )

    result = _extract_matches_from_transport(P, N, M)
    result["transport_matrix"] = P
    return result


def _extract_matches_from_transport(
    P: np.ndarray,
    N: int,
    M: int,
    top_k: Optional[int] = None,
) -> Dict:
    """
    Extract correspondences from transport matrix.

    Uses mutual maximum assignment.

    Returns dict with matches_src_idx, matches_tgt_idx, weights.
    """
    # Forward: each source's best target
    fwd_idx = np.argmax(P, axis=1)  # (N,)
    fwd_val = P[np.arange(N), fwd_idx]

    # Backward: each target's best source
    bwd_idx = np.argmax(P, axis=0)  # (M,)

    # Mutual consistency
    src_indices = np.arange(N)
    mutual = bwd_idx[fwd_idx] == src_indices

    matches_src = src_indices[mutual]
    matches_tgt = fwd_idx[mutual]
    weights = fwd_val[mutual]

    if top_k is not None and len(matches_src) > top_k:
        top_indices = np.argsort(-weights)[:top_k]
        matches_src = matches_src[top_indices]
        matches_tgt = matches_tgt[top_indices]
        weights = weights[top_indices]

    return {
        "matches_src_idx": matches_src,
        "matches_tgt_idx": matches_tgt,
        "weights": weights,
    }


def match(
    desc_src: np.ndarray,
    desc_tgt: np.ndarray,
    points_src: np.ndarray,
    points_tgt: np.ndarray,
    method: Literal["nn", "ot", "gwot"] = "gwot",
    **kwargs,
) -> Dict:
    """
    Dispatch to the appropriate matching method.

    Args:
        method: 'nn', 'ot', or 'gwot'
        **kwargs: passed to the specific matcher

    Returns:
        Match result dict with matches_src_idx, matches_tgt_idx, weights
    """
    if method == "nn":
        return nn_matching(desc_src, desc_tgt, points_src, points_tgt)
    elif method == "ot":
        return ot_matching(desc_src, desc_tgt, points_src, points_tgt, **kwargs)
    elif method == "gwot":
        return gwot_matching(desc_src, desc_tgt, points_src, points_tgt, **kwargs)
    else:
        raise ValueError(f"Unknown matching method: {method}")
