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


def _check_ot_memory(N: int, M: int, max_n: int = 4000) -> None:
    """
    Guard against N² memory explosion in dense OT/GWOT.

    For N=M=4000: NxN float64 matrix ≈ 128MB. Two GW structural matrices
    + cost + transport ≈ 512MB. Beyond that, POT will either OOM or be
    impractically slow. Raise early with a clear message.
    """
    if N > max_n or M > max_n:
        raise ValueError(
            f"GWOT point set too large (N={N}, M={M}, max={max_n}). "
            f"Dense GW requires O(N²) memory ({N*N*8/1e6:.0f}+{M*M*8/1e6:.0f} MB). "
            "Reduce n_points or switch to 'nn' matcher."
        )


def build_local_distance_matrix(
    points: np.ndarray,
    radius: float,
) -> np.ndarray:
    """
    Build local distance matrix, capped at radius.

    D[i,j] = min(|p_i - p_j|, radius)

    *** IMPORTANT: cap (not zero) far-apart pairs. ***
    In GW, D[i,j]=0 means points i and j are STRUCTURALLY IDENTICAL.
    Setting D=0 for far pairs destroys the structural term — it tells the
    solver these distant points have zero separation, which is the opposite
    of what we want. Capping at radius says "at least this far apart".

    Args:
        points: (N, 3) point coordinates
        radius: cap value (mm); all distances clipped to [0, radius]

    Returns:
        D: (N, N) capped distance matrix
    """
    D = cdist(points, points, metric="euclidean")
    np.clip(D, 0.0, radius, out=D)   # cap — NOT zero
    return D




def nn_matching(
    desc_src: np.ndarray,
    desc_tgt: np.ndarray,
    points_src: np.ndarray,
    points_tgt: np.ndarray,
    sim_threshold: float = 0.3,
    max_displacement: float = 150.0,
    margin_threshold: float = 0.0,
) -> Dict:
    """
    Nearest-neighbor matching with mutual consistency and similarity threshold.

    For each source point, find the closest target by cosine similarity.
    Uses mutual consistency (forward+backward) + minimum similarity threshold.

    Args:
        sim_threshold: minimum cosine similarity to keep
        max_displacement: maximum spatial distance in voxels
        margin_threshold: minimum gap between the best and second-best cosine match

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

    if M >= 2:
        top2 = np.partition(cos_sim, kth=M - 2, axis=1)[:, -2:]
        best = top2.max(axis=1)
        second = top2.min(axis=1)
        margin = best - second
    else:
        margin = np.full(N, np.inf, dtype=cos_sim.dtype)
    margin_ok = margin >= margin_threshold

    # Spatial displacement limit
    if max_displacement is not None and max_displacement > 0:
        disp = np.linalg.norm(
            points_src[src_indices] - points_tgt[fwd_idx], axis=1
        )
        spatial_ok = disp < max_displacement
    else:
        spatial_ok = np.ones(N, dtype=bool)

    # Combine: mutual + sim threshold + spatial
    valid = mutual & sim_ok & margin_ok & spatial_ok

    logger.info(f"  NN matching: {valid.sum()}/{N} matches "
                f"(mutual: {mutual.sum()}, "
                f"sim>{sim_threshold}: {sim_ok.sum()}, "
                f"margin>{margin_threshold}: {margin_ok.sum()}, "
                f"spatial<{max_displacement}: {spatial_ok.sum()})")

    if valid.any():
        scale = max(margin_threshold, 1e-6)
        distinctiveness = np.clip(margin[valid] / scale, 0.0, 1.0)
        weights = fwd_sim[valid] * distinctiveness
    else:
        weights = fwd_sim[valid]

    return {
        "matches_src_idx": src_indices[valid],
        "matches_tgt_idx": fwd_idx[valid],
        "weights": weights,
    }


def ot_matching(
    desc_src: np.ndarray,
    desc_tgt: np.ndarray,
    points_src: np.ndarray,
    points_tgt: np.ndarray,
    lambda_prior: float = 0.2,
    epsilon: float = 0.05,
    lambda_mass: float = 0.0,
    max_iter: int = 100,
) -> Dict:
    """
    Standard (non-Gromov) Optimal Transport matching.

    Uses balanced Sinkhorn (lambda_mass=0) or unbalanced Sinkhorn
    (lambda_mass>0) with feature cost + spatial prior.

    lambda_mass > 0 enables partial overlap / occlusion handling via
    KL-divergence penalty on marginals (ot.unbalanced.sinkhorn_unbalanced).
    Smaller lambda_mass = more unbalanced = more mass can appear/disappear.

    Returns dict with matches and transport matrix.
    """
    if not HAS_POT:
        raise ImportError("POT is required for OT matching: pip install pot")

    N, M = len(desc_src), len(desc_tgt)
    _check_ot_memory(N, M)

    # Cost matrix
    C_feat = compute_feature_cost(desc_src, desc_tgt)
    C_prior = compute_spatial_prior(points_src, points_tgt)
    C = C_feat + lambda_prior * C_prior
    C = C / (C.max() + 1e-8)

    # Uniform marginals
    a = np.ones(N) / N
    b = np.ones(M) / M

    if lambda_mass > 0:
        # Unbalanced: KL penalty on marginals handles partial visibility
        P = ot.unbalanced.sinkhorn_unbalanced(
            a, b, C, reg=epsilon, reg_m=lambda_mass,
            numItermax=max_iter, warn=False,
        )
    else:
        P = ot.sinkhorn(a, b, C, reg=epsilon, numItermax=max_iter, warn=False)

    result = _extract_matches_from_transport(P, N, M, C_feat=C_feat)
    result["transport_matrix"] = P
    return result



def gwot_matching(
    desc_src: np.ndarray,
    desc_tgt: np.ndarray,
    points_src: np.ndarray,
    points_tgt: np.ndarray,
    lambda_gw: float = 0.5,
    lambda_prior: float = 0.2,
    epsilon: float = 0.05,
    lambda_mass: float = 0.0,
    local_radius: float = 12.0,
    max_iter: int = 100,
) -> Dict:
    """
    Full GWOT (Gromov-Wasserstein Optimal Transport) matching.

    Combines:
    - Feature similarity (cosine distance)
    - Spatial structure preservation (GW term — distance matrices CAPPED at radius)
    - Anatomical plausibility (spatial prior)
    - Entropic regularization

    lambda_mass > 0 enables unbalanced/partial GW for partial overlap scenarios.
    Balanced (lambda_mass=0) is the default.

    Memory: O(N²) + O(M²). Use _check_ot_memory guard. Max N~4000.

    Args:
        desc_src: (N, D) source descriptors
        desc_tgt: (M, D) target descriptors
        points_src: (N, 3) source coordinates (mm)
        points_tgt: (M, 3) target coordinates (mm)
        lambda_gw: weight for GW structural term (alpha = lambda_gw/(1+lambda_gw))
        lambda_prior: weight for spatial correspondence prior
        epsilon: entropic regularization
        lambda_mass: >0 enables partial/unbalanced GW for partial overlap
        local_radius: cap radius for structural distance matrices (mm)
        max_iter: max iterations

    Returns dict with:
        matches_src_idx, matches_tgt_idx, weights, transport_matrix
    """
    if not HAS_POT:
        raise ImportError("POT is required for GWOT matching: pip install pot")

    N, M = len(desc_src), len(desc_tgt)
    _check_ot_memory(N, M)

    # 1) Feature cost + spatial prior
    C_feat = compute_feature_cost(desc_src, desc_tgt)
    C_prior = compute_spatial_prior(points_src, points_tgt)
    C = C_feat + lambda_prior * C_prior
    C = C / (C.max() + 1e-8)

    # 2) Structural matrices — CAPPED (not zeroed) at radius
    # Zero-ing would tell GW that far-apart pairs are structurally identical.
    logger.info(f"Building local distance graphs (radius={local_radius}mm, capped)...")
    D_src = build_local_distance_matrix(points_src, local_radius)
    D_tgt = build_local_distance_matrix(points_tgt, local_radius)

    # Normalize to [0, 1]
    D_src = D_src / (D_src.max() + 1e-8)
    D_tgt = D_tgt / (D_tgt.max() + 1e-8)

    # 3) Marginals
    a = np.ones(N) / N
    b = np.ones(M) / M

    # 4) Fused GW — balanced or unbalanced
    alpha = lambda_gw / (1.0 + lambda_gw)  # POT: alpha = weight of GW vs linear term
    logger.info(f"Running fused GW-OT (N={N}, M={M}, λ_gw={lambda_gw}, ε={epsilon}, "
                f"unbalanced={'yes' if lambda_mass > 0 else 'no'})...")

    if lambda_mass > 0:
        # Partial/unbalanced FGW — allows some points to go unmatched
        # m = fraction of mass to transport (smaller = more unbalanced)
        m = min(0.99, 1.0 / (1.0 + lambda_mass))
        try:
            P, _ = ot.gromov.partial_fused_gromov_wasserstein(
                M=C, C1=D_src, C2=D_tgt, p=a, q=b,
                m=m, loss_fun="square_loss", alpha=alpha,
                numItermax=max_iter, log=True,
            )
        except Exception as e:
            logger.warning(f"partial_fgw failed ({e}), falling back to balanced FGW")
            P = ot.gromov.fused_gromov_wasserstein(
                M=C, C1=D_src, C2=D_tgt, p=a, q=b,
                loss_fun="square_loss", alpha=alpha,
                armijo=False, log=False, numItermax=max_iter,
            )
    else:
        # Use entropic FGW so epsilon is actually applied.
        # ot.gromov.fused_gromov_wasserstein is conditional-gradient (no epsilon).
        # ot.gromov.entropic_fused_gromov_wasserstein is the entropy-regularized version.
        try:
            P = ot.gromov.entropic_fused_gromov_wasserstein(
                M=C, C1=D_src, C2=D_tgt, p=a, q=b,
                loss_fun="square_loss", epsilon=epsilon, alpha=alpha,
                numItermax=max_iter,
            )
        except AttributeError:
            # Older POT versions may not have this — fall back with a warning
            logger.warning(
                "ot.gromov.entropic_fused_gromov_wasserstein not found in this POT version. "
                "Falling back to conditional-gradient FGW (epsilon will be ignored). "
                "Upgrade POT: pip install pot --upgrade"
            )
            P = ot.gromov.fused_gromov_wasserstein(
                M=C, C1=D_src, C2=D_tgt, p=a, q=b,
                loss_fun="square_loss", alpha=alpha,
                armijo=False, log=False, numItermax=max_iter,
            )

    result = _extract_matches_from_transport(P, N, M, C_feat=C_feat)
    result["transport_matrix"] = P
    return result


def _extract_matches_from_transport(
    P: np.ndarray,
    N: int,
    M: int,
    C_feat: Optional[np.ndarray] = None,
    top_k: Optional[int] = None,
) -> Dict:
    """
    Extract correspondences from OT/GWOT transport matrix.

    Strategy (in order of preference):
    1. LAP (Linear Assignment Problem) on the transport plan — finds the
       optimal 1:1 assignment that maximizes total transported mass.
       Much better than mutual argmax for entropic plans which spread mass.
    2. Confidence filter — keep only assignments where the transport plan
       concentrates enough mass (relative to uniform 1/M).

    Returns dict with matches_src_idx, matches_tgt_idx, weights.
    """
    from scipy.optimize import linear_sum_assignment

    # --- LAP assignment on negative transport plan ---
    # Maximize total mass → minimize negative mass.
    # LAP alone is too permissive for diffuse transport plans: it will always
    # return a dense 1:1 assignment even when the plan is nearly uniform.
    row_ind, col_ind = linear_sum_assignment(-P)

    # Transport mass at each assignment
    assignment_mass = P[row_ind, col_ind]

    row_mass = P.sum(axis=1)
    col_mass = P.sum(axis=0)
    expected_mass = row_mass[row_ind] * col_mass[col_ind]

    # Concentration above independent marginals. For balanced OT this reduces
    # to the old uniform baseline; for unbalanced OT it stays meaningful.
    confidence = assignment_mass / (expected_mass + 1e-15)

    # How much of the row/column transport is explained by this assignment.
    # Diffuse plans have tiny shares even if LAP forces an assignment.
    row_share = assignment_mass / (row_mass[row_ind] + 1e-15)
    col_share = assignment_mass / (col_mass[col_ind] + 1e-15)

    MIN_CONCENTRATION = 1.5
    MIN_ROW_SHARE = 2.5 / max(M, 1)
    MIN_COL_SHARE = 2.5 / max(N, 1)
    valid = (
        (confidence > MIN_CONCENTRATION)
        & (row_share > MIN_ROW_SHARE)
        & (col_share > MIN_COL_SHARE)
    )

    logger.info(
        "  LAP extraction: %d/%d assignments kept "
        "(lift>%.1fx expected: %d, row_share>%.4f: %d, col_share>%.4f: %d)"
        % (
            int(valid.sum()),
            len(row_ind),
            MIN_CONCENTRATION,
            int((confidence > MIN_CONCENTRATION).sum()),
            MIN_ROW_SHARE,
            int((row_share > MIN_ROW_SHARE).sum()),
            MIN_COL_SHARE,
            int((col_share > MIN_COL_SHARE).sum()),
        )
    )

    matches_src = row_ind[valid]
    matches_tgt = col_ind[valid]

    # Also log mutual-argmax count for comparison
    fwd_idx = np.argmax(P, axis=1)
    bwd_idx = np.argmax(P, axis=0)
    n_mutual = np.sum(bwd_idx[fwd_idx] == np.arange(N))
    logger.info(f"  (For reference: mutual-argmax would give {n_mutual} matches)")

    # Weights: use feature similarity when available, slightly boosted by
    # transport concentration so diffuse-but-similar pairs do not dominate.
    if len(matches_src) == 0:
        return {
            "matches_src_idx": matches_src,
            "matches_tgt_idx": matches_tgt,
            "weights": np.zeros(0, dtype=np.float64),
        }

    if C_feat is not None:
        feat_sim = 1.0 - C_feat[matches_src, matches_tgt]
        trans_conf = confidence[valid]
        trans_conf = trans_conf / (trans_conf.max() + 1e-15)
        weights = feat_sim * np.sqrt(np.clip(trans_conf, 0.0, 1.0))
    else:
        weights = confidence[valid] / confidence[valid].max()  # normalize to [0, 1]

    if top_k is None:
        top_k = min(max(200, int(0.35 * min(N, M))), 1000)

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
        return nn_matching(desc_src, desc_tgt, points_src, points_tgt, **kwargs)
    elif method == "ot":
        return ot_matching(desc_src, desc_tgt, points_src, points_tgt, **kwargs)
    elif method == "gwot":
        return gwot_matching(desc_src, desc_tgt, points_src, points_tgt, **kwargs)
    else:
        raise ValueError(f"Unknown matching method: {method}")
