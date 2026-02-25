"""
Intensity-based refinement using local NCC.

Final optional refinement step: after correspondence-driven fitting,
optimizes a residual SVF using local Normalized Cross-Correlation
between fixed and warped-moving images.

Uses strong regularization and few iterations to prevent destabilization.
"""
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .svf import SVFField
from .integrate import scaling_and_squaring, compose_displacements
from .warp import warp_volume

logger = logging.getLogger(__name__)


def local_ncc_loss(
    fixed: torch.Tensor,
    warped: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    win_size: int = 9,
) -> torch.Tensor:
    """
    Local Normalized Cross-Correlation loss.

    Args:
        fixed: (1, 1, D, H, W) fixed image
        warped: (1, 1, D, H, W) warped moving image
        mask: optional (1, 1, D, H, W) trunk mask
        win_size: window size for local NCC

    Returns:
        loss: scalar (1 - mean NCC), lower is better
    """
    ndims = 3
    pad = win_size // 2

    # Local sums using average pooling
    pool = F.avg_pool3d
    
    # Sum of elements in window
    f = fixed
    w = warped
    
    if mask is not None:
        f = f * mask
        w = w * mask
    
    f2 = f * f
    w2 = w * w
    fw = f * w

    f_sum = pool(f, win_size, stride=1, padding=pad)
    w_sum = pool(w, win_size, stride=1, padding=pad)
    f2_sum = pool(f2, win_size, stride=1, padding=pad)
    w2_sum = pool(w2, win_size, stride=1, padding=pad)
    fw_sum = pool(fw, win_size, stride=1, padding=pad)

    win_vol = win_size ** ndims

    f_mean = f_sum
    w_mean = w_sum

    cross = fw_sum - f_mean * w_mean * win_vol
    f_var = f2_sum - f_mean * f_mean * win_vol
    w_var = w2_sum - w_mean * w_mean * win_vol

    denom = torch.sqrt(f_var.clamp(min=1e-5) * w_var.clamp(min=1e-5))
    ncc = cross / denom

    if mask is not None:
        mask_pool = pool(mask, win_size, stride=1, padding=pad)
        ncc = ncc * (mask_pool > 0.5).float()
        return 1.0 - ncc[mask_pool > 0.5].mean()
    
    return 1.0 - ncc.mean()


def intensity_refinement(
    fixed_img: np.ndarray,
    moving_img: np.ndarray,
    current_displacement: torch.Tensor,
    fixed_mask: Optional[np.ndarray] = None,
    grid_spacing: float = 4.0,
    lambda_smooth: float = 5.0,
    lr: float = 1e-3,
    n_iters: int = 50,
    n_squaring_steps: int = 7,
    win_size: int = 9,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Refine displacement field using intensity-based NCC loss.

    Fits a small residual SVF to improve alignment using image similarity.
    Uses strong regularization to prevent destabilization.

    Args:
        fixed_img: (D, H, W) fixed volume
        moving_img: (D, H, W) moving volume
        current_displacement: (1, 3, D, H, W) current displacement from correspondence fitting
        fixed_mask: optional (D, H, W) trunk mask
        grid_spacing: SVF control grid spacing
        lambda_smooth: smoothness weight (high = conservative)
        lr: learning rate
        n_iters: number of optimization iterations
        n_squaring_steps: scaling-and-squaring steps
        win_size: NCC window size
        device: torch device

    Returns:
        refined_displacement: (1, 3, D, H, W) refined displacement field
    """
    volume_shape = fixed_img.shape
    logger.info(f"Intensity refinement: grid_spacing={grid_spacing}, "
                f"iters={n_iters}, λ_smooth={lambda_smooth}")

    # Prepare tensors
    fixed_t = torch.from_numpy(fixed_img).float().unsqueeze(0).unsqueeze(0).to(device)
    moving_t = torch.from_numpy(moving_img).float().unsqueeze(0).unsqueeze(0).to(device)
    
    mask_t = None
    if fixed_mask is not None:
        mask_t = torch.from_numpy(fixed_mask).float().unsqueeze(0).unsqueeze(0).to(device)

    # Normalize images to [0, 1]
    fixed_t = (fixed_t - fixed_t.min()) / (fixed_t.max() - fixed_t.min() + 1e-8)
    moving_t = (moving_t - moving_t.min()) / (moving_t.max() - moving_t.min() + 1e-8)

    # Residual SVF (fine grid)
    svf = SVFField(volume_shape, grid_spacing=grid_spacing, device=device)
    optimizer = torch.optim.Adam(svf.parameters(), lr=lr)

    best_loss = float("inf")
    best_disp = current_displacement.clone()

    for it in range(n_iters):
        optimizer.zero_grad()

        # Get residual displacement
        v = svf.get_velocity_field()
        residual_disp = scaling_and_squaring(v, n_squaring_steps)

        # Compose: total = current + residual
        total_disp = compose_displacements(current_displacement, residual_disp)

        # Warp moving image
        warped = warp_volume(moving_t, total_disp)

        # NCC loss
        loss_ncc = local_ncc_loss(fixed_t, warped, mask_t, win_size)

        # Smoothness regularization (strong)
        loss_smooth = svf.regularization_loss()

        loss = loss_ncc + lambda_smooth * loss_smooth

        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_disp = total_disp.detach().clone()

        if (it + 1) % 10 == 0:
            logger.info(f"  iter {it+1}/{n_iters}: "
                        f"loss={loss.item():.4f} "
                        f"(ncc={loss_ncc.item():.4f}, "
                        f"smooth={loss_smooth.item():.4f})")

    return best_disp
