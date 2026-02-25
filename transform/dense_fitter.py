"""
Dense feature-based fitter: optimizes displacement by directly minimizing
dense feature L2 distance between warped moving and fixed feature volumes.

Inspired by DINO-Reg's Adam instance optimization (convex_adam_MIND.py).
Instead of sparse point-to-point correspondences, uses dense feature
alignment at a downsampled resolution with diffusion regularization.

This avoids the fundamental failure mode of sparse matching: with noisy
features, explicit correspondences are unreliable. Dense optimization
lets the whole field be guided by the feature landscape gradient.
"""
import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def dense_feature_registration(
    fixed_feats: np.ndarray,
    moving_feats: np.ndarray,
    volume_shape: tuple,
    grid_sp_adam: int = 2,
    lambda_weight: float = 1.25,
    lr: float = 1.0,
    n_iters: int = 300,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Dense feature-based registration using Adam optimization.

    Directly optimizes a displacement field to minimize the L2 distance
    between fixed features and warped-moving features at every voxel.

    Following DINO-Reg's convex_adam approach:
    1. Downsample features by grid_sp_adam
    2. Initialize displacement (optionally from coarse stage)
    3. Optimize with Adam: feature L2 loss + diffusion regularization

    Args:
        fixed_feats: (C, fD, fH, fW) fixed feature volume
        moving_feats: (C, fD, fH, fW) moving feature volume
        volume_shape: (D, H, W) original volume shape for output displacement
        grid_sp_adam: downsampling factor for optimization grid
        lambda_weight: diffusion regularization weight
        lr: learning rate (1.0 like DINO-Reg, very aggressive)
        n_iters: number of Adam iterations
        device: torch device

    Returns:
        displacement: (1, 3, D, H, W) displacement field in voxel coordinates
    """
    C, fD, fH, fW = fixed_feats.shape
    D, H, W = volume_shape

    logger.info(f"Dense feature registration: feat=({C},{fD},{fH},{fW}), "
                f"vol=({D},{H},{W}), grid_sp={grid_sp_adam}, "
                f"lr={lr}, iters={n_iters}, λ={lambda_weight}")

    # Convert to torch tensors on GPU
    fix_feat_t = torch.from_numpy(fixed_feats).float().unsqueeze(0).to(device)  # (1, C, fD, fH, fW)
    mov_feat_t = torch.from_numpy(moving_feats).float().unsqueeze(0).to(device)

    # Downsample features for optimization
    if grid_sp_adam > 1:
        fix_down = F.avg_pool3d(fix_feat_t, grid_sp_adam, stride=grid_sp_adam)
        mov_down = F.avg_pool3d(mov_feat_t, grid_sp_adam, stride=grid_sp_adam)
    else:
        fix_down = fix_feat_t
        mov_down = mov_feat_t

    _, _, gD, gH, gW = fix_down.shape
    logger.info(f"  Optimization grid: ({gD}, {gH}, {gW})")

    # Create optimizable displacement grid
    # Following DINO-Reg: use a Conv3d layer to hold the displacement
    net = nn.Sequential(nn.Conv3d(3, 1, (gD, gH, gW), bias=False))
    net[0].weight.data.zero_()  # Initialize to zero displacement
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # Identity grid for grid_sample
    grid0 = F.affine_grid(
        torch.eye(3, 4).unsqueeze(0).to(device),
        (1, 1, gD, gH, gW),
        align_corners=False,
    )

    best_loss = float("inf")
    best_disp = None

    for it in range(n_iters):
        optimizer.zero_grad()

        # Get smoothed displacement (triple avg_pool3d for B-spline-like smoothing)
        disp_sample = F.avg_pool3d(
            F.avg_pool3d(
                F.avg_pool3d(net[0].weight, 3, stride=1, padding=1),
                3, stride=1, padding=1),
            3, stride=1, padding=1
        ).permute(0, 2, 3, 4, 1)  # (1, gD, gH, gW, 3)

        # Diffusion regularization: penalize spatial gradients of displacement
        reg_loss = lambda_weight * (
            ((disp_sample[0, :, 1:, :] - disp_sample[0, :, :-1, :]) ** 2).mean() +
            ((disp_sample[0, 1:, :, :] - disp_sample[0, :-1, :, :]) ** 2).mean() +
            ((disp_sample[0, :, :, 1:] - disp_sample[0, :, :, :-1]) ** 2).mean()
        )

        # Convert displacement to normalized grid coordinates for grid_sample
        scale = torch.tensor(
            [(gD - 1) / 2, (gH - 1) / 2, (gW - 1) / 2],
            device=device,
        ).unsqueeze(0)
        grid_disp = grid0.view(-1, 3).float() + (disp_sample.view(-1, 3) / scale).flip(1).float()
        grid_disp = grid_disp.view(1, gD, gH, gW, 3)

        # Warp moving features
        mov_warped = F.grid_sample(
            mov_down.float(), grid_disp,
            align_corners=False, mode='bilinear',
        )

        # Feature L2 loss (SSD)
        feat_loss = ((mov_warped - fix_down.float()) ** 2).mean(1).mean() * 12

        loss = feat_loss + reg_loss
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_disp = disp_sample.detach().clone()

        if (it + 1) % 50 == 0 or it == 0:
            logger.info(f"  iter {it+1}/{n_iters}: loss={loss.item():.4f} "
                        f"(feat={feat_loss.item():.4f}, reg={reg_loss.item():.4f})")

    # Scale displacement to voxel coordinates and upsample to full resolution
    # best_disp is in optimization-grid units, convert to feature-grid units
    fitted_grid = best_disp.permute(0, 4, 1, 2, 3)  # (1, 3, gD, gH, gW)

    # Scale from optimization-grid to feature-grid coordinates
    disp_feat_res = fitted_grid * grid_sp_adam  # (1, 3, gD, gH, gW)

    # Now upsample to feature grid resolution
    disp_feat = F.interpolate(
        disp_feat_res, size=(fD, fH, fW),
        mode='trilinear', align_corners=False,
    )

    # Convert from feature-grid coordinates to voxel coordinates
    # Feature grid (fD, fH, fW) maps to volume (D, H, W)
    scale_z = D / fD
    scale_y = H / fH
    scale_x = W / fW

    disp_voxel = torch.zeros(1, 3, fD, fH, fW, device=device)
    disp_voxel[:, 0] = disp_feat[:, 0] * scale_z  # z
    disp_voxel[:, 1] = disp_feat[:, 1] * scale_y  # y
    disp_voxel[:, 2] = disp_feat[:, 2] * scale_x  # x

    # Note: the optimizer minimizes ||grid_sample(moving, identity+disp) - fixed||^2
    # From the diagnostics, x-axis had correct sign WITHOUT negation (corr=0.707),
    # indicating the displacement convention already matches warp_points (moving + d → fixed).

    # Upsample to full volume resolution
    displacement = F.interpolate(
        disp_voxel, size=(D, H, W),
        mode='trilinear', align_corners=False,
    )

    logger.info(f"  Final displacement: max={displacement.abs().max().item():.1f} voxels, "
                f"mean={displacement.abs().mean().item():.1f} voxels")

    return displacement
