"""
MIND-SSC feature extraction and ConvexAdam registration.

Implements the proven baseline from DINO-Reg (convex_adam_MIND.py):
  1. MIND-SSC features at full voxel resolution (not patch-level!)
  2. Coupled convex optimization for coarse initialization 
  3. Adam instance optimization with diffusion regularization

This operates at FULL volume resolution (390×280×300), giving 33M feature
points vs 25K from DINOv3 triplanar. This is the key to making registration work.

Adapted from: https://github.com/RPIDIAL/DINO-Reg/blob/main/convex_adam_MIND.py
"""
import logging
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def pdist_squared(x):
    """Pairwise squared distances."""
    xx = (x ** 2).sum(dim=2).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x, x.permute(0, 2, 1))
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, np.inf)
    return dist


def MINDSSC(img, radius=2, dilation=2):
    """
    Compute MIND-SSC descriptor at every voxel.
    
    See: http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf
    
    Args:
        img: (1, 1, D, H, W) input volume
        radius: patch radius for averaging
        dilation: dilation for self-similarity computation
        
    Returns:
        mind: (1, 12, D, H, W) MIND-SSC descriptor
    """
    device = img.device
    kernel_size = radius * 2 + 1
    
    # Define 6-neighbourhood for self-similarity pattern
    six_neighbourhood = torch.tensor([[0,1,1],
                                      [1,1,0],
                                      [1,0,1],
                                      [1,1,2],
                                      [2,1,1],
                                      [1,2,1]], dtype=torch.long)
    
    # Squared distances between neighbourhood points (6×6 pairwise)
    dist = pdist_squared(six_neighbourhood.float().unsqueeze(0)).squeeze(0)
    
    # Comparison mask: pairs at distance 2
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6), indexing='ij')
    mask = ((x > y).view(-1) & (dist == 2).view(-1))
    
    # Build shift kernels
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1,6,1).view(-1,3)[mask,:]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6,1,1).view(-1,3)[mask,:]
    
    mshift1 = torch.zeros(12, 1, 3, 3, 3, device=device)
    mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:,0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
    mshift2 = torch.zeros(12, 1, 3, 3, 3, device=device)
    mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:,0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
    
    rpad1 = nn.ReplicationPad3d(dilation)
    rpad2 = nn.ReplicationPad3d(radius)
    
    # Compute patch-SSD
    ssd = F.avg_pool3d(
        rpad2((F.conv3d(rpad1(img), mshift1, dilation=dilation) - 
               F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2),
        kernel_size, stride=1
    )
    
    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(mind_var, mind_var.mean().item()*0.001, mind_var.mean().item()*1000)
    mind /= mind_var
    mind = torch.exp(-mind)
    
    # Permute to canonical ordering
    mind = mind[:, torch.tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3], dtype=torch.long), :, :, :]
    
    return mind


def correlate(mind_fix, mind_mov, disp_hw, grid_sp, shape, ch=12):
    """Compute dense SSD cost volume between fixed and moving MIND features."""
    H, W, D = int(shape[0]), int(shape[1]), int(shape[2])
    
    with torch.no_grad():
        mind_unfold = F.unfold(
            F.pad(mind_mov, (disp_hw,)*6).squeeze(0),
            disp_hw * 2 + 1
        )
        mind_unfold = mind_unfold.view(ch, -1, (disp_hw*2+1)**2, W//grid_sp, D//grid_sp)

    ssd = torch.zeros(
        (disp_hw*2+1)**3, H//grid_sp, W//grid_sp, D//grid_sp,
        dtype=mind_fix.dtype, device=mind_fix.device
    )
    
    with torch.no_grad():
        for i in range(disp_hw*2+1):
            mind_sum = (mind_fix.permute(1,2,0,3,4) - mind_unfold[:, i:i+H//grid_sp]).pow(2).sum(0, keepdim=True)
            ssd[i::(disp_hw*2+1)] = F.avg_pool3d(
                F.avg_pool3d(mind_sum.transpose(2,1), 3, stride=1, padding=1),
                3, stride=1, padding=1
            ).squeeze(1)
        
        ssd = ssd.view(
            disp_hw*2+1, disp_hw*2+1, disp_hw*2+1,
            H//grid_sp, W//grid_sp, D//grid_sp
        ).transpose(1, 0).reshape((disp_hw*2+1)**3, H//grid_sp, W//grid_sp, D//grid_sp)
        ssd_argmin = torch.argmin(ssd, 0)
    
    return ssd, ssd_argmin


def coupled_convex(ssd, ssd_argmin, disp_mesh_t, grid_sp, shape):
    """Coupled convex optimization for global regularization."""
    H, W, D = int(shape[0]), int(shape[1]), int(shape[2])
    
    disp_soft = F.avg_pool3d(
        disp_mesh_t.view(3,-1)[:, ssd_argmin.view(-1)].reshape(
            1, 3, H//grid_sp, W//grid_sp, D//grid_sp
        ),
        3, padding=1, stride=1
    )
    
    coeffs = torch.tensor([0.003, 0.01, 0.03, 0.1, 0.3, 1])
    for j in range(6):
        ssd_coupled_argmin = torch.zeros_like(ssd_argmin)
        with torch.no_grad():
            for i in range(H//grid_sp):
                coupled = ssd[:, i, :, :] + coeffs[j] * (
                    disp_mesh_t - disp_soft[:, :, i].view(3, 1, -1)
                ).pow(2).sum(0).view(-1, W//grid_sp, D//grid_sp)
                ssd_coupled_argmin[i] = torch.argmin(coupled, 0)
        
        disp_soft = F.avg_pool3d(
            disp_mesh_t.view(3,-1)[:, ssd_coupled_argmin.view(-1)].reshape(
                1, 3, H//grid_sp, W//grid_sp, D//grid_sp
            ),
            3, padding=1, stride=1
        )
    
    return disp_soft


def inverse_consistency(disp_field1s, disp_field2s, iter=20):
    """Enforce inverse consistency between forward and backward displacement."""
    B, C, H, W, D = disp_field1s.size()
    
    # identity grid
    grid_sp = 1
    identity = F.affine_grid(
        torch.eye(3, 4).unsqueeze(0).to(disp_field1s.device).half(),
        (B, 1, H, W, D), align_corners=True
    )
    
    for i in range(iter):
        disp_field1i = F.grid_sample(
            disp_field1s, (identity - disp_field2s.permute(0,2,3,4,1)),
            align_corners=True, padding_mode='border'
        )
        disp_field2i = F.grid_sample(
            disp_field2s, (identity - disp_field1s.permute(0,2,3,4,1)),
            align_corners=True, padding_mode='border'
        )
        disp_field1s = (disp_field1i - disp_field2i) / 2
        disp_field2s = (disp_field2i - disp_field1i) / 2
    
    return disp_field1s, disp_field2s


def mind_convex_adam(
    fixed_img: np.ndarray,
    moving_img: np.ndarray,
    mind_r: int = 1,
    mind_d: int = 2,
    lambda_weight: float = 1.25,
    grid_sp: int = 6,
    disp_hw: int = 4,
    n_iter_adam: int = 300,
    grid_sp_adam: int = 2,
    ic: bool = True,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Full ConvexAdam registration with MIND-SSC features.
    
    Direct port of DINO-Reg's proven baseline.
    
    Args:
        fixed_img: (D, H, W) fixed volume, float32
        moving_img: (D, H, W) moving volume, float32
        mind_r: MIND radius
        mind_d: MIND dilation
        lambda_weight: Adam regularization weight
        grid_sp: grid spacing for convex init
        disp_hw: displacement half-width for correlation
        n_iter_adam: Adam optimization iterations
        grid_sp_adam: Adam grid spacing
        ic: inverse consistency
        device: torch device
        
    Returns:
        displacement: (1, 3, D, H, W) displacement field in voxels
                      Convention: d(x_fixed) = moving_pos - fixed_pos
                      (to be negated for warp_points which does moving + d → fixed)
    """
    t0 = time.time()
    D, H, W = fixed_img.shape
    logger.info(f"MIND ConvexAdam: vol=({D},{H},{W}), grid_sp={grid_sp}, "
                f"disp_hw={disp_hw}, adam_iters={n_iter_adam}")
    
    # Prepare tensors
    img_fixed = torch.from_numpy(fixed_img).float().unsqueeze(0).unsqueeze(0).to(device)
    img_moving = torch.from_numpy(moving_img).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # Extract MIND-SSC features
    with torch.no_grad():
        features_fix = MINDSSC(img_fixed, mind_r, mind_d).half()
        features_mov = MINDSSC(img_moving, mind_r, mind_d).half()
        logger.info(f"  MIND features: {features_fix.shape}")
        
        features_fix_smooth = F.avg_pool3d(features_fix, grid_sp, stride=grid_sp)
        features_mov_smooth = F.avg_pool3d(features_mov, grid_sp, stride=grid_sp)
        n_ch = features_fix_smooth.shape[1]
    logger.info(f"  Smoothed features: {features_fix_smooth.shape}")
    
    # Step 1: Compute correlation volume
    ssd, ssd_argmin = correlate(
        features_fix_smooth, features_mov_smooth,
        disp_hw, grid_sp, (D, H, W), n_ch
    )
    
    # Displacement mesh
    disp_mesh_t = F.affine_grid(
        disp_hw * torch.eye(3, 4, device=device).half().unsqueeze(0),
        (1, 1, disp_hw*2+1, disp_hw*2+1, disp_hw*2+1),
        align_corners=True
    ).permute(0, 4, 1, 2, 3).reshape(3, -1, 1)
    
    # Step 2: Coupled convex optimization
    disp_soft = coupled_convex(ssd, ssd_argmin, disp_mesh_t, grid_sp, (D, H, W))
    logger.info(f"  Convex init: max_disp={disp_soft.abs().max().item():.1f}")
    
    # Step 3: Inverse consistency
    if ic:
        scale = torch.tensor(
            [D//grid_sp-1, H//grid_sp-1, W//grid_sp-1],
            device=device
        ).view(1, 3, 1, 1, 1).half() / 2
        
        ssd_, ssd_argmin_ = correlate(
            features_mov_smooth, features_fix_smooth,
            disp_hw, grid_sp, (D, H, W), n_ch
        )
        disp_soft_ = coupled_convex(ssd_, ssd_argmin_, disp_mesh_t, grid_sp, (D, H, W))
        disp_ice, _ = inverse_consistency(
            (disp_soft / scale).flip(1),
            (disp_soft_ / scale).flip(1),
            iter=15
        )
        disp_hr = F.interpolate(
            disp_ice.flip(1) * scale * grid_sp,
            size=(D, H, W), mode='trilinear', align_corners=False
        )
        logger.info(f"  After IC: max_disp={disp_hr.abs().max().item():.1f}")
    else:
        disp_hr = F.interpolate(
            disp_soft * grid_sp, size=(D, H, W), mode='trilinear', align_corners=False
        )
    
    # Step 4: Adam instance optimization
    if lambda_weight > 0:
        with torch.no_grad():
            patch_features_fix = F.avg_pool3d(features_fix, grid_sp_adam, stride=grid_sp_adam)
            patch_features_mov = F.avg_pool3d(features_mov, grid_sp_adam, stride=grid_sp_adam)
        
        # Create optimizable displacement grid
        disp_lr = F.interpolate(
            disp_hr,
            size=(D//grid_sp_adam, H//grid_sp_adam, W//grid_sp_adam),
            mode='trilinear', align_corners=False
        )
        
        net = nn.Sequential(nn.Conv3d(
            3, 1, (D//grid_sp_adam, H//grid_sp_adam, W//grid_sp_adam), bias=False
        ))
        net[0].weight.data[:] = disp_lr.float().cpu().data / grid_sp_adam
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=1)
        
        grid0 = F.affine_grid(
            torch.eye(3, 4).unsqueeze(0).to(device),
            (1, 1, D//grid_sp_adam, H//grid_sp_adam, W//grid_sp_adam),
            align_corners=False
        )
        
        # Adam optimization with diffusion regularization
        for it in range(n_iter_adam):
            optimizer.zero_grad()
            
            disp_sample = F.avg_pool3d(
                F.avg_pool3d(
                    F.avg_pool3d(net[0].weight, 3, stride=1, padding=1),
                    3, stride=1, padding=1),
                3, stride=1, padding=1
            ).permute(0, 2, 3, 4, 1)
            
            reg_loss = lambda_weight * (
                ((disp_sample[0,:,1:,:] - disp_sample[0,:,:-1,:]) ** 2).mean() +
                ((disp_sample[0,1:,:,:] - disp_sample[0,:-1,:,:]) ** 2).mean() +
                ((disp_sample[0,:,:,1:] - disp_sample[0,:,:,:-1]) ** 2).mean()
            )
            
            scale_adam = torch.tensor([
                (D//grid_sp_adam - 1) / 2,
                (H//grid_sp_adam - 1) / 2,
                (W//grid_sp_adam - 1) / 2,
            ], device=device).unsqueeze(0)
            
            grid_disp = grid0.view(-1, 3).float() + (
                (disp_sample.view(-1, 3)) / scale_adam
            ).flip(1).float()
            
            patch_mov_sampled = F.grid_sample(
                patch_features_mov.float(),
                grid_disp.view(1, D//grid_sp_adam, H//grid_sp_adam, W//grid_sp_adam, 3),
                align_corners=False, mode='bilinear'
            )
            
            sampled_cost = (patch_mov_sampled - patch_features_fix).pow(2).mean(1) * 12
            loss = sampled_cost.mean()
            (loss + reg_loss).backward()
            optimizer.step()
            
            if (it + 1) % 50 == 0 or it == 0:
                logger.info(f"  Adam iter {it+1}/{n_iter_adam}: "
                           f"loss={loss.item():.4f}, reg={reg_loss.item():.4f}")
        
        fitted_grid = disp_sample.detach().permute(0, 4, 1, 2, 3)
        disp_hr = F.interpolate(
            fitted_grid * grid_sp_adam,
            size=(D, H, W), mode='trilinear', align_corners=False
        )
    
    elapsed = time.time() - t0
    logger.info(f"  Final: max_disp={disp_hr.abs().max().item():.1f} voxels, "
                f"mean={disp_hr.abs().mean().item():.1f} voxels, time={elapsed:.1f}s")
    
    return disp_hr.float()


def convex_adam_on_features(
    fixed_feats: torch.Tensor,
    moving_feats: torch.Tensor,
    volume_shape: tuple,
    grid_sp: int = 6,
    disp_hw: int = 4,
    lambda_weight: float = 1.25,
    n_iter_adam: int = 300,
    grid_sp_adam: int = 2,
    ic: bool = True,
    device: str = "cuda",
) -> torch.Tensor:
    """
    ConvexAdam registration on pre-computed feature volumes.
    
    Same algorithm as mind_convex_adam but takes pre-computed features
    (e.g., upsampled DINO PCA features) instead of computing MIND-SSC.
    
    Args:
        fixed_feats: (1, C, D, H, W) fixed feature tensor (already on device or CPU)
        moving_feats: (1, C, D, H, W) moving feature tensor
        volume_shape: (D, H, W) 
        grid_sp: grid spacing for convex initialization
        disp_hw: displacement half-width for correlation volume
        lambda_weight: Adam regularization weight
        n_iter_adam: Adam optimization iterations
        grid_sp_adam: Adam grid spacing
        ic: use inverse consistency
        device: torch device
        
    Returns:
        displacement: (1, 3, D, H, W) displacement field in voxels.
                      Convention: negate before use with warp_points.
    """
    t0 = time.time()
    D, H, W = volume_shape
    
    features_fix = fixed_feats.half().to(device)
    features_mov = moving_feats.half().to(device)
    n_ch = features_fix.shape[1]
    
    logger.info(f"ConvexAdam on features: shape={features_fix.shape}, "
                f"grid_sp={grid_sp}, disp_hw={disp_hw}, adam_iters={n_iter_adam}")
    
    # Downsample for convex initialization
    with torch.no_grad():
        features_fix_smooth = F.avg_pool3d(features_fix, grid_sp, stride=grid_sp)
        features_mov_smooth = F.avg_pool3d(features_mov, grid_sp, stride=grid_sp)
    logger.info(f"  Smoothed features: {features_fix_smooth.shape}")
    
    # Step 1: Correlation volume
    ssd, ssd_argmin = correlate(
        features_fix_smooth, features_mov_smooth,
        disp_hw, grid_sp, (D, H, W), n_ch
    )
    
    disp_mesh_t = F.affine_grid(
        disp_hw * torch.eye(3, 4, device=device).half().unsqueeze(0),
        (1, 1, disp_hw*2+1, disp_hw*2+1, disp_hw*2+1),
        align_corners=True
    ).permute(0, 4, 1, 2, 3).reshape(3, -1, 1)
    
    # Step 2: Coupled convex optimization
    disp_soft = coupled_convex(ssd, ssd_argmin, disp_mesh_t, grid_sp, (D, H, W))
    logger.info(f"  Convex init: max_disp={disp_soft.abs().max().item():.1f}")
    
    # Step 3: Inverse consistency
    if ic:
        scale = torch.tensor(
            [D//grid_sp-1, H//grid_sp-1, W//grid_sp-1], device=device
        ).view(1, 3, 1, 1, 1).half() / 2
        
        ssd_, ssd_argmin_ = correlate(
            features_mov_smooth, features_fix_smooth,
            disp_hw, grid_sp, (D, H, W), n_ch
        )
        disp_soft_ = coupled_convex(ssd_, ssd_argmin_, disp_mesh_t, grid_sp, (D, H, W))
        disp_ice, _ = inverse_consistency(
            (disp_soft / scale).flip(1),
            (disp_soft_ / scale).flip(1),
            iter=15
        )
        disp_hr = F.interpolate(
            disp_ice.flip(1) * scale * grid_sp,
            size=(D, H, W), mode='trilinear', align_corners=False
        )
        logger.info(f"  After IC: max_disp={disp_hr.abs().max().item():.1f}")
    else:
        disp_hr = F.interpolate(
            disp_soft * grid_sp, size=(D, H, W), mode='trilinear', align_corners=False
        )
    
    # Step 4: Adam instance optimization
    if lambda_weight > 0:
        with torch.no_grad():
            patch_features_fix = F.avg_pool3d(features_fix, grid_sp_adam, stride=grid_sp_adam)
            patch_features_mov = F.avg_pool3d(features_mov, grid_sp_adam, stride=grid_sp_adam)
        
        gD, gH, gW = D//grid_sp_adam, H//grid_sp_adam, W//grid_sp_adam
        
        disp_lr = F.interpolate(
            disp_hr, size=(gD, gH, gW), mode='trilinear', align_corners=False
        )
        
        net = nn.Sequential(nn.Conv3d(3, 1, (gD, gH, gW), bias=False))
        net[0].weight.data[:] = disp_lr.float().cpu().data / grid_sp_adam
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=1)
        
        grid0 = F.affine_grid(
            torch.eye(3, 4).unsqueeze(0).to(device),
            (1, 1, gD, gH, gW), align_corners=False
        )
        
        for it in range(n_iter_adam):
            optimizer.zero_grad()
            
            disp_sample = F.avg_pool3d(
                F.avg_pool3d(
                    F.avg_pool3d(net[0].weight, 3, stride=1, padding=1),
                    3, stride=1, padding=1),
                3, stride=1, padding=1
            ).permute(0, 2, 3, 4, 1)
            
            reg_loss = lambda_weight * (
                ((disp_sample[0,:,1:,:] - disp_sample[0,:,:-1,:]) ** 2).mean() +
                ((disp_sample[0,1:,:,:] - disp_sample[0,:-1,:,:]) ** 2).mean() +
                ((disp_sample[0,:,:,1:] - disp_sample[0,:,:,:-1]) ** 2).mean()
            )
            
            scale_adam = torch.tensor(
                [(gD-1)/2, (gH-1)/2, (gW-1)/2], device=device
            ).unsqueeze(0)
            
            grid_disp = grid0.view(-1, 3).float() + (
                disp_sample.view(-1, 3) / scale_adam
            ).flip(1).float()
            
            patch_mov_sampled = F.grid_sample(
                patch_features_mov.float(),
                grid_disp.view(1, gD, gH, gW, 3),
                align_corners=False, mode='bilinear'
            )
            
            sampled_cost = (patch_mov_sampled - patch_features_fix).pow(2).mean(1) * 12
            loss = sampled_cost.mean()
            (loss + reg_loss).backward()
            optimizer.step()
            
            if (it + 1) % 50 == 0 or it == 0:
                logger.info(f"  Adam iter {it+1}/{n_iter_adam}: "
                           f"loss={loss.item():.4f}, reg={reg_loss.item():.4f}")
        
        fitted_grid = disp_sample.detach().permute(0, 4, 1, 2, 3)
        disp_hr = F.interpolate(
            fitted_grid * grid_sp_adam,
            size=(D, H, W), mode='trilinear', align_corners=False
        )
    
    # DO NOT negate. Displacement convention follows DINO-Reg:
    # d(x_fixed) = x_moving - x_fixed
    # Evaluation: sample d at FIXED keypoints, warped_fixed = fixed + d, compare to moving.
    # Channel meaning: ch0=first_spatial_dim, ch1=second, ch2=third of the input feature volume.
    
    elapsed = time.time() - t0
    logger.info(f"  Final: max_disp={disp_hr.abs().max().item():.1f} voxels, "
                f"mean={disp_hr.abs().mean().item():.1f} voxels, time={elapsed:.1f}s")
    for c in range(3):
        logger.info(f"    ch{c}: mean={disp_hr[0,c].mean().item():+.2f}, "
                    f"std={disp_hr[0,c].std().item():.2f}, "
                    f"absmax={disp_hr[0,c].abs().max().item():.1f}")
    
    return disp_hr.float()

