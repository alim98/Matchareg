"""
Warping utilities for volumes and points.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional

from .integrate import create_identity_grid


def warp_volume(
    volume: torch.Tensor,
    displacement: torch.Tensor,
) -> torch.Tensor:
    """
    Warp a 3D volume using a displacement field.

    Args:
        volume: (1, 1, D, H, W) volume to warp
        displacement: (1, 3, D, H, W) displacement field in voxel units

    Returns:
        warped: (1, 1, D, H, W) warped volume
    """
    _, _, D, H, W = displacement.shape
    identity = create_identity_grid((D, H, W), device=displacement.device)

    # Convert displacement to normalized coordinates
    disp_norm = torch.zeros_like(identity)
    disp_norm[..., 0] = displacement[:, 0] / (D - 1) * 2
    disp_norm[..., 1] = displacement[:, 1] / (H - 1) * 2
    disp_norm[..., 2] = displacement[:, 2] / (W - 1) * 2

    grid = identity + disp_norm

    # grid_sample expects (x, y, z) ordering, we have (z, y, x)
    grid = grid.flip(-1)

    warped = F.grid_sample(
        volume, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return warped


def warp_points(
    points: np.ndarray,
    displacement: torch.Tensor,
) -> np.ndarray:
    """
    Warp 3D points using a displacement field.

    Samples the displacement field at the point locations using
    trilinear interpolation.

    Args:
        points: (N, 3) points in voxel coordinates (z, y, x)
        displacement: (1, 3, D, H, W) displacement field in voxel units

    Returns:
        warped_points: (N, 3) warped point coordinates
    """
    _, _, D, H, W = displacement.shape
    device = displacement.device

    # Convert points to normalized grid coordinates [-1, 1]
    pts = torch.from_numpy(points).float().to(device)
    pts_norm = torch.zeros_like(pts)
    pts_norm[:, 0] = pts[:, 2] / (W - 1) * 2 - 1  # x → normalized x
    pts_norm[:, 1] = pts[:, 1] / (H - 1) * 2 - 1  # y → normalized y
    pts_norm[:, 2] = pts[:, 0] / (D - 1) * 2 - 1  # z → normalized z

    # grid_sample expects (B, C, D, H, W) input and (B, 1, 1, N, 3) grid
    grid = pts_norm.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, N, 3)

    sampled = F.grid_sample(
        displacement, grid, mode="bilinear", padding_mode="border", align_corners=True
    )  # (1, 3, 1, 1, N)

    disp_at_pts = sampled.squeeze().T  # (N, 3) in (z, y, x)

    warped = pts + disp_at_pts
    return warped.cpu().numpy()


def compute_jacobian_determinant(
    displacement: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the Jacobian determinant of a displacement field.

    |J(φ)| where φ(x) = x + displacement(x).

    Positive values = valid diffeomorphism.
    Values ≤ 0 = folding (invalid).

    Args:
        displacement: (1, 3, D, H, W) in voxel units

    Returns:
        jac_det: (1, 1, D, H, W) Jacobian determinant
    """
    # Compute spatial gradients of displacement
    # d(disp_z)/dz, d(disp_z)/dy, d(disp_z)/dx, etc.
    disp = displacement

    # Central differences
    dz = torch.zeros_like(disp)
    dy = torch.zeros_like(disp)
    dx = torch.zeros_like(disp)

    # d/dz (dim=2)
    dz[:, :, 1:-1] = (disp[:, :, 2:] - disp[:, :, :-2]) / 2
    dz[:, :, 0] = disp[:, :, 1] - disp[:, :, 0]
    dz[:, :, -1] = disp[:, :, -1] - disp[:, :, -2]

    # d/dy (dim=3)
    dy[:, :, :, 1:-1] = (disp[:, :, :, 2:] - disp[:, :, :, :-2]) / 2
    dy[:, :, :, 0] = disp[:, :, :, 1] - disp[:, :, :, 0]
    dy[:, :, :, -1] = disp[:, :, :, -1] - disp[:, :, :, -2]

    # d/dx (dim=4)
    dx[:, :, :, :, 1:-1] = (disp[:, :, :, :, 2:] - disp[:, :, :, :, :-2]) / 2
    dx[:, :, :, :, 0] = disp[:, :, :, :, 1] - disp[:, :, :, :, 0]
    dx[:, :, :, :, -1] = disp[:, :, :, :, -1] - disp[:, :, :, :, -2]

    # Jacobian matrix (3x3 at each voxel)
    # J = I + grad(displacement)
    # J[i,j] = δ_ij + d(disp_i)/d(x_j)
    J00 = 1 + dz[:, 0:1]  # d(disp_z)/dz
    J01 = dy[:, 0:1]       # d(disp_z)/dy
    J02 = dx[:, 0:1]       # d(disp_z)/dx
    J10 = dz[:, 1:2]       # d(disp_y)/dz
    J11 = 1 + dy[:, 1:2]  # d(disp_y)/dy
    J12 = dx[:, 1:2]       # d(disp_y)/dx
    J20 = dz[:, 2:3]       # d(disp_x)/dz
    J21 = dy[:, 2:3]       # d(disp_x)/dy
    J22 = 1 + dx[:, 2:3]  # d(disp_x)/dx

    # Determinant of 3x3 matrix
    det = (
        J00 * (J11 * J22 - J12 * J21)
        - J01 * (J10 * J22 - J12 * J20)
        + J02 * (J10 * J21 - J11 * J20)
    )

    return det
