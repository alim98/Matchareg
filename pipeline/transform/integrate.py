"""
Scaling-and-squaring integration for SVF → diffeomorphism.

Computes φ = exp(v) via iterative squaring:
  φ = v/2^N
  for i in range(N):
      φ = φ ∘ φ
"""
import torch
import torch.nn.functional as F
from typing import Optional


def create_identity_grid(
    shape: tuple,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Create identity displacement grid in voxel coordinates.

    Args:
        shape: (D, H, W)
        device: torch device

    Returns:
        grid: (1, D, H, W, 3) identity grid with values in [-1, 1]
              for use with F.grid_sample
    """
    D, H, W = shape
    vectors = [
        torch.linspace(-1, 1, s, device=device)
        for s in [D, H, W]
    ]
    grids = torch.meshgrid(vectors, indexing="ij")
    grid = torch.stack(grids, dim=-1)  # (D, H, W, 3)
    grid = grid.unsqueeze(0)  # (1, D, H, W, 3)
    return grid


def scaling_and_squaring(
    velocity: torch.Tensor,
    n_steps: int = 7,
) -> torch.Tensor:
    """
    Integrate a stationary velocity field via scaling and squaring.

    φ = exp(v) computed as:
    1. φ_0 = v / 2^n_steps
    2. φ_{i+1} = φ_i ∘ φ_i  (compose with itself)

    Args:
        velocity: (1, 3, D, H, W) velocity field in voxel units
        n_steps: number of squaring steps

    Returns:
        displacement: (1, 3, D, H, W) displacement field (NOT deformation)
    """
    disp = velocity / (2 ** n_steps)

    for _ in range(n_steps):
        disp = compose_displacements(disp, disp)

    return disp


def compose_displacements(
    disp1: torch.Tensor,
    disp2: torch.Tensor,
) -> torch.Tensor:
    """
    Compose two displacement fields: disp1 ∘ disp2.

    This computes: φ_composed(x) = φ_1(φ_2(x))
    As displacement: d_composed(x) = d_2(x) + d_1(x + d_2(x))

    Args:
        disp1: (1, 3, D, H, W) first displacement
        disp2: (1, 3, D, H, W) second displacement

    Returns:
        composed: (1, 3, D, H, W) composed displacement
    """
    _, _, D, H, W = disp1.shape

    # Create identity grid in [-1, 1]
    identity = create_identity_grid((D, H, W), device=disp1.device)

    # Convert disp2 to grid_sample format: add to identity grid
    # disp2 is in voxel coordinates, convert to [-1, 1]
    disp2_normalized = torch.zeros_like(identity)
    disp2_normalized[..., 0] = disp2[:, 0] / (D - 1) * 2  # z
    disp2_normalized[..., 1] = disp2[:, 1] / (H - 1) * 2  # y
    disp2_normalized[..., 2] = disp2[:, 2] / (W - 1) * 2  # x

    sample_grid = identity + disp2_normalized

    # Sample disp1 at displaced locations
    # grid_sample expects (B, C, D, H, W) input and (B, D, H, W, 3) grid
    # but grid ordering is (x, y, z) → (W, H, D) in grid_sample
    # We need to flip the grid coordinates for grid_sample
    sample_grid_flipped = sample_grid.flip(-1)  # (z,y,x) → (x,y,z)

    disp1_warped = F.grid_sample(
        disp1,
        sample_grid_flipped,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    return disp2 + disp1_warped


def displacement_to_deformation(
    displacement: torch.Tensor,
) -> torch.Tensor:
    """
    Convert displacement field to deformation field (absolute coordinates).

    deformation(x) = x + displacement(x)

    Args:
        displacement: (1, 3, D, H, W) in voxel units

    Returns:
        deformation: (1, D, H, W, 3) in grid_sample format [-1, 1]
    """
    _, _, D, H, W = displacement.shape
    identity = create_identity_grid((D, H, W), device=displacement.device)

    # Convert displacement to normalized coordinates
    disp_norm = torch.zeros_like(identity)
    disp_norm[..., 0] = displacement[:, 0] / (D - 1) * 2
    disp_norm[..., 1] = displacement[:, 1] / (H - 1) * 2
    disp_norm[..., 2] = displacement[:, 2] / (W - 1) * 2

    return identity + disp_norm
