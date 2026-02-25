"""
Stationary Velocity Field (SVF) parameterization.

Represents diffeomorphic deformations as the exponential of a
smooth velocity field on a 3D grid.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SVFField(nn.Module):
    """
    Stationary Velocity Field on a 3D control-point grid.

    The velocity field is defined on a coarse grid and interpolated
    to full resolution via trilinear interpolation.
    """

    def __init__(
        self,
        volume_shape: Tuple[int, int, int],
        grid_spacing: float = 8.0,
        device: str = "cuda",
    ):
        """
        Args:
            volume_shape: (D, H, W) of the target volume
            grid_spacing: control point spacing in voxels
            device: torch device
        """
        super().__init__()
        self.volume_shape = volume_shape
        self.grid_spacing = grid_spacing
        self.device = device

        # Compute control grid dimensions
        D, H, W = volume_shape
        self.grid_shape = (
            max(int(D / grid_spacing) + 1, 4),
            max(int(H / grid_spacing) + 1, 4),
            max(int(W / grid_spacing) + 1, 4),
        )

        # Velocity field parameters: (1, 3, gD, gH, gW)
        self.velocity = nn.Parameter(
            torch.zeros(1, 3, *self.grid_shape, device=device)
        )

    def get_velocity_field(self) -> torch.Tensor:
        """
        Interpolate velocity from control grid to full resolution.

        Returns:
            v: (1, 3, D, H, W) velocity field at full resolution
        """
        v = F.interpolate(
            self.velocity,
            size=self.volume_shape,
            mode="trilinear",
            align_corners=True,
        )
        return v

    def regularization_loss(self) -> torch.Tensor:
        """
        Bending energy regularization on the velocity field.

        Computes the sum of squared second derivatives (approximated
        by finite differences on the control grid).

        Returns:
            reg: scalar regularization loss
        """
        v = self.velocity  # (1, 3, gD, gH, gW)
        reg = torch.tensor(0.0, device=v.device)

        for dim in range(3):
            # Second-order finite difference along each spatial dim
            # dim+2 because first two dims are batch and channel
            d2 = torch.diff(v, n=2, dim=dim + 2)
            reg = reg + (d2 ** 2).mean()

        return reg

    def reset(self):
        """Reset velocity field to zero."""
        with torch.no_grad():
            self.velocity.zero_()
