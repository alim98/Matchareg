"""
Diffeomorphic fitter: fits an SVF to weighted correspondences.

Multi-resolution optimization of:
    min_v  Σ w_k |φ(x_k) - y_k|^2 + λ_s |Lv|^2 + λ_j P_jac(φ)
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .svf import SVFField
from .integrate import scaling_and_squaring
from .warp import warp_points, compute_jacobian_determinant

logger = logging.getLogger(__name__)


class DiffeomorphicFitter:
    """
    Multi-resolution diffeomorphic fitting from sparse correspondences.

    Fits an SVF to weighted 3D correspondences using gradient descent
    with bending energy and Jacobian regularization.
    """

    def __init__(
        self,
        volume_shape: tuple,
        grid_spacings: List[float] = [10.0, 6.0, 3.0],
        lambda_smooth: float = 1.0,
        lambda_jac: float = 0.1,
        lr: float = 1e-3,
        n_iters_per_level: int = 200,
        n_squaring_steps: int = 7,
        device: str = "cuda",
    ):
        """
        Args:
            volume_shape: (D, H, W) target volume shape
            grid_spacings: control grid spacings for each resolution level
            lambda_smooth: bending energy weight
            lambda_jac: Jacobian determinant penalty weight
            lr: learning rate
            n_iters_per_level: iterations per resolution level
            n_squaring_steps: scaling-and-squaring integration steps
            device: torch device
        """
        self.volume_shape = volume_shape
        self.grid_spacings = grid_spacings
        self.lambda_smooth = lambda_smooth
        self.lambda_jac = lambda_jac
        self.lr = lr
        self.n_iters_per_level = n_iters_per_level
        self.n_squaring_steps = n_squaring_steps
        self.device = device

    def fit(
        self,
        matched_src: np.ndarray,
        matched_tgt: np.ndarray,
        weights: np.ndarray,
    ) -> torch.Tensor:
        """
        Fit a diffeomorphic transformation to correspondences.

        Args:
            matched_src: (K, 3) source (moving) matched points, voxel coords
            matched_tgt: (K, 3) target (fixed) matched points, voxel coords
            weights: (K,) confidence weights

        Returns:
            displacement: (1, 3, D, H, W) final displacement field
        """
        # Convert to torch
        src_pts = torch.from_numpy(matched_src).float().to(self.device)
        tgt_pts = torch.from_numpy(matched_tgt).float().to(self.device)
        w = torch.from_numpy(weights).float().to(self.device)
        w = w / w.sum()  # normalize weights

        displacement = None

        for level, spacing in enumerate(self.grid_spacings):
            logger.info(
                f"Level {level+1}/{len(self.grid_spacings)}: "
                f"grid_spacing={spacing}, iters={self.n_iters_per_level}"
            )

            # Current source points (warped if we have a previous displacement)
            if displacement is not None:
                current_src = torch.from_numpy(
                    warp_points(matched_src, displacement)
                ).float().to(self.device)
            else:
                current_src = src_pts.clone()

            # Fit residual at this level
            residual_disp = self._fit_level(
                current_src, tgt_pts, w, spacing
            )

            # Compose with previous displacement
            if displacement is not None:
                displacement = self._compose(displacement, residual_disp)
            else:
                displacement = residual_disp

            # Log progress
            with torch.no_grad():
                warped = torch.from_numpy(
                    warp_points(matched_src, displacement)
                ).float().to(self.device)
                tre = torch.norm(warped - tgt_pts, dim=1).mean()
                logger.info(f"  Mean correspondence error: {tre:.3f} voxels")

        return displacement

    def _fit_level(
        self,
        src_pts: torch.Tensor,
        tgt_pts: torch.Tensor,
        weights: torch.Tensor,
        grid_spacing: float,
    ) -> torch.Tensor:
        """Fit SVF at a single resolution level."""
        svf = SVFField(
            self.volume_shape,
            grid_spacing=grid_spacing,
            device=self.device,
        )

        optimizer = torch.optim.Adam(svf.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_iters_per_level, eta_min=self.lr * 0.01
        )

        best_loss = float("inf")
        best_disp = None

        for it in range(self.n_iters_per_level):
            optimizer.zero_grad()

            # Get velocity and integrate
            v = svf.get_velocity_field()
            disp = scaling_and_squaring(v, self.n_squaring_steps)

            # Correspondence loss: weighted Huber loss (robust to outlier matches)
            warped = self._warp_points_differentiable(src_pts, disp)
            residuals = torch.norm(warped - tgt_pts, dim=1)  # per-point distance
            # Huber: quadratic for small residuals, linear for large (outliers)
            delta = 10.0  # voxels — transition point
            huber = torch.where(
                residuals < delta,
                0.5 * residuals ** 2,
                delta * (residuals - 0.5 * delta),
            )
            loss_corr = (weights * huber).sum()

            # Smoothness regularization
            loss_smooth = svf.regularization_loss()

            # Jacobian penalty (subsample for efficiency)
            loss_jac = self._jacobian_penalty(disp)

            # Total loss
            loss = (
                loss_corr
                + self.lambda_smooth * loss_smooth
                + self.lambda_jac * loss_jac
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_disp = disp.detach().clone()

            if (it + 1) % 50 == 0:
                logger.info(
                    f"  iter {it+1}/{self.n_iters_per_level}: "
                    f"loss={loss.item():.4f} "
                    f"(corr={loss_corr.item():.4f}, "
                    f"smooth={loss_smooth.item():.4f}, "
                    f"jac={loss_jac.item():.4f})"
                )

        return best_disp

    def _warp_points_differentiable(
        self,
        points: torch.Tensor,
        displacement: torch.Tensor,
    ) -> torch.Tensor:
        """
        Differentiable point warping (for gradient computation).

        Args:
            points: (N, 3) in voxel coords (z, y, x)
            displacement: (1, 3, D, H, W)

        Returns:
            warped: (N, 3)
        """
        _, _, D, H, W = displacement.shape

        # Normalize point coordinates to [-1, 1] for grid_sample
        pts_norm = torch.zeros_like(points)
        pts_norm[:, 0] = points[:, 2] / (W - 1) * 2 - 1  # x
        pts_norm[:, 1] = points[:, 1] / (H - 1) * 2 - 1  # y
        pts_norm[:, 2] = points[:, 0] / (D - 1) * 2 - 1  # z

        # Reshape for grid_sample: (1, 1, 1, N, 3)
        grid = pts_norm.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Sample displacement at points
        sampled = F.grid_sample(
            displacement, grid, mode="bilinear",
            padding_mode="border", align_corners=True
        )  # (1, 3, 1, 1, N)

        disp_at_pts = sampled.squeeze().T  # (N, 3): (z, y, x) displacement

        return points + disp_at_pts

    def _jacobian_penalty(
        self,
        displacement: torch.Tensor,
        subsample: int = 8,
    ) -> torch.Tensor:
        """
        Penalty for non-positive Jacobian determinant (folding).

        Subsampled for efficiency.
        """
        # Subsample the displacement field
        disp_sub = displacement[:, :, ::subsample, ::subsample, ::subsample]
        jac_det = compute_jacobian_determinant(disp_sub)

        # Penalize values below 0 (folding)
        neg_jac = F.relu(-jac_det)
        return neg_jac.mean()

    def _compose(
        self,
        disp1: torch.Tensor,
        disp2: torch.Tensor,
    ) -> torch.Tensor:
        """Compose two displacement fields."""
        from .integrate import compose_displacements
        return compose_displacements(disp1, disp2)
