"""
Tests for transform module (SVF, integration, warping).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.transform.svf import SVFField
from pipeline.transform.integrate import (
    create_identity_grid,
    scaling_and_squaring,
)
from pipeline.transform.warp import (
    warp_volume,
    warp_points,
    compute_jacobian_determinant,
)
from pipeline.transform.fitter import DiffeomorphicFitter


DEVICE = "cpu"  # Use CPU for testing


class TestSVF:
    def test_svf_init(self):
        svf = SVFField((32, 32, 32), grid_spacing=8, device=DEVICE)
        assert svf.velocity.shape[0] == 1
        assert svf.velocity.shape[1] == 3

    def test_svf_velocity_field(self):
        svf = SVFField((32, 32, 32), grid_spacing=8, device=DEVICE)
        v = svf.get_velocity_field()
        assert v.shape == (1, 3, 32, 32, 32)
        # Zero initialized → should be all zeros
        assert torch.allclose(v, torch.zeros_like(v))

    def test_svf_regularization(self):
        svf = SVFField((32, 32, 32), grid_spacing=8, device=DEVICE)
        reg = svf.regularization_loss()
        assert reg.item() == 0.0  # Zero field → zero regularization

    def test_svf_nonzero_reg(self):
        svf = SVFField((32, 32, 32), grid_spacing=8, device=DEVICE)
        with torch.no_grad():
            svf.velocity += torch.randn_like(svf.velocity) * 0.1
        reg = svf.regularization_loss()
        assert reg.item() > 0


class TestIntegration:
    def test_identity_grid(self):
        grid = create_identity_grid((8, 8, 8), device=DEVICE)
        assert grid.shape == (1, 8, 8, 8, 3)
        # Center should be at 0
        center = grid[0, 4, 4, 4]
        assert torch.allclose(center, torch.tensor([0.0, 0.0, 0.0]), atol=0.3)

    def test_zero_velocity_integration(self):
        v = torch.zeros(1, 3, 16, 16, 16, device=DEVICE)
        disp = scaling_and_squaring(v, n_steps=5)
        assert disp.shape == (1, 3, 16, 16, 16)
        assert torch.allclose(disp, torch.zeros_like(disp), atol=1e-6)


class TestWarping:
    def test_identity_warp(self):
        vol = torch.randn(1, 1, 16, 16, 16, device=DEVICE)
        disp = torch.zeros(1, 3, 16, 16, 16, device=DEVICE)
        warped = warp_volume(vol, disp)
        assert torch.allclose(vol, warped, atol=1e-5)

    def test_point_warp_identity(self):
        pts = np.array([[8, 8, 8], [4, 4, 4]], dtype=np.float64)
        disp = torch.zeros(1, 3, 16, 16, 16, device=DEVICE)
        warped = warp_points(pts, disp)
        np.testing.assert_allclose(pts, warped, atol=1e-4)

    def test_jacobian_identity(self):
        disp = torch.zeros(1, 3, 16, 16, 16, device=DEVICE)
        jac = compute_jacobian_determinant(disp)
        # Identity → Jacobian = 1
        assert torch.allclose(jac, torch.ones_like(jac), atol=0.1)

    def test_jacobian_positive_for_smooth(self):
        # Small smooth displacement → Jacobian should be mostly positive
        disp = torch.randn(1, 3, 16, 16, 16, device=DEVICE) * 0.1
        jac = compute_jacobian_determinant(disp)
        pct_neg = (jac <= 0).float().mean().item()
        assert pct_neg < 0.1  # Less than 10% folding for small displacements


class TestFitter:
    def test_fitter_trivial(self):
        """Test fitter with known correspondence that's already aligned."""
        fitter = DiffeomorphicFitter(
            volume_shape=(16, 16, 16),
            grid_spacings=[4.0],
            n_iters_per_level=20,
            device=DEVICE,
        )
        # Points already at correct positions
        src = np.array([[8, 8, 8], [4, 4, 4], [12, 12, 12]], dtype=np.float64)
        tgt = src.copy()
        weights = np.ones(3)

        disp = fitter.fit(src, tgt, weights)
        assert disp.shape == (1, 3, 16, 16, 16)
        # Displacement should be near zero for identity transform
        assert disp.abs().max().item() < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
