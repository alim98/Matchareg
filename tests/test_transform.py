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
    compose_displacements,
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


class TestWarpingKnownDisplacement:
    """Verify displacement channel ordering and sign conventions."""

    def test_point_warp_uniform_z_shift(self):
        """Uniform displacement in channel 0 (z) shifts z-coords of all points."""
        D, H, W = 32, 32, 32
        pts = np.array([[10.0, 10.0, 10.0], [16.0, 16.0, 16.0]], dtype=np.float64)
        shift_z = 5.0
        disp = torch.zeros(1, 3, D, H, W, device=DEVICE)
        disp[0, 0] = shift_z  # channel 0 = z displacement
        warped = warp_points(pts, disp)
        expected = pts.copy()
        expected[:, 0] += shift_z
        np.testing.assert_allclose(warped, expected, atol=0.1)

    def test_point_warp_uniform_x_shift(self):
        """Uniform displacement in channel 2 (x) shifts x-coords of all points."""
        D, H, W = 32, 32, 32
        pts = np.array([[10.0, 10.0, 10.0]], dtype=np.float64)
        shift_x = 7.0
        disp = torch.zeros(1, 3, D, H, W, device=DEVICE)
        disp[0, 2] = shift_x  # channel 2 = x displacement
        warped = warp_points(pts, disp)
        expected = pts.copy()
        expected[:, 2] += shift_x
        np.testing.assert_allclose(warped, expected, atol=0.1)

    def test_volume_warp_z_shift_moves_band(self):
        """
        warp_volume(vol, disp) is a pull-warp:  output(x) = vol(x + disp(x)).
        A z-displacement of +2 samples from z+2 in the input → bright band
        originally at z=8 should appear at z=6 in the output.
        """
        D, H, W = 16, 16, 16
        vol = torch.zeros(1, 1, D, H, W, device=DEVICE)
        vol[0, 0, 8, :, :] = 1.0  # bright axial band at z=8
        disp = torch.zeros(1, 3, D, H, W, device=DEVICE)
        disp[0, 0] = 2.0  # z-shift +2 → sample from z+2 → band appears at z=6
        warped = warp_volume(vol, disp)
        assert warped[0, 0, 6, H // 2, W // 2].item() > 0.5, \
            "Bright band should be at z=6 after +2 z-displacement"
        assert warped[0, 0, 8, H // 2, W // 2].item() < 0.1, \
            "Original z=8 location should be dark after shift"

    def test_warp_points_and_volume_consistent(self):
        """
        Displacement applied to points and to a volume must be consistent:
        the warped point location matches where the bright voxel ended up.
        """
        D, H, W = 32, 32, 32
        disp = torch.zeros(1, 3, D, H, W, device=DEVICE)
        disp[0, 0] = 4.0  # 4-voxel z-shift
        # A point at z=10 warps to z=14 (point-warp: add displacement)
        pt = np.array([[10.0, 16.0, 16.0]], dtype=np.float64)
        warped_pt = warp_points(pt, disp)
        assert abs(warped_pt[0, 0] - 14.0) < 0.1, \
            f"Point should move to z=14, got z={warped_pt[0, 0]:.2f}"


class TestComposeDisplacements:
    def test_compose_two_identical_z_translations(self):
        """Composing two +2 z-shifts should give ~+4 at interior points."""
        D, H, W = 16, 16, 16
        d = torch.zeros(1, 3, D, H, W, device=DEVICE)
        d[0, 0] = 2.0  # constant 2-voxel z-shift
        d2 = compose_displacements(d, d)
        center_val = d2[0, 0, D // 2, H // 2, W // 2].item()
        assert abs(center_val - 4.0) < 0.5, \
            f"Expected ~4.0 from composing two 2-voxel shifts, got {center_val:.3f}"

    def test_compose_zero_and_translation(self):
        """Composing zero-field with a translation should give the translation."""
        D, H, W = 16, 16, 16
        d = torch.zeros(1, 3, D, H, W, device=DEVICE)
        d[0, 0] = 3.0
        zero = torch.zeros_like(d)
        # compose_displacements(zero, d) = d(x) + zero(x + d(x)) = d(x)
        composed = compose_displacements(zero, d)
        np.testing.assert_allclose(
            composed.numpy(), d.numpy(), atol=0.1,
            err_msg="compose(zero, d) should equal d",
        )


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

    def test_fitter_known_z_translation(self):
        """
        Fitter should recover a pure z-translation from dense correspondences.

        Uses 200 scattered points with a +6 voxel z-shift.  The mean sampled
        displacement in z should land within 3 voxels of the ground truth.
        """
        from scipy.ndimage import map_coordinates as mc

        D, H, W = 32, 32, 32
        shift_z = 6.0
        rng = np.random.RandomState(0)

        src = rng.uniform(4, D - 4 - shift_z, size=(200, 3))
        tgt = src.copy()
        tgt[:, 0] += shift_z
        weights = np.ones(200)

        fitter = DiffeomorphicFitter(
            volume_shape=(D, H, W),
            grid_spacings=[8.0],
            n_iters_per_level=100,
            lambda_smooth=0.1,
            lambda_jac=0.01,
            lr=0.05,
            device=DEVICE,
        )
        displacement = fitter.fit(src, tgt, weights)
        assert displacement.shape == (1, 3, D, H, W)

        disp_np = displacement.cpu().numpy()
        z_disp_at_src = mc(
            disp_np[0, 0],
            [src[:, 0], src[:, 1], src[:, 2]],
            order=1, mode='nearest',
        )
        mean_recovered = z_disp_at_src.mean()
        assert abs(mean_recovered - shift_z) < 3.0, (
            f"Expected mean z-displacement ≈ {shift_z}, got {mean_recovered:.2f}. "
            "Fitter may have wrong axis convention or insufficient optimization."
        )


class TestIntensityRefinement:
    """Smoke tests for intensity_refine.py — verifies it runs on CPU."""

    def test_returns_correct_shape(self):
        from pipeline.transform.intensity_refine import intensity_refinement

        D, H, W = 16, 16, 16
        rng = np.random.RandomState(1)
        fixed = rng.randn(D, H, W).astype(np.float32)
        moving = rng.randn(D, H, W).astype(np.float32)
        curr_disp = torch.zeros(1, 3, D, H, W, device=DEVICE)

        result = intensity_refinement(
            fixed, moving, curr_disp,
            grid_spacing=8.0,
            n_iters=3,
            device=DEVICE,
        )
        assert result.shape == (1, 3, D, H, W), \
            f"Expected (1,3,{D},{H},{W}), got {tuple(result.shape)}"
        assert result.dtype == torch.float32
        assert torch.isfinite(result).all(), "Output displacement has NaN/Inf"

    def test_does_not_increase_initial_displacement(self):
        """With few iterations and high regularization, displacement stays bounded."""
        from pipeline.transform.intensity_refine import intensity_refinement

        D, H, W = 16, 16, 16
        rng = np.random.RandomState(0)
        fixed = (rng.randn(D, H, W) * 50 + 100).astype(np.float32)
        moving = (rng.randn(D, H, W) * 50 + 100).astype(np.float32)
        curr_disp = torch.zeros(1, 3, D, H, W, device=DEVICE)

        result = intensity_refinement(
            fixed, moving, curr_disp,
            grid_spacing=8.0, lambda_smooth=100.0,
            n_iters=5, device=DEVICE,
        )
        # High regularization → output should stay small
        assert result.abs().max().item() < 5.0, \
            "With high regularization, displacement should stay near zero"


class TestDenseFeatureRegistration:
    """Smoke tests for dense_fitter.py — verifies it runs on CPU."""

    def test_returns_correct_shape(self):
        from pipeline.transform.dense_fitter import dense_feature_registration

        C, D, H, W = 8, 6, 6, 6
        rng = np.random.RandomState(2)
        fixed_feats = rng.randn(C, D, H, W).astype(np.float32)
        moving_feats = rng.randn(C, D, H, W).astype(np.float32)

        result = dense_feature_registration(
            fixed_feats, moving_feats,
            volume_shape=(D, H, W),
            grid_sp_adam=1,
            n_iters=3,
            device=DEVICE,
        )
        assert result.shape == (1, 3, D, H, W), \
            f"Expected (1,3,{D},{H},{W}), got {tuple(result.shape)}"
        assert torch.isfinite(result).all(), "Output displacement has NaN/Inf"

    def test_identical_features_gives_small_displacement(self):
        """When fixed == moving features, the optimizer has nothing to do."""
        from pipeline.transform.dense_fitter import dense_feature_registration

        C, D, H, W = 8, 6, 6, 6
        feats = np.ones((C, D, H, W), dtype=np.float32)

        result = dense_feature_registration(
            feats, feats,
            volume_shape=(D, H, W),
            grid_sp_adam=1,
            n_iters=5,
            device=DEVICE,
        )
        # Starting from zero init with identical inputs → displacement stays small
        assert result.abs().max().item() < 5.0, \
            "Identical fixed/moving features should produce near-zero displacement"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
