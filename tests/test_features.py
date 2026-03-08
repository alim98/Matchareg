"""
Unit tests for feature extraction (TriplanarFuser).

Uses a mock extractor so no GPU or pretrained weights are required.
Tests verify:
  - Output shapes are correct for all three planes
  - concat_norm fusion produces L2-unit-norm per-voxel descriptors
  - Feature coordinate conversion is consistent
  - save/load roundtrip preserves data
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.features.triplanar_fuser import TriplanarFuser, save_features, load_features
from pipeline.matching.sampling import sample_descriptors_at_points


# ---------------------------------------------------------------------------
# Mock extractor: returns constant non-zero patches so norms are well-defined
# ---------------------------------------------------------------------------

class MockExtractor:
    """
    Minimal extractor that satisfies the TriplanarFuser API.

    Returns channel-first feature maps filled with ones.
    patch_size=8 keeps spatial dims manageable in tests.
    embed_dim must be > spatial patch dim (the assertion in TriplanarFuser checks this).
    """
    patch_size = 8
    embed_dim = 64  # 64 >> any patch grid size we'll see in tests

    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        """Return (B, embed_dim, H//patch_size, W//patch_size) of ones."""
        B, _, H, W = batch.shape
        ph = max(H // self.patch_size, 1)
        pw = max(W // self.patch_size, 1)
        return torch.ones(B, self.embed_dim, ph, pw)


class MockExtractorRandom:
    """Like MockExtractor but returns distinct random features per slice-batch."""
    patch_size = 8
    embed_dim = 64

    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        B, _, H, W = batch.shape
        ph = max(H // self.patch_size, 1)
        pw = max(W // self.patch_size, 1)
        out = torch.randn(B, self.embed_dim, ph, pw)
        # L2-normalise across channel dim so cosine similarity is well-defined
        norms = out.norm(dim=1, keepdim=True).clamp(min=1e-8)
        return out / norms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fuser(extractor=None, downsample=1, fusion="concat_norm", batch_size=2):
    if extractor is None:
        extractor = MockExtractor()
    return TriplanarFuser(
        extractor,
        batch_size=batch_size,
        fusion=fusion,
        downsample=downsample,
        device="cpu",
    )


def _small_extract(fuser, vol, axis, min_input_size=32):
    """Call extract_plane_features with a small min_input_size to skip upsampling."""
    return fuser.extract_plane_features(vol, axis=axis, min_input_size=min_input_size)


# ---------------------------------------------------------------------------
# Tests: extract_plane_features per axis
# ---------------------------------------------------------------------------

class TestExtractPlaneFeatures:

    def test_axial_output_shape(self):
        """Axial plane: n_slices == D, spatial grid comes from (H, W)."""
        fuser = _make_fuser()
        vol = np.zeros((10, 16, 24), dtype=np.float32)
        feats = _small_extract(fuser, vol, axis=0)
        # (n_slices=10, embed_dim=64, pH=16//8=2, pW=24//8=3)
        assert feats.ndim == 4
        assert feats.shape[0] == 10  # D slices
        assert feats.shape[1] == 64  # embed_dim

    def test_coronal_output_shape(self):
        """Coronal plane: n_slices == H."""
        fuser = _make_fuser()
        vol = np.zeros((10, 16, 24), dtype=np.float32)
        feats = _small_extract(fuser, vol, axis=1)
        assert feats.shape[0] == 16  # H slices

    def test_sagittal_output_shape(self):
        """Sagittal plane: n_slices == W."""
        fuser = _make_fuser()
        vol = np.zeros((10, 16, 24), dtype=np.float32)
        feats = _small_extract(fuser, vol, axis=2)
        assert feats.shape[0] == 24  # W slices

    def test_channel_first_assertion_fires_for_bad_extractor(self):
        """TriplanarFuser asserts channel-first output — bad extractor should raise."""
        class BadExtractor:
            patch_size = 8
            embed_dim = 2  # embed_dim < spatial → violates assertion

            def extract_features(self, batch):
                B, _, H, W = batch.shape
                ph = H // 8
                pw = W // 8
                # Returns channel-LAST (B, pH, pW, embed_dim) — will fail assert
                # We simulate this by making shape[1] < shape[2]
                return torch.ones(B, 2, 100, 100)  # shape[1]=2 < shape[2]=100

        fuser = _make_fuser(extractor=BadExtractor())
        vol = np.zeros((4, 16, 16), dtype=np.float32)
        with pytest.raises(AssertionError):
            _small_extract(fuser, vol, axis=0)


# ---------------------------------------------------------------------------
# Tests: fuse_triplanar end-to-end
# ---------------------------------------------------------------------------

class TestFuseTriplanar:

    def _run_fuse(self, vol_shape, fusion="concat_norm"):
        fuser = _make_fuser(fusion=fusion)
        vol = np.ones(vol_shape, dtype=np.float32)
        # Patch extract_plane_features to use small min_input_size (avoids 448 upscaling)
        orig = fuser.extract_plane_features
        fuser.extract_plane_features = lambda v, axis, min_input_size=32: \
            orig(v, axis, min_input_size=32)
        return fuser.fuse_triplanar(vol)

    def test_return_types_and_shapes(self):
        fused, feat_shape, orig_shape = self._run_fuse((10, 16, 20))
        assert isinstance(fused, np.ndarray)
        assert len(feat_shape) == 3
        assert orig_shape == (10, 16, 20)
        # Feature channels = 3 × embed_dim
        assert fused.shape[0] == 3 * 64

    def test_concat_norm_produces_unit_norms(self):
        """After concat_norm, every voxel should have L2 norm = 1."""
        fused, _, _ = self._run_fuse((8, 12, 16), fusion="concat_norm")
        norms = np.linalg.norm(fused, axis=0)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-4,
                                   err_msg="concat_norm should produce unit-norm voxels")

    def test_feature_volume_is_finite(self):
        fused, _, _ = self._run_fuse((8, 8, 8))
        assert np.isfinite(fused).all(), "Feature volume should not contain NaN/Inf"

    def test_downsample_reduces_input_size(self):
        """downsample=2 halves the input before feature extraction."""
        extractor = MockExtractor()
        fuser_ds1 = TriplanarFuser(extractor, batch_size=2, downsample=1, device="cpu")
        fuser_ds2 = TriplanarFuser(extractor, batch_size=2, downsample=2, device="cpu")
        vol = np.ones((16, 16, 16), dtype=np.float32)

        orig1 = fuser_ds1.extract_plane_features
        fuser_ds1.extract_plane_features = lambda v, axis, min_input_size=32: \
            orig1(v, axis, min_input_size=32)
        orig2 = fuser_ds2.extract_plane_features
        fuser_ds2.extract_plane_features = lambda v, axis, min_input_size=32: \
            orig2(v, axis, min_input_size=32)

        _, shape1, _ = fuser_ds1.fuse_triplanar(vol)
        _, shape2, _ = fuser_ds2.fuse_triplanar(vol)

        # With downsample=2, the feature grid should be smaller or equal
        assert shape2[0] <= shape1[0], "Downsampling should reduce z-grid size"


# ---------------------------------------------------------------------------
# Tests: save / load roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoadFeatures:

    def test_roundtrip_preserves_data(self):
        fused = np.random.randn(192, 5, 6, 7).astype(np.float32)
        feat_shape = (5, 6, 7)
        orig_shape = (10, 12, 14)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = Path(f.name)

        try:
            save_features((fused, feat_shape, orig_shape), path)
            loaded_feats, loaded_feat_shape, loaded_orig_shape = load_features(path)

            np.testing.assert_array_equal(fused, loaded_feats)
            assert loaded_feat_shape == feat_shape
            assert loaded_orig_shape == orig_shape
        finally:
            path.unlink(missing_ok=True)

    def test_roundtrip_array_only(self):
        """save_features with a plain array (not a tuple) should also round-trip."""
        arr = np.random.randn(64, 4, 4, 4).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = Path(f.name)
        try:
            save_features(arr, path)
            loaded = load_features(path)
            # When saved as plain array, load_features returns the array directly
            if isinstance(loaded, tuple):
                loaded = loaded[0]
            np.testing.assert_array_equal(arr, loaded)
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tests: sample_descriptors_at_points (used extensively in test_C / test_D)
# ---------------------------------------------------------------------------

class TestSampleDescriptors:

    def test_shape(self):
        feat_vol = np.random.randn(32, 8, 8, 8).astype(np.float32)
        pts = np.array([[3.5, 3.5, 3.5], [1.0, 2.0, 6.0]], dtype=np.float64)
        desc = sample_descriptors_at_points(feat_vol, pts)
        assert desc.shape == (2, 32)

    def test_corner_value_matches_voxel(self):
        """Sampling at integer corner should return exactly that voxel's feature."""
        feat_vol = np.zeros((4, 5, 5, 5), dtype=np.float32)
        feat_vol[:, 2, 3, 4] = np.arange(4, dtype=np.float32)  # voxel (2,3,4)
        pts = np.array([[2.0, 3.0, 4.0]], dtype=np.float64)
        desc = sample_descriptors_at_points(feat_vol, pts)
        np.testing.assert_allclose(desc[0], np.arange(4), atol=1e-4)

    def test_interpolation_between_voxels(self):
        """At the midpoint between two known voxels, should get their average."""
        feat_vol = np.zeros((1, 4, 4, 4), dtype=np.float32)
        feat_vol[0, 0, 0, 0] = 0.0
        feat_vol[0, 2, 0, 0] = 2.0
        pts = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)  # midpoint in z
        desc = sample_descriptors_at_points(feat_vol, pts)
        np.testing.assert_allclose(desc[0, 0], 1.0, atol=0.1)

    def test_out_of_bounds_clips_to_border(self):
        """Points outside the volume should be handled (padding_mode='border')."""
        feat_vol = np.ones((4, 5, 5, 5), dtype=np.float32)
        pts = np.array([[-1.0, -1.0, -1.0], [10.0, 10.0, 10.0]], dtype=np.float64)
        desc = sample_descriptors_at_points(feat_vol, pts)
        # Should not raise and all values should be 1.0 (border clamping)
        np.testing.assert_allclose(desc, np.ones_like(desc), atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
