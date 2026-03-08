"""
Tests for data loading and preprocessing.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.data.dataset_thoraxcbct import (
    ThoraxCBCTDataset,
    load_keypoints_csv,
    get_paired_images,
    load_dataset_json,
)
from pipeline.data.preprocessing import (
    robust_intensity_normalize,
    generate_trunk_mask,
    volume_slice_to_pseudo_rgb,
    extract_all_pseudo_rgb_slices,
    rescale_to_model_range,
)
from pipeline.config import PipelineConfig

DATA_ROOT = Path("/nexus/posix0/MBR-neuralsystems/alim/regdata/ThoraxCBCT/ThoraxCBCT")


@pytest.fixture
def config():
    return PipelineConfig()


class TestDatasetLoading:
    """Test NIfTI and keypoint loading."""

    @pytest.mark.skipif(not DATA_ROOT.exists(), reason="Data not available")
    def test_dataset_json_parsing(self):
        ds_json = load_dataset_json(DATA_ROOT / "ThoraxCBCT_dataset.json")
        assert ds_json["name"] == "ThoraxCBCT"
        assert len(ds_json["training_paired_images"]) == 22
        assert len(ds_json["registration_val"]) == 6

    @pytest.mark.skipif(not DATA_ROOT.exists(), reason="Data not available")
    def test_dataset_train_split(self):
        ds = ThoraxCBCTDataset(DATA_ROOT, split="train")
        assert len(ds) == 22

    @pytest.mark.skipif(not DATA_ROOT.exists(), reason="Data not available")
    def test_dataset_val_split(self):
        ds = ThoraxCBCTDataset(DATA_ROOT, split="val")
        assert len(ds) == 6

    @pytest.mark.skipif(not DATA_ROOT.exists(), reason="Data not available")
    def test_load_single_pair(self):
        ds = ThoraxCBCTDataset(DATA_ROOT, split="train")
        data = ds[0]
        assert data["fixed_img"].ndim == 3
        assert data["moving_img"].ndim == 3
        assert data["fixed_img"].shape == (390, 280, 300)

    @pytest.mark.skipif(not DATA_ROOT.exists(), reason="Data not available")
    def test_keypoints_loaded(self):
        ds = ThoraxCBCTDataset(DATA_ROOT, split="train")
        data = ds[0]
        assert data["fixed_keypoints"] is not None
        assert data["moving_keypoints"] is not None
        assert data["fixed_keypoints"].shape[1] == 3
        assert data["fixed_keypoints"].shape[0] > 1000


class TestPreprocessing:
    """Test preprocessing utilities."""

    def test_intensity_normalization(self):
        vol = np.random.randn(32, 32, 32).astype(np.float32) * 100 + 500
        normed = robust_intensity_normalize(vol)
        assert abs(normed.mean()) < 1.0
        assert abs(normed.std() - 1.0) < 0.5

    def test_intensity_normalization_with_mask(self):
        vol = np.random.randn(32, 32, 32).astype(np.float32) * 100 + 500
        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        mask[8:24, 8:24, 8:24] = 1
        normed = robust_intensity_normalize(vol, mask=mask)
        assert normed.dtype == np.float32

    def test_trunk_mask_generation(self):
        vol = np.zeros((32, 32, 32), dtype=np.float32)
        vol[8:24, 8:24, 8:24] = 100.0  # body in center
        vol += np.random.randn(32, 32, 32).astype(np.float32) * 5
        mask = generate_trunk_mask(vol, method="percentile", percentile=80.0)
        assert mask.dtype == np.uint8
        assert mask.sum() > 0
        # Center should be in mask
        assert mask[16, 16, 16] == 1

    def test_pseudo_rgb_conversion(self):
        vol = np.random.randn(32, 32, 32).astype(np.float32)
        rgb = volume_slice_to_pseudo_rgb(vol, slice_idx=16, axis=0)
        assert rgb.shape == (3, 32, 32)

    def test_pseudo_rgb_edges(self):
        vol = np.random.randn(32, 32, 32).astype(np.float32)
        # First slice — should clamp idx_prev to 0
        rgb = volume_slice_to_pseudo_rgb(vol, slice_idx=0, axis=0)
        assert rgb.shape == (3, 32, 32)
        # Last slice
        rgb = volume_slice_to_pseudo_rgb(vol, slice_idx=31, axis=0)
        assert rgb.shape == (3, 32, 32)

    def test_extract_all_slices(self):
        vol = np.random.randn(10, 12, 14).astype(np.float32)
        all_rgb = extract_all_pseudo_rgb_slices(vol, axis=0)
        assert all_rgb.shape == (10, 3, 12, 14)

    def test_rescale_to_model_range(self):
        # Input: z-scored slices (mean≈0, std≈1)
        rng = np.random.RandomState(0)
        slices = rng.randn(5, 3, 32, 32).astype(np.float32)
        rescaled = rescale_to_model_range(slices)

        assert rescaled.shape == slices.shape
        assert rescaled.dtype == np.float32
        assert np.isfinite(rescaled).all(), "rescale_to_model_range produced NaN/Inf"

        # After clipping to ±3σ, scaling to [0,1], then ImageNet normalisation:
        # ImageNet mean ≈ [0.485, 0.456, 0.406], std ≈ [0.229, 0.224, 0.225].
        # The midpoint of the [0,1] range (z-score=0) maps to ≈ (0.5-mean)/std:
        #   channel 0: (0.5 - 0.485) / 0.229 ≈  0.065
        #   channel 1: (0.5 - 0.456) / 0.224 ≈  0.196
        #   channel 2: (0.5 - 0.406) / 0.225 ≈  0.418
        # So values around the median of the input should land near these numbers.
        for c in range(3):
            ch_median = float(np.median(rescaled[:, c]))
            assert abs(ch_median) < 3.0, (
                f"Channel {c} median {ch_median:.3f} is far from expected range "
                "(should be within ±3 after ImageNet normalisation)"
            )

    def test_rescale_known_midpoint(self):
        """z-score = 0 should map to (0.5 - ImageNet_mean) / ImageNet_std per channel."""
        slices = np.zeros((1, 3, 4, 4), dtype=np.float32)  # all zeros (z-score=0)
        rescaled = rescale_to_model_range(slices, clip_sigma=3.0,
                                          target_mean=(0.485, 0.456, 0.406),
                                          target_std=(0.229, 0.224, 0.225))
        expected = [(0.5 - 0.485) / 0.229,
                    (0.5 - 0.456) / 0.224,
                    (0.5 - 0.406) / 0.225]
        for c in range(3):
            np.testing.assert_allclose(rescaled[0, c, 0, 0], expected[c], atol=1e-4,
                                       err_msg=f"Channel {c} midpoint mismatch")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
