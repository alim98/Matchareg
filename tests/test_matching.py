"""
Tests for matching module (sampling, GWOT, filtering).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.matching.sampling import (
    sample_points_in_mask,
    sample_descriptors_at_points,
    include_keypoints,
)
from pipeline.matching.gwot3d import (
    compute_feature_cost,
    compute_spatial_prior,
    build_local_distance_matrix,
    nn_matching,
    match,
)
from pipeline.matching.filters import filter_matches


class TestSampling:
    def test_uniform_sampling(self):
        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        mask[8:24, 8:24, 8:24] = 1
        pts = sample_points_in_mask(mask, n_points=100, z_stratified=False)
        assert pts.shape == (100, 3)
        # All points should be inside mask
        for p in pts.astype(int):
            assert mask[p[0], p[1], p[2]] == 1

    def test_stratified_sampling(self):
        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        mask[8:24, 8:24, 8:24] = 1
        pts = sample_points_in_mask(mask, n_points=100, z_stratified=True)
        assert len(pts) == 100

    def test_descriptor_sampling(self):
        feat_vol = np.random.randn(64, 10, 10, 10).astype(np.float32)
        points = np.array([[5, 5, 5], [3, 3, 3]], dtype=np.float64)
        desc = sample_descriptors_at_points(feat_vol, points)
        assert desc.shape == (2, 64)

    def test_include_keypoints(self):
        pts = np.random.rand(50, 3) * 30
        kps = np.random.rand(10, 3) * 30
        combined = include_keypoints(pts, kps)
        assert len(combined) == 60


class TestMatching:
    def test_feature_cost(self):
        d1 = np.random.randn(10, 64).astype(np.float64)
        d2 = np.random.randn(15, 64).astype(np.float64)
        d1 = d1 / np.linalg.norm(d1, axis=1, keepdims=True)
        d2 = d2 / np.linalg.norm(d2, axis=1, keepdims=True)
        C = compute_feature_cost(d1, d2)
        assert C.shape == (10, 15)
        assert np.all(C >= -0.01)  # cosine dist >= 0 for normalized vectors

    def test_spatial_prior(self):
        p1 = np.random.rand(10, 3) * 100
        p2 = np.random.rand(15, 3) * 100
        C = compute_spatial_prior(p1, p2, sigma=50)
        assert C.shape == (10, 15)
        assert np.all(C >= 0)

    def test_local_distance_matrix(self):
        pts = np.random.rand(20, 3) * 100
        D = build_local_distance_matrix(pts, radius=50)
        assert D.shape == (20, 20)
        assert np.all(D >= 0)
        # Diagonal should be 0
        assert np.allclose(np.diag(D), 0)

    def test_nn_matching_known(self):
        """NN matching with known correspondences."""
        N = 50
        D = 32
        # Create identical descriptors (shifted by a known permutation)
        desc = np.random.randn(N, D).astype(np.float64)
        desc = desc / np.linalg.norm(desc, axis=1, keepdims=True)
        pts1 = np.random.rand(N, 3) * 100
        pts2 = pts1 + np.random.randn(N, 3) * 2

        result = nn_matching(desc, desc, pts1, pts2)
        # Should match identity
        assert len(result["matches_src_idx"]) > 0
        # All matches should be self-matches
        for s, t in zip(result["matches_src_idx"], result["matches_tgt_idx"]):
            assert s == t

    def test_nn_dispatch(self):
        d1 = np.random.randn(20, 32).astype(np.float64)
        d2 = np.random.randn(20, 32).astype(np.float64)
        d1 /= np.linalg.norm(d1, axis=1, keepdims=True)
        d2 /= np.linalg.norm(d2, axis=1, keepdims=True)
        p1 = np.random.rand(20, 3) * 100
        p2 = np.random.rand(20, 3) * 100
        result = match(d1, d2, p1, p2, method="nn")
        assert "matches_src_idx" in result


class TestFiltering:
    def test_confidence_filter(self):
        result = {
            "matches_src_idx": np.array([0, 1, 2, 3]),
            "matches_tgt_idx": np.array([0, 1, 2, 3]),
            "weights": np.array([0.5, 0.005, 0.3, 0.001]),
        }
        pts = np.random.rand(4, 3) * 100
        filtered = filter_matches(result, pts, pts, confidence_threshold=0.01)
        assert filtered["n_matches"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
