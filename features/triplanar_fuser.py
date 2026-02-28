"""
Tri-planar feature fusion for 3D volumes.

Extracts 2D features from axial, coronal, and sagittal slices using a
foundation model, then fuses them into per-voxel 3D descriptors.

Memory-efficient: works at patch resolution (1/16), not full volume.
"""
import logging
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import torch
import torch.nn.functional as F

from ..data.preprocessing import (
    extract_all_pseudo_rgb_slices,
    rescale_to_model_range,
)

logger = logging.getLogger(__name__)


class TriplanarFuser:
    """
    Tri-planar 3D feature extraction pipeline.

    For each volume:
    1. Downsample volume to manageable size
    2. Extract pseudo-RGB slices along axial (0), coronal (1), sagittal (2)
    3. Run 2D feature extractor on each slice batch
    4. Assemble into 3D feature volumes at PATCH resolution (not full res)
    5. Fuse per-voxel: concatenate 3 planes → L2 normalize
    """

    def __init__(
        self,
        extractor,
        batch_size: int = 4,
        fusion: Literal["concat_norm", "concat_pca"] = "concat_norm",
        pca_dim: int = 256,
        downsample: int = 2,
        device: str = "cpu",
    ):
        """
        Args:
            extractor: feature extractor with extract_features(Tensor) -> Tensor
            batch_size: number of slices per GPU batch (keep low for memory)
            fusion: 'concat_norm' or 'concat_pca'
            pca_dim: output dim if using PCA fusion
            downsample: spatial downsample factor before slicing (2=half res)
            device: torch device
        """
        self.extractor = extractor
        self.batch_size = batch_size
        self.fusion = fusion
        self.pca_dim = pca_dim
        self.downsample = downsample
        self.device = device
        self.patch_size = getattr(extractor, 'patch_size', 16)

    def _downsample_volume(self, volume: np.ndarray) -> np.ndarray:
        """Downsample volume by factor self.downsample using strided slicing."""
        if self.downsample <= 1:
            return volume
        s = self.downsample
        return volume[::s, ::s, ::s].copy()

    def extract_plane_features(
        self,
        volume: np.ndarray,
        axis: int,
        min_input_size: int = 448,
    ) -> np.ndarray:
        """
        Extract 2D features for all slices along one axis.
        Returns features at patch resolution.

        Slices are resized to at least min_input_size to ensure sufficient
        spatial resolution in the patch feature grid (ViT-B/16 divides by 16).

        Args:
            volume: (D, H, W) normalized volume (already downsampled)
            axis: 0=axial, 1=coronal, 2=sagittal
            min_input_size: minimum spatial dimension for ViT input
                           (448 → 28×28 patches, 224 → 14×14 patches)

        Returns:
            features: np.ndarray of shape (N_slices, feat_dim, h_patch, w_patch)
        """
        # Get pseudo-RGB slices: (N_slices, 3, H', W')
        slices_rgb = extract_all_pseudo_rgb_slices(volume, axis=axis)
        slices_rgb = rescale_to_model_range(slices_rgb)

        n_slices = slices_rgb.shape[0]
        _, _, h_orig, w_orig = slices_rgb.shape

        # Resize slices if too small for meaningful patch features
        # ViT-B/16 produces h/16 × w/16 patches. We want at least 14×14 patches.
        need_resize = h_orig < min_input_size or w_orig < min_input_size
        if need_resize:
            scale = max(min_input_size / h_orig, min_input_size / w_orig)
            new_h = int(np.ceil(h_orig * scale / self.patch_size) * self.patch_size)
            new_w = int(np.ceil(w_orig * scale / self.patch_size) * self.patch_size)
            logger.info(f"    Resizing slices {h_orig}×{w_orig} → {new_h}×{new_w} "
                        f"(→ {new_h//self.patch_size}×{new_w//self.patch_size} patches)")

        all_feats = []

        for start in range(0, n_slices, self.batch_size):
            end = min(start + self.batch_size, n_slices)
            batch = torch.from_numpy(slices_rgb[start:end]).to(self.device)

            # Resize if necessary
            if need_resize:
                batch = F.interpolate(
                    batch, size=(new_h, new_w),
                    mode='bilinear', align_corners=False,
                )

            feats = self.extractor.extract_features(batch)
            # Sanity: extractor must return channel-first (B, C, h, w).
            # DINOv3's get_intermediate_layers(reshape=True) applies permute(0,3,1,2)
            # → C == embed_dim (768 for ViT-B) >> spatial dims.
            # If this fires, the extractor is returning channel-last and the whole
            # triplanar geometry will be silently corrupted.
            assert feats.ndim == 4 and feats.shape[1] > feats.shape[2], (
                f"extract_features returned shape {tuple(feats.shape)}. "
                "Expected channel-first (B, C, H, W) with C >> H. "
                "Check get_intermediate_layers uses reshape=True with permute(0,3,1,2)."
            )
            # Move to CPU immediately to save device memory
            all_feats.append(feats.cpu().numpy())


            if (start // self.batch_size) % 20 == 0:
                logger.info(f"    slice {start}/{n_slices}")

        return np.concatenate(all_feats, axis=0)

    def fuse_triplanar(
        self,
        volume: np.ndarray,
    ) -> tuple:
        """
        Full tri-planar feature extraction and fusion.

        Returns features at PATCH resolution to save memory.

        Args:
            volume: (D, H, W) preprocessed volume

        Returns:
            (fused_features, feature_shape, original_shape)
            fused_features: (fused_dim, fD, fH, fW) per-voxel descriptors
                           at patch resolution
            feature_shape: (fD, fH, fW) shape of feature grid
            original_shape: (D, H, W) original volume shape
        """
        original_shape = volume.shape
        D, H, W = original_shape
        logger.info(f"Original volume: ({D}, {H}, {W})")

        # Downsample
        vol_ds = self._downsample_volume(volume)
        dD, dH, dW = vol_ds.shape
        logger.info(f"Downsampled to: ({dD}, {dH}, {dW}) (factor={self.downsample})")

        # Extract features for each plane
        axis_names = ["axial", "coronal", "sagittal"]
        plane_features = []

        for axis in range(3):
            n_slices = vol_ds.shape[axis]
            logger.info(f"  {axis_names[axis]}: {n_slices} slices...")
            plane_feats = self.extract_plane_features(vol_ds, axis=axis)
            # plane_feats: (N_slices, feat_dim, h_patch, w_patch)
            plane_features.append(plane_feats)

        feat_dim = plane_features[0].shape[1]

        # Determine unified feature grid resolution.
        # Each plane has (n_slices, feat_dim, patch_h, patch_w).
        # Axial (axis=0):    n_slices=dD, patches represent (dH, dW)
        # Coronal (axis=1):  n_slices=dH, patches represent (dD, dW)
        # Sagittal (axis=2): n_slices=dW, patches represent (dD, dH)
        #
        # Use patch resolution for the unified grid in each dimension.
        # Get the patch size from each plane's output.
        axial_pf = plane_features[0]    # (dD, dim, pH_a, pW_a)  pH_a ≈ dH→patch, pW_a ≈ dW→patch
        coronal_pf = plane_features[1]  # (dH, dim, pH_c, pW_c)  pH_c ≈ dD→patch, pW_c ≈ dW→patch
        sagittal_pf = plane_features[2] # (dW, dim, pH_s, pW_s)  pH_s ≈ dD→patch, pW_s ≈ dH→patch

        # Unified grid: keep z at full slice resolution from axial plane,
        # since each axial slice produces one feature map. Only y and x
        # are at patch resolution. This gives (dD, dH/ps, dW/ps) grid.
        fD = dD                        # full z-resolution from axial slices
        fH = axial_pf.shape[2]        # dH at patch resolution (from axial's h_patch)
        fW = axial_pf.shape[3]        # dW at patch resolution (from axial's w_patch)
        logger.info(f"  Unified feature grid: ({fD}, {fH}, {fW})")

        feat_volumes = []

        # --- Axial: (dD, feat_dim, pH_a, pW_a) → need (feat_dim, fD, fH, fW)
        # Spatial axes: z=dD slices, y=pH_a patches, x=pW_a patches
        # pH_a should ≈ fH, pW_a should ≈ fW (both come from same slice dims)
        feat_vol = axial_pf.transpose(1, 0, 2, 3)  # (feat_dim, dD, pH_a, pW_a)
        feat_t = torch.from_numpy(feat_vol).unsqueeze(0)  # (1, feat_dim, dD, pH_a, pW_a)
        feat_t = F.interpolate(feat_t, size=(fD, fH, fW), mode="trilinear", align_corners=False)
        feat_volumes.append(feat_t[0].numpy())
        logger.info(f"  axial → {feat_volumes[-1].shape}")

        # --- Coronal: (dH, feat_dim, pH_c, pW_c) → need (feat_dim, fD, fH, fW)
        # Spatial axes: y=dH slices, z=pH_c patches, x=pW_c patches
        # First transpose to (feat_dim, dH, pH_c, pW_c)
        feat_vol = coronal_pf.transpose(1, 0, 2, 3)  # (feat_dim, dH, pH_c, pW_c)
        # Rearrange: (feat_dim, y=dH, z=pH_c, x=pW_c) → (feat_dim, z=pH_c, y=dH, x=pW_c)
        feat_vol = feat_vol.transpose(0, 2, 1, 3)
        feat_t = torch.from_numpy(feat_vol).unsqueeze(0)  # (1, feat_dim, pH_c, dH, pW_c)
        feat_t = F.interpolate(feat_t, size=(fD, fH, fW), mode="trilinear", align_corners=False)
        feat_volumes.append(feat_t[0].numpy())
        logger.info(f"  coronal → {feat_volumes[-1].shape}")

        # --- Sagittal: (dW, feat_dim, pH_s, pW_s) → need (feat_dim, fD, fH, fW)
        # Spatial axes: x=dW slices, z=pH_s patches, y=pW_s patches
        # First transpose to (feat_dim, dW, pH_s, pW_s)
        feat_vol = sagittal_pf.transpose(1, 0, 2, 3)  # (feat_dim, dW, pH_s, pW_s)
        # Rearrange: (feat_dim, x=dW, z=pH_s, y=pW_s) → (feat_dim, z=pH_s, y=pW_s, x=dW)
        feat_vol = feat_vol.transpose(0, 2, 3, 1)
        feat_t = torch.from_numpy(feat_vol).unsqueeze(0)  # (1, feat_dim, pH_s, pW_s, dW)
        feat_t = F.interpolate(feat_t, size=(fD, fH, fW), mode="trilinear", align_corners=False)
        feat_volumes.append(feat_t[0].numpy())
        logger.info(f"  sagittal → {feat_volumes[-1].shape}")

        # Concatenate along feature dimension
        fused = np.concatenate(feat_volumes, axis=0)  # (3*feat_dim, fD, fH, fW)
        logger.info(f"Concatenated: {fused.shape}")

        if self.fusion == "concat_norm":
            norms = np.linalg.norm(fused, axis=0, keepdims=True) + 1e-8
            fused = fused / norms
        elif self.fusion == "concat_pca":
            fused = self._apply_pca(fused)

        feature_shape = fused.shape[1:]
        logger.info(f"Final fused: {fused.shape} "
                     f"(memory: {fused.nbytes / 1e6:.0f} MB)")
        return fused, feature_shape, original_shape

    def _apply_pca(self, features: np.ndarray) -> np.ndarray:
        """Apply PCA dimensionality reduction."""
        from sklearn.decomposition import PCA

        feat_dim = features.shape[0]
        spatial = features.shape[1:]
        flat = features.reshape(feat_dim, -1).T

        n_voxels = flat.shape[0]
        n_samples = min(50000, n_voxels)
        indices = np.random.choice(n_voxels, n_samples, replace=False)

        pca = PCA(n_components=self.pca_dim)
        pca.fit(flat[indices])
        reduced = pca.transform(flat)

        norms = np.linalg.norm(reduced, axis=1, keepdims=True) + 1e-8
        reduced = reduced / norms
        return reduced.T.reshape(self.pca_dim, *spatial)


def save_features(features, path: Path):
    """Save feature data to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(features, tuple):
        fused, feat_shape, orig_shape = features
        np.savez_compressed(
            str(path),
            features=fused,
            feat_shape=np.array(feat_shape),
            orig_shape=np.array(orig_shape),
        )
    else:
        np.savez_compressed(str(path), features=features)
    logger.info(f"Saved features to {path}")


def load_features(path: Path):
    """Load feature data from disk."""
    data = np.load(str(path))
    if "feat_shape" in data:
        return (
            data["features"],
            tuple(data["feat_shape"]),
            tuple(data["orig_shape"]),
        )
    return data["features"]
