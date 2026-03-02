"""
Pipeline visualization utilities.

Saves diagnostic PNG panels for each stage of the registration pipeline.
All visualizations are saved under log_dir/pair_XX/.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class PipelineVisualizer:
    """
    Diagnostic visualizer for the registration pipeline.

    Saves PNG panels for inputs, preprocessing, features, matches,
    displacement fields, and final outputs to a per-pair subdirectory.
    """

    def __init__(self, log_dir: Path, pair_idx: int, enabled: bool = True):
        """
        Args:
            log_dir: base directory for saving visualizations
            pair_idx: index of the current pair (used to name subdirectory)
            enabled: if False, all methods are no-ops (useful for benchmarking)
        """
        self.enabled = enabled
        self.pair_idx = pair_idx
        if enabled:
            self.out_dir = Path(log_dir) / f"pair_{pair_idx:02d}"
            self.out_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Visualizer saving to {self.out_dir}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save(self, name: str, fig) -> None:
        """Save a matplotlib figure and close it."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            path = self.out_dir / f"{name}.png"
            fig.savefig(str(path), dpi=100, bbox_inches="tight")
            fig.clf()
            import matplotlib.pyplot as plt
            plt.close(fig)
            logger.info(f"  Saved: {path.name}")
        except Exception as e:
            logger.warning(f"  Visualization save failed ({name}): {e}")

    @staticmethod
    def _mid_slice(vol: np.ndarray) -> np.ndarray:
        """Return the mid-axial slice of a 3D volume."""
        return vol[vol.shape[0] // 2]

    # ------------------------------------------------------------------
    # Stage visualizations
    # ------------------------------------------------------------------

    def input(
        self,
        fixed_img: np.ndarray,
        moving_img: np.ndarray,
        fixed_kp: Optional[np.ndarray] = None,
        moving_kp: Optional[np.ndarray] = None,
    ) -> None:
        """Visualize raw input volumes with optional keypoints."""
        if not self.enabled:
            return
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            z = fixed_img.shape[0] // 2
            axes[0].imshow(fixed_img[z], cmap="gray")
            axes[0].set_title("Fixed (axial mid-slice)")
            axes[1].imshow(moving_img[moving_img.shape[0] // 2], cmap="gray")
            axes[1].set_title("Moving (axial mid-slice)")
            if fixed_kp is not None:
                kp_z = fixed_kp[np.abs(fixed_kp[:, 0] - z) < 5]
                if len(kp_z):
                    axes[0].scatter(kp_z[:, 2], kp_z[:, 1], s=5, c="red", alpha=0.5)
            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            self._save("00_input", fig)
        except Exception as e:
            logger.warning(f"  input viz failed: {e}")

    def preprocessing(
        self,
        fixed_img: np.ndarray,
        moving_img: np.ndarray,
        fixed_mask: np.ndarray,
        moving_mask: np.ndarray,
        fixed_norm: Optional[np.ndarray] = None,
        moving_norm: Optional[np.ndarray] = None,
    ) -> None:
        """Visualize trunk masks and normalized volumes."""
        if not self.enabled:
            return
        try:
            import matplotlib.pyplot as plt
            n_cols = 4 if fixed_norm is not None else 2
            fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
            z = fixed_img.shape[0] // 2
            axes[0].imshow(fixed_img[z], cmap="gray")
            axes[0].imshow(fixed_mask[z], cmap="Reds", alpha=0.3)
            axes[0].set_title("Fixed + mask")
            axes[1].imshow(moving_img[moving_img.shape[0] // 2], cmap="gray")
            axes[1].imshow(moving_mask[moving_mask.shape[0] // 2], cmap="Reds", alpha=0.3)
            axes[1].set_title("Moving + mask")
            if fixed_norm is not None and n_cols == 4:
                axes[2].imshow(fixed_norm[z], cmap="gray")
                axes[2].set_title("Fixed (normalized)")
                axes[3].imshow(moving_norm[moving_norm.shape[0] // 2], cmap="gray")
                axes[3].set_title("Moving (normalized)")
            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            self._save("01_preprocessing", fig)
        except Exception as e:
            logger.warning(f"  preprocessing viz failed: {e}")

    def features(
        self,
        fixed_feats: np.ndarray,
        moving_feats: np.ndarray,
    ) -> None:
        """Visualize PCA projection of feature maps (first 3 PCs → RGB)."""
        if not self.enabled:
            return
        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA

            def feat_to_rgb(feats: np.ndarray) -> np.ndarray:
                """(fD, fH, fW, C) → mid-slice RGB."""
                fD, fH, fW, C = feats.shape
                z = fD // 2
                flat = feats[z].reshape(-1, C)
                pca = PCA(n_components=3)
                rgb = pca.fit_transform(flat)
                mn, mx = rgb.min(axis=0, keepdims=True), rgb.max(axis=0, keepdims=True)
                rgb = (rgb - mn) / (mx - mn + 1e-8)
                return rgb.reshape(fH, fW, 3)

            rgb_f = feat_to_rgb(fixed_feats)
            rgb_m = feat_to_rgb(moving_feats)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(rgb_f)
            axes[0].set_title("Fixed features (PCA-RGB, mid-slice)")
            axes[1].imshow(rgb_m)
            axes[1].set_title("Moving features (PCA-RGB, mid-slice)")
            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            self._save("02_features", fig)
        except Exception as e:
            logger.warning(f"  features viz failed: {e}")

    def sampled_points(
        self,
        fixed_img: np.ndarray,
        moving_img: np.ndarray,
        fixed_pts: np.ndarray,
        moving_pts: np.ndarray,
        stage_name: str = "",
    ) -> None:
        """Overlay sampled points on mid-axial slices."""
        if not self.enabled:
            return
        try:
            import matplotlib.pyplot as plt
            z = fixed_img.shape[0] // 2
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(fixed_img[z], cmap="gray")
            fp = fixed_pts[np.abs(fixed_pts[:, 0] - z) < 5]
            if len(fp):
                axes[0].scatter(fp[:, 2], fp[:, 1], s=3, c="lime", alpha=0.6)
            axes[0].set_title(f"Fixed sampled pts ({len(fixed_pts)}) [{stage_name}]")
            axes[1].imshow(moving_img[moving_img.shape[0] // 2], cmap="gray")
            mp = moving_pts[np.abs(moving_pts[:, 0] - moving_img.shape[0] // 2) < 5]
            if len(mp):
                axes[1].scatter(mp[:, 2], mp[:, 1], s=3, c="lime", alpha=0.6)
            axes[1].set_title(f"Moving sampled pts ({len(moving_pts)}) [{stage_name}]")
            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            safe_name = stage_name.replace("/", "_")
            self._save(f"03_sampled_{safe_name}", fig)
        except Exception as e:
            logger.warning(f"  sampled_points viz failed: {e}")

    def matches(
        self,
        match_result: dict,
        fixed_img: np.ndarray,
        moving_img: np.ndarray,
    ) -> None:
        """Visualize matched point pairs projected onto mid-axial slice."""
        if not self.enabled:
            return
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            z = fixed_img.shape[0] // 2
            axes[0].imshow(fixed_img[z], cmap="gray")
            axes[1].imshow(moving_img[moving_img.shape[0] // 2], cmap="gray")

            src = match_result.get("matched_points_src")
            tgt = match_result.get("matched_points_tgt")
            if src is not None and tgt is not None:
                near_z = np.abs(src[:, 0] - z) < 5
                fs = src[near_z]
                ft = tgt[near_z]
                if len(fs):
                    axes[1].scatter(fs[:, 2], fs[:, 1], s=8, c="cyan", alpha=0.5,
                                    label="matched src")
                    axes[0].scatter(ft[:, 2], ft[:, 1], s=8, c="yellow", alpha=0.5,
                                    label="matched tgt")
            n = match_result.get("n_matches", "?")
            axes[0].set_title(f"Fixed matched ({n} matches)")
            axes[1].set_title("Moving matched")
            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            self._save("04_matches", fig)
        except Exception as e:
            logger.warning(f"  matches viz failed: {e}")

    def displacement(
        self,
        displacement,
        fixed_img: np.ndarray,
    ) -> None:
        """Visualize displacement field magnitude on mid-axial slice."""
        if not self.enabled:
            return
        try:
            import matplotlib.pyplot as plt
            import torch
            disp_np = displacement.cpu().numpy() if hasattr(displacement, "cpu") else displacement
            # disp_np: (1, 3, D, H, W)
            z = disp_np.shape[2] // 2
            mag = np.sqrt(
                disp_np[0, 0, z] ** 2 +
                disp_np[0, 1, z] ** 2 +
                disp_np[0, 2, z] ** 2
            )
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(fixed_img[fixed_img.shape[0] // 2], cmap="gray")
            axes[0].set_title("Fixed (reference)")
            im = axes[1].imshow(mag, cmap="hot")
            axes[1].set_title(f"Displacement magnitude (mm) at z={z}")
            plt.colorbar(im, ax=axes[1])
            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            self._save("05_displacement", fig)
        except Exception as e:
            logger.warning(f"  displacement viz failed: {e}")

    def output(
        self,
        fixed_img: np.ndarray,
        moving_img: np.ndarray,
        displacement,
        fixed_kp: Optional[np.ndarray] = None,
        moving_kp: Optional[np.ndarray] = None,
    ) -> None:
        """Visualize warped moving image alongside fixed for qualitative check."""
        if not self.enabled:
            return
        try:
            import matplotlib.pyplot as plt
            from scipy.ndimage import map_coordinates as mc

            disp_np = displacement.cpu().numpy() if hasattr(displacement, "cpu") else displacement
            D, H, W = fixed_img.shape
            z = D // 2

            # Warp moving image: for each fixed voxel sample from moving
            zz, yy, xx = np.meshgrid(
                np.arange(D), np.arange(H), np.arange(W), indexing="ij"
            )
            coords = [
                zz + disp_np[0, 0],
                yy + disp_np[0, 1],
                xx + disp_np[0, 2],
            ]
            warped = mc(moving_img, coords, order=1, mode="nearest")

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            axes[0].imshow(fixed_img[z], cmap="gray")
            axes[0].set_title("Fixed")
            axes[1].imshow(moving_img[moving_img.shape[0] // 2], cmap="gray")
            axes[1].set_title("Moving (unregistered)")
            axes[2].imshow(warped[z], cmap="gray")
            axes[2].set_title("Moving (warped to fixed)")

            if fixed_kp is not None and moving_kp is not None:
                fkp_z = fixed_kp[np.abs(fixed_kp[:, 0] - z) < 5]
                mkp_z = moving_kp[np.abs(moving_kp[:, 0] - z) < 5]
                if len(fkp_z):
                    axes[0].scatter(fkp_z[:, 2], fkp_z[:, 1], s=8, c="red", alpha=0.7)
                if len(mkp_z):
                    axes[1].scatter(mkp_z[:, 2], mkp_z[:, 1], s=8, c="blue", alpha=0.7)

            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            self._save("06_output", fig)
        except Exception as e:
            logger.warning(f"  output viz failed: {e}")
