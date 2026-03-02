import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.data.dataset_thoraxcbct import ThoraxCBCTDataset
from pipeline.config import PipelineConfig
from pipeline.features.dinov3_extractor import DINOv3Extractor
from pipeline.features.triplanar_fuser import TriplanarFuser
from pipeline.data.preprocessing import generate_trunk_mask
from pipeline.matching.sampling import sample_descriptors_at_points, sample_points_in_mask
from pipeline.matching.gwot3d import match

def visualize_correspondences(
    fixed_img: np.ndarray,
    moving_img: np.ndarray,
    fixed_pts: np.ndarray,
    moving_pts: np.ndarray,
    axis: int = 0,
    max_pts: int = 200,
    output_path: str = "correspondences.png"
):
    """Plot fixed and moving images side-by-side with lines connecting matching points."""
    D, H, W = fixed_img.shape
    
    # Take mid-slice along the requested axis
    slice_idx = int(fixed_img.shape[axis] / 2)
    
    # Filter points near this slice for clearer visualization
    # We allow points within ±20 voxels of the slice
    tolerance = 20
    
    valid_indices = []
    for i in range(len(fixed_pts)):
        p_f = fixed_pts[i]
        p_m = moving_pts[i]
        
        # Check if BOTH points are somewhat near the visualization plane
        if abs(p_f[axis] - slice_idx) < tolerance and abs(p_m[axis] - slice_idx) < tolerance:
            valid_indices.append(i)
            
    # Subsample if too many
    if len(valid_indices) > max_pts:
        valid_indices = np.random.choice(valid_indices, max_pts, replace=False)
        
    print(f"Visualizing {len(valid_indices)} correspondences near slice {slice_idx} (axis {axis})")
        
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Extract slices
    if axis == 0:
        f_slice = fixed_img[slice_idx, :, :]
        m_slice = moving_img[slice_idx, :, :]
        # For plotting: x is dim 2, y is dim 1
        x_idx, y_idx = 2, 1
    elif axis == 1:
        f_slice = fixed_img[:, slice_idx, :]
        m_slice = moving_img[:, slice_idx, :]
        # For plotting: x is dim 2, y is dim 0
        x_idx, y_idx = 2, 0
    else:
        f_slice = fixed_img[:, :, slice_idx]
        m_slice = moving_img[:, :, slice_idx]
        # For plotting: x is dim 1, y is dim 0
        x_idx, y_idx = 1, 0
        
    axes[0].imshow(f_slice, cmap="gray", origin="lower")
    axes[0].set_title(f"Fixed (Slice {slice_idx})")
    axes[0].axis("off")
    
    axes[1].imshow(m_slice, cmap="gray", origin="lower")
    axes[1].set_title(f"Moving (Slice {slice_idx})")
    axes[1].axis("off")
    
    # Plot connecting lines
    for i in valid_indices:
        p_f = fixed_pts[i]
        p_m = moving_pts[i]
        
        # Plot points
        axes[0].plot(p_f[x_idx], p_f[y_idx], 'r.', markersize=8)
        axes[1].plot(p_m[x_idx], p_m[y_idx], 'b.', markersize=8)
        
        # Draw line across subplots
        # Need to use figure coordinate transformations for cross-axes lines
        con = plt.matplotlib.patches.ConnectionPatch(
            xyA=(p_m[x_idx], p_m[y_idx]), coordsA=axes[1].transData,
            xyB=(p_f[x_idx], p_f[y_idx]), coordsB=axes[0].transData,
            color="green", alpha=0.6, linewidth=1.5
        )
        fig.add_artist(con)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()

def main():
    print("Loading dataset...")
    config = PipelineConfig()
    dataset = ThoraxCBCTDataset(config.paths.data_root, split="train")
    data = dataset[0]
    
    fixed_img = data["fixed_img"]
    moving_img = data["moving_img"]
    
    # 1. Generate Masks
    print("Generating masks...")
    mask_fixed = generate_trunk_mask(fixed_img, method="percentile", percentile=30)
    mask_moving = generate_trunk_mask(moving_img, method="percentile", percentile=30)
    
    # 2. Sample Points
    print("Sampling random points uniformly within masks...")
    n_points = 5000
    pts_fixed = sample_points_in_mask(mask_fixed, n_points, z_stratified=True)
    pts_moving = sample_points_in_mask(mask_moving, n_points, z_stratified=True)
    
    # 3. Extract Dense Features using TriPlanarSlicer
    print("Extracting DINOv3 features...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = DINOv3Extractor(weights_path=str(config.paths.dinov3_weights), device=device)
    fuser = TriplanarFuser(
        extractor, 
        batch_size=config.features.slice_batch_size, 
        fusion=config.features.fusion_method, 
        pca_dim=config.features.pca_dim
    )
    
    print("  Processing Fixed...")
    feat_fixed, _, _ = fuser.fuse_triplanar(fixed_img)
    print("  Processing Moving...")
    feat_moving, _, _ = fuser.fuse_triplanar(moving_img)
    
    # 4. Sample Descriptors at Points
    print("Sampling descriptors at points...")
    valid_f, desc_f = sample_descriptors_at_points(feat_fixed, pts_fixed, fixed_img.shape)
    valid_m, desc_m = sample_descriptors_at_points(feat_moving, pts_moving, moving_img.shape)
    
    pts_fixed = pts_fixed[valid_f]
    pts_moving = pts_moving[valid_m]
    
    # 5. NN Matcher
    print("Matching (NN)...")
    res = match(desc_f, desc_m, pts_fixed, pts_moving, method="nn", sim_threshold=0.3, max_displacement=150.0)
    idx_f = res["matches_src_idx"]
    idx_m = res["matches_tgt_idx"]
    
    matched_pts_fixed = pts_fixed[idx_f]
    matched_pts_moving = pts_moving[idx_m]
    
    print(f"Total Matches: {len(idx_f)}")
    
    # 6. Visualize
    out_dir = PROJECT_ROOT / "pipeline" / "output" / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    visualize_correspondences(
        fixed_img, moving_img, 
        matched_pts_fixed, matched_pts_moving, 
        axis=0, max_pts=150, 
        output_path=str(out_dir / "sanity_matches_ax0.png")
    )
    visualize_correspondences(
        fixed_img, moving_img, 
        matched_pts_fixed, matched_pts_moving, 
        axis=1, max_pts=150, 
        output_path=str(out_dir / "sanity_matches_ax1.png")
    )
    
if __name__ == "__main__":
    import torch
    # Limit memory explicitly just for this debug script if needed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()
