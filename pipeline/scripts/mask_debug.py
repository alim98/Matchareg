import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.data.dataset_thoraxcbct import ThoraxCBCTDataset
from pipeline.data.preprocessing import generate_trunk_mask

def plot_mask_overlay(volume: np.ndarray, mask: np.ndarray, axis: int = 0, case_name: str = ""):
    """Plot an overlay of the binary mask on the anatomical background at 3 depth slices."""
    D, H, W = volume.shape
    depth = volume.shape[axis]
    
    slices = [
        int(depth * 0.25),  # 1/4
        int(depth * 0.50),  # Center
        int(depth * 0.75)   # 3/4
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(f"Mask Overlay - {case_name} (Axis {axis})", fontsize=16)
    
    for i, idx in enumerate(slices):
        if axis == 0:
            vol_slice = volume[idx, :, :]
            mask_slice = mask[idx, :, :]
        elif axis == 1:
            vol_slice = volume[:, idx, :]
            mask_slice = mask[:, idx, :]
        else:
            vol_slice = volume[:, :, idx]
            mask_slice = mask[:, :, idx]
            
        # Plot anatomy in grayscale
        axes[i].imshow(vol_slice, cmap="gray", origin="lower")
        # Overlay mask in translucent red
        axes[i].imshow(mask_slice, cmap="Reds", alpha=0.4 * mask_slice, origin="lower")
        
        axes[i].set_title(f"Slice {idx}")
        axes[i].axis("off")
        
    plt.tight_layout()
    
    out_dir = PROJECT_ROOT / "pipeline" / "output" / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mask_overlay_{case_name}_ax{axis}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()

def main():
    print("Loading dataset...")
    # Setup paths - assuming default ThoraxCBCT
    data_root = Path("/nexus/posix0/MBR-neuralsystems/alim/regdata/ThoraxCBCT/ThoraxCBCT")
    dataset = ThoraxCBCTDataset(data_root, split="train")
    
    pair_idx = 0
    data = dataset[pair_idx]
    
    print("\n[Fixed Image]")
    fixed_img = data["fixed_img"]
    print(f"Shape: {fixed_img.shape}, Data bounds: [{fixed_img.min():.1f}, {fixed_img.max():.1f}]")
    mask_fixed = generate_trunk_mask(fixed_img, method="percentile", percentile=30)
    
    print("\n[Moving Image]")
    moving_img = data["moving_img"]
    print(f"Shape: {moving_img.shape}, Data bounds: [{moving_img.min():.1f}, {moving_img.max():.1f}]")
    mask_moving = generate_trunk_mask(moving_img, method="percentile", percentile=30)
    
    # Generate coronal (y-axis) and axial (z-axis) views
    plot_mask_overlay(fixed_img, mask_fixed, axis=0, case_name="Fixed_Axial")
    plot_mask_overlay(fixed_img, mask_fixed, axis=1, case_name="Fixed_Coronal")
    
    plot_mask_overlay(moving_img, mask_moving, axis=0, case_name="Moving_Axial")
    plot_mask_overlay(moving_img, mask_moving, axis=1, case_name="Moving_Coronal")
    print("\nDone! Visualizations saved to pipeline/output/debug/")

if __name__ == "__main__":
    main()
