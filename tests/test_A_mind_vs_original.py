#!/usr/bin/env python3
"""
Direct comparison: our mind_convex_adam module vs original DINO-Reg code.

Tests the CONVEX-ONLY path (no Adam) on pair 0 to see if output matches.

Usage:
    python -m pipeline.tests.test_A_mind_vs_original
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, '/u/almik/feb25/dinoreg/DINO-Reg')

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import map_coordinates

from pipeline.data.dataset_thoraxcbct import ThoraxCBCTDataset

dataset = ThoraxCBCTDataset('/nexus/posix0/MBR-neuralsystems/alim/regdata/ThoraxCBCT/ThoraxCBCT', split='train')
data = dataset[0]
fixed_img = data["fixed_img"]
moving_img = data["moving_img"]
fkp = data["fixed_keypoints"]
mkp = data["moving_keypoints"]
ideal_disp = mkp - fkp

print(f"Volume: {fixed_img.shape}")
print(f"Initial TRE: {np.linalg.norm(fkp - mkp, axis=1).mean():.3f} mm")

device = 'cuda'
grid_sp = 4
disp_hw = 4

# ==========================================================
# 1) OUR MODULE
# ==========================================================
print("\n" + "="*60)
print("OUR MODULE (mind_convex_adam)")
print("="*60)

from pipeline.transform.mind_convex_adam import mind_convex_adam
disp_ours = mind_convex_adam(
    fixed_img=fixed_img,
    moving_img=moving_img,
    mind_r=1, mind_d=2,
    lambda_weight=0, grid_sp=grid_sp, disp_hw=disp_hw,
    n_iter_adam=0, grid_sp_adam=2,
    ic=True, device=device,
)

disp_ours_np = disp_ours.cpu().numpy()
print(f"disp_ours shape: {disp_ours_np.shape}")
for ch in range(3):
    print(f"  ch{ch}: mean={disp_ours_np[0,ch].mean():+.4f}, std={disp_ours_np[0,ch].std():.4f}, "
          f"absmax={np.abs(disp_ours_np[0,ch]).max():.4f}")

# Evaluate our module
disp_at_kp_ours = np.zeros((len(fkp), 3))
for ch in range(3):
    disp_at_kp_ours[:, ch] = map_coordinates(disp_ours_np[0, ch], fkp.T, order=1, mode='nearest')
warped = fkp + disp_at_kp_ours
tre_ours = np.linalg.norm(warped - mkp, axis=1).mean()
print(f"\n  TRE (our module): {tre_ours:.3f} mm")
for ax in range(3):
    corr = np.corrcoef(ideal_disp[:, ax], disp_at_kp_ours[:, ax])[0, 1]
    print(f"  Axis {ax}: ideal={ideal_disp[:, ax].mean():+.3f}, ours={disp_at_kp_ours[:, ax].mean():+.3f}, corr={corr:+.3f}")

# ==========================================================
# 2) ORIGINAL CODE (if available)
# ==========================================================
print("\n" + "="*60)
print("ORIGINAL CODE (convex_adam_utils)")
print("="*60)

try:
    from convex_adam_utils import MINDSSC as orig_MINDSSC, correlate as orig_correlate
    from convex_adam_utils import coupled_convex as orig_coupled_convex, inverse_consistency as orig_ic
    
    H, W, D = fixed_img.shape  # Original naming convention
    
    img_fixed = torch.from_numpy(fixed_img).float().unsqueeze(0).unsqueeze(0).to(device)
    img_moving = torch.from_numpy(moving_img).float().unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features_fix = orig_MINDSSC(img_fixed, 1, 2).half()
        features_mov = orig_MINDSSC(img_moving, 1, 2).half()
        features_fix_smooth = F.avg_pool3d(features_fix, grid_sp, stride=grid_sp)
        features_mov_smooth = F.avg_pool3d(features_mov, grid_sp, stride=grid_sp)
        n_ch = 12

    ssd, ssd_argmin = orig_correlate(features_fix_smooth, features_mov_smooth, disp_hw, grid_sp, (H, W, D), n_ch)
    
    disp_mesh_t = F.affine_grid(
        disp_hw * torch.eye(3, 4).to(device).half().unsqueeze(0),
        (1, 1, disp_hw*2+1, disp_hw*2+1, disp_hw*2+1), align_corners=True
    ).permute(0, 4, 1, 2, 3).reshape(3, -1, 1)
    
    disp_soft = orig_coupled_convex(ssd, ssd_argmin, disp_mesh_t, grid_sp, (H, W, D))

    # IC
    scale = torch.tensor([H//grid_sp-1, W//grid_sp-1, D//grid_sp-1]).view(1,3,1,1,1).to(device).half()/2
    ssd_, ssd_argmin_ = orig_correlate(features_mov_smooth, features_fix_smooth, disp_hw, grid_sp, (H, W, D), n_ch)
    disp_soft_ = orig_coupled_convex(ssd_, ssd_argmin_, disp_mesh_t, grid_sp, (H, W, D))
    disp_ice, _ = orig_ic((disp_soft/scale).flip(1), (disp_soft_/scale).flip(1), iter=15)
    disp_orig = F.interpolate(disp_ice.flip(1)*scale*grid_sp, size=(H, W, D), mode='trilinear', align_corners=False)

    disp_orig_np = disp_orig.float().cpu().numpy()
    print(f"disp_orig shape: {disp_orig_np.shape}")
    for ch in range(3):
        print(f"  ch{ch}: mean={disp_orig_np[0,ch].mean():+.4f}, std={disp_orig_np[0,ch].std():.4f}, "
              f"absmax={np.abs(disp_orig_np[0,ch]).max():.4f}")

    # Evaluate original
    disp_at_kp_orig = np.zeros((len(fkp), 3))
    for ch in range(3):
        disp_at_kp_orig[:, ch] = map_coordinates(disp_orig_np[0, ch], fkp.T, order=1, mode='nearest')
    warped = fkp + disp_at_kp_orig
    tre_orig = np.linalg.norm(warped - mkp, axis=1).mean()
    print(f"\n  TRE (original): {tre_orig:.3f} mm")
    for ax in range(3):
        corr = np.corrcoef(ideal_disp[:, ax], disp_at_kp_orig[:, ax])[0, 1]
        print(f"  Axis {ax}: ideal={ideal_disp[:, ax].mean():+.3f}, orig={disp_at_kp_orig[:, ax].mean():+.3f}, corr={corr:+.3f}")

    # ==========================================================
    # 3) COMPARE
    # ==========================================================
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    diff = np.abs(disp_ours_np - disp_orig_np)
    print(f"Max absolute difference: {diff.max():.6f}")
    print(f"Mean absolute difference: {diff.mean():.6f}")
    for ch in range(3):
        print(f"  ch{ch}: max_diff={diff[0,ch].max():.6f}, mean_diff={diff[0,ch].mean():.6f}")
    
    if diff.max() < 0.01:
        print("\n✅ Outputs MATCH — our module is identical to original")
        print("   The ~5% improvement is just how MIND performs on this easy pair")
    else:
        print(f"\n❌ Outputs DIFFER — investigating channel mismatch...")
        
        # Try swapped channel comparison
        for perm_name, perm in [
            ("swap ch0↔ch2", [2, 1, 0]),
            ("reverse all", [2, 1, 0]),
        ]:
            diff_perm = np.abs(disp_ours_np[0, perm] - disp_orig_np[0])
            print(f"  With {perm_name}: max_diff={diff_perm.max():.6f}, mean_diff={diff_perm.mean():.6f}")

except ImportError as e:
    print(f"Original code not available: {e}")
    print("Cannot do direct comparison")
except Exception as e:
    print(f"Error running original: {e}")
    import traceback
    traceback.print_exc()
