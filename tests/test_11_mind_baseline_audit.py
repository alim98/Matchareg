#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import map_coordinates

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DINOREG_ROOT = Path("/u/almik/feb25/dinoreg/DINO-Reg")
if str(DINOREG_ROOT) not in sys.path:
    sys.path.insert(0, str(DINOREG_ROOT))

from pipeline.config import PipelineConfig
from pipeline.data.dataset_thoraxcbct import ThoraxCBCTDataset
from pipeline.tests._stage8_common import cleanup_cuda, create_synthetic_case, maybe_import_matplotlib, setup_logging
from pipeline.transform.mind_convex_adam import mind_convex_adam


def evaluate_tre(displacement, fixed_kp, moving_kp):
    disp_np = displacement.detach().cpu().numpy()
    sampled = np.zeros_like(fixed_kp)
    for ax in range(3):
        sampled[:, ax] = map_coordinates(disp_np[0, ax], fixed_kp.T, order=1, mode="nearest")
    warped = fixed_kp + sampled
    tre = np.linalg.norm(warped - moving_kp, axis=1)
    return {
        "tre_mean": float(tre.mean()),
        "tre_median": float(np.median(tre)),
        "tre": tre,
        "sampled_disp": sampled,
    }


def run_original(fixed_img, moving_img, device):
    from convex_adam_utils import MINDSSC as orig_MINDSSC
    from convex_adam_utils import correlate as orig_correlate
    from convex_adam_utils import coupled_convex as orig_coupled_convex
    from convex_adam_utils import inverse_consistency as orig_ic

    H, W, D = fixed_img.shape
    grid_sp = 4
    disp_hw = 4

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
        (1, 1, disp_hw * 2 + 1, disp_hw * 2 + 1, disp_hw * 2 + 1),
        align_corners=True,
    ).permute(0, 4, 1, 2, 3).reshape(3, -1, 1)
    disp_soft = orig_coupled_convex(ssd, ssd_argmin, disp_mesh_t, grid_sp, (H, W, D))

    scale = torch.tensor([H // grid_sp - 1, W // grid_sp - 1, D // grid_sp - 1]).view(1, 3, 1, 1, 1).to(device).half() / 2
    ssd2, ssd_argmin2 = orig_correlate(features_mov_smooth, features_fix_smooth, disp_hw, grid_sp, (H, W, D), n_ch)
    disp_soft2 = orig_coupled_convex(ssd2, ssd_argmin2, disp_mesh_t, grid_sp, (H, W, D))
    disp_ice, _ = orig_ic((disp_soft / scale).flip(1), (disp_soft2 / scale).flip(1), iter=15)
    disp_orig = F.interpolate(disp_ice.flip(1) * scale * grid_sp, size=(H, W, D), mode="trilinear", align_corners=False)
    return disp_orig.float()


def run_case(case_name, fixed_img, moving_img, fixed_kp, moving_kp, device, logger):
    t0 = time.time()
    disp_ours = mind_convex_adam(
        fixed_img=fixed_img,
        moving_img=moving_img,
        mind_r=1,
        mind_d=2,
        lambda_weight=0,
        grid_sp=4,
        disp_hw=4,
        n_iter_adam=0,
        grid_sp_adam=2,
        ic=True,
        device=device,
    )
    ours_time = time.time() - t0
    ours_eval = evaluate_tre(disp_ours, fixed_kp, moving_kp)

    orig_disp = None
    orig_eval = None
    diff_stats = None
    try:
        t0 = time.time()
        orig_disp = run_original(fixed_img, moving_img, device)
        orig_time = time.time() - t0
        orig_eval = evaluate_tre(orig_disp, fixed_kp, moving_kp)
        diff = np.abs(disp_ours.detach().cpu().numpy() - orig_disp.detach().cpu().numpy())
        diff_stats = {
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "runtime_s": orig_time,
        }
    except Exception as exc:
        diff_stats = {"error": str(exc)}

    orig_tre = orig_eval["tre_mean"] if orig_eval is not None else float("nan")
    logger.info(f"{case_name}: ours_tre={ours_eval['tre_mean']:.3f} orig_tre={orig_tre:.3f}")

    return {
        "ours_disp": disp_ours,
        "ours_eval": ours_eval,
        "ours_runtime_s": ours_time,
        "orig_disp": orig_disp,
        "orig_eval": orig_eval,
        "diff_stats": diff_stats,
    }


def visualize(viz_dir, name, result):
    plt = maybe_import_matplotlib()
    if plt is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ours = result["ours_eval"]["sampled_disp"]
    ideal = None
    if result["orig_eval"] is not None:
        ideal = result["orig_eval"]["sampled_disp"]

    ax = axes[0]
    ax.hist(result["ours_eval"]["tre"], bins=50, alpha=0.7, label="ours")
    if result["orig_eval"] is not None:
        ax.hist(result["orig_eval"]["tre"], bins=50, alpha=0.5, label="original")
    ax.set_title("TRE distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(ours[:, 0], ours[:, 2], s=2, alpha=0.2)
    ax.set_title("ours disp at keypoints")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if ideal is not None:
        ax.scatter(ideal[:, 2], ours[:, 2], s=2, alpha=0.2)
        lo = min(ideal[:, 2].min(), ours[:, 2].min())
        hi = max(ideal[:, 2].max(), ours[:, 2].max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_title("original vs ours (x-axis disp)")
        ax.grid(True, alpha=0.3)
    else:
        ax.axis("off")

    path = viz_dir / f"test_11_{name}_mind_audit.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Test 11: MIND baseline audit")
    parser.add_argument("--pair", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--synthetic-max-displacement", type=float, default=15.0)
    parser.add_argument("--synthetic-smoothness", type=float, default=20.0)
    args = parser.parse_args()

    logger, _, viz_dir, _ = setup_logging("test_11_mind_baseline_audit")
    config = PipelineConfig()
    config.device = args.device
    dataset = ThoraxCBCTDataset(config.paths.data_root, split="train")
    data = dataset[args.pair]
    if data["fixed_keypoints"] is None or data["moving_keypoints"] is None or len(data["fixed_keypoints"]) == 0:
        raise ValueError("No keypoints available for this pair")

    real = run_case(
        "real",
        data["fixed_img"],
        data["moving_img"],
        data["fixed_keypoints"],
        data["moving_keypoints"],
        args.device,
        logger,
    )
    visualize(viz_dir, "real", real)
    cleanup_cuda()

    synthetic = create_synthetic_case(
        data,
        args.device,
        max_displacement=args.synthetic_max_displacement,
        smoothness=args.synthetic_smoothness,
    )
    synth = run_case(
        "synthetic",
        synthetic["fixed_img"],
        synthetic["moving_img"],
        synthetic["fixed_keypoints"],
        synthetic["moving_keypoints"],
        args.device,
        logger,
    )
    visualize(viz_dir, "synthetic", synth)
    cleanup_cuda()

    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    for name, result in (("real", real), ("synthetic", synth)):
        line = f"{name}: ours_tre={result['ours_eval']['tre_mean']:.3f} runtime={result['ours_runtime_s']:.1f}s"
        if result["orig_eval"] is not None:
            line += (
                f" orig_tre={result['orig_eval']['tre_mean']:.3f} "
                f"field_diff_max={result['diff_stats']['max_abs_diff']:.6f} "
                f"field_diff_mean={result['diff_stats']['mean_abs_diff']:.6f}"
            )
        else:
            line += f" original=ERROR({result['diff_stats']['error']})"
        logger.info(line)


if __name__ == "__main__":
    main()
