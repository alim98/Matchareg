#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import PipelineConfig
from pipeline.data.dataset_thoraxcbct import ThoraxCBCTDataset
from pipeline.data.preprocessing import generate_trunk_mask
from pipeline.matching.sampling import sample_points_in_mask
from pipeline.tests._stage8_common import create_synthetic_case, maybe_import_matplotlib, setup_logging


def legacy_sample(mask, n_points, rng):
    D = mask.shape[0]
    slice_counts = np.array([np.sum(mask[z] > 0) for z in range(D)])
    total_fg = slice_counts.sum()
    if total_fg == 0:
        raise ValueError("Mask is empty")
    points_per_slice = np.round(slice_counts / total_fg * n_points).astype(int)
    diff = n_points - points_per_slice.sum()
    if diff > 0:
        top = np.argsort(-slice_counts)
        for i in range(diff):
            points_per_slice[top[i % D]] += 1
    elif diff < 0:
        top = np.argsort(-points_per_slice)
        for i in range(-diff):
            if points_per_slice[top[i % D]] > 0:
                points_per_slice[top[i % D]] -= 1

    all_points = []
    for z in range(D):
        n_z = points_per_slice[z]
        if n_z <= 0:
            continue
        yx = np.argwhere(mask[z] > 0)
        if len(yx) == 0:
            continue
        if len(yx) <= n_z:
            chosen = yx
        else:
            y_min, x_min = yx.min(axis=0)
            area = max(len(yx), 1)
            step = max(1, int(np.sqrt(area / max(n_z, 1))))
            y_mod = step // 2
            x_mod = step // 2
            lattice_mask = (
                ((yx[:, 0] - y_min) % step == y_mod % step)
                & ((yx[:, 1] - x_min) % step == x_mod % step)
            )
            lattice = yx[lattice_mask]
            if len(lattice) >= n_z:
                idx = np.linspace(0, len(lattice) - 1, n_z, dtype=int)
                chosen = lattice[idx]
            else:
                remaining = yx[~lattice_mask]
                need = n_z - len(lattice)
                extra_idx = rng.choice(len(remaining), need, replace=False)
                chosen = np.vstack([lattice, remaining[extra_idx]])
        z_col = np.full((len(chosen), 1), z)
        all_points.append(np.hstack([z_col, chosen]))

    return np.vstack(all_points).astype(np.float64)


def coverage_metrics(sampled_fixed, sampled_moving, fixed_kp, moving_kp, eval_n):
    rng = np.random.RandomState(0)
    idx = rng.choice(len(fixed_kp), min(eval_n, len(fixed_kp)), replace=False)
    fkp = fixed_kp[idx]
    mkp = moving_kp[idx]

    tree_fixed = cKDTree(sampled_fixed)
    tree_moving = cKDTree(sampled_moving)
    dist_fixed, _ = tree_fixed.query(fkp, k=1)
    dist_moving, _ = tree_moving.query(mkp, k=1)

    thresholds = [2, 4, 6, 8, 10, 12, 16]
    joint = {t: float(((dist_fixed <= t) & (dist_moving <= t)).mean() * 100.0) for t in thresholds}
    single_fixed = {t: float((dist_fixed <= t).mean() * 100.0) for t in thresholds}
    single_moving = {t: float((dist_moving <= t).mean() * 100.0) for t in thresholds}

    return {
        "thresholds": thresholds,
        "joint": joint,
        "fixed": single_fixed,
        "moving": single_moving,
        "fixed_mean": float(dist_fixed.mean()),
        "moving_mean": float(dist_moving.mean()),
        "fixed_median": float(np.median(dist_fixed)),
        "moving_median": float(np.median(dist_moving)),
    }


def evaluate_case(case_name, fixed_img, moving_img, fixed_kp, moving_kp, n_points, eval_n, logger):
    fixed_mask = generate_trunk_mask(fixed_img)
    moving_mask = generate_trunk_mask(moving_img)

    current_fixed = sample_points_in_mask(fixed_mask, n_points, z_stratified=True, rng=np.random.RandomState(42))
    current_moving = sample_points_in_mask(moving_mask, n_points, z_stratified=True, rng=np.random.RandomState(123))
    legacy_fixed = legacy_sample(fixed_mask, n_points, np.random.RandomState(42))
    legacy_moving = legacy_sample(moving_mask, n_points, np.random.RandomState(123))

    current = coverage_metrics(current_fixed, current_moving, fixed_kp, moving_kp, eval_n)
    legacy = coverage_metrics(legacy_fixed, legacy_moving, fixed_kp, moving_kp, eval_n)

    logger.info(
        f"{case_name} current fixed_mean={current['fixed_mean']:.2f} moving_mean={current['moving_mean']:.2f} "
        f"joint@8={current['joint'][8]:.1f}% joint@10={current['joint'][10]:.1f}% joint@12={current['joint'][12]:.1f}%"
    )
    logger.info(
        f"{case_name} legacy  fixed_mean={legacy['fixed_mean']:.2f} moving_mean={legacy['moving_mean']:.2f} "
        f"joint@8={legacy['joint'][8]:.1f}% joint@10={legacy['joint'][10]:.1f}% joint@12={legacy['joint'][12]:.1f}%"
    )

    return {"current": current, "legacy": legacy}


def visualize(viz_dir, results):
    plt = maybe_import_matplotlib()
    if plt is None:
        return

    thresholds = results["real"]["current"]["thresholds"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, case_name in zip(axes, ("real", "synthetic")):
        cur = [results[case_name]["current"]["joint"][t] for t in thresholds]
        old = [results[case_name]["legacy"]["joint"][t] for t in thresholds]
        ax.plot(thresholds, cur, marker="o", label="current")
        ax.plot(thresholds, old, marker="o", label="legacy")
        ax.set_title(f"{case_name} joint coverage")
        ax.set_xlabel("distance threshold (mm)")
        ax.set_ylabel("GT pair coverage (%)")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend()

    path = viz_dir / "test_13_sampling_recall_audit.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Test 13: sampling recall audit")
    parser.add_argument("--pair", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-points", type=int, default=8000)
    parser.add_argument("--eval-n", type=int, default=4000)
    parser.add_argument("--synthetic-max-displacement", type=float, default=15.0)
    parser.add_argument("--synthetic-smoothness", type=float, default=20.0)
    args = parser.parse_args()

    logger, _, viz_dir, _ = setup_logging("test_13_sampling_recall_audit")
    config = PipelineConfig()
    device = args.device
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            logger.info("CUDA unavailable, falling back to cpu")
            device = "cpu"
    dataset = ThoraxCBCTDataset(config.paths.data_root, split="train")
    data = dataset[args.pair]
    if data["fixed_keypoints"] is None or data["moving_keypoints"] is None or len(data["fixed_keypoints"]) == 0:
        raise ValueError("No keypoints available for this pair")

    synthetic = create_synthetic_case(
        data,
        device,
        max_displacement=args.synthetic_max_displacement,
        smoothness=args.synthetic_smoothness,
    )

    real = evaluate_case(
        "real",
        data["fixed_img"],
        data["moving_img"],
        data["fixed_keypoints"],
        data["moving_keypoints"],
        args.n_points,
        args.eval_n,
        logger,
    )
    synthetic_result = evaluate_case(
        "synthetic",
        synthetic["fixed_img"],
        synthetic["moving_img"],
        synthetic["fixed_keypoints"],
        synthetic["moving_keypoints"],
        args.n_points,
        args.eval_n,
        logger,
    )

    results = {"real": real, "synthetic": synthetic_result}
    visualize(viz_dir, results)

    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    for case_name in ("real", "synthetic"):
        cur = results[case_name]["current"]
        old = results[case_name]["legacy"]
        logger.info(
            f"{case_name}: current joint@8={cur['joint'][8]:.1f}% joint@10={cur['joint'][10]:.1f}% "
            f"joint@12={cur['joint'][12]:.1f}% | legacy joint@8={old['joint'][8]:.1f}% "
            f"joint@10={old['joint'][10]:.1f}% joint@12={old['joint'][12]:.1f}%"
        )


if __name__ == "__main__":
    main()
