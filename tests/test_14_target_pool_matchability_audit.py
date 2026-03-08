#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import PipelineConfig
from pipeline.data.dataset_thoraxcbct import ThoraxCBCTDataset
from pipeline.data.preprocessing import generate_trunk_mask
from pipeline.features.triplanar_fuser import load_features
from pipeline.matching.gwot3d import match
from pipeline.matching.sampling import sample_points_in_mask
from pipeline.tests._stage8_common import (
    create_backend,
    extract_features,
    maybe_import_matplotlib,
    sample_descriptors,
    setup_logging,
)


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
            lattice_mask = (
                ((yx[:, 0] - y_min) % step == (step // 2) % step)
                & ((yx[:, 1] - x_min) % step == (step // 2) % step)
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


def evaluate_sampler(config, feature, backend, data, downsample, n_points, n_sources, sampler):
    fixed_mask = generate_trunk_mask(data["fixed_img"])
    moving_mask = generate_trunk_mask(data["moving_img"])

    fixed_generic_cache = config.paths.feature_cache_dir / f"{data['fixed_id']}_{feature}.npz"
    moving_generic_cache = config.paths.feature_cache_dir / f"{data['moving_id']}_{feature}.npz"
    if config.features.use_cache and fixed_generic_cache.exists() and moving_generic_cache.exists():
        fixed_feats, fixed_feat_shape, fixed_orig_shape = load_features(fixed_generic_cache)
        moving_feats, moving_feat_shape, moving_orig_shape = load_features(moving_generic_cache)
    else:
        if backend is None:
            raise RuntimeError("Backend required because cached features are unavailable")
        fixed_feats, fixed_feat_shape, fixed_orig_shape = extract_features(
            config, data["fixed_img"], data["fixed_id"], backend, downsample, "target_pool_fixed", setup_dummy_logger(), fixed_mask
        )
        moving_feats, moving_feat_shape, moving_orig_shape = extract_features(
            config, data["moving_img"], data["moving_id"], backend, downsample, "target_pool_moving", setup_dummy_logger(), moving_mask
        )

    fkp = data["fixed_keypoints"]
    mkp = data["moving_keypoints"]
    idx = np.random.RandomState(0).choice(len(fkp), min(n_sources, len(fkp)), replace=False)
    fkp_src = fkp[idx]
    mkp_gt = mkp[idx]

    moving_candidates = sampler(moving_mask, n_points, np.random.RandomState(123))
    fixed_desc = sample_descriptors(fixed_feats, fixed_feat_shape, fixed_orig_shape, fkp_src, downsample)
    moving_desc = sample_descriptors(moving_feats, moving_feat_shape, moving_orig_shape, moving_candidates, downsample)

    result = match(
        fixed_desc,
        moving_desc,
        fkp_src,
        moving_candidates,
        method="nn",
        max_displacement=config.sampling.max_displacement,
    )
    src_idx = result["matches_src_idx"]
    tgt_idx = result["matches_tgt_idx"]
    errors = np.linalg.norm(moving_candidates[tgt_idx] - mkp_gt[src_idx], axis=1) if len(src_idx) else np.zeros(0)
    availability = np.min(np.linalg.norm(moving_candidates[:, None, :] - mkp_gt[None, :, :], axis=2), axis=0)

    return {
        "n_matches": int(len(src_idx)),
        "target_avail_8": float((availability < 8).mean() * 100.0),
        "target_avail_10": float((availability < 10).mean() * 100.0),
        "target_avail_12": float((availability < 12).mean() * 100.0),
        "pck_8": float((errors < 8).mean() * 100.0) if len(errors) else 0.0,
        "pck_10": float((errors < 10).mean() * 100.0) if len(errors) else 0.0,
        "pck_12": float((errors < 12).mean() * 100.0) if len(errors) else 0.0,
    }


class _DummyLogger:
    def info(self, *_args, **_kwargs):
        return None


def setup_dummy_logger():
    return _DummyLogger()


def visualize(viz_dir, current, legacy):
    plt = maybe_import_matplotlib()
    if plt is None:
        return

    labels = ["avail@8", "avail@10", "avail@12", "pck@8", "pck@10", "pck@12"]
    cur_vals = [
        current["target_avail_8"],
        current["target_avail_10"],
        current["target_avail_12"],
        current["pck_8"],
        current["pck_10"],
        current["pck_12"],
    ]
    old_vals = [
        legacy["target_avail_8"],
        legacy["target_avail_10"],
        legacy["target_avail_12"],
        legacy["pck_8"],
        legacy["pck_10"],
        legacy["pck_12"],
    ]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, cur_vals, width=width, label="current")
    ax.bar(x + width / 2, old_vals, width=width, label="legacy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    path = viz_dir / "test_14_target_pool_matchability_audit.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Test 14: target pool matchability audit")
    parser.add_argument("--pair", type=int, default=0)
    parser.add_argument("--feature", type=str, default="matcha", choices=["dinov3", "matcha", "mind"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-points", type=int, default=4000)
    parser.add_argument("--n-sources", type=int, default=200)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    logger, _, viz_dir, _ = setup_logging("test_14_target_pool_matchability_audit")
    config = PipelineConfig()
    device = args.device
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            logger.info("CUDA unavailable, falling back to cpu")
            device = "cpu"
    config.device = device
    config.features.backend = args.feature
    config.features.use_cache = not args.no_cache

    dataset = ThoraxCBCTDataset(config.paths.data_root, split="train")
    data = dataset[args.pair]
    if data["fixed_keypoints"] is None or data["moving_keypoints"] is None or len(data["fixed_keypoints"]) == 0:
        raise ValueError("No keypoints available for this pair")

    fixed_generic_cache = config.paths.feature_cache_dir / f"{data['fixed_id']}_{args.feature}.npz"
    moving_generic_cache = config.paths.feature_cache_dir / f"{data['moving_id']}_{args.feature}.npz"
    backend = None
    if not (config.features.use_cache and fixed_generic_cache.exists() and moving_generic_cache.exists()):
        backend = create_backend(config, args.feature, device, args.downsample, args.batch_size)
    logger.info("Evaluating current sampler")
    current = evaluate_sampler(
        config,
        args.feature,
        backend,
        data,
        args.downsample,
        args.n_points,
        args.n_sources,
        lambda mask, n_points, rng: sample_points_in_mask(mask, n_points, z_stratified=True, rng=rng),
    )
    logger.info("Evaluating legacy sampler")
    legacy = evaluate_sampler(
        config,
        args.feature,
        backend,
        data,
        args.downsample,
        args.n_points,
        args.n_sources,
        legacy_sample,
    )

    logger.info(
        f"current: matches={current['n_matches']} avail@10={current['target_avail_10']:.1f}% "
        f"avail@12={current['target_avail_12']:.1f}% pck@10={current['pck_10']:.1f}% pck@12={current['pck_12']:.1f}%"
    )
    logger.info(
        f"legacy:  matches={legacy['n_matches']} avail@10={legacy['target_avail_10']:.1f}% "
        f"avail@12={legacy['target_avail_12']:.1f}% pck@10={legacy['pck_10']:.1f}% pck@12={legacy['pck_12']:.1f}%"
    )

    visualize(viz_dir, current, legacy)

    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(
        f"current matches={current['n_matches']} pck@10={current['pck_10']:.1f}% pck@12={current['pck_12']:.1f}% "
        f"| legacy matches={legacy['n_matches']} pck@10={legacy['pck_10']:.1f}% pck@12={legacy['pck_12']:.1f}%"
    )


if __name__ == "__main__":
    main()
