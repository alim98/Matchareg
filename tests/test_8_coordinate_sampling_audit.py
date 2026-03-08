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
from pipeline.matching.sampling import sample_descriptors_at_points, sample_points_in_mask
from pipeline.tests._stage8_common import (
    cleanup_cuda,
    create_backend,
    extract_features,
    maybe_import_matplotlib,
    retrieval_metrics,
    sample_descriptors,
    setup_logging,
    voxel_to_feature_coords,
)


def synthetic_sampling_error():
    grid = np.zeros((3, 12, 11, 10), dtype=np.float32)
    zz, yy, xx = np.meshgrid(
        np.arange(12, dtype=np.float32),
        np.arange(11, dtype=np.float32),
        np.arange(10, dtype=np.float32),
        indexing="ij",
    )
    grid[0] = zz
    grid[1] = yy
    grid[2] = xx

    points = np.array(
        [
            [1.25, 2.50, 3.75],
            [7.10, 5.20, 4.80],
            [10.40, 9.10, 8.60],
            [0.00, 0.00, 0.00],
            [11.00, 10.00, 9.00],
        ],
        dtype=np.float64,
    )
    sampled = sample_descriptors_at_points(grid, points)
    expected = points.astype(np.float32)
    err = np.abs(sampled - expected)
    return {
        "points": points,
        "sampled": sampled,
        "expected": expected,
        "max_abs_err": float(err.max()),
        "mean_abs_err": float(err.mean()),
    }


def mapping_consistency(points, original_shape, feature_shape, downsample):
    ours = voxel_to_feature_coords(points, original_shape, feature_shape, downsample)
    prod = points.astype(np.float64).copy()
    D, H, W = original_shape
    fD, fH, fW = feature_shape
    if (D, H, W) == (fD, fH, fW):
        prod[:, 0] = np.clip(prod[:, 0], 0, fD - 1)
        prod[:, 1] = np.clip(prod[:, 1], 0, fH - 1)
        prod[:, 2] = np.clip(prod[:, 2], 0, fW - 1)
    else:
        prod[:, 0] = prod[:, 0] / downsample * fD / (D // downsample)
        prod[:, 1] = prod[:, 1] / downsample * fH / (H // downsample)
        prod[:, 2] = prod[:, 2] / downsample * fW / (W // downsample)
        prod[:, 0] = np.clip(prod[:, 0], 0, fD - 1)
        prod[:, 1] = np.clip(prod[:, 1], 0, fH - 1)
        prod[:, 2] = np.clip(prod[:, 2], 0, fW - 1)
    diff = np.abs(ours - prod)
    return {
        "ours": ours,
        "prod": prod,
        "max_abs_err": float(diff.max()),
        "mean_abs_err": float(diff.mean()),
    }


def backend_identity_audit(config, dataset, pair_idx, feature, downsample, batch_size, logger):
    backend = create_backend(config, feature, config.device, downsample, batch_size)
    data = dataset[pair_idx]
    fixed_img = data["fixed_img"]
    fixed_id = data["fixed_id"]
    fixed_mask = (fixed_img > -2000).astype(np.uint8)

    feats, feat_shape, orig_shape = extract_features(
        config,
        fixed_img,
        fixed_id,
        backend,
        downsample,
        "coord_identity",
        logger,
        mask=fixed_mask,
    )

    rng = np.random.RandomState(7)
    points = sample_points_in_mask(fixed_mask, 500, z_stratified=True, rng=rng)
    desc = sample_descriptors(feats, feat_shape, orig_shape, points, downsample)
    metrics = retrieval_metrics(desc, desc, max_eval=300, seed=13)

    jitter = points.copy()
    jitter[:, 0] = np.clip(jitter[:, 0] + 2.0, 0, orig_shape[0] - 1)
    jitter_desc = sample_descriptors(feats, feat_shape, orig_shape, jitter, downsample)
    pointwise_same = np.sum(desc * desc, axis=1)
    pointwise_shift = np.sum(desc * jitter_desc, axis=1)

    mapping = mapping_consistency(points, orig_shape, feat_shape, downsample)
    logger.info(
        f"{feature}: identity nn@1={metrics['nn_accuracy']:.1f}% "
        f"same_mean={pointwise_same.mean():.3f} shifted_mean={pointwise_shift.mean():.3f} "
        f"mapping_max_err={mapping['max_abs_err']:.6f}"
    )

    return {
        "feature_shape": feat_shape,
        "metrics": metrics,
        "same_sims": pointwise_same,
        "shift_sims": pointwise_shift,
        "mapping": mapping,
    }


def visualize(viz_dir, results, synthetic_error):
    plt = maybe_import_matplotlib()
    if plt is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax = axes[0, 0]
    labels = list(results.keys())
    nn = [results[k]["metrics"]["nn_accuracy"] for k in labels]
    ax.bar(labels, nn, color=["steelblue", "darkorange", "seagreen"][: len(labels)])
    ax.set_ylim(0, 105)
    ax.set_title("Identity Retrieval")
    ax.set_ylabel("Top-1 (%)")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[0, 1]
    same = [results[k]["same_sims"].mean() for k in labels]
    shift = [results[k]["shift_sims"].mean() for k in labels]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, same, width, label="same-point")
    ax.bar(x + width / 2, shift, width, label="shifted +2z")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Point Similarity")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 0]
    for label in labels:
        ranks = results[label]["metrics"]["ranks"]
        ax.hist(ranks, bins=40, alpha=0.5, label=label)
    ax.set_title("Identity Rank Distribution")
    ax.set_xlabel("GT rank")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    sampled = synthetic_error["sampled"]
    expected = synthetic_error["expected"]
    for i, name in enumerate(["z", "y", "x"]):
        ax.scatter(expected[:, i], sampled[:, i], label=name, s=40)
    lo = min(expected.min(), sampled.min())
    hi = max(expected.max(), sampled.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_title(
        f"Synthetic interpolation\nmean_err={synthetic_error['mean_abs_err']:.4f}, "
        f"max_err={synthetic_error['max_abs_err']:.4f}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = viz_dir / "test_8_coordinate_sampling_summary.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Test 8: coordinate and sampling audit")
    parser.add_argument("--pair", type=int, default=0)
    parser.add_argument("--feature", type=str, default="all", choices=["all", "dinov3", "matcha", "mind"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    logger, _, viz_dir, _ = setup_logging("test_8_coordinate_sampling")
    config = PipelineConfig()
    config.device = args.device
    config.features.use_cache = not args.no_cache

    dataset = ThoraxCBCTDataset(config.paths.data_root, split="train")
    backends = ["dinov3", "matcha", "mind"] if args.feature == "all" else [args.feature]

    synthetic_error = synthetic_sampling_error()
    logger.info(
        f"synthetic sampling: mean_abs_err={synthetic_error['mean_abs_err']:.6f}, "
        f"max_abs_err={synthetic_error['max_abs_err']:.6f}"
    )

    results = {}
    for feature in backends:
        config.features.backend = feature
        results[feature] = backend_identity_audit(
            config, dataset, args.pair, feature, args.downsample, args.batch_size, logger
        )
        cleanup_cuda()

    visualize(viz_dir, results, synthetic_error)

    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    for feature, result in results.items():
        logger.info(
            f"{feature}: nn@1={result['metrics']['nn_accuracy']:.1f}% "
            f"same={result['same_sims'].mean():.3f} "
            f"shifted={result['shift_sims'].mean():.3f} "
            f"mapping_err={result['mapping']['max_abs_err']:.6f}"
        )


if __name__ == "__main__":
    main()
