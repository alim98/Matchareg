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
from pipeline.tests._stage8_common import (
    cleanup_cuda,
    create_backend,
    create_synthetic_case,
    extract_features,
    maybe_import_matplotlib,
    retrieval_metrics,
    sample_descriptors,
    setup_logging,
)


def evaluate_case(config, backend, case_data, downsample, cache_tag, logger, n_points):
    fixed_feats, fixed_feat_shape, fixed_orig_shape = extract_features(
        config, case_data["fixed_img"], case_data["fixed_id"], backend, downsample, f"{cache_tag}_fixed", logger
    )
    moving_feats, moving_feat_shape, moving_orig_shape = extract_features(
        config, case_data["moving_img"], case_data["moving_id"], backend, downsample, f"{cache_tag}_moving", logger
    )

    fkp = case_data["fixed_keypoints"]
    mkp = case_data["moving_keypoints"]
    if fkp is None or mkp is None or len(fkp) == 0:
        raise ValueError(f"{cache_tag}: no keypoints available")

    if len(fkp) > n_points:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(fkp), n_points, replace=False)
        fkp = fkp[idx]
        mkp = mkp[idx]

    fixed_desc = sample_descriptors(fixed_feats, fixed_feat_shape, fixed_orig_shape, fkp, downsample)
    moving_desc = sample_descriptors(moving_feats, moving_feat_shape, moving_orig_shape, mkp, downsample)
    metrics = retrieval_metrics(fixed_desc, moving_desc, max_eval=500, seed=42)

    logger.info(
        f"{cache_tag}: pos={metrics['positive_mean']:.3f} neg={metrics['negative_mean']:.3f} "
        f"sep={metrics['separation']:.3f} nn@1={metrics['nn_accuracy']:.1f}%"
    )

    return metrics


def classify(real_metrics, synthetic_metrics):
    labels = []
    if synthetic_metrics["nn_accuracy"] < 20.0:
        labels.append("feature path broken even on synthetic")
    if synthetic_metrics["nn_accuracy"] >= 50.0 and real_metrics["nn_accuracy"] < 10.0:
        labels.append("domain transfer bottleneck")
    if real_metrics["separation"] < 0.1:
        labels.append("weak feature separation on real data")
    if synthetic_metrics["separation"] > real_metrics["separation"] + 0.1:
        labels.append("feature signal collapses on real CT/CBCT")
    if not labels:
        labels.append("feature stage looks usable")
    return labels


def visualize(viz_dir, backend_name, real_metrics, synthetic_metrics):
    plt = maybe_import_matplotlib()
    if plt is None:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    cases = [("real", real_metrics), ("synthetic", synthetic_metrics)]

    for row, (name, metrics) in enumerate(cases):
        ax = axes[row, 0]
        ax.hist(metrics["positive_all"], bins=50, alpha=0.6, label="positive")
        ax.hist(metrics["negative_all"], bins=50, alpha=0.6, label="negative")
        ax.set_title(f"{name}: similarity")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[row, 1]
        sim = metrics["sim_matrix"]
        n = min(100, sim.shape[0])
        im = ax.imshow(sim[:n, :n], aspect="auto", cmap="viridis")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{name}: sim matrix")

        ax = axes[row, 2]
        ax.hist(metrics["ranks"], bins=50, color="steelblue", alpha=0.8)
        ax.set_title(f"{name}: GT rank")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{backend_name}: real nn@1={real_metrics['nn_accuracy']:.1f}%, "
        f"synthetic nn@1={synthetic_metrics['nn_accuracy']:.1f}%"
    )
    path = viz_dir / f"test_9_{backend_name}_feature_audit.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Test 9: backend feature audit")
    parser.add_argument("--pair", type=int, default=0)
    parser.add_argument("--feature", type=str, default="all", choices=["all", "dinov3", "matcha", "mind"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--synthetic-max-displacement", type=float, default=15.0)
    parser.add_argument("--synthetic-smoothness", type=float, default=20.0)
    parser.add_argument("--n-points", type=int, default=6000)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    logger, _, viz_dir, _ = setup_logging("test_9_backend_feature_audit")
    config = PipelineConfig()
    config.device = args.device
    config.features.use_cache = not args.no_cache

    dataset = ThoraxCBCTDataset(config.paths.data_root, split="train")
    data = dataset[args.pair]
    synthetic_case = create_synthetic_case(
        data,
        args.device,
        max_displacement=args.synthetic_max_displacement,
        smoothness=args.synthetic_smoothness,
    )

    backends = ["dinov3", "matcha", "mind"] if args.feature == "all" else [args.feature]
    all_results = {}

    for feature in backends:
        logger.info("=" * 80)
        logger.info(f"BACKEND {feature}")
        logger.info("=" * 80)
        config.features.backend = feature
        backend = create_backend(config, feature, args.device, args.downsample, args.batch_size)
        real_metrics = evaluate_case(config, backend, data, args.downsample, "real", logger, args.n_points)
        synthetic_metrics = evaluate_case(
            config, backend, synthetic_case, args.downsample, "synthetic", logger, args.n_points
        )
        labels = classify(real_metrics, synthetic_metrics)
        visualize(viz_dir, feature, real_metrics, synthetic_metrics)
        all_results[feature] = {
            "real": real_metrics,
            "synthetic": synthetic_metrics,
            "labels": labels,
        }
        logger.info(f"labels: {', '.join(labels)}")
        del backend
        cleanup_cuda()

    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    for feature, result in all_results.items():
        logger.info(
            f"{feature}: real nn@1={result['real']['nn_accuracy']:.1f}% "
            f"synthetic nn@1={result['synthetic']['nn_accuracy']:.1f}% "
            f"real sep={result['real']['separation']:.3f} "
            f"synthetic sep={result['synthetic']['separation']:.3f}"
        )
        logger.info(f"  labels: {', '.join(result['labels'])}")


if __name__ == "__main__":
    main()
