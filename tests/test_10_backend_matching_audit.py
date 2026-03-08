#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import PipelineConfig
from pipeline.data.dataset_thoraxcbct import ThoraxCBCTDataset
from pipeline.data.preprocessing import generate_trunk_mask
from pipeline.matching.sampling import include_keypoints, sample_points_in_mask
from pipeline.tests._stage8_common import (
    cleanup_cuda,
    compute_candidate_pck,
    compute_index_pck,
    create_backend,
    create_synthetic_case,
    extract_features,
    maybe_import_matplotlib,
    run_matching,
    sample_descriptors,
    setup_logging,
)


def evaluate_matching_case(config, backend, case_data, downsample, cache_tag, logger):
    fixed_mask = generate_trunk_mask(case_data["fixed_img"])
    moving_mask = generate_trunk_mask(case_data["moving_img"])

    fixed_feats, fixed_feat_shape, fixed_orig_shape = extract_features(
        config, case_data["fixed_img"], case_data["fixed_id"], backend, downsample, f"{cache_tag}_fixed", logger, fixed_mask
    )
    moving_feats, moving_feat_shape, moving_orig_shape = extract_features(
        config, case_data["moving_img"], case_data["moving_id"], backend, downsample, f"{cache_tag}_moving", logger, moving_mask
    )

    fkp = case_data["fixed_keypoints"]
    mkp = case_data["moving_keypoints"]
    if fkp is None or mkp is None or len(fkp) == 0:
        raise ValueError(f"{cache_tag}: no keypoints available")

    rng = np.random.RandomState(42)
    idx = rng.choice(len(fkp), min(2000, len(fkp)), replace=False)
    fkp_sub = fkp[idx]
    mkp_sub = mkp[idx]

    fixed_desc = sample_descriptors(fixed_feats, fixed_feat_shape, fixed_orig_shape, fkp_sub, downsample)
    moving_desc = sample_descriptors(moving_feats, moving_feat_shape, moving_orig_shape, mkp_sub, downsample)

    gt_results = {}
    for method in ("nn", "ot", "gwot"):
        t0 = time.time()
        result = run_matching(config, method, moving_desc, fixed_desc, mkp_sub, fkp_sub)
        elapsed = time.time() - t0
        pck, errors = compute_index_pck(fkp_sub, result["matches_src_idx"], result["matches_tgt_idx"])
        gt_results[method] = {
            "n_matches": int(len(result["matches_src_idx"])),
            "pck": pck,
            "mean_error_mm": float(errors.mean()) if len(errors) else 0.0,
            "runtime_s": elapsed,
        }

    src_idx = rng.choice(len(fkp), min(500, len(fkp)), replace=False)
    fkp_src = fkp[src_idx]
    mkp_gt = mkp[src_idx]
    moving_random = sample_points_in_mask(moving_mask, 2000, z_stratified=True, rng=np.random.RandomState(123))
    moving_candidates = include_keypoints(moving_random, mkp_gt, mask=moving_mask)

    fixed_desc = sample_descriptors(fixed_feats, fixed_feat_shape, fixed_orig_shape, fkp_src, downsample)
    moving_desc = sample_descriptors(moving_feats, moving_feat_shape, moving_orig_shape, moving_candidates, downsample)

    distractor_results = {}
    for method in ("nn", "gwot"):
        t0 = time.time()
        result = run_matching(config, method, fixed_desc, moving_desc, fkp_src, moving_candidates)
        elapsed = time.time() - t0
        pck, errors = compute_candidate_pck(moving_candidates, mkp_gt, result["matches_src_idx"], result["matches_tgt_idx"])
        distractor_results[method] = {
            "n_matches": int(len(result["matches_src_idx"])),
            "pck": pck,
            "mean_error_mm": float(errors.mean()) if len(errors) else 0.0,
            "runtime_s": elapsed,
        }

    logger.info(
        f"{cache_tag}: gt(nn/ot/gwot)={gt_results['nn']['pck']['pck@10mm']:.1f}/"
        f"{gt_results['ot']['pck']['pck@10mm']:.1f}/{gt_results['gwot']['pck']['pck@10mm']:.1f} "
        f"distractor(nn/gwot)={distractor_results['nn']['pck']['pck@10mm']:.1f}/"
        f"{distractor_results['gwot']['pck']['pck@10mm']:.1f}"
    )

    return {
        "gt": gt_results,
        "distractor": distractor_results,
    }


def classify(result_real, result_synth):
    labels = []
    real_d2_nn = result_real["distractor"]["nn"]["pck"]["pck@10mm"]
    real_d2_gwot = result_real["distractor"]["gwot"]["pck"]["pck@10mm"]
    synth_d2_nn = result_synth["distractor"]["nn"]["pck"]["pck@10mm"]
    synth_d2_gwot = result_synth["distractor"]["gwot"]["pck"]["pck@10mm"]

    if real_d2_gwot + 5.0 < real_d2_nn:
        labels.append("gwot hurts realistic matching")
    if synth_d2_nn >= 70.0 and real_d2_nn < 40.0:
        labels.append("matching degrades on real distractors")
    if result_real["gt"]["nn"]["pck"]["pck@10mm"] < 40.0:
        labels.append("nn weak even at gt-aligned points")
    if synth_d2_gwot + 5.0 < synth_d2_nn:
        labels.append("gwot extraction/objective weaker than nn")
    if not labels:
        labels.append("matching stage looks usable")
    return labels


def visualize(viz_dir, feature, real_result, synth_result):
    plt = maybe_import_matplotlib()
    if plt is None:
        return

    thresholds = [2, 5, 10, 20, 50]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    cases = [("real", real_result), ("synthetic", synth_result)]

    for row, (name, result) in enumerate(cases):
        ax = axes[row, 0]
        for method in ("nn", "ot", "gwot"):
            p = [result["gt"][method]["pck"][f"pck@{t}mm"] for t in thresholds]
            ax.plot(thresholds, p, marker="o", label=method)
        ax.set_title(f"{name}: gt matching")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[row, 1]
        for method in ("nn", "gwot"):
            p = [result["distractor"][method]["pck"][f"pck@{t}mm"] for t in thresholds]
            ax.plot(thresholds, p, marker="o", label=method)
        ax.set_title(f"{name}: distractor matching")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend()

    path = viz_dir / f"test_10_{feature}_matching_audit.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Test 10: backend matching audit")
    parser.add_argument("--pair", type=int, default=0)
    parser.add_argument("--feature", type=str, default="all", choices=["all", "dinov3", "matcha", "mind"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--synthetic-max-displacement", type=float, default=15.0)
    parser.add_argument("--synthetic-smoothness", type=float, default=20.0)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    logger, _, viz_dir, _ = setup_logging("test_10_backend_matching_audit")
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
        real_result = evaluate_matching_case(config, backend, data, args.downsample, "real", logger)
        synth_result = evaluate_matching_case(config, backend, synthetic_case, args.downsample, "synthetic", logger)
        labels = classify(real_result, synth_result)
        visualize(viz_dir, feature, real_result, synth_result)
        all_results[feature] = {
            "real": real_result,
            "synthetic": synth_result,
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
            f"{feature}: real D2 nn={result['real']['distractor']['nn']['pck']['pck@10mm']:.1f}% "
            f"gwot={result['real']['distractor']['gwot']['pck']['pck@10mm']:.1f}% "
            f"synthetic D2 nn={result['synthetic']['distractor']['nn']['pck']['pck@10mm']:.1f}% "
            f"gwot={result['synthetic']['distractor']['gwot']['pck']['pck@10mm']:.1f}%"
        )
        logger.info(f"  labels: {', '.join(result['labels'])}")


if __name__ == "__main__":
    main()
