#!/usr/bin/env python3
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import PipelineConfig
from pipeline.data.dataset_thoraxcbct import ThoraxCBCTDataset
from pipeline.data.preprocessing import generate_trunk_mask, robust_intensity_normalize
from pipeline.features.triplanar_fuser import TriplanarFuser, load_features, save_features
from pipeline.matching.gwot3d import match
from pipeline.matching.sampling import include_keypoints, sample_descriptors_at_points, sample_points_in_mask
from pipeline.transform.integrate import scaling_and_squaring
from pipeline.transform.warp import warp_points, warp_volume

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def create_backend(config, feature, device, downsample, batch_size):
    if feature == "dinov3":
        from pipeline.features.dinov3_extractor import DINOv3Extractor

        extractor = DINOv3Extractor(
            repo_path=str(config.paths.dinov3_repo),
            weights_path=str(config.paths.dinov3_weights),
            device=device,
        )
        return TriplanarFuser(
            extractor,
            batch_size=batch_size,
            fusion=config.features.fusion_method,
            downsample=downsample,
            device=device,
        )
    if feature == "matcha":
        from pipeline.features.matcha_extractor import MATCHAExtractor

        extractor = MATCHAExtractor(
            repo_path=str(config.paths.matcha_repo),
            weights_path=str(config.paths.matcha_weights),
            device=device,
        )
        return TriplanarFuser(
            extractor,
            batch_size=batch_size,
            fusion=config.features.fusion_method,
            downsample=downsample,
            device=device,
        )
    if feature == "mind":
        from pipeline.features.mind_extractor import MINDExtractor

        return MINDExtractor(device=device)
    raise ValueError(feature)


def voxel_to_feature_coords(points, original_shape, feature_shape, downsample):
    D, H, W = original_shape
    fD, fH, fW = feature_shape

    feat_points = points.astype(np.float64).copy()
    if (D, H, W) == (fD, fH, fW):
        feat_points[:, 0] = np.clip(feat_points[:, 0], 0, fD - 1)
        feat_points[:, 1] = np.clip(feat_points[:, 1], 0, fH - 1)
        feat_points[:, 2] = np.clip(feat_points[:, 2], 0, fW - 1)
        return feat_points

    feat_points[:, 0] /= downsample
    feat_points[:, 1] /= downsample
    feat_points[:, 2] /= downsample

    dD = D // downsample
    dH = H // downsample
    dW = W // downsample

    feat_points[:, 0] = feat_points[:, 0] * fD / dD
    feat_points[:, 1] = feat_points[:, 1] * fH / dH
    feat_points[:, 2] = feat_points[:, 2] * fW / dW

    feat_points[:, 0] = np.clip(feat_points[:, 0], 0, fD - 1)
    feat_points[:, 1] = np.clip(feat_points[:, 1], 0, fH - 1)
    feat_points[:, 2] = np.clip(feat_points[:, 2], 0, fW - 1)
    return feat_points


def extract_features(config, volume, case_id, backend, downsample, cache_suffix, mask=None):
    cache_name = f"{case_id}_{config.features.backend}_{cache_suffix}_ds{downsample}.npz"
    cache_path = config.paths.feature_cache_dir / cache_name

    if config.features.use_cache and cache_path.exists():
        logger.info(f"  Loading cached features: {cache_path}")
        return load_features(cache_path)

    if mask is None:
        mask = generate_trunk_mask(volume)

    if hasattr(backend, "extract_volume_features"):
        result = backend.extract_volume_features(volume.astype(np.float32))
    else:
        vol_norm = robust_intensity_normalize(volume, mask=mask)
        result = backend.fuse_triplanar(vol_norm)

    if config.features.use_cache:
        save_features(result, cache_path)
    return result


def l2_normalize(desc):
    desc = desc.astype(np.float32, copy=False)
    return desc / (np.linalg.norm(desc, axis=1, keepdims=True) + 1e-8)


def descriptor_set(feats, feat_shape, orig_shape, points, downsample):
    feat_points = voxel_to_feature_coords(points, orig_shape, feat_shape, downsample)
    return l2_normalize(sample_descriptors_at_points(feats, feat_points))


def compute_retrieval_metrics(fixed_desc, moving_desc, max_eval=500, seed=42):
    n = min(len(fixed_desc), len(moving_desc))
    if n == 0:
        return {
            "positive_mean": 0.0,
            "negative_mean": 0.0,
            "separation": 0.0,
            "nn_accuracy": 0.0,
            "n_eval": 0,
        }

    fixed_desc = fixed_desc[:n]
    moving_desc = moving_desc[:n]
    positive = np.sum(fixed_desc * moving_desc, axis=1)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    same = perm == np.arange(n)
    if same.any():
        perm[same] = (perm[same] + 1) % n
    negative = np.sum(fixed_desc * moving_desc[perm], axis=1)

    n_eval = min(max_eval, n)
    sim = fixed_desc[:n_eval] @ moving_desc[:n_eval].T
    nn_accuracy = float((np.argmax(sim, axis=1) == np.arange(n_eval)).mean() * 100.0)

    return {
        "positive_mean": float(positive.mean()),
        "negative_mean": float(negative.mean()),
        "separation": float(positive.mean() - negative.mean()),
        "nn_accuracy": nn_accuracy,
        "n_eval": int(n_eval),
    }


def compute_index_pck(target_points, src_indices, tgt_indices, thresholds=(2, 5, 10, 20, 50)):
    if len(src_indices) == 0:
        return {f"pck@{t}mm": 0.0 for t in thresholds}, np.zeros(0, dtype=np.float32)
    errors = np.linalg.norm(target_points[tgt_indices] - target_points[src_indices], axis=1)
    pck = {f"pck@{t}mm": float((errors < t).mean() * 100.0) for t in thresholds}
    return pck, errors


def compute_candidate_pck(candidate_points, gt_targets, src_indices, tgt_indices, thresholds=(2, 5, 10, 20, 50)):
    if len(src_indices) == 0:
        return {f"pck@{t}mm": 0.0 for t in thresholds}, np.zeros(0, dtype=np.float32)
    errors = np.linalg.norm(candidate_points[tgt_indices] - gt_targets[src_indices], axis=1)
    pck = {f"pck@{t}mm": float((errors < t).mean() * 100.0) for t in thresholds}
    return pck, errors


def run_gt_matching(config, fixed_desc, moving_desc, fixed_points, moving_points):
    results = {}
    for method in ("nn", "ot", "gwot"):
        kwargs = {}
        if method == "nn":
            kwargs["max_displacement"] = config.sampling.max_displacement
            kwargs["margin_threshold"] = config.matcher.nn_margin_threshold
        elif method == "ot":
            kwargs.update(
                lambda_prior=config.gwot.lambda_prior,
                epsilon=config.gwot.epsilon,
                lambda_mass=config.gwot.lambda_mass,
                max_iter=config.gwot.max_iter,
            )
        else:
            kwargs.update(
                lambda_gw=config.gwot.lambda_gw,
                lambda_prior=config.gwot.lambda_prior,
                epsilon=config.gwot.epsilon,
                lambda_mass=config.gwot.lambda_mass,
                local_radius=config.gwot.local_radius,
                max_iter=config.gwot.max_iter,
            )
        t0 = time.time()
        result = match(moving_desc, fixed_desc, moving_points, fixed_points, method=method, **kwargs)
        elapsed = time.time() - t0
        src_idx = result["matches_src_idx"]
        tgt_idx = result["matches_tgt_idx"]
        pck, errors = compute_index_pck(fixed_points, src_idx, tgt_idx)
        results[method] = {
            "n_matches": int(len(src_idx)),
            "pck": pck,
            "mean_error_mm": float(errors.mean()) if len(errors) else 0.0,
            "runtime_s": elapsed,
        }
    return results


def run_distractor_matching(config, fixed_desc, moving_desc_candidates, fixed_points, moving_candidates, gt_targets):
    results = {}
    for method in ("nn", "gwot"):
        kwargs = {}
        if method == "nn":
            kwargs["max_displacement"] = config.sampling.max_displacement
            kwargs["margin_threshold"] = config.matcher.nn_margin_threshold
        else:
            kwargs.update(
                lambda_gw=config.gwot.lambda_gw,
                lambda_prior=config.gwot.lambda_prior,
                epsilon=config.gwot.epsilon,
                lambda_mass=config.gwot.lambda_mass,
                local_radius=config.gwot.local_radius,
                max_iter=config.gwot.max_iter,
            )
        t0 = time.time()
        result = match(fixed_desc, moving_desc_candidates, fixed_points, moving_candidates, method=method, **kwargs)
        elapsed = time.time() - t0
        src_idx = result["matches_src_idx"]
        tgt_idx = result["matches_tgt_idx"]
        pck, errors = compute_candidate_pck(moving_candidates, gt_targets, src_idx, tgt_idx)
        results[method] = {
            "n_matches": int(len(src_idx)),
            "pck": pck,
            "mean_error_mm": float(errors.mean()) if len(errors) else 0.0,
            "runtime_s": elapsed,
        }
    return results


def create_synthetic_case(data, device, max_displacement, smoothness, seed):
    from scipy.ndimage import gaussian_filter

    fixed_img = data["fixed_img"]
    fixed_kp = data["fixed_keypoints"]
    D, H, W = fixed_img.shape

    torch.manual_seed(seed)
    v_coarse = torch.randn(1, 3, 8, 8, 8, device=device) * 3.0
    velocity = torch.nn.functional.interpolate(v_coarse, size=(D, H, W), mode="trilinear", align_corners=True)

    velocity_np = velocity.detach().cpu().numpy()
    for c in range(3):
        velocity_np[0, c] = gaussian_filter(velocity_np[0, c], sigma=smoothness)
    velocity = torch.from_numpy(velocity_np).float().to(device)

    current_max = velocity.abs().max().item()
    if current_max > 0:
        velocity = velocity * (max_displacement / current_max)

    disp_fwd = scaling_and_squaring(velocity, n_steps=7)
    disp_inv = scaling_and_squaring(-velocity, n_steps=7)

    fixed_tensor = torch.from_numpy(fixed_img).float().unsqueeze(0).unsqueeze(0).to(device)
    moving_img = warp_volume(fixed_tensor, disp_inv).squeeze().detach().cpu().numpy()
    moving_kp = warp_points(fixed_kp, disp_fwd)

    return {
        "fixed_img": fixed_img,
        "moving_img": moving_img,
        "fixed_keypoints": fixed_kp,
        "moving_keypoints": moving_kp,
        "fixed_id": f"{data['fixed_id']}_synthetic_fixed",
        "moving_id": f"{data['fixed_id']}_synthetic_moving",
    }


def evaluate_case(config, backend, case_name, case_data, downsample):
    fixed_mask = generate_trunk_mask(case_data["fixed_img"])
    moving_mask = generate_trunk_mask(case_data["moving_img"])

    fixed_feats, fixed_feat_shape, fixed_orig_shape = extract_features(
        config,
        case_data["fixed_img"],
        case_data["fixed_id"],
        backend,
        downsample,
        f"{case_name}_fixed",
        fixed_mask,
    )
    moving_feats, moving_feat_shape, moving_orig_shape = extract_features(
        config,
        case_data["moving_img"],
        case_data["moving_id"],
        backend,
        downsample,
        f"{case_name}_moving",
        moving_mask,
    )

    fkp = case_data["fixed_keypoints"]
    mkp = case_data["moving_keypoints"]

    rng = np.random.RandomState(42)

    n_gt = min(2000, len(fkp))
    gt_idx = rng.choice(len(fkp), n_gt, replace=False)
    fkp_gt = fkp[gt_idx]
    mkp_gt = mkp[gt_idx]

    fixed_desc_gt = descriptor_set(fixed_feats, fixed_feat_shape, fixed_orig_shape, fkp_gt, downsample)
    moving_desc_gt = descriptor_set(moving_feats, moving_feat_shape, moving_orig_shape, mkp_gt, downsample)

    retrieval = compute_retrieval_metrics(fixed_desc_gt, moving_desc_gt)
    gt_matching = run_gt_matching(config, fixed_desc_gt, moving_desc_gt, fkp_gt, mkp_gt)

    n_src = min(500, len(fkp))
    src_idx = rng.choice(len(fkp), n_src, replace=False)
    fkp_src = fkp[src_idx]
    mkp_src = mkp[src_idx]

    moving_random = sample_points_in_mask(moving_mask, 2000, z_stratified=True, rng=np.random.RandomState(123))
    moving_candidates = include_keypoints(moving_random, mkp_src, mask=moving_mask)

    fixed_desc_src = descriptor_set(fixed_feats, fixed_feat_shape, fixed_orig_shape, fkp_src, downsample)
    moving_desc_candidates = descriptor_set(
        moving_feats,
        moving_feat_shape,
        moving_orig_shape,
        moving_candidates,
        downsample,
    )

    distractor_matching = run_distractor_matching(
        config,
        fixed_desc_src,
        moving_desc_candidates,
        fkp_src,
        moving_candidates,
        mkp_src,
    )

    return {
        "retrieval": retrieval,
        "gt_matching": gt_matching,
        "distractor_matching": distractor_matching,
        "fixed_shape": tuple(fixed_feats.shape),
        "moving_shape": tuple(moving_feats.shape),
    }


def classify_bottleneck(result):
    real_feat = result["real"]["retrieval"]["nn_accuracy"]
    synth_feat = result["synthetic"]["retrieval"]["nn_accuracy"]
    real_d1_nn = result["real"]["gt_matching"]["nn"]["pck"]["pck@10mm"]
    synth_d1_nn = result["synthetic"]["gt_matching"]["nn"]["pck"]["pck@10mm"]
    real_d2_nn = result["real"]["distractor_matching"]["nn"]["pck"]["pck@10mm"]
    synth_d2_nn = result["synthetic"]["distractor_matching"]["nn"]["pck"]["pck@10mm"]
    real_d2_gwot = result["real"]["distractor_matching"]["gwot"]["pck"]["pck@10mm"]

    labels = []

    if synth_feat < 10.0 or synth_d1_nn < 30.0:
        labels.append("fails even on synthetic geometry")
    if synth_feat >= 20.0 and real_feat < 10.0:
        labels.append("domain transfer bottleneck")
    if synth_d1_nn >= 40.0 and synth_d2_nn + 20.0 < synth_d1_nn:
        labels.append("candidate search / distractor bottleneck")
    if real_d2_nn > 0.0 and real_d2_gwot + 5.0 < real_d2_nn:
        labels.append("gwot hurts realistic matching")
    if real_feat < 10.0 and real_d1_nn < 30.0:
        labels.append("feature discrimination bottleneck on real data")
    if not labels:
        labels.append("mixed or inconclusive")

    return labels


def log_case_summary(case_name, result):
    retrieval = result["retrieval"]
    gt_match = result["gt_matching"]
    d2 = result["distractor_matching"]

    logger.info(f"  {case_name} retrieval: "
                f"pos={retrieval['positive_mean']:.3f}, "
                f"neg={retrieval['negative_mean']:.3f}, "
                f"sep={retrieval['separation']:.3f}, "
                f"nn@1={retrieval['nn_accuracy']:.1f}%")
    logger.info(f"  {case_name} gt matching: "
                f"nn={gt_match['nn']['pck']['pck@10mm']:.1f}%, "
                f"ot={gt_match['ot']['pck']['pck@10mm']:.1f}%, "
                f"gwot={gt_match['gwot']['pck']['pck@10mm']:.1f}%")
    logger.info(f"  {case_name} distractor matching: "
                f"nn={d2['nn']['pck']['pck@10mm']:.1f}%, "
                f"gwot={d2['gwot']['pck']['pck@10mm']:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Backend bottleneck diagnostics")
    parser.add_argument("--pair", type=int, default=0)
    parser.add_argument("--feature", type=str, default="all", choices=["all", "dinov3", "matcha", "mind"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--synthetic-max-displacement", type=float, default=15.0)
    parser.add_argument("--synthetic-smoothness", type=float, default=20.0)
    parser.add_argument("--synthetic-seed", type=int, default=42)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    config = PipelineConfig()
    config.device = args.device
    config.features.use_cache = not args.no_cache

    dataset = ThoraxCBCTDataset(config.paths.data_root, split="train")
    data = dataset[args.pair]

    backends = ["dinov3", "matcha", "mind"] if args.feature == "all" else [args.feature]

    all_results = {}

    for feature in backends:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"BACKEND: {feature}")
        logger.info(f"{'=' * 80}")
        config.features.backend = feature

        try:
            backend = create_backend(config, feature, args.device, args.downsample, args.batch_size)
        except Exception as exc:
            logger.error(f"  failed to create backend {feature}: {exc}")
            all_results[feature] = {"error": str(exc)}
            continue

        try:
            real_case = {
                "fixed_img": data["fixed_img"],
                "moving_img": data["moving_img"],
                "fixed_keypoints": data["fixed_keypoints"],
                "moving_keypoints": data["moving_keypoints"],
                "fixed_id": data["fixed_id"],
                "moving_id": data["moving_id"],
            }
            synthetic_case = create_synthetic_case(
                data,
                args.device,
                args.synthetic_max_displacement,
                args.synthetic_smoothness,
                args.synthetic_seed,
            )

            real_result = evaluate_case(config, backend, "real", real_case, args.downsample)
            synthetic_result = evaluate_case(config, backend, "synthetic", synthetic_case, args.downsample)

            combined = {
                "real": real_result,
                "synthetic": synthetic_result,
                "bottlenecks": classify_bottleneck({"real": real_result, "synthetic": synthetic_result}),
            }
            all_results[feature] = combined

            log_case_summary("real", real_result)
            log_case_summary("synthetic", synthetic_result)
            logger.info(f"  bottlenecks: {', '.join(combined['bottlenecks'])}")
        except Exception as exc:
            logger.error(f"  backend {feature} failed: {exc}")
            import traceback

            traceback.print_exc()
            all_results[feature] = {"error": str(exc)}

    logger.info(f"\n{'=' * 80}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 80}")
    for feature, result in all_results.items():
        if "error" in result:
            logger.info(f"  {feature}: ERROR")
            continue
        real_feat = result["real"]["retrieval"]["nn_accuracy"]
        synth_feat = result["synthetic"]["retrieval"]["nn_accuracy"]
        real_d2_nn = result["real"]["distractor_matching"]["nn"]["pck"]["pck@10mm"]
        real_d2_gwot = result["real"]["distractor_matching"]["gwot"]["pck"]["pck@10mm"]
        logger.info(
            f"  {feature}: real nn@1={real_feat:.1f}%, synthetic nn@1={synth_feat:.1f}%, "
            f"real D2 nn={real_d2_nn:.1f}%, real D2 gwot={real_d2_gwot:.1f}%"
        )
        logger.info(f"    bottlenecks: {', '.join(result['bottlenecks'])}")


if __name__ == "__main__":
    main()
