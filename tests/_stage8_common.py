import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.data.preprocessing import generate_trunk_mask, robust_intensity_normalize
from pipeline.features.triplanar_fuser import TriplanarFuser, load_features, save_features
from pipeline.matching.gwot3d import match
from pipeline.matching.sampling import include_keypoints, sample_descriptors_at_points, sample_points_in_mask
from pipeline.transform.integrate import scaling_and_squaring
from pipeline.transform.warp import warp_points, warp_volume


def setup_logging(test_name: str):
    results_dir = PROJECT_ROOT / "pipeline" / "tests" / "results" / test_name
    logs_dir = results_dir / "logs"
    viz_dir = results_dir / "viz"
    logs_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(test_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{test_name}_{ts}.log"

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.info(f"Logging to {log_path}")
    return logger, results_dir, viz_dir, log_path


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
    points = points.astype(np.float64).copy()
    D, H, W = original_shape
    fD, fH, fW = feature_shape

    if (D, H, W) == (fD, fH, fW):
        points[:, 0] = np.clip(points[:, 0], 0, fD - 1)
        points[:, 1] = np.clip(points[:, 1], 0, fH - 1)
        points[:, 2] = np.clip(points[:, 2], 0, fW - 1)
        return points

    points[:, 0] /= downsample
    points[:, 1] /= downsample
    points[:, 2] /= downsample

    dD = D // downsample
    dH = H // downsample
    dW = W // downsample

    points[:, 0] = points[:, 0] * fD / dD
    points[:, 1] = points[:, 1] * fH / dH
    points[:, 2] = points[:, 2] * fW / dW

    points[:, 0] = np.clip(points[:, 0], 0, fD - 1)
    points[:, 1] = np.clip(points[:, 1], 0, fH - 1)
    points[:, 2] = np.clip(points[:, 2], 0, fW - 1)
    return points


def extract_features(config, volume, case_id, backend, downsample, cache_tag, logger, mask=None):
    cache_name = f"{case_id}_{config.features.backend}_{cache_tag}_ds{downsample}.npz"
    cache_path = config.paths.feature_cache_dir / cache_name
    if config.features.use_cache and cache_path.exists():
        logger.info(f"Loading cached features: {cache_path}")
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


def sample_descriptors(feats, feat_shape, orig_shape, points, downsample):
    feat_points = voxel_to_feature_coords(points, orig_shape, feat_shape, downsample)
    return l2_normalize(sample_descriptors_at_points(feats, feat_points))


def retrieval_metrics(fixed_desc, moving_desc, max_eval=500, seed=42):
    n = min(len(fixed_desc), len(moving_desc))
    if n == 0:
        return {
            "positive_mean": 0.0,
            "negative_mean": 0.0,
            "separation": 0.0,
            "nn_accuracy": 0.0,
            "n_eval": 0,
            "ranks": np.zeros(0, dtype=np.int32),
            "sim_matrix": np.zeros((0, 0), dtype=np.float32),
        }

    fixed_desc = fixed_desc[:n]
    moving_desc = moving_desc[:n]
    positive = np.sum(fixed_desc * moving_desc, axis=1)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    clash = perm == np.arange(n)
    if clash.any():
        perm[clash] = (perm[clash] + 1) % n
    negative = np.sum(fixed_desc * moving_desc[perm], axis=1)

    n_eval = min(max_eval, n)
    sim = fixed_desc[:n_eval] @ moving_desc[:n_eval].T
    best = np.argmax(sim, axis=1)
    ranks = np.array([(sim[i] > sim[i, i]).sum() for i in range(n_eval)], dtype=np.int32)

    return {
        "positive_mean": float(positive.mean()),
        "negative_mean": float(negative.mean()),
        "separation": float(positive.mean() - negative.mean()),
        "nn_accuracy": float((best == np.arange(n_eval)).mean() * 100.0),
        "n_eval": int(n_eval),
        "ranks": ranks,
        "sim_matrix": sim,
        "positive_all": positive,
        "negative_all": negative,
    }


def compute_index_pck(target_points, src_indices, tgt_indices, thresholds=(2, 5, 10, 20, 50)):
    if len(src_indices) == 0:
        return {f"pck@{t}mm": 0.0 for t in thresholds}, np.zeros(0, dtype=np.float32)
    errors = np.linalg.norm(target_points[tgt_indices] - target_points[src_indices], axis=1)
    return {f"pck@{t}mm": float((errors < t).mean() * 100.0) for t in thresholds}, errors


def compute_candidate_pck(candidate_points, gt_targets, src_indices, tgt_indices, thresholds=(2, 5, 10, 20, 50)):
    if len(src_indices) == 0:
        return {f"pck@{t}mm": 0.0 for t in thresholds}, np.zeros(0, dtype=np.float32)
    errors = np.linalg.norm(candidate_points[tgt_indices] - gt_targets[src_indices], axis=1)
    return {f"pck@{t}mm": float((errors < t).mean() * 100.0) for t in thresholds}, errors


def run_matching(config, method, desc_src, desc_tgt, points_src, points_tgt):
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
    elif method == "gwot":
        kwargs.update(
            lambda_gw=config.gwot.lambda_gw,
            lambda_prior=config.gwot.lambda_prior,
            epsilon=config.gwot.epsilon,
            lambda_mass=config.gwot.lambda_mass,
            local_radius=config.gwot.local_radius,
            max_iter=config.gwot.max_iter,
        )
    else:
        raise ValueError(method)
    return match(desc_src, desc_tgt, points_src, points_tgt, method=method, **kwargs)


def create_synthetic_case(data, device, max_displacement=15.0, smoothness=20.0, seed=42):
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

    cur_max = velocity.abs().max().item()
    if cur_max > 0:
        velocity = velocity * (max_displacement / cur_max)

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


def maybe_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
