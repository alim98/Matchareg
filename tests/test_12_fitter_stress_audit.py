#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import PipelineConfig
from pipeline.data.dataset_thoraxcbct import ThoraxCBCTDataset
from pipeline.eval.metrics import compute_jacobian_stats, compute_tre
from pipeline.tests._stage8_common import cleanup_cuda, maybe_import_matplotlib, setup_logging
from pipeline.transform.fitter import DiffeomorphicFitter


def run_fitter(volume_shape, src, tgt, weights, device):
    fitter = DiffeomorphicFitter(
        volume_shape=volume_shape,
        grid_spacings=[10.0, 6.0, 3.0],
        n_iters_per_level=200,
        n_squaring_steps=7,
        device=device,
    )
    t0 = time.time()
    disp = fitter.fit(src, tgt, weights)
    return disp, time.time() - t0


def evaluate_displacement(displacement, fkp, mkp):
    final = compute_tre(mkp, fkp, displacement=displacement)
    jac = compute_jacobian_stats(displacement)
    disp_np = displacement.detach().cpu().numpy()
    sampled = np.zeros_like(fkp)
    for ax in range(3):
        sampled[:, ax] = map_coordinates(disp_np[0, ax], fkp.T, order=1, mode="nearest")
    ideal = mkp - fkp
    corr = []
    for ax in range(3):
        c = np.corrcoef(ideal[:, ax], sampled[:, ax])[0, 1] if ideal[:, ax].std() > 0 else 0.0
        corr.append(float(c))
    return final, jac, corr


def make_scenarios(fkp, mkp):
    rng = np.random.RandomState(42)
    scenarios = {}

    scenarios["perfect_all"] = (fkp, mkp, np.ones(len(fkp), dtype=np.float64))

    idx = rng.choice(len(fkp), min(2000, len(fkp)), replace=False)
    scenarios["perfect_subset_2k"] = (fkp[idx], mkp[idx], np.ones(len(idx), dtype=np.float64))

    for sigma in (2.0, 5.0, 10.0):
        noisy = mkp[idx] + rng.normal(scale=sigma, size=mkp[idx].shape)
        scenarios[f"noisy_sigma_{int(sigma)}"] = (fkp[idx], noisy, np.ones(len(idx), dtype=np.float64))

    for ratio in (0.1, 0.3, 0.5):
        src = fkp[idx].copy()
        tgt = mkp[idx].copy()
        n_out = int(len(idx) * ratio)
        out_idx = rng.choice(len(idx), n_out, replace=False)
        perm = rng.permutation(len(idx))
        tgt[out_idx] = tgt[perm[out_idx]]
        w = np.ones(len(idx), dtype=np.float64)
        scenarios[f"outlier_{int(ratio*100)}pct"] = (src, tgt, w)

    center = np.median(fkp, axis=0)
    mask = np.linalg.norm(fkp - center[None], axis=1) < 45.0
    cluster_src = fkp[mask]
    cluster_tgt = mkp[mask]
    if len(cluster_src) == 0:
        d = np.linalg.norm(fkp - center[None], axis=1)
        nearest = np.argsort(d)[: min(2500, len(fkp))]
        cluster_src = fkp[nearest]
        cluster_tgt = mkp[nearest]
    if len(cluster_src) > 2500:
        sub = rng.choice(len(cluster_src), 2500, replace=False)
        cluster_src = cluster_src[sub]
        cluster_tgt = cluster_tgt[sub]
    scenarios["clustered_local_region"] = (
        cluster_src,
        cluster_tgt,
        np.ones(len(cluster_src), dtype=np.float64),
    )

    return scenarios


def visualize(viz_dir, names, final_tres, jac_negs, corr_x):
    plt = maybe_import_matplotlib()
    if plt is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(len(names))

    axes[0].bar(x, final_tres, color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha="right")
    axes[0].set_title("Final TRE")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(x, jac_negs, color="darkorange")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha="right")
    axes[1].set_title("Negative Jacobian %")
    axes[1].grid(True, alpha=0.3, axis="y")

    axes[2].bar(x, corr_x, color="seagreen")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=45, ha="right")
    axes[2].set_title("Max axis correlation")
    axes[2].grid(True, alpha=0.3, axis="y")

    path = viz_dir / "test_12_fitter_stress_summary.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Test 12: fitter stress audit")
    parser.add_argument("--pair", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logger, _, viz_dir, _ = setup_logging("test_12_fitter_stress_audit")
    config = PipelineConfig()
    config.device = args.device
    dataset = ThoraxCBCTDataset(config.paths.data_root, split="train")
    data = dataset[args.pair]
    fkp = data["fixed_keypoints"]
    mkp = data["moving_keypoints"]
    if fkp is None or mkp is None or len(fkp) == 0:
        raise ValueError("No keypoints available for this pair")
    volume_shape = data["fixed_img"].shape

    initial = compute_tre(mkp, fkp, displacement=None)
    logger.info(f"initial_tre={initial['mean_tre']:.3f}")

    scenarios = make_scenarios(fkp, mkp)
    results = {}

    for name, (src, tgt, weights) in scenarios.items():
        logger.info("=" * 80)
        logger.info(f"SCENARIO {name} n={len(src)}")
        logger.info("=" * 80)
        displacement, runtime_s = run_fitter(volume_shape, src, tgt, weights, args.device)
        final, jac, corr = evaluate_displacement(displacement, fkp, mkp)
        results[name] = {
            "final_tre": final["mean_tre"],
            "runtime_s": runtime_s,
            "jac_pct_negative": jac["jac_pct_negative"],
            "max_axis_corr": max(corr),
        }
        logger.info(
            f"{name}: final_tre={final['mean_tre']:.3f} "
            f"improvement={(1 - final['mean_tre'] / initial['mean_tre']) * 100:.1f}% "
            f"jac_neg={jac['jac_pct_negative']:.2f}% "
            f"max_axis_corr={max(corr):.3f} runtime={runtime_s:.1f}s"
        )
        cleanup_cuda()

    names = list(results.keys())
    visualize(
        viz_dir,
        names,
        [results[n]["final_tre"] for n in names],
        [results[n]["jac_pct_negative"] for n in names],
        [results[n]["max_axis_corr"] for n in names],
    )

    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    for name in names:
        r = results[name]
        logger.info(
            f"{name}: final_tre={r['final_tre']:.3f} "
            f"jac_neg={r['jac_pct_negative']:.2f}% "
            f"max_corr={r['max_axis_corr']:.3f} runtime={r['runtime_s']:.1f}s"
        )


if __name__ == "__main__":
    main()
