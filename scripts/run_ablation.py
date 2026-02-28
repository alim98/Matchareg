#!/usr/bin/env python3
"""
Ablation runner.

Runs the full ablation grid from plan.md:
1. DINOv2 + NN + diffeo (DINO-Reg baseline)
2. DINOv3 + NN + diffeo
3. MATCHA + NN + diffeo
4. DINOv3 + GWOT + diffeo
5. MATCHA + GWOT + diffeo  ← main method
6. MATCHA + OT (no GW) + diffeo (ablation)

Usage:
    python -m pipeline.scripts.run_ablation --split val
"""
import argparse
import csv
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import PipelineConfig
from pipeline.data.dataset_thoraxcbct import ThoraxCBCTDataset
from pipeline.scripts.run_pipeline import create_feature_extractor, run_pair
from pipeline.features.triplanar_fuser import TriplanarFuser

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Ablation variants
ABLATION_GRID = [
    {"name": "DINOv3_NN_diffeo", "feature": "dinov3", "matcher": "nn"},
    {"name": "DINOv3_GWOT_diffeo", "feature": "dinov3", "matcher": "gwot"},
    {"name": "DINOv3_OT_diffeo", "feature": "dinov3", "matcher": "ot"},
    {"name": "MATCHA_NN_diffeo", "feature": "matcha", "matcher": "nn"},
    {"name": "MATCHA_GWOT_diffeo", "feature": "matcha", "matcher": "gwot"},
    {"name": "MATCHA_OT_diffeo", "feature": "matcha", "matcher": "ot"},
]


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Subset of variants to run (by name)")
    args = parser.parse_args()

    config = PipelineConfig()
    config.device = args.device
    dataset = ThoraxCBCTDataset(config.paths.data_root, split=args.split)

    # Filter variants
    variants = ABLATION_GRID
    if args.variants:
        variants = [v for v in ABLATION_GRID if v["name"] in args.variants]

    # Output CSV
    output_path = args.output or str(
        config.paths.output_dir / f"ablation_{args.split}.csv"
    )

    all_results = []

    for variant in variants:
        logger.info(f"\n{'#'*60}")
        logger.info(f"VARIANT: {variant['name']}")
        logger.info(f"{'#'*60}")

        config.features.backend = variant["feature"]
        config.matcher.method = variant["matcher"]

        # Create extractor for this variant
        extractor = create_feature_extractor(config)
        fuser = TriplanarFuser(
            extractor,
            batch_size=config.features.slice_batch_size,
            fusion=config.features.fusion_method,
            downsample=2,
            device=config.device,
        )

        for i in range(len(dataset)):
            r = run_pair(config, dataset, i, method="sparse", fuser=fuser, downsample=2)
            r["variant"] = variant["name"]
            all_results.append(r)

    # Save results
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "variant", "pair_idx", "moving_id", "fixed_id",
            "initial_tre", "final_tre", "n_matches",
            "jac_pct_negative", "runtime_s"
        ])
        for r in all_results:
            if "error" in r:
                writer.writerow([
                    r.get("variant", ""), r["pair_idx"], "", "",
                    "", "", "", "", ""
                ])
            else:
                writer.writerow([
                    r.get("variant", ""),
                    r["pair_idx"],
                    r.get("moving_id", ""),
                    r.get("fixed_id", ""),
                    r.get("initial_tre", {}).get("mean_tre", ""),
                    r.get("final_tre", {}).get("mean_tre", ""),
                    r.get("n_matches", ""),
                    r.get("jac_stats", {}).get("jac_pct_negative", ""),
                    f"{r.get('runtime_seconds', 0):.1f}",
                ])

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
