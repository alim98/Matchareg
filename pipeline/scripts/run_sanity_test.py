import numpy as np
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import PipelineConfig
from pipeline.data.dataset_thoraxcbct import ThoraxCBCTDataset
from pipeline.transform.fitter import DiffeomorphicFitter
from pipeline.eval.metrics import compute_tre

config = PipelineConfig()
dataset = ThoraxCBCTDataset(config.paths.data_root, split='train')
data = dataset[0]

fkp = data["fixed_keypoints"]
mkp = data["moving_keypoints"]

w = np.ones(len(fkp), dtype=np.float32)

fitter = DiffeomorphicFitter(
    volume_shape=data["fixed_img"].shape,
    grid_spacings=config.fitting.grid_spacings,
    n_iters_per_level=config.fitting.n_iters_per_level,
    lambda_smooth=config.fitting.lambda_smooth,
    lambda_jac=config.fitting.lambda_jac,
    lr=config.fitting.lr,
    n_squaring_steps=config.fitting.n_squaring_steps,
    device=config.device,
)

print(f"Initial Mean TRE: {np.linalg.norm(fkp - mkp, axis=1).mean():.3f} mm")
disp = fitter.fit(matched_src=fkp, matched_tgt=mkp, weights=w)
tre = compute_tre(mkp, fkp, disp)
print(f"Post-Fitter SVF Mean TRE: {tre['mean_tre']:.3f} mm")
