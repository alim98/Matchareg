import logging

import numpy as np
import torch

from pipeline.transform.mind_convex_adam import MINDSSC

logger = logging.getLogger(__name__)


class MINDExtractor:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.patch_size = 1
        self.embed_dim = 12

    @torch.no_grad()
    def extract_volume_features(self, volume: np.ndarray) -> tuple:
        vol_t = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0).to(self.device)
        feats = MINDSSC(vol_t, radius=1, dilation=2)
        feats = feats.squeeze(0).cpu().numpy().astype(np.float16)
        feat_shape = tuple(feats.shape[1:])
        orig_shape = tuple(volume.shape)
        logger.info(f"MIND features extracted: {feats.shape}")
        return feats, feat_shape, orig_shape
