"""
MATCHA dense feature extractor.

Uses MATCHA (DIFT + DINOv2 fusion) to extract dense correspondence features
from 2D slices. Loaded from the local repo with pretrained weights.
"""
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MATCHAExtractor:
    """
    Extract dense features from 2D images using MATCHA.

    MATCHA fuses diffusion (DIFT) and DINOv2 features through an
    attention-based fusion network. The `describe()` method returns
    dense feature maps.
    """

    def __init__(
        self,
        repo_path: str = "/u/almik/feb25/matcha",
        weights_path: str = "/u/almik/feb25/matcha/weights/matcha_pretrained.pth",
        device: str = "cuda",
    ):
        self.device = device
        self.repo_path = Path(repo_path)

        # Add repo to path for imports
        if str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))

        self.model = self._load_model(weights_path)
        self.model.eval()
        # Model components are already moved to device via MatchaFeature

    def _load_model(self, weights_path: str):
        """Load MATCHA feature model with pretrained weights."""
        from matcha.feature.matcha_feature import MatchaFeature

        config = {
            "topK": 4096,
            "upsampling": 0,
            "image_size": None,
            "scale_factor": 32,
            "keypoint_method": None,  # We don't need keypoint detection
            "max_length": None,
            "device": self.device,
        }

        model = MatchaFeature(config=config)

        # Load pretrained weights
        state_dict = torch.load(weights_path, map_location=self.device)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)

        return model

    @torch.no_grad()
    def extract_features(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract dense features from a batch of images.

        Args:
            images: Tensor of shape (B, 3, H, W)

        Returns:
            features: Tensor of shape (B, D, H', W') where D is the
                      MATCHA descriptor dimension and H', W' depend on
                      the internal scale factor.
        """
        images = images.to(self.device)
        # MATCHA describe() returns dense feature map
        feat = self.model.describe(images)
        return feat
