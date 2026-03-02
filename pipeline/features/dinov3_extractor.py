"""
DINOv3 dense feature extractor.

Uses DINOv3 ViT-B/16 to extract dense patch-level features from 2D slices.
The model is loaded from the local repo with pretrained weights.
Supports loading from HuggingFace-format safetensors checkpoints.
"""
import sys
import re
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)


def convert_hf_to_native(hf_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert HuggingFace-format DINOv3 state_dict keys to native DINOv3 format.

    HF format uses:
      embeddings.cls_token, embeddings.patch_embeddings.weight,
      layer.{i}.attention.{q,k,v}_proj, layer.{i}.mlp.{up,down}_proj, ...

    Native format uses:
      cls_token, patch_embed.proj.weight,
      blocks.{i}.attn.qkv (merged), blocks.{i}.mlp.fc1/fc2, ...
    """
    native = {}

    # Collect per-block q/k/v weights and biases for merging
    block_qkv = {}  # block_idx -> {q_weight, k_weight, v_weight, q_bias, v_bias}

    for hf_key, val in hf_state.items():
        # --- Global tokens ---
        if hf_key == "embeddings.cls_token":
            native["cls_token"] = val
            continue
        if hf_key == "embeddings.mask_token":
            native["mask_token"] = val.squeeze(0)  # (1,1,768) -> (1,768)
            continue
        if hf_key == "embeddings.register_tokens":
            native["storage_tokens"] = val
            continue

        # --- Patch embedding ---
        if hf_key == "embeddings.patch_embeddings.weight":
            native["patch_embed.proj.weight"] = val
            continue
        if hf_key == "embeddings.patch_embeddings.bias":
            native["patch_embed.proj.bias"] = val
            continue

        # --- Transformer blocks ---
        m = re.match(r"layer\.(\d+)\.(.*)", hf_key)
        if m:
            block_idx = m.group(1)
            rest = m.group(2)

            # Attention: q/k/v projections (need to merge into qkv)
            if rest == "attention.q_proj.weight":
                block_qkv.setdefault(block_idx, {})["q_weight"] = val
                continue
            if rest == "attention.k_proj.weight":
                block_qkv.setdefault(block_idx, {})["k_weight"] = val
                continue
            if rest == "attention.v_proj.weight":
                block_qkv.setdefault(block_idx, {})["v_weight"] = val
                continue
            if rest == "attention.q_proj.bias":
                block_qkv.setdefault(block_idx, {})["q_bias"] = val
                continue
            if rest == "attention.v_proj.bias":
                block_qkv.setdefault(block_idx, {})["v_bias"] = val
                continue

            # Output projection
            if rest == "attention.o_proj.weight":
                native[f"blocks.{block_idx}.attn.proj.weight"] = val
                continue
            if rest == "attention.o_proj.bias":
                native[f"blocks.{block_idx}.attn.proj.bias"] = val
                continue

            # Layer scale
            if rest == "layer_scale1.lambda1":
                native[f"blocks.{block_idx}.ls1.gamma"] = val
                continue
            if rest == "layer_scale2.lambda1":
                native[f"blocks.{block_idx}.ls2.gamma"] = val
                continue

            # MLP
            if rest == "mlp.up_proj.weight":
                native[f"blocks.{block_idx}.mlp.fc1.weight"] = val
                continue
            if rest == "mlp.up_proj.bias":
                native[f"blocks.{block_idx}.mlp.fc1.bias"] = val
                continue
            if rest == "mlp.down_proj.weight":
                native[f"blocks.{block_idx}.mlp.fc2.weight"] = val
                continue
            if rest == "mlp.down_proj.bias":
                native[f"blocks.{block_idx}.mlp.fc2.bias"] = val
                continue

            # Norms
            if rest == "norm1.weight":
                native[f"blocks.{block_idx}.norm1.weight"] = val
                continue
            if rest == "norm1.bias":
                native[f"blocks.{block_idx}.norm1.bias"] = val
                continue
            if rest == "norm2.weight":
                native[f"blocks.{block_idx}.norm2.weight"] = val
                continue
            if rest == "norm2.bias":
                native[f"blocks.{block_idx}.norm2.bias"] = val
                continue

        # Anything else: pass through
        native[hf_key] = val

    # Merge q/k/v into qkv for each block
    for block_idx, parts in block_qkv.items():
        q_w = parts["q_weight"]
        k_w = parts["k_weight"]
        v_w = parts["v_weight"]
        native[f"blocks.{block_idx}.attn.qkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)

        # Bias: q and v have bias, k does not (in DINOv3 with mask_k_bias)
        q_b = parts.get("q_bias", torch.zeros(q_w.shape[0]))
        k_b = torch.zeros(k_w.shape[0])  # k has no bias
        v_b = parts.get("v_bias", torch.zeros(v_w.shape[0]))
        native[f"blocks.{block_idx}.attn.qkv.bias"] = torch.cat([q_b, k_b, v_b], dim=0)

        # bias_mask: 1 for q, 0 for k, 1 for v
        mask = torch.cat([
            torch.ones(q_w.shape[0]),
            torch.zeros(k_w.shape[0]),
            torch.ones(v_w.shape[0]),
        ])
        native[f"blocks.{block_idx}.attn.qkv.bias_mask"] = mask

    return native


def is_hf_format(state_dict: Dict[str, torch.Tensor]) -> bool:
    """Check if a state dict is in HuggingFace format."""
    return any(k.startswith("embeddings.") or k.startswith("layer.") for k in state_dict)


class DINOv3Extractor:
    """
    Extract dense features from 2D images using DINOv3 ViT-B/16.

    Features are extracted via `get_intermediate_layers` which returns
    patch-level tokens of shape (B, H/16, W/16, 768) when reshaped.
    """

    def __init__(
        self,
        repo_path: str = "/u/almik/feb25/dinov3",
        weights_path: Optional[str] = None,
        model_name: str = "vitb16",
        device: str = "cuda",
    ):
        self.device = device
        self.patch_size = 16
        self.embed_dim = 768

        # Add repo to path for imports
        repo = Path(repo_path)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        # Clear any cached 'dinov3' namespace package from PYTHONPATH
        # so the real package inside the repo is found
        for key in list(sys.modules.keys()):
            if key == "dinov3" or key.startswith("dinov3."):
                del sys.modules[key]

        self.model = self._load_model(model_name, weights_path)
        self.model.eval()
        self.model.to(device)

    def _load_model(self, model_name: str, weights_path: Optional[str]) -> nn.Module:
        """Load DINOv3 backbone with local weights (HF or native format)."""
        from dinov3.hub.backbones import dinov3_vitb16

        if weights_path is not None and str(weights_path).strip() != "None":
            weights_p = Path(weights_path)
            if weights_p.is_dir():
                if (weights_p / "model.safetensors").exists():
                    weights_p = weights_p / "model.safetensors"
                elif (weights_p / "pytorch_model.bin").exists():
                    weights_p = weights_p / "pytorch_model.bin"

            # Create model without pretrained weights
            model = dinov3_vitb16(pretrained=False)

            # Load weights file
            weights_file = str(weights_p)
            if weights_file.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(weights_file)
            else:
                state_dict = torch.load(weights_file, map_location="cpu")

            # Convert HF format if needed
            if is_hf_format(state_dict):
                logger.info("Converting HuggingFace weights to native DINOv3 format...")
                state_dict = convert_hf_to_native(state_dict)

            # Load with strict=False to skip buffers like rope_embed.periods
            # which are computed at init, not learned
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                # Filter out expected missing buffers
                real_missing = [k for k in missing if "rope_embed" not in k]
                if real_missing:
                    logger.warning(f"Warning: missing keys: {real_missing}")
            if unexpected:
                logger.warning(f"Warning: unexpected keys: {unexpected}")

            logger.info("DINOv3 weights loaded successfully.")
        else:
            model = dinov3_vitb16(pretrained=True)
            logger.info("DINOv3 weights loaded from default pretrained repository.")

        return model

    @torch.no_grad()
    def extract_features(
        self,
        images: torch.Tensor,
        n_layers: int = 1,
    ) -> torch.Tensor:
        """
        Extract dense patch features from a batch of images.

        Args:
            images: Tensor of shape (B, 3, H, W) in ImageNet-normalized range
            n_layers: number of last layers to return (fused by averaging)

        Returns:
            features: Tensor of shape (B, embed_dim, H//16, W//16)  — channel-first
        """
        B, C, H, W = images.shape
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            images = F.pad(images, (0, pad_w, 0, pad_h), mode="reflect")

        images = images.to(self.device)

        outputs = self.model.get_intermediate_layers(
            images, n=n_layers, reshape=True, norm=True
        )

        if n_layers == 1:
            features = outputs[0]
        else:
            features = torch.stack(outputs, dim=0).mean(dim=0)

        return features

    @torch.no_grad()
    def extract_features_multiscale(
        self,
        images: torch.Tensor,
        scales: tuple = (1.0,),
    ) -> torch.Tensor:
        """Multi-scale feature extraction with scale averaging."""
        B, C, H, W = images.shape
        target_h = H // self.patch_size
        target_w = W // self.patch_size

        all_feats = []
        for s in scales:
            if s != 1.0:
                scaled = F.interpolate(
                    images, scale_factor=s, mode="bilinear", align_corners=False
                )
            else:
                scaled = images

            feat = self.extract_features(scaled)
            if feat.shape[-2:] != (target_h, target_w):
                feat = F.interpolate(
                    feat, size=(target_h, target_w), mode="bilinear", align_corners=False
                )
            all_feats.append(feat)

        return torch.stack(all_feats, dim=0).mean(dim=0)
