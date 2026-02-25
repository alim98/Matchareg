"""
Centralized configuration for the registration pipeline.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal


@dataclass
class PathConfig:
    """Data and model paths."""
    # Dataset
    data_root: Path = Path("/nexus/posix0/MBR-neuralsystems/alim/regdata/ThoraxCBCT/ThoraxCBCT")
    dataset_json: Path = Path("/nexus/posix0/MBR-neuralsystems/alim/regdata/ThoraxCBCT/ThoraxCBCT/ThoraxCBCT_dataset.json")

    # Model repos
    dinov3_repo: Path = Path("/u/almik/feb25/dinov3")
    matcha_repo: Path = Path("/u/almik/feb25/matcha")

    # Pretrained weights
    dinov3_weights: Optional[Path] = Path("/u/almik/feb25/dino_weights")
    matcha_weights: Path = Path("/u/almik/feb25/matcha/weights/matcha_pretrained.pth")

    # Output / cache
    output_dir: Path = Path("/u/almik/feb25/pipeline/output")
    feature_cache_dir: Path = Path("/u/almik/feb25/pipeline/output/feature_cache")


@dataclass
class FeatureConfig:
    """Feature extraction settings."""
    backend: Literal["dinov3", "matcha"] = "dinov3"
    dinov3_model: str = "vitb16"  # ViT-B/16
    patch_size: int = 16
    embed_dim: int = 768  # for ViT-B/16

    # Tri-planar settings
    slice_context: int = 1  # stack [s-1, s, s+1] as 3 channels
    fusion_method: Literal["concat_norm", "concat_pca"] = "concat_norm"
    pca_dim: int = 256  # only used if fusion_method == "concat_pca"

    # Processing
    slice_batch_size: int = 16  # num slices per forward pass
    use_cache: bool = True


@dataclass
class SamplingConfig:
    """Point sampling settings."""
    n_points_coarse: int = 8000
    n_points_medium: int = 15000
    n_points_fine: int = 25000
    z_stratified: bool = True
    include_keypoints: bool = True  # include provided Förstner keypoints


@dataclass
class GWOTConfig:
    """GWOT matching settings."""
    local_radius: float = 12.0  # mm, for spatial distance graphs
    lambda_gw: float = 0.5  # spatial smoothness strength
    lambda_prior: float = 0.2  # anatomical plausibility prior
    epsilon: float = 0.05  # entropic regularization
    lambda_mass: float = 1.0  # unbalanced mass regularization
    max_iter: int = 100  # GWOT solver iterations

    # Confidence filtering
    mutual_filter: bool = True
    confidence_threshold: float = 0.01


@dataclass
class FittingConfig:
    """Diffeomorphic fitting settings."""
    # SVF grid spacings (mm) for multi-resolution
    grid_spacings: list = field(default_factory=lambda: [10.0, 6.0, 3.0])

    # Regularization
    lambda_smooth: float = 1.0
    lambda_jac: float = 0.1

    # Optimization
    optimizer: Literal["lbfgs", "adam"] = "adam"
    lr: float = 1e-3
    n_iters_per_level: int = 200
    n_squaring_steps: int = 7

    # Alternation
    n_outer_iters: int = 3  # GWOT ↔ fit alternations


@dataclass
class MatcherConfig:
    """Matcher type for ablation."""
    method: Literal["nn", "gwot", "ot"] = "gwot"
    # nn = nearest-neighbor matching
    # gwot = full Gromov-Wasserstein OT
    # ot = OT without GW term (ablation)


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    paths: PathConfig = field(default_factory=PathConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    gwot: GWOTConfig = field(default_factory=GWOTConfig)
    fitting: FittingConfig = field(default_factory=FittingConfig)
    matcher: MatcherConfig = field(default_factory=MatcherConfig)

    # Device
    device: str = "cuda"

    def __post_init__(self):
        """Ensure output dirs exist."""
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths.feature_cache_dir.mkdir(parents=True, exist_ok=True)
