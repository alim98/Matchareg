"""
ThoraxCBCT dataset loader.

Loads NIfTI volumes, parses dataset JSON for train/val pair lists,
and reads keypoint CSV files (Förstner keypoints).
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel is required: pip install nibabel")


def load_dataset_json(json_path: Path) -> dict:
    """Load and parse ThoraxCBCT_dataset.json."""
    with open(json_path, "r") as f:
        return json.load(f)


def load_nifti(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a NIfTI volume.

    Returns:
        volume: ndarray of shape (D, H, W) — float32
        affine: 4x4 affine matrix
    """
    img = nib.load(str(path))
    volume = img.get_fdata().astype(np.float32)
    affine = img.affine.copy()
    return volume, affine


def load_keypoints_csv(csv_path: Path) -> np.ndarray:
    """
    Load keypoint CSV (no header, 3 columns: x, y, z).

    Returns:
        keypoints: ndarray of shape (N, 3) — float64
    """
    return np.loadtxt(str(csv_path), delimiter=",")


def get_paired_images(
    dataset_json: dict,
    split: str = "train",
) -> List[Dict[str, str]]:
    """
    Get list of paired image dicts from dataset JSON.

    Args:
        dataset_json: parsed ThoraxCBCT_dataset.json
        split: 'train' or 'val'

    Returns:
        List of dicts with keys 'fixed', 'moving' (relative paths).
    """
    if split == "train":
        return dataset_json["training_paired_images"]
    elif split == "val":
        return dataset_json["registration_val"]
    else:
        raise ValueError(f"Unknown split: {split}. Use 'train' or 'val'.")


def _resolve_path(rel_path: str, data_root: Path) -> Path:
    """Resolve a relative path like './imagesTr/...' to an absolute path."""
    return data_root / rel_path.lstrip("./")


def _case_id_from_path(path: str) -> str:
    """Extract case ID like 'ThoraxCBCT_0000_0001' from path."""
    return Path(path).stem.replace(".nii", "")


def _find_keypoints(case_id: str, data_root: Path) -> Optional[Path]:
    """
    Find keypoint CSV for a given case ID.
    Searches in keypoints01Tr and keypoints02Tr.
    """
    for kp_dir in ["keypoints01Tr", "keypoints02Tr"]:
        csv_path = data_root / kp_dir / f"{case_id}.csv"
        if csv_path.exists():
            return csv_path
    return None


class ThoraxCBCTDataset:
    """
    ThoraxCBCT dataset with paired FBCT↔CBCT volumes.

    Naming convention:
    - ThoraxCBCT_XXXX_0000 = FBCT (moving)
    - ThoraxCBCT_XXXX_0001 = CBCT session 1 (fixed)
    - ThoraxCBCT_XXXX_0002 = CBCT session 2 (fixed)

    Registration direction: moving (FBCT, _0000) → fixed (CBCT, _0001 or _0002)
    """

    def __init__(self, data_root: Path, split: str = "train"):
        self.data_root = Path(data_root)
        self.split = split
        self.dataset_json = load_dataset_json(self.data_root / "ThoraxCBCT_dataset.json")
        self.pairs = get_paired_images(self.dataset_json, split)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        """
        Load a single registration pair.

        Returns dict with:
            - fixed_img: ndarray (D, H, W)
            - moving_img: ndarray (D, H, W)
            - fixed_affine: 4x4 ndarray
            - moving_affine: 4x4 ndarray
            - fixed_keypoints: ndarray (N, 3) or None
            - moving_keypoints: ndarray (N, 3) or None
            - fixed_id: str
            - moving_id: str
        """
        pair = self.pairs[idx]
        fixed_path = _resolve_path(pair["fixed"], self.data_root)
        moving_path = _resolve_path(pair["moving"], self.data_root)

        fixed_img, fixed_aff = load_nifti(fixed_path)
        moving_img, moving_aff = load_nifti(moving_path)

        fixed_id = _case_id_from_path(pair["fixed"])
        moving_id = _case_id_from_path(pair["moving"])

        # Load keypoints if available
        fixed_kp_path = _find_keypoints(fixed_id, self.data_root)
        moving_kp_path = _find_keypoints(moving_id, self.data_root)

        fixed_kp = load_keypoints_csv(fixed_kp_path) if fixed_kp_path else None
        moving_kp = load_keypoints_csv(moving_kp_path) if moving_kp_path else None

        return {
            "fixed_img": fixed_img,
            "moving_img": moving_img,
            "fixed_affine": fixed_aff,
            "moving_affine": moving_aff,
            "fixed_keypoints": fixed_kp,
            "moving_keypoints": moving_kp,
            "fixed_id": fixed_id,
            "moving_id": moving_id,
        }

    def get_pair_info(self, idx: int) -> Dict[str, str]:
        """Get pair metadata without loading volumes."""
        pair = self.pairs[idx]
        return {
            "fixed": pair["fixed"],
            "moving": pair["moving"],
            "fixed_id": _case_id_from_path(pair["fixed"]),
            "moving_id": _case_id_from_path(pair["moving"]),
        }

    def __repr__(self) -> str:
        return f"ThoraxCBCTDataset(split={self.split}, n_pairs={len(self)})"
