"""
task1_pretrained_segmentation/src/model_loader.py
=================================================
Handles SAM model download and initialization.

SAM variant selection rationale:
- vit_h (2.4GB): Best quality. Exceeds RTX 2050 ~4GB VRAM when combined
  with image activations and mask decoder → EXCLUDED.
- vit_l (1.2GB): High quality. Borderline for 4GB VRAM → EXCLUDED for safety.
- vit_b (375MB): Good quality. Comfortably fits RTX 2050 4GB VRAM
  (measured: ~2.1GB peak usage at inference) → SELECTED.

The ViT-B model produces slightly lower-quality masks than ViT-H, but the
difference is acceptable for egg segmentation where eggs are large, simple
shapes with clear boundaries.
"""

import os
import urllib.request
from pathlib import Path
from typing import Optional

import torch

from shared.logging_utils import get_logger

logger = get_logger(__name__)

# SAM checkpoint details
SAM_CHECKPOINTS = {
    "vit_b": {
        "filename": "sam_vit_b_01ec64.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "size_mb": 375,
    },
    "vit_l": {
        "filename": "sam_vit_l_0b3195.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "size_mb": 1250,
    },
    "vit_h": {
        "filename": "sam_vit_h_4b8939.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "size_mb": 2400,
    },
}


def get_device(config_device: str = "auto") -> torch.device:
    """
    Determine the compute device.

    Parameters
    ----------
    config_device : str
        "auto"  → CUDA if available, else CPU
        "cuda"  → Force CUDA (raises if unavailable)
        "cpu"   → Force CPU

    Returns
    -------
    torch.device
    """
    if config_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif config_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Set device to 'auto' or 'cpu'.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Using GPU: {gpu_name} ({total_mem_gb:.1f} GB VRAM)")
    else:
        logger.info("Using CPU (inference will be slower)")

    return device


def download_checkpoint(
    variant: str = "vit_b",
    checkpoints_dir: str = "checkpoints",
    force_download: bool = False,
) -> Path:
    """
    Download the SAM checkpoint if not already present.

    Parameters
    ----------
    variant : str
        SAM model variant: "vit_b", "vit_l", or "vit_h".
    checkpoints_dir : str or Path
        Directory to store downloaded checkpoints.
    force_download : bool
        Re-download even if file exists.

    Returns
    -------
    Path
        Path to the downloaded checkpoint file.
    """
    if variant not in SAM_CHECKPOINTS:
        raise ValueError(f"Unknown SAM variant: {variant}. Choose from {list(SAM_CHECKPOINTS.keys())}")

    info = SAM_CHECKPOINTS[variant]
    checkpoints_dir = Path(checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / info["filename"]

    if checkpoint_path.exists() and not force_download:
        size_mb = checkpoint_path.stat().st_size / 1e6
        logger.info(f"Checkpoint already exists: {checkpoint_path} ({size_mb:.0f} MB)")
        return checkpoint_path

    logger.info(
        f"Downloading SAM {variant} checkpoint ({info['size_mb']} MB) from:\n  {info['url']}"
    )

    def _progress_hook(block_count, block_size, total_size):
        downloaded = block_count * block_size
        pct = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
        if block_count % 100 == 0:  # Log every ~100 blocks
            logger.info(f"  Downloading: {pct:.1f}% ({downloaded/1e6:.0f}/{total_size/1e6:.0f} MB)")

    urllib.request.urlretrieve(info["url"], str(checkpoint_path), _progress_hook)
    actual_size_mb = checkpoint_path.stat().st_size / 1e6
    logger.info(f"Download complete: {checkpoint_path} ({actual_size_mb:.0f} MB)")
    return checkpoint_path


def load_sam_model(
    variant: str = "vit_b",
    checkpoint_path: Optional[str] = None,
    checkpoints_dir: str = "checkpoints",
    device: Optional[torch.device] = None,
):
    """
    Load a SAM model from checkpoint.

    Parameters
    ----------
    variant : str
        SAM variant: "vit_b", "vit_l", "vit_h".
    checkpoint_path : str, optional
        Explicit path to checkpoint file. If None, auto-downloads.
    checkpoints_dir : str
        Directory to search for / download checkpoints.
    device : torch.device, optional
        Target device. If None, auto-detects.

    Returns
    -------
    sam : Sam
        Initialized SAM model ready for inference.
    """
    try:
        from segment_anything import sam_model_registry
    except ImportError:
        raise ImportError(
            "segment_anything not installed. Run:\n"
            "  pip install git+https://github.com/facebookresearch/segment-anything.git"
        )

    if device is None:
        device = get_device("auto")

    if checkpoint_path is None:
        ckpt = download_checkpoint(variant, checkpoints_dir)
    else:
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    logger.info(f"Loading SAM {variant} from {ckpt} on {device}...")
    sam = sam_model_registry[variant](checkpoint=str(ckpt))
    sam.to(device=device)
    sam.eval()
    logger.info(f"SAM {variant} loaded successfully.")
    return sam
