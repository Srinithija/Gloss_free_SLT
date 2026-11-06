"""Shared preprocessing, caption filtering and paths for the optimized iSign pipeline.

Frame preprocessing contract (identical for caching, training and inference):
  stored LMDB frame (256x256 RGB JPEG q80, uniformly sampled <=128 frames)
  -> centre crop 224x224 -> float /255 -> ImageNet mean/std normalisation
  -> torchvision ResNet18 (ImageNet weights, frozen, fc removed) -> 512-d feature per frame.
"""
from __future__ import annotations

import gzip
import io
import pickle
import re
from pathlib import Path

import cv2
import lmdb
import numpy as np
import torch
import torchvision

REPO = Path(__file__).resolve().parents[1]
EXP_ROOT = Path(r"D:\iSign_Experiments\GFSLT\optimized_v2_20261023")
FULL_LMDB = Path(r"D:\iSign_LMDB")
SUBSET_LMDB = Path(r"D:\iSign_LMDB_subset_16k")
MBART_ROOT = Path(r"E:\FINAL YEAR PROJECT\GLOSS FREE SLT\mbart_models")
TOKENIZER_DIR = MBART_ROOT / "MBart_trimmed"
LANG = "en_XX"

FRAME_SIZE = 256
CROP = 224
MAX_FRAMES = 128
JPEG_QUALITY = 80
FEATURE_DIM = 512
PREPROC_VERSION = "lmdb256-q80_centercrop224_imagenetnorm_resnet18-f37072fd_fp16_v1"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_labels(path) -> dict:
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


_JUNK = re.compile(r"^(blank|dash|\[music\]|music|\W*)\.?$", re.IGNORECASE)


def caption_exclusion_reason(text: str) -> str | None:
    """Return why a caption is excluded, or None if it is kept.

    Only captions carrying no translatable content are excluded: empty strings,
    subtitle placeholders ('blank', 'dash', '[music]') and pure punctuation.
    """
    t = " ".join(str(text).split())
    if not t:
        return "empty"
    if _JUNK.fullmatch(t):
        return "placeholder"
    return None


def read_lmdb_frames(sample_dir) -> np.ndarray:
    """Return all stored frames of one per-video LMDB as uint8 [T, 256, 256, 3] RGB, in order."""
    env = lmdb.open(str(sample_dir), readonly=True, lock=False, readahead=False, meminit=False)
    try:
        with env.begin(write=False) as txn:
            protocol = pickle.loads(txn.get(b"protocol"))
            keys = pickle.loads(txn.get(pickle.dumps("keys", protocol=protocol)))
            frames = [cv2.imdecode(np.frombuffer(txn.get(k), np.uint8), cv2.IMREAD_COLOR) for k in keys]
    finally:
        env.close()
    if not frames:
        raise ValueError(f"No frames in {sample_dir}")
    arr = np.stack(frames)[..., ::-1]  # BGR -> RGB
    if arr.shape[1:3] != (FRAME_SIZE, FRAME_SIZE):
        raise ValueError(f"Unexpected frame size {arr.shape} in {sample_dir}")
    return np.ascontiguousarray(arr)


def center_crop(frames: np.ndarray) -> np.ndarray:
    o = (FRAME_SIZE - CROP) // 2
    return frames[:, o:o + CROP, o:o + CROP]


def video_to_lmdb_equivalent_frames(video_path, max_frames: int = MAX_FRAMES) -> tuple[np.ndarray, int]:
    """Decode an MP4 exactly as scripts/prepare_isign_lmdb.py did (uniform sampling,
    bilinear 256x256 resize, JPEG q80 round trip). Returns (uint8 [T,256,256,3] RGB, original_frame_count)."""
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    raw = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw.append(frame)
    cap.release()
    if not raw:
        raise ValueError(f"Video has no decodable frames: {video_path}")
    n = len(raw)
    idx = np.linspace(0, n - 1, min(n, max_frames), dtype=np.int64)
    out = []
    for i in idx:
        img = Image.fromarray(cv2.cvtColor(raw[i], cv2.COLOR_BGR2RGB)).resize((FRAME_SIZE, FRAME_SIZE), Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
        dec = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        out.append(dec[..., ::-1])
    return np.ascontiguousarray(np.stack(out)), n


class FrameEncoder(torch.nn.Module):
    """Frozen ImageNet ResNet18 producing 512-d per-frame features from uint8 crops."""

    def __init__(self):
        super().__init__()
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = torch.nn.Identity()
        self.net = net.eval()
        for p in self.net.parameters():
            p.requires_grad = False
        self.register_buffer("mean", MEAN)
        self.register_buffer("std", STD)

    @torch.no_grad()
    def forward(self, frames_uint8: torch.Tensor) -> torch.Tensor:
        """frames_uint8: [T, 224, 224, 3] uint8 on device -> [T, 512] float16."""
        x = frames_uint8.permute(0, 3, 1, 2).float().div_(255)
        x = ((x - self.mean) / self.std).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16):
            return self.net(x).half()
