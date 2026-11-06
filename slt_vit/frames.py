"""Frame preprocessing shared by caching, training, evaluation and video inference.

Every frame goes through ``canonicalize``: frames wider than 4:3 are centre-cropped to 4:3 (the signer stands
in the middle of the 1920x1080 recordings), then padded with black to a square and resized. Nothing is
stretched, so hand and body proportions are kept. A clip is resampled to a fixed number of frames spread
uniformly over its whole duration, so frame folders (6-25 fps) and raw 25 fps videos share one timeline.
"""
import os

import cv2
import numpy as np
import torch

MAX_ASPECT = 4 / 3
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v')


def canonicalize(img_bgr, size):
    """BGR image of any shape -> RGB uint8 [size, size, 3]."""
    if img_bgr is None or img_bgr.ndim != 3 or min(img_bgr.shape[:2]) < 8:
        raise ValueError('invalid or empty image')
    h, w = img_bgr.shape[:2]
    if w > MAX_ASPECT * h:
        cw = int(round(MAX_ASPECT * h))
        x0 = (w - cw) // 2
        img_bgr, w = img_bgr[:, x0:x0 + cw], cw
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    img_bgr = cv2.copyMakeBorder(img_bgr, top, side - h - top, left, side - w - left, cv2.BORDER_CONSTANT, value=0)
    img_bgr = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def load_frame_folder(folder, files, size):
    out = []
    for f in files:
        img = cv2.imread(os.path.join(folder, f))
        if img is None:
            raise ValueError(f'unreadable frame {os.path.join(folder, f)}')
        out.append(canonicalize(img, size))
    return np.stack(out)


def read_video(path, size, max_frames=2000):
    """Decode a video -> (canonical uint8 frames [N,S,S,3], info). Raises ValueError with a readable reason."""
    if not os.path.isfile(path):
        raise ValueError(f'file not found: {path}')
    if not path.lower().endswith(VIDEO_EXTENSIONS):
        raise ValueError(f'unsupported file type {os.path.splitext(path)[1]!r}; use one of {", ".join(VIDEO_EXTENSIONS)}')
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError('could not open the video (corrupt file or unsupported codec)')
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames, shape = [], None
    while len(frames) < max_frames:
        ok, img = cap.read()
        if not ok:
            break
        shape = img.shape
        frames.append(canonicalize(img, size))
    cap.release()
    if len(frames) < 2:
        raise ValueError(f'the video has {len(frames)} decodable frame(s); at least 2 are needed')
    return np.stack(frames), {'decoded_frames': len(frames), 'fps': fps, 'width': shape[1], 'height': shape[0],
                              'duration_s': len(frames) / fps if fps else None, 'truncated': len(frames) >= max_frames}


def sample_indices(n, num_frames, rng=None, jitter=0.0):
    """``num_frames`` indices spread uniformly over [0, n), in order. Deterministic without ``rng``; with it, the
    clip ends move by up to ``jitter`` of its length and each index moves within its own segment."""
    if rng is None:
        return np.round(np.linspace(0, n - 1, num_frames)).astype(np.int64)
    start = rng.uniform(-jitter, jitter) * n
    end = n - 1 + rng.uniform(-jitter, jitter) * n
    pos = np.linspace(start, end, num_frames) + rng.uniform(-0.5, 0.5, num_frames) * (end - start) / max(num_frames - 1, 1)
    return np.clip(np.round(np.sort(pos)), 0, n - 1).astype(np.int64)


def to_unit_float(frames_uint8, device=None):
    """uint8 [T,H,W,3] -> float [T,3,H,W] in [0, 1]."""
    return torch.as_tensor(frames_uint8).to(device).permute(0, 3, 1, 2).float().div_(255.0)
