"""Frozen SigLIP / CLIP per-frame pooled features from the LMDB frames (trim_5 positions, <=116 per clip).

Run in envs/main:
  python scripts/extract_visual.py --manifest manifests/pilot_clips.csv --out cache/pilot/siglip_v1 --encoder siglip
  python scripts/extract_visual.py ... --aug_views 2   # adds augmented views for train clips

Preprocessing (documented contract, identical for every encoder):
  LMDB 256x256 RGB frame -> (no signer crop: audit found full-frame signers; frame is already square,
  so "centre-crop to <=4:3, pad to square" is the identity) -> resize 224x224 bicubic (antialias)
  -> /255 -> encoder's own mean/std (SigLIP 0.5/0.5; CLIP OpenAI stats) -> vision tower -> pooler_output [768].
Augmented view k (train only, seeded by (uid, k)): one random-resized-crop (scale 0.8-1.0, ratio 0.9-1.1),
rotation +-5 deg, shift +-4%, brightness/contrast/saturation jitter 0.2, hue 0.02 applied identically to all
frames of the clip. Temporal offsets are not baked in: every view keeps all trim_5 positions, and training
picks shifted subsets, so landmarks stay frame-aligned.
Stored as fp16 shards (slt/shards.py), arrays 'v0' (clean) and 'v1'..'vK' (augmented).
"""
import argparse
import csv
import hashlib
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slt.encoders import load_vision_encoder  # noqa: E402
from slt.shards import shard_done, write_shard  # noqa: E402
from slt.video import read_lmdb_clip, read_lmdb_meta, trim_positions  # noqa: E402


def clip_seed(uid, k):
    return int(hashlib.sha256(f"{uid}|{k}".encode()).hexdigest()[:8], 16)


def augment(x, seed):
    """x: [L,3,256,256] float 0..1 on GPU. Same random transform for every frame of the clip."""
    g = torch.Generator().manual_seed(seed)
    r = lambda lo, hi: float(lo + (hi - lo) * torch.rand(1, generator=g))
    L, _, H, W = x.shape
    scale, ratio = r(0.8, 1.0), r(0.9, 1.1)
    cw, ch = min(W, int(round(W * (scale * ratio) ** 0.5))), min(H, int(round(H * (scale / ratio) ** 0.5)))
    x0, y0 = int(r(0, W - cw + 1e-6)), int(r(0, H - ch + 1e-6))
    x = x[:, :, y0:y0 + ch, x0:x0 + cw]
    ang, tx, ty = np.deg2rad(r(-5, 5)), r(-0.04, 0.04) * 2, r(-0.04, 0.04) * 2
    theta = torch.tensor([[np.cos(ang), -np.sin(ang), tx], [np.sin(ang), np.cos(ang), ty]], dtype=x.dtype, device=x.device)
    grid = F.affine_grid(theta.expand(L, 2, 3), list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    b, c, s, hue = r(0.8, 1.2), r(0.8, 1.2), r(0.8, 1.2), r(-0.02, 0.02)
    x = x * b
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - m) * c + m
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = (x - gray) * s + gray
    x = _hue_shift(x, hue)
    return x.clamp(0, 1)


def _hue_shift(x, h):
    # rotate chroma in YIQ space by 2*pi*h
    a = 2 * np.pi * h
    yiq = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], dtype=x.dtype, device=x.device)
    rot = torch.tensor([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]], dtype=x.dtype, device=x.device)
    m = torch.linalg.inv(yiq) @ rot @ yiq
    return torch.einsum("ij,ljhw->lihw", m, x)


def loader(jobs, q):
    for split, uid, pos in jobs:
        if pos is None:
            pos = trim_positions(len(read_lmdb_meta(split, uid)["selected_indices"]))
        pos = np.asarray(pos)
        frames, _ = read_lmdb_clip(split, uid, pos)
        q.put((split, uid, pos, frames))
    q.put(None)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="siglip", choices=["siglip", "clip"])
    ap.add_argument("--aug_views", type=int, default=0)
    ap.add_argument("--shard_size", type=int, default=500)
    ap.add_argument("--max_shards", type=int, default=0)
    ap.add_argument("--batch_frames", type=int, default=256)
    ap.add_argument("--positions", default=None, help="keyframes json {uid: {pos: [...]}}; default all trim_5 frames")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda"
    rev = json.loads((Path(__file__).resolve().parents[1] / "configs/model_revisions.json").read_text())
    from slt.encoders import MODELS
    enc, mean, std, info = load_vision_encoder(a.encoder, dev, revision=rev[MODELS[a.encoder]])
    kf = json.loads(Path(a.positions).read_text()) if a.positions else {}
    jobs = [(r["split"], r["uid"], kf[r["uid"]]["pos"] if kf else None)
            for r in csv.DictReader(open(a.manifest, encoding="utf-8"))]
    if a.aug_views and any(s != "train" for s, _, _ in jobs):
        raise SystemExit("augmented views are for train clips only; pass a train-only manifest")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "encoder_info.json").write_text(json.dumps(info | {"aug_views": a.aug_views}, indent=1))
    log = open(out / "extract_log.txt", "a", encoding="utf-8")
    n_sh = (len(jobs) + a.shard_size - 1) // a.shard_size
    todo = [i for i in range(n_sh) if not shard_done(out, i)][: a.max_shards or None]
    t_start = time.time()

    def encode(x):
        x = F.interpolate(x, size=(224, 224), mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
        x = (x - mean) / std
        outs = []
        for s in range(0, len(x), a.batch_frames):
            with torch.autocast("cuda", dtype=torch.float16):
                outs.append(enc(pixel_values=x[s:s + a.batch_frames]).pooler_output.float())
        return torch.cat(outs).half().cpu().numpy()

    for i in todo:
        ts = time.time()
        chunk = jobs[i * a.shard_size:(i + 1) * a.shard_size]
        q = queue.Queue(maxsize=16)
        th = threading.Thread(target=loader, args=(chunk, q), daemon=True)
        th.start()
        uids, arrays, extra = [], {f"v{k}": [] for k in range(a.aug_views + 1)}, {"positions": [], "split": []}
        n_frames = 0
        while (item := q.get()) is not None:
            split, uid, pos, frames = item
            x = torch.from_numpy(frames).to(dev).permute(0, 3, 1, 2).float().div_(255)
            arrays["v0"].append(encode(x))
            for k in range(1, a.aug_views + 1):
                arrays[f"v{k}"].append(encode(augment(x, clip_seed(uid, k))))
            uids.append(uid)
            extra["positions"].append(pos.tolist())
            extra["split"].append(split)
            n_frames += len(pos)
        write_shard(out, i, uids, arrays, extra)
        dt = time.time() - ts
        stats = {"shard": i, "clips": len(uids), "frames": n_frames, "views": a.aug_views + 1, "wall_s": round(dt, 1),
                 "frames_per_s": round(n_frames * (a.aug_views + 1) / dt, 1),
                 "peak_vram_GiB": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                 "elapsed_min": round((time.time() - t_start) / 60, 1)}
        print(json.dumps(stats), file=log, flush=True)
        print(json.dumps(stats), flush=True)
    print("done", file=log, flush=True)


if __name__ == "__main__":
    main()
