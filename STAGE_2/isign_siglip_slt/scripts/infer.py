"""Single clip -> English sentence (beam search). Input: an MP4 path or an LMDB key "split/uid".

  envs/main/Scripts/python.exe scripts/infer.py --run runs/<selected> --video clip.mp4
  envs/main/Scripts/python.exe scripts/infer.py --run runs/<selected> --lmdb_key dev/0fb8102d6c95-18

Preprocessing reproduces the training pipeline exactly:
  MP4: uniform <=128 frames (np.linspace over the decoded frame count), PIL bilinear 256x256, JPEG q80
       (= scripts/prepare_isign_lmdb.py), then trim_5 -> T evenly spaced positions (bin centres).
  Visual: same tensor path as scripts/extract_visual.py (bicubic 224, encoder mean/std, pooler_output, fp16).
  Landmarks: MediaPipe Holistic in envs/mediapipe (subprocess) on the ORIGINAL frames at the same indices.
"""
import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate import load_model  # noqa: E402
from slt.encoders import MODELS, load_vision_encoder  # noqa: E402
from slt.landmarks import frame_features  # noqa: E402
from slt.paths import PROJECT, VIDEOS  # noqa: E402
from slt.video import read_lmdb_clip, subsample, trim_positions  # noqa: E402

MP_PY = PROJECT / "envs" / "mediapipe" / "Scripts" / "python.exe"


def video_like_lmdb(path, max_frames=128):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    if n == 0:
        raise SystemExit(f"no decodable frames: {path}")
    sel = np.linspace(0, n - 1, min(n, max_frames), dtype=np.int64)
    from slt.video import decode_video_frames
    raw = decode_video_frames(None, sel, video_path=path)
    out = []
    for f in raw:
        buf = io.BytesIO()
        Image.fromarray(f).resize((256, 256), Image.Resampling.BILINEAR).save(buf, format="JPEG", quality=80)
        out.append(np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB")))
    return np.stack(out), sel


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--video")
    g.add_argument("--lmdb_key")
    ap.add_argument("--run", required=True)
    ap.add_argument("--decoding", default=None, help='json, e.g. {"num_beams":4,"length_penalty":1.0,"no_repeat_ngram_size":3}')
    a = ap.parse_args()
    t0 = time.time()
    model, vocab, cfg, _ = load_model(a.run)
    dec = json.loads(a.decoding) if a.decoding else json.loads((Path(a.run) / "selected_decoding.json").read_text())

    if a.video:
        frames, sel = video_like_lmdb(a.video)
        video_path = a.video
    else:
        split, uid = a.lmdb_key.split("/", 1)
        frames, meta = read_lmdb_clip(split, uid)
        sel = np.asarray(meta["selected_indices"])
        video_path = str(VIDEOS / f"{uid}.mp4")
    pos = trim_positions(len(frames))[subsample(len(trim_positions(len(frames))), cfg["T"], 0.5)]

    vis = lm = None
    if cfg["use_visual"]:
        enc, mean, std, _ = load_vision_encoder("siglip", "cuda",
                                                json.loads((PROJECT / "configs/model_revisions.json").read_text())[MODELS["siglip"]])
        x = torch.from_numpy(frames[pos]).cuda().permute(0, 3, 1, 2).float().div_(255)
        x = F.interpolate(x, size=(224, 224), mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
        with torch.autocast("cuda", dtype=torch.float16):
            v = enc(pixel_values=(x - mean) / std).pooler_output.float()
        vis = v.half().float()[None]
    if cfg["use_landmarks"]:
        all_pos = trim_positions(len(frames))
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "lm.npz"
            env = dict(os.environ, PYTHONNOUSERSITE="1", GLOG_minloglevel="2")
            subprocess.run([str(MP_PY), str(PROJECT / "scripts/mp_frames_worker.py"), video_path, str(out)]
                           + [str(int(i)) for i in sel[all_pos]], check=True, env=env, capture_output=True)
            with np.load(out) as z:
                feats = frame_features(z["pose"].astype(np.float32), z["lh"].astype(np.float32), z["rh"].astype(np.float32),
                                       int(z["width"]), int(z["height"]))
        lm = torch.from_numpy(feats[subsample(len(all_pos), cfg["T"], 0.5)])[None].cuda()
    mask = torch.ones(1, len(pos), dtype=torch.bool, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(vis.cuda() if vis is not None else None, lm, mask, **dec)
    print(json.dumps({"input": a.video or a.lmdb_key, "translation": vocab.decode(out[0, 1:].tolist()),
                      "frames_used": int(len(pos)), "decoding": dec, "seconds": round(time.time() - t0, 2)}))


if __name__ == "__main__":
    main()
