"""Video -> English inference with exactly the training preprocessing.

python -m isign_opt.infer path/to/video.mp4 [more.mp4 ...] [--ckpt runs/main_full/best.pt]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import MBartTokenizer

from isign_opt.common import EXP_ROOT, LANG, TOKENIZER_DIR, FrameEncoder, center_crop, video_to_lmdb_equivalent_frames
from isign_opt.model import FeatureSLT, pad_batch

DEFAULT_CKPT = EXP_ROOT / "reports" / "final_eval" / "selected_model.pt"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MIN_FRAMES = 8


class Translator:
    def __init__(self, ckpt=DEFAULT_CKPT, decoding=None, device="cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required for this model.")
        ckpt = Path(ckpt)
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        self.device = torch.device(device)
        self.tok = MBartTokenizer.from_pretrained(TOKENIZER_DIR, src_lang=LANG, tgt_lang=LANG)
        ck = torch.load(ckpt, map_location="cpu")
        self.model = FeatureSLT(pretrained=False).to(self.device)
        self.model.load_state_dict(ck["model"]); self.model.eval()
        self.encoder = FrameEncoder().to(self.device)
        self.ckpt, self.ckpt_epoch = ckpt, ck.get("epoch")
        if decoding is None:  # decoding chosen on dev during final evaluation (stored with the selected model)
            decoding = ck.get("decoding", "beam4")
        from isign_opt.evaluate import DECODE_CANDIDATES
        self.decoding = decoding
        self.num_beams, self.gen_kw = DECODE_CANDIDATES[decoding]["num_beams"], DECODE_CANDIDATES[decoding]["gen_kw"]

    def validate(self, path):
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Video not found: {p}")
        if p.suffix.lower() not in VIDEO_EXTS:
            raise ValueError(f"Unsupported file type '{p.suffix}'. Supported: {sorted(VIDEO_EXTS)}")
        if p.stat().st_size == 0:
            raise ValueError("The video file is empty.")
        return p

    @torch.no_grad()
    def translate(self, path, return_frames=False) -> dict:
        p = self.validate(path)
        t0 = time.time()
        frames, n_orig = video_to_lmdb_equivalent_frames(p)
        if len(frames) < MIN_FRAMES:
            raise ValueError(f"Video too short: {len(frames)} frames (need at least {MIN_FRAMES}).")
        t1 = time.time()
        crops = torch.from_numpy(center_crop(frames).copy()).to(self.device)
        feats = self.encoder(crops).cpu()
        x, lens = pad_batch([feats])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model.generate(x.to(self.device), lens, self.tok.lang_code_to_id[LANG],
                                      num_beams=self.num_beams, max_new_tokens=60, **self.gen_kw)
        torch.cuda.synchronize(); t2 = time.time()
        res = {"video": str(p), "prediction": self.tok.batch_decode(out, skip_special_tokens=True)[0].strip(),
               "original_frames": n_orig, "sampled_frames": len(frames), "decoding": self.decoding,
               "decode_seconds": round(t1 - t0, 3), "model_seconds": round(t2 - t1, 3)}
        if return_frames:
            res["frames"] = frames
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--decoding", default=None)
    a = ap.parse_args()
    tr = Translator(a.ckpt, a.decoding)
    for v in a.videos:
        try:
            print(json.dumps(tr.translate(v), ensure_ascii=False))
        except (ValueError, FileNotFoundError) as e:
            print(json.dumps({"video": v, "error": str(e)}))


if __name__ == "__main__":
    main()
