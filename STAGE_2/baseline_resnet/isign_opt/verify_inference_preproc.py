"""Check that raw-MP4 inference preprocessing reproduces the cached training features (held-out test videos)."""
import json, random
import torch
from isign_opt.common import EXP_ROOT, FrameEncoder, center_crop, video_to_lmdb_equivalent_frames
from isign_opt.data import FeatureStore
VID = r"E:\FINAL YEAR PROJECT\GLOSS FREE SLT\dataset\iSign\videos\iSign-videos_v1.1"
store = FeatureStore("test"); enc = FrameEncoder().cuda(); random.seed(5); out = []
for u in random.sample(sorted(store.index), 8):
    fr, n = video_to_lmdb_equivalent_frames(f"{VID}\{u}.mp4")
    f = enc(torch.from_numpy(center_crop(fr).copy()).cuda()).float().cpu(); c = torch.from_numpy(store.get(u)).float()
    cos = torch.nn.functional.cosine_similarity(f, c, dim=1) if f.shape == c.shape else torch.tensor([float("nan")])
    out.append({"uid": u, "orig_frames": n, "inference_frames": len(f), "cached_frames": len(c), "min_cos": float(cos.min()), "mean_cos": float(cos.mean())})
    print(out[-1], flush=True)
ok = all(o["inference_frames"] == o["cached_frames"] and o["min_cos"] > 0.98 for o in out)
json.dump({"pass": ok, "samples": out}, open(EXP_ROOT / "reports" / "inference_preprocessing_check.json", "w"), indent=1)
print("PASS" if ok else "FAIL")
