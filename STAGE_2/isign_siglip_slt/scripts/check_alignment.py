"""Verify that original-MP4 frame k = selected_indices[i] is the frame stored at LMDB position i.

For each of N clips (stratified over 300x300 / large / small, train+dev only) and up to 16 stored
positions, re-create the builder's 256x256 image from MP4 frames at offsets d in -2..2 and measure the
mean absolute pixel difference to the stored LMDB frame. Aligned => the minimum is at d = 0 and
near the JPEG-noise floor. Run in the mediapipe env (PIL, cv2, lmdb).
"""
import csv
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slt.paths import MANIFESTS, REPORTS  # noqa: E402
from slt.video import decode_video_frames, read_lmdb_clip  # noqa: E402

OFFSETS = (-2, -1, 0, 1, 2)


def builder_like(rgb):
    img = Image.fromarray(rgb).resize((256, 256), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def main(n_clips=20, seed=0):
    rows = [r for r in csv.DictReader(open(MANIFESTS / "audit_clips.csv", encoding="utf-8")) if r["split"] != "test"]
    grp = lambda r: ("300sq" if (r["video_w"], r["video_h"]) == ("300", "300")
                     else "small" if min(int(r["video_w"]), int(r["video_h"])) < 256 else "large")
    rng = random.Random(seed)
    picks = (rng.sample([r for r in rows if grp(r) == "300sq"], 12) + rng.sample([r for r in rows if grp(r) == "large"], 6)
             + rng.sample([r for r in rows if grp(r) == "small"], 2))[:n_clips]
    out = []
    for r in picks:
        lmdb_frames, meta = read_lmdb_clip(r["split"], r["uid"])
        sel = meta["selected_indices"]
        pos = np.unique(np.linspace(0, len(sel) - 1, 16).astype(int))
        n_video = int(meta["original_frame_count"])
        need = sorted({sel[p] + d for p in pos for d in OFFSETS if 0 <= sel[p] + d < n_video})
        vid = dict(zip(need, decode_video_frames(r["uid"], need)))
        per_pos = []
        for p in pos:
            ref = lmdb_frames[p].astype(np.int16)
            per_pos.append({d: float(np.abs(builder_like(vid[sel[p] + d]).astype(np.int16) - ref).mean())
                            for d in OFFSETS if sel[p] + d in vid})
        mad = {d: [m[d] for m in per_pos if d in m] for d in OFFSETS}
        mean = {d: round(float(np.mean(v)), 3) for d, v in mad.items() if v}
        per_frame_best0 = float(np.mean([min(m, key=m.get) == 0 for m in per_pos]))
        rec = {"uid": r["uid"], "split": r["split"], "group": grp(r), "size": f'{r["video_w"]}x{r["video_h"]}',
               "n_stored": len(sel), "meta_original_frames": n_video, "header_frames": int(r["video_frames"]),
               "mad_by_offset": mean, "best_offset": min(mean, key=mean.get),
               "frac_frames_best_at_0": round(per_frame_best0, 3), "mad_at_0_max": round(max(mad[0]), 3)}
        out.append(rec)
        print(json.dumps(rec), flush=True)
    s = {"n_clips": len(out), "all_best_offset_0": all(o["best_offset"] == 0 for o in out),
         "mean_frac_frames_best_at_0": round(float(np.mean([o["frac_frames_best_at_0"] for o in out])), 3),
         "mad_at_0_mean": round(float(np.mean([o["mad_by_offset"][0] for o in out])), 3),
         "mad_at_pm1_mean": round(float(np.mean([(o["mad_by_offset"].get(-1, np.nan) + o["mad_by_offset"].get(1, np.nan)) / 2
                                                for o in out])), 3),
         "clips": out}
    (REPORTS / "audit" / "alignment_check.json").write_text(json.dumps(s, indent=1))
    print({k: v for k, v in s.items() if k != "clips"})


if __name__ == "__main__":
    main()
