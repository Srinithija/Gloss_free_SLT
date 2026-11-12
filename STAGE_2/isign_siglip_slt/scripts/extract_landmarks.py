"""MediaPipe Holistic landmarks on the original MP4 frames at the LMDB's stored frame indices.

Run in envs/mediapipe with PYTHONNOUSERSITE=1:
  python scripts/extract_landmarks.py --manifest manifests/pilot_clips.csv --out cache/landmarks_v1 --workers 12

Frames per clip = trim_5 of the stored LMDB frames (<=116), original index = metadata.selected_indices.
Holistic(static_image_mode=True, model_complexity=1) on CPU, one instance per worker process.
Stored raw (fp16, NaN = not detected): pose [L,33,4] (x,y,z,visibility), lh [L,21,3], rh [L,21,3].
Resumable by shard; output is independent of worker count and of interruptions.
"""
import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slt.shards import shard_done, write_shard  # noqa: E402
from slt.video import decode_video_frames, read_lmdb_meta, trim_positions  # noqa: E402

_MP = None


def _init():
    global _MP
    os.environ.setdefault("GLOG_minloglevel", "2")
    import mediapipe as mp
    _MP = mp


def _lm(lms, n, with_vis):
    if lms is None:
        return np.full((n, 4 if with_vis else 3), np.nan, np.float32)
    if with_vis:
        return np.array([[p.x, p.y, p.z, p.visibility] for p in lms.landmark], np.float32)
    return np.array([[p.x, p.y, p.z] for p in lms.landmark], np.float32)


def process_clip(job):
    split, uid, pos, complexity = job
    t0 = time.perf_counter()
    meta = read_lmdb_meta(split, uid)
    sel = np.asarray(meta["selected_indices"])
    pos = trim_positions(len(sel)) if pos is None else np.asarray(pos)
    frames = decode_video_frames(uid, sel[pos])
    t1 = time.perf_counter()
    P, L, R = [], [], []
    # fresh instance per clip (~0.05 s): a clip's output cannot depend on which clips a worker saw before
    with _MP.solutions.holistic.Holistic(static_image_mode=True, model_complexity=complexity) as holistic:
        for f in frames:
            r = holistic.process(f)
            P.append(_lm(r.pose_landmarks, 33, True))
            L.append(_lm(r.left_hand_landmarks, 21, False))
            R.append(_lm(r.right_hand_landmarks, 21, False))
    h, w = frames.shape[1:3]
    return {"uid": uid, "pose": np.stack(P).astype(np.float16), "lh": np.stack(L).astype(np.float16),
            "rh": np.stack(R).astype(np.float16), "positions": pos.tolist(), "width": int(w), "height": int(h),
            "decode_s": t1 - t0, "mp_s": time.perf_counter() - t1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="csv with uid,split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--shard_size", type=int, default=500)
    ap.add_argument("--max_shards", type=int, default=0)
    ap.add_argument("--positions", default=None, help="keyframes json {uid: {pos: [...]}}; default all trim_5 frames")
    ap.add_argument("--complexity", type=int, default=1)
    a = ap.parse_args()
    kf = json.loads(Path(a.positions).read_text()) if a.positions else {}
    jobs = [(r["split"], r["uid"], kf[r["uid"]]["pos"] if kf else None, a.complexity)
            for r in csv.DictReader(open(a.manifest, encoding="utf-8"))]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "extract_log.txt", "a", encoding="utf-8")
    n_sh = (len(jobs) + a.shard_size - 1) // a.shard_size
    todo = [i for i in range(n_sh) if not shard_done(out, i)][: a.max_shards or None]
    print(f"{len(jobs)} clips, {n_sh} shards, {len(todo)} to do, workers={a.workers}", file=log, flush=True)
    t_start = time.time()
    with Pool(a.workers, initializer=_init) as pool:
        for i in todo:
            ts = time.time()
            chunk = jobs[i * a.shard_size:(i + 1) * a.shard_size]
            res = list(pool.imap(process_clip, chunk, chunksize=2))  # ordered => deterministic layout
            write_shard(out, i, [r["uid"] for r in res],
                        {"pose": [r["pose"] for r in res], "lh": [r["lh"] for r in res], "rh": [r["rh"] for r in res]},
                        extra={k: [r[k] for r in res] for k in ("positions", "width", "height")})
            nf = sum(len(r["positions"]) for r in res)
            dt = time.time() - ts
            stats = {"shard": i, "clips": len(res), "frames": nf, "wall_s": round(dt, 1), "fps": round(nf / dt, 1),
                     "decode_s_sum": round(sum(r["decode_s"] for r in res), 1),
                     "mp_s_sum": round(sum(r["mp_s"] for r in res), 1),
                     "lh_rate": round(float(np.mean(np.concatenate([~np.isnan(r["lh"][:, 0, 0]) for r in res]))), 4),
                     "rh_rate": round(float(np.mean(np.concatenate([~np.isnan(r["rh"][:, 0, 0]) for r in res]))), 4),
                     "pose_rate": round(float(np.mean(np.concatenate([~np.isnan(r["pose"][:, 0, 0]) for r in res]))), 4),
                     "elapsed_min": round((time.time() - t_start) / 60, 1)}
            print(json.dumps(stats), file=log, flush=True)
            print(json.dumps(stats), flush=True)
    print("done", file=log, flush=True)


if __name__ == "__main__":
    main()
