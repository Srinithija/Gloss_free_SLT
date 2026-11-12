"""Motion-based "meaningful frame" selection, computed once from the LMDB (read-only), before any heavy extraction.

For each clip (stored LMDB frames, <=128):
  1. motion m[t] = mean |gray64(t) - gray64(t-1)|, using JPEG reduced decoding (64x64 grayscale, ~0.2 ms/frame);
     smoothed with a 5-frame moving average.
  2. active span: inside the trim_5 range, drop leading/trailing frames whose smoothed motion is below
     30% of the clip's 90th-percentile motion (the signer at rest before/after signing). The span keeps at
     least 50% of the trimmed range (centred on the motion mass) so a slow clip is never cut to nothing.
  3. K frames inside the span, spaced evenly in cumulative weight w = 0.5*uniform + 0.5*motion: denser where
     the hands move, but still covering holds (a pure motion weighting would skip static sign holds).
Output: manifests/keyframes_K{K}.json  {uid: {"pos": [...stored-frame indices...], "n": n_stored, "span": [a, b]}}
"""
import argparse
import csv
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import lmdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slt.paths import LMDB_ROOT, MANIFESTS  # noqa: E402

K = 32


def motion(split, uid):
    env = lmdb.open(str(LMDB_ROOT / split / uid), readonly=True, lock=False, readahead=False, meminit=False)
    try:
        with env.begin() as t:
            proto = pickle.loads(t.get(b"protocol"))
            keys = pickle.loads(t.get(pickle.dumps("keys", protocol=proto)))
            g = np.stack([cv2.imdecode(np.frombuffer(t.get(k), np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_4)
                          for k in keys]).astype(np.float32)
    finally:
        env.close()
    m = np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))
    return np.concatenate([[m[0] if len(m) else 0.0], m])


def select(m, k=K, trim=0.05, rest_frac=0.3, min_keep=0.5):
    n = len(m)
    c = int(np.floor(trim * n))
    lo, hi = c, n - c                       # trim_5 range [lo, hi)
    sm = np.convolve(m, np.ones(5) / 5, mode="same")[lo:hi]
    L = hi - lo
    thr = rest_frac * np.percentile(sm, 90) if L else 0
    act = np.flatnonzero(sm > thr)
    a, b = (act[0], act[-1] + 1) if len(act) else (0, L)
    need = int(np.ceil(min_keep * L))
    if b - a < need:                        # widen symmetrically around the span centre
        mid = (a + b) / 2
        a = int(max(0, min(L - need, round(mid - need / 2))))
        b = a + need
    span = np.arange(lo + a, lo + b)
    if len(span) <= k:
        return span.tolist(), [int(lo + a), int(lo + b)]
    w = 0.5 / len(span) + 0.5 * sm[a:b] / max(sm[a:b].sum(), 1e-6)
    cdf = np.cumsum(w) / w.sum()
    pos = np.searchsorted(cdf, (np.arange(k) + 0.5) / k)
    pos = np.unique(np.clip(pos, 0, len(span) - 1))
    if len(pos) < k:                        # fill collisions with evenly spaced unused frames
        rest = np.setdiff1d(np.arange(len(span)), pos)
        extra = rest[np.linspace(0, len(rest) - 1, k - len(pos)).astype(int)]
        pos = np.sort(np.concatenate([pos, extra]))
    return span[pos].tolist(), [int(lo + a), int(lo + b)]


def job(args):
    split, uid = args
    m = motion(split, uid)
    pos, span = select(m)
    return uid, {"pos": pos, "n": int(len(m)), "span": span, "motion_mean": round(float(m.mean()), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(MANIFESTS / "audit_clips.csv", encoding="utf-8")))
    jobs = [(r["split"], r["uid"]) for r in rows][: a.limit or None]
    t0 = time.time()
    out = {}
    with Pool(a.workers) as p:
        for i, (u, r) in enumerate(p.imap_unordered(job, jobs, chunksize=64), 1):
            out[u] = r
            if i % 20000 == 0:
                print(f"{i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    path = MANIFESTS / (f"keyframes_K{K}.json" if not a.limit else f"keyframes_K{K}_test{a.limit}.json")
    path.write_text(json.dumps(dict(sorted(out.items()))))
    kept = np.array([r["span"][1] - r["span"][0] for r in out.values()]) / np.array(
        [r["n"] - 2 * int(np.floor(0.05 * r["n"])) for r in out.values()])
    print(json.dumps({"clips": len(out), "seconds": round(time.time() - t0, 1),
                      "active_span_frac_of_trimmed": {q: round(float(np.percentile(kept, q)), 3) for q in (10, 50, 90)},
                      "clips_with_lt_K_frames": int(sum(len(r["pos"]) < K for r in out.values()))}))


if __name__ == "__main__":
    main()
