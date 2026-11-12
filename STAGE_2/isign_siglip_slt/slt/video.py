"""Read-only access to the per-clip LMDBs and the original MP4s (numpy/cv2/lmdb only, usable in both envs)."""
import pickle

import cv2
import lmdb
import numpy as np

from slt.paths import LMDB_ROOT, VIDEOS


def read_lmdb_clip(split, uid, positions=None):
    """Return (frames uint8 [N, 256, 256, 3] RGB, metadata dict). positions: indices into the stored frames."""
    env = lmdb.open(str(LMDB_ROOT / split / uid), readonly=True, lock=False, readahead=False, meminit=False)
    try:
        with env.begin() as t:
            proto = pickle.loads(t.get(b"protocol"))
            keys = pickle.loads(t.get(pickle.dumps("keys", protocol=proto)))
            meta = pickle.loads(t.get(b"metadata"))
            if positions is None:
                positions = range(len(keys))
            frames = [cv2.imdecode(np.frombuffer(t.get(keys[p]), np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
                      for p in positions]
    finally:
        env.close()
    return np.ascontiguousarray(np.stack(frames)), meta


def read_lmdb_meta(split, uid):
    env = lmdb.open(str(LMDB_ROOT / split / uid), readonly=True, lock=False, readahead=False, meminit=False)
    try:
        with env.begin() as t:
            return pickle.loads(t.get(b"metadata"))
    finally:
        env.close()


def decode_video_frames(uid, frame_indices, video_path=None):
    """Decode the original MP4 sequentially (as the LMDB builder did; no seeking) and return the
    requested original frame indices as uint8 RGB at native resolution, in the given order."""
    want = sorted(set(int(i) for i in frame_indices))
    cap = cv2.VideoCapture(str(video_path or VIDEOS / f"{uid}.mp4"))
    if not cap.isOpened():
        raise ValueError(f"cannot open video for {uid}")
    got, i, j = {}, 0, 0
    try:
        while j < len(want):
            ok, frame = cap.read()
            if not ok:
                break
            if i == want[j]:
                got[i] = frame[..., ::-1].copy()
                j += 1
            i += 1
    finally:
        cap.release()
    missing = [k for k in want if k not in got]
    if missing:
        raise ValueError(f"{uid}: {len(missing)} requested frames beyond decodable length {i}")
    return np.stack([got[int(k)] for k in frame_indices])


def trim_positions(n_stored, trim=0.05):
    """Stored-frame positions kept after dropping `trim` of the frames at each end (trim_5)."""
    k = int(np.floor(trim * n_stored))
    return np.arange(k, n_stored - k)


def subsample(n, T, offset=0.5):
    """Evenly spaced T positions out of n (all n if n <= T). offset in [0,1) shifts within each bin;
    0.5 = bin centres (deterministic eval), other values give temporally shifted training views."""
    if n <= T:
        return np.arange(n)
    return np.minimum(((np.arange(T) + offset) * n / T).astype(int), n - 1)
