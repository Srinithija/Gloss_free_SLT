"""Sharded, resumable, memory-mapped per-frame caches shared by the SigLIP and MediaPipe extractors.

A cache directory holds shard_XXXXX.json (uids, offsets, lengths, extra per-clip info) plus one
.npy per array name with all frames of the shard concatenated along axis 0. A shard is written to
*.tmp names first and renamed only when complete, so an interrupted run leaves no partial shard and
restarting skips exactly the finished shards.
"""
import json
import os
from pathlib import Path

import numpy as np


def shard_name(i):
    return f"shard_{i:05d}"


def shard_done(root, i):
    return (Path(root) / f"{shard_name(i)}.json").exists()


def write_shard(root, i, uids, arrays_per_clip, extra=None):
    """arrays_per_clip: {name: [array [L_c, ...] per clip]} with equal L_c across names."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lengths = [len(a) for a in next(iter(arrays_per_clip.values()))]
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(int).tolist()
    final = []
    for name, lst in arrays_per_clip.items():
        assert [len(a) for a in lst] == lengths, name
        tmp = root / f"{shard_name(i)}.{name}.tmp.npy"
        np.save(tmp, np.concatenate(lst, 0))
        final.append((tmp, root / f"{shard_name(i)}.{name}.npy"))
    meta = {"uids": list(uids), "offsets": offsets, "lengths": lengths, "arrays": list(arrays_per_clip),
            "extra": extra or {}}
    tmpj = root / f"{shard_name(i)}.json.tmp"
    tmpj.write_text(json.dumps(meta))
    for tmp, dst in final:
        os.replace(tmp, dst)
    os.replace(tmpj, root / f"{shard_name(i)}.json")  # the json marks the shard complete


class ShardStore:
    """uid -> per-frame arrays, memory-mapped lazily (safe to pickle into DataLoader workers)."""

    def __init__(self, root):
        self.root = Path(root)
        self.index, self.extra = {}, {}
        self._mm = {}
        for js in sorted(self.root.glob("shard_*[0-9].json")):
            m = json.loads(js.read_text())
            sid = js.stem
            for k, (u, o, n) in enumerate(zip(m["uids"], m["offsets"], m["lengths"])):
                self.index[u] = (sid, o, n)
                self.extra[u] = {key: v[k] for key, v in m["extra"].items()}

    def __getstate__(self):
        return {"root": self.root, "index": self.index, "extra": self.extra, "_mm": {}}

    def __contains__(self, uid):
        return uid in self.index

    def __len__(self):
        return len(self.index)

    def get(self, uid, name, positions=None):
        sid, o, n = self.index[uid]
        key = (sid, name)
        if key not in self._mm:
            self._mm[key] = np.load(self.root / f"{sid}.{name}.npy", mmap_mode="r")
        a = self._mm[key][o:o + n]
        return np.array(a if positions is None else a[positions])


class KeyframeView:
    """Read-only view of a cache holding ALL stored LMDB frames per clip (e.g. the old ResNet-18 cache, or a
    cache made by scripts/extract_resnet.py), exposing only the clip's keyframe positions, in order.
    Old-cache layout: <root>/<split>/shard_*.json {uids, offsets, lengths} + shard_*.npy [frames, D]."""

    def __init__(self, roots, keyframes):
        import json as _json
        self.root = roots
        self.index, self._paths, self._mm, self._kf = {}, {}, {}, {}
        for root in roots:
            for js in sorted(Path(root).glob("shard_*[0-9].json")):
                m = _json.loads(js.read_text())
                if "arrays" in m:  # new-format shard written by write_shard
                    path = str(js.with_name(f"{js.stem}.v0.npy"))
                else:
                    path = str(js.with_suffix(".npy"))
                for u, o, n in zip(m["uids"], m["offsets"], m["lengths"]):
                    if u in keyframes:
                        kf = np.asarray(keyframes[u]["pos"])
                        assert kf.max() < n, (u, n)
                        self.index[u] = (path, o, len(kf))
                        self._kf[u] = kf
        self.extra = {}

    def __getstate__(self):
        return {"root": self.root, "index": self.index, "_kf": self._kf, "_paths": {}, "_mm": {}, "extra": {}}

    def __contains__(self, uid):
        return uid in self.index

    def __len__(self):
        return len(self.index)

    def get(self, uid, name, positions=None):
        path, o, _ = self.index[uid]
        if path not in self._mm:
            self._mm[path] = np.load(path, mmap_mode="r")
        a = self._mm[path][o + self._kf[uid]]
        return np.array(a if positions is None else a[positions])
