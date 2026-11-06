"""Feature-store helpers for this experiment (read-only views over the existing caches).

PositionsView: a cache that stores many frame positions per clip (extra['positions'], e.g. the pilot caches with
all trimmed frames) exposed at the clip's K=32 keyframe positions only, so it matches cache/full_k32 exactly.
MergedStore: uid -> first store that has it.
"""
import json
from pathlib import Path

import numpy as np

from slt.shards import ShardStore


class PositionsView:
    def __init__(self, root, keyframes):
        self.base = ShardStore(root)
        self.root = self.base.root
        self.index, self.extra, self._rows = {}, {}, {}
        for u, (sid, o, n) in self.base.index.items():
            if u not in keyframes:
                continue
            ex = self.base.extra[u]
            where = {p: i for i, p in enumerate(ex["positions"])}
            kf = keyframes[u]["pos"]
            if not all(p in where for p in kf):
                continue  # keyframe outside the cached range: skip the clip rather than misalign
            self._rows[u] = np.asarray([where[p] for p in kf])
            self.index[u] = (sid, o, len(kf))
            self.extra[u] = dict(ex, positions=list(kf))

    def __contains__(self, u):
        return u in self.index

    def get(self, uid, name, positions=None):
        a = self.base.get(uid, name, self._rows[uid])
        return a if positions is None else a[positions]


class MergedStore:
    def __init__(self, stores):
        self.stores = stores
        self.root = [str(s.root) for s in stores]
        self.index, self.extra, self._owner = {}, {}, {}
        for k, s in enumerate(stores):
            for u in s.index:
                if u not in self.index:
                    self.index[u], self.extra[u], self._owner[u] = s.index[u], s.extra[u], k

    def __contains__(self, u):
        return u in self.index

    def get(self, uid, name, positions=None):
        return self.stores[self._owner[uid]].get(uid, name, positions)


def build(spec, keyframes_path):
    """spec: a path, or a list of {"root": path, "kind": "shard"|"positions"}."""
    if isinstance(spec, str):
        return ShardStore(spec)
    kf = json.loads(Path(keyframes_path).read_text())
    return MergedStore([ShardStore(s["root"]) if s["kind"] == "shard" else PositionsView(s["root"], kf) for s in spec])
