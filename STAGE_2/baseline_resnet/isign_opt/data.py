"""Cached-feature datasets, length-bucketed batching and caption filtering."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from isign_opt.common import EXP_ROOT, FULL_LMDB, SUBSET_LMDB, caption_exclusion_reason, load_labels
from isign_opt.model import pad_batch

CACHE = EXP_ROOT / "cache" / "resnet18_frames"


class FeatureStore:
    """uid -> [L, 512] float16 frame features, memory-mapped from completed cache shards."""

    def __init__(self, split: str, cache_root: Path = CACHE):
        self.index, self.paths, self._arrays = {}, [], None
        for js in sorted((cache_root / split).glob("shard_*[0-9].json")):
            meta = json.loads(js.read_text())
            a = len(self.paths)
            self.paths.append(str(js.with_suffix(".npy")))
            for u, o, n in zip(meta["uids"], meta["offsets"], meta["lengths"]):
                self.index[u] = (a, o, n)

    def __getstate__(self):
        # only paths and the index cross process boundaries; each DataLoader worker reopens the memmaps
        return {"index": self.index, "paths": self.paths, "_arrays": None}

    def __contains__(self, uid):
        return uid in self.index

    def get(self, uid) -> np.ndarray:
        if self._arrays is None:
            self._arrays = [np.load(p, mmap_mode="r") for p in self.paths]
        a, o, n = self.index[uid]
        return np.array(self._arrays[a][o:o + n])


def build_split(split: str, source: str):
    """Return (kept_records, excluded) for a split. source: 'subset' (16k/2k/2k) or 'full'."""
    root = SUBSET_LMDB if source == "subset" else FULL_LMDB
    labels = load_labels(root / "labels" / f"labels.{split}")
    kept, excluded = {}, {}
    for uid in sorted(labels):
        reason = caption_exclusion_reason(labels[uid]["text"])
        if reason:
            excluded[uid] = {"text": labels[uid]["text"], "reason": reason}
        else:
            kept[uid] = " ".join(str(labels[uid]["text"]).split())
    return kept, excluded


class FeatureCaptionDataset(Dataset):
    def __init__(self, records: dict, store: FeatureStore, train: bool, keep_min: float = 0.8):
        missing = [u for u in records if u not in store]
        if missing:
            raise RuntimeError(f"{len(missing)} uids have no cached features (first: {missing[:3]})")
        self.uids = list(records)
        self.texts = [records[u] for u in self.uids]
        self.lengths = [store.index[u][2] for u in self.uids]
        self.store, self.train, self.keep_min = store, train, keep_min

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        f = self.store.get(self.uids[i])
        if self.train and len(f) > 16:
            # temporal augmentation: keep a random ordered subset of 80-100% of frames
            keep = max(16, int(len(f) * random.uniform(self.keep_min, 1.0)))
            f = f[np.sort(np.random.choice(len(f), keep, replace=False))]
        return self.uids[i], torch.from_numpy(f), self.texts[i]


class BucketBatchSampler(Sampler):
    """Batches of similar frame length; order of buckets and batches shuffled each epoch."""

    def __init__(self, lengths, batch_size, shuffle=True, seed=0, bucket_mult=50):
        self.lengths, self.bs, self.shuffle, self.seed, self.mult, self.epoch = lengths, batch_size, shuffle, seed, bucket_mult, 0

    def set_epoch(self, e):
        self.epoch = e

    def batches(self):
        rng = random.Random(self.seed + self.epoch)
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(idx)
        chunk = self.bs * self.mult
        out = []
        for c in range(0, len(idx), chunk):
            part = sorted(idx[c:c + chunk], key=lambda i: self.lengths[i])
            out += [part[b:b + self.bs] for b in range(0, len(part), self.bs)]
        if self.shuffle:
            rng.shuffle(out)
        return out

    def __iter__(self):
        return iter(self.batches())

    def __len__(self):
        return (len(self.lengths) + self.bs - 1) // self.bs


class Collate:
    def __init__(self, tokenizer, max_tokens=64):
        self.tok, self.max_tokens = tokenizer, max_tokens

    def __call__(self, batch):
        uids, feats, texts = zip(*batch)
        x, lens = pad_batch(list(feats))
        t = self.tok(text_target=list(texts), return_tensors="pt", padding=True, truncation=True, max_length=self.max_tokens)
        return {"uids": list(uids), "feats": x, "lengths": lens, "labels": t["input_ids"], "label_mask": t["attention_mask"], "texts": list(texts)}
