"""Datasets over the memory-mapped feature caches (nothing is loaded fully into RAM)."""
import csv
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from slt.landmarks import FEATURE_DIM, frame_features
from slt.paths import MANIFESTS, SPLITS
from slt.shards import ShardStore
from slt.text import exclusion_reason
from slt.video import subsample


def load_records(split, uids=None, drop_flagged=False):
    """uid -> caption for kept captions. Returns (records, excluded {uid: reason})."""
    flagged = {r["uid"] for r in csv.DictReader(open(MANIFESTS / "flag_small_resolution.csv", encoding="utf-8"))}
    want = set(uids) if uids is not None else None
    recs, exc = {}, {}
    for r in csv.DictReader(open(SPLITS / f"{split}.csv", encoding="utf-8")):
        u = r["uid"]
        if want is not None and u not in want:
            continue
        why = exclusion_reason(r["text"]) or ("flagged_small_resolution" if drop_flagged and u in flagged else None)
        if why:
            exc[u] = why
        else:
            recs[u] = " ".join(r["text"].split())
    return recs, exc


class ClipDataset(Dataset):
    def __init__(self, records, vocab, vis_store=None, lm_store=None, T=64, train=False, n_views=1, max_label_len=64,
                 frame_keep_min=1.0):
        self.uids = sorted(records)
        self.records, self.vocab = records, vocab
        self.vis, self.lm = vis_store, lm_store
        self.T, self.train, self.n_views, self.max_label_len = T, train, n_views, max_label_len
        self.frame_keep_min = frame_keep_min
        for s in (vis_store, lm_store):
            if s is not None:
                missing = [u for u in self.uids if u not in s]
                if missing:
                    raise KeyError(f"{len(missing)} clips missing from {s.root} (e.g. {missing[:3]})")

    def __len__(self):
        return len(self.uids)

    def n_frames(self, uid):
        s = self.vis if self.vis is not None else self.lm
        return s.index[uid][2]

    def __getitem__(self, i):
        uid = self.uids[i]
        L = self.n_frames(uid)
        offset = random.random() if self.train else 0.5
        pos = subsample(L, self.T, offset)
        if self.train and self.frame_keep_min < 1.0 and len(pos) > 4:
            # temporal frame dropping (as in the ResNet baseline): keep an ordered random 80-100% of frames
            keep = max(4, int(round(random.uniform(self.frame_keep_min, 1.0) * len(pos))))
            pos = np.sort(np.asarray(random.sample(list(pos), keep)))  # python RNG: reseeded per worker by torch
        item = {"uid": uid, "labels": self.vocab.encode(self.records[uid], self.max_label_len)}
        if self.vis is not None:
            view = random.randrange(self.n_views) if self.train else 0
            item["vis"] = self.vis.get(uid, f"v{view}", pos)
        if self.lm is not None:
            ex = self.lm.extra[uid]
            feats = frame_features(self.lm.get(uid, "pose").astype(np.float32), self.lm.get(uid, "lh").astype(np.float32),
                                   self.lm.get(uid, "rh").astype(np.float32), ex["width"], ex["height"])
            item["lm"] = feats[pos]
        return item


class Collate:
    """Picklable (Windows spawn) batch collation: pads frames (mask) and labels."""

    def __init__(self, pad, lang):
        self.pad, self.lang = pad, lang

    def __call__(self, items):
        B = len(items)
        T = max(len(it.get("vis", it.get("lm"))) for it in items)
        mask = torch.zeros(B, T, dtype=torch.bool)
        vis = torch.zeros(B, T, items[0]["vis"].shape[1], dtype=torch.float16) if "vis" in items[0] else None
        lm = torch.zeros(B, T, FEATURE_DIM) if "lm" in items[0] else None
        Lmax = max(len(it["labels"]) for it in items)
        labels = torch.full((B, Lmax), -100, dtype=torch.long)
        dec = torch.full((B, Lmax), self.pad, dtype=torch.long)
        for b, it in enumerate(items):
            n = len(it.get("vis", it.get("lm")))
            mask[b, :n] = True
            if vis is not None:
                vis[b, :n] = torch.from_numpy(np.asarray(it["vis"]))
            if lm is not None:
                lm[b, :n] = torch.from_numpy(it["lm"])
            y = torch.tensor(it["labels"])
            labels[b, :len(y)] = y
            dec[b, 0] = self.lang
            dec[b, 1:len(y)] = y[:-1]
        return {"uids": [it["uid"] for it in items], "vis": vis, "lm": lm, "mask": mask, "labels": labels,
                "decoder_input_ids": dec}


def make_collate(vocab):
    return Collate(vocab.pad, vocab.lang)


class LengthBucketSampler(Sampler):
    """Shuffled batches of similar caption length; every index exactly once per epoch."""

    def __init__(self, lengths, batch_size, seed=0, bucket=50):
        self.lengths, self.bs, self.seed, self.bucket, self.epoch = np.asarray(lengths), batch_size, seed, bucket, 0

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        idx = rng.permutation(len(self.lengths))
        chunks = [idx[i:i + self.bs * self.bucket] for i in range(0, len(idx), self.bs * self.bucket)]
        batches = []
        for c in chunks:
            c = c[np.argsort(self.lengths[c], kind="stable")]
            batches += [c[i:i + self.bs].tolist() for i in range(0, len(c), self.bs)]
        rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        n, step = len(self.lengths), self.bs * self.bucket
        return sum((min(step, n - i) + self.bs - 1) // self.bs for i in range(0, n, step))
