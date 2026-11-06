"""Resumable frozen-ResNet18 feature cache for iSign LMDB videos.

Output layout (per split):  <cache_root>/<split>/shard_XXXXX.npy   float16 [frames_in_shard, 512]
                            <cache_root>/<split>/shard_XXXXX.json  {"uids": [...], "offsets": [...], "lengths": [...]}
A shard is written to a temporary name and renamed only when complete, so reruns skip finished shards.

Usage: python -m isign_opt.cache_features --split train --order subset_first [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from isign_opt.common import (EXP_ROOT, FULL_LMDB, PREPROC_VERSION, SUBSET_LMDB, FrameEncoder,
                              center_crop, load_labels, read_lmdb_frames)

SHARD_SIZE = int(os.environ.get("ISIGN_SHARD_SIZE", 1000))


class LmdbVideos(Dataset):
    def __init__(self, split, uids):
        self.split, self.uids = split, uids

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        uid = self.uids[i]
        frames = center_crop(read_lmdb_frames(FULL_LMDB / self.split / uid))
        return uid, torch.from_numpy(np.ascontiguousarray(frames))


def ordered_uids(split, order):
    full = sorted(load_labels(FULL_LMDB / "labels" / f"labels.{split}"))
    subset = sorted(load_labels(SUBSET_LMDB / "labels" / f"labels.{split}"))
    if order == "subset_only":
        return subset
    sset = set(subset)
    return subset + [u for u in full if u not in sset]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--order", default="subset_first", choices=["subset_first", "subset_only"])
    ap.add_argument("--limit", type=int, default=None, help="cache only the first N uids of the ordering")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cache_root", type=Path, default=EXP_ROOT / "cache" / "resnet18_frames")
    args = ap.parse_args()

    uids = ordered_uids(args.split, args.order)
    if args.limit:
        uids = uids[: args.limit]
    out = args.cache_root / args.split
    out.mkdir(parents=True, exist_ok=True)
    meta_path = args.cache_root / "cache_meta.json"
    meta = {"preprocessing_version": PREPROC_VERSION, "source_lmdb": str(FULL_LMDB), "shard_size": SHARD_SIZE,
            "ordering": "sorted subset-16k uids first, then remaining sorted full-split uids"}
    if meta_path.exists():
        if json.loads(meta_path.read_text())["preprocessing_version"] != PREPROC_VERSION:
            raise RuntimeError("Existing cache was built with a different preprocessing version")
    else:
        meta_path.write_text(json.dumps(meta, indent=1))

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    enc = FrameEncoder().to(device).to(memory_format=torch.channels_last)
    shards = [uids[i:i + SHARD_SIZE] for i in range(0, len(uids), SHARD_SIZE)]
    todo = []
    for si, shard_uids in enumerate(shards):
        final_json = out / f"shard_{si:05d}.json"
        if final_json.exists():
            if json.loads(final_json.read_text())["uids"] != shard_uids:
                raise RuntimeError(f"Existing {final_json} holds different uids; refusing to mix caches")
        else:
            todo.append(si)
    # one loader over all pending videos, so worker start-up is paid once
    pending = [u for si in todo for u in shards[si]]
    loader = iter(DataLoader(LmdbVideos(args.split, pending), batch_size=None, num_workers=args.workers,
                             pin_memory=True, prefetch_factor=4 if args.workers else None))
    t0, done = time.time(), 0
    for si in todo:
        shard_uids = shards[si]
        final_npy, final_json = out / f"shard_{si:05d}.npy", out / f"shard_{si:05d}.json"
        feats, offsets, lengths, got, off = [], [], [], [], 0
        for _ in shard_uids:
            uid, frames = next(loader)
            f = enc(frames.to(device, non_blocking=True)).cpu().numpy()
            feats.append(f); got.append(uid); offsets.append(off); lengths.append(len(f)); off += len(f)
        if got != shard_uids:
            raise RuntimeError(f"Loader returned uids out of order for shard {si}")
        tmp_npy = out / f"shard_{si:05d}.tmp.npy"
        np.save(tmp_npy, np.concatenate(feats).astype(np.float16))
        os.replace(tmp_npy, final_npy)
        tmp_json = out / f"shard_{si:05d}.tmp.json"
        tmp_json.write_text(json.dumps({"uids": got, "offsets": offsets, "lengths": lengths}))
        os.replace(tmp_json, final_json)
        done += len(got)
        el = time.time() - t0
        print(f"[{args.split}] shard {si + 1}/{len(shards)} written ({len(todo)} pending at start); {done} videos in {el / 60:.1f} min "
              f"({el / done:.3f} s/video)", flush=True)
    print(f"[{args.split}] cache complete: {len(uids)} videos in {out}", flush=True)


if __name__ == "__main__":
    main()
