"""Control: dev loss (no label smoothing) with each clip's own features vs features of another clip.
A clear gap means the decoder actually uses the visual/landmark input rather than a language prior.

  python scripts/shuffle_control.py --run runs/X --manifest manifests/subset2k_dev.csv --vis_cache ... --lm_cache ...
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate import load_model  # noqa: E402
from scripts.train import make_store  # noqa: E402
from slt.data import ClipDataset, load_records, make_collate  # noqa: E402
from slt.paths import PROJECT  # noqa: E402
from slt.shards import ShardStore  # noqa: E402


@torch.no_grad()
def dev_loss(model, dl, shuffle, seed=0):
    g = torch.Generator().manual_seed(seed)
    tot, n = 0.0, 0
    for b in dl:
        vis, lm, mask = b["vis"], b["lm"], b["mask"]
        if shuffle:  # roll the batch so every clip gets another clip's features (mask travels with them)
            k = int(torch.randint(1, len(mask), (1,), generator=g)) if len(mask) > 1 else 0
            vis = torch.roll(vis, k, 0) if vis is not None else None
            lm = torch.roll(lm, k, 0) if lm is not None else None
            mask = torch.roll(mask, k, 0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model(vis.cuda() if vis is not None else None, lm.cuda() if lm is not None else None, mask.cuda(),
                            b["labels"].cuda(), b["decoder_input_ids"].cuda(), 0.0)
        ntok = int((b["labels"] != -100).sum())
        tot += loss.item() * ntok
        n += ntok
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--vis_cache")
    ap.add_argument("--lm_cache")
    a = ap.parse_args()
    model, vocab, cfg, ck = load_model(a.run)
    recs, _ = load_records("dev", [r["uid"] for r in csv.DictReader(open(PROJECT / a.manifest, encoding="utf-8"))])
    ecfg = dict(cfg, vis_cache_eval=(a.vis_cache.split(",") if cfg.get("vis_kind") == "keyframe_view" else a.vis_cache),
                lm_cache_eval=a.lm_cache)
    ds = ClipDataset(recs, vocab, make_store(ecfg, "vis_cache_eval") if cfg["use_visual"] else None,
                     make_store(ecfg, "lm_cache_eval") if cfg["use_landmarks"] else None, cfg["T"], False)
    dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=make_collate(vocab), num_workers=2)
    res = {"run": a.run, "epoch": ck["epoch"], "dev_loss_true": dev_loss(model, dl, False),
           "dev_loss_shuffled": dev_loss(model, dl, True)}
    res["gap"] = res["dev_loss_shuffled"] - res["dev_loss_true"]
    print(json.dumps(res))
    (Path(a.run) / "shuffle_control.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
