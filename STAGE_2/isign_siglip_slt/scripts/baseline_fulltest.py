"""ResNet baseline on the FULL test split, using the old pipeline's code unchanged (imported, read-only).

Run with the OLD conda env (its torch/transformers versions):
  "C:/Users/Revetha K/anaconda3/envs/sltbaselines/python.exe" scripts/baseline_fulltest.py cache
  "C:/Users/Revetha K/anaconda3/envs/sltbaselines/python.exe" scripts/baseline_fulltest.py decode
cache : ResNet-18 features of every stored frame of every test clip, written in the OLD cache format
        (shard_XXXXX.json {uids, offsets, lengths} + shard_XXXXX.npy) under isign_siglip_slt/cache/resnet18_fulltest/test.
        Same preprocessing as the old cache (isign_opt.common: centre crop 224, ImageNet norm, frozen ResNet-18, fp16).
decode: the old selected_model.pt with its dev-selected decoding (beam 4, no_repeat_ngram_size 3), old collate/padding.
        Writes runs/baseline_resnet_old/test_full_predictions.tsv. Scoring + ledger: scripts/score_predictions.py.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
OLD_REPO = Path(r"E:\FINAL YEAR PROJECT\GLOSS FREE SLT\sltbaselines")
sys.path.insert(0, str(OLD_REPO))
from isign_opt.common import FULL_LMDB, EXP_ROOT, FrameEncoder, center_crop, read_lmdb_frames  # noqa: E402

OUT = PROJECT / "cache" / "resnet18_fulltest" / "test"
SHARD = 1000


def cache():
    import csv
    uids = [r["uid"] for r in csv.DictReader(open(r"E:\FINAL YEAR PROJECT\GLOSS FREE SLT\dataset\iSign\splits\v1\test.csv",
                                               encoding="utf-8"))]
    uids = sorted(uids)
    OUT.mkdir(parents=True, exist_ok=True)
    enc = FrameEncoder().cuda()
    t0 = time.time()
    for s in range(0, len(uids), SHARD):
        js = OUT / f"shard_{s // SHARD:05d}.json"
        if js.exists():
            continue
        feats, lens = [], []
        for u in uids[s:s + SHARD]:
            fr = torch.from_numpy(center_crop(read_lmdb_frames(FULL_LMDB / "test" / u))).cuda()
            feats.append(enc(fr).cpu().numpy())
            lens.append(len(fr))
        offs = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(int).tolist()
        np.save(OUT / f"shard_{s // SHARD:05d}.tmp.npy", np.concatenate(feats))
        (OUT / f"shard_{s // SHARD:05d}.tmp.npy").replace(OUT / f"shard_{s // SHARD:05d}.npy")
        js.write_text(json.dumps({"uids": uids[s:s + SHARD], "offsets": offs, "lengths": lens}))
        print(f"cached {min(s + SHARD, len(uids))}/{len(uids)} in {time.time() - t0:.0f}s", flush=True)


@torch.no_grad()
def decode():
    from torch.utils.data import DataLoader
    from transformers import MBartTokenizer
    from isign_opt.common import LANG, TOKENIZER_DIR
    from isign_opt.data import BucketBatchSampler, Collate, FeatureCaptionDataset, FeatureStore, build_split
    from isign_opt.model import FeatureSLT

    sel = EXP_ROOT / "reports" / "final_eval" / "selected_model.pt"
    ck = torch.load(sel, map_location="cpu")
    tok = MBartTokenizer.from_pretrained(TOKENIZER_DIR, src_lang=LANG, tgt_lang=LANG)
    model = FeatureSLT(pretrained=False).cuda()
    model.load_state_dict(ck["model"])
    model.eval()
    recs, exc = build_split("test", "full")
    store = FeatureStore("test", cache_root=OUT.parent)
    ds = FeatureCaptionDataset(recs, store, train=False)
    dl = DataLoader(ds, batch_sampler=BucketBatchSampler(ds.lengths, 32, shuffle=False), collate_fn=Collate(tok))
    uids, preds = [], []
    t0 = time.time()
    for b in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(b["feats"].cuda(), b["lengths"], tok.lang_code_to_id[LANG], num_beams=4,
                                 no_repeat_ngram_size=3)
        preds += [p.strip() for p in tok.batch_decode(out, skip_special_tokens=True)]
        uids += b["uids"]
    run = PROJECT / "runs" / "baseline_resnet_old"
    run.mkdir(parents=True, exist_ok=True)
    with open(run / "test_full_predictions.tsv", "w", encoding="utf-8") as f:
        f.write("uid\treference\tprediction\n")
        for u, p in zip(uids, preds):
            f.write(f"{u}\t{recs[u]}\t{p}\n")
    (run / "test_full_decode_info.json").write_text(json.dumps({
        "checkpoint": str(sel), "checkpoint_epoch": ck.get("epoch"), "decoding": "beam4_norepeat3 (selected on dev, 2026-10-23)",
        "n": len(uids), "excluded_captions": len(exc), "minutes": round((time.time() - t0) / 60, 1)}, indent=1))
    print("decoded", len(uids), "in", round((time.time() - t0) / 60, 1), "min")


if __name__ == "__main__":
    {"cache": cache, "decode": decode}[sys.argv[1]]()
