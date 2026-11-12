"""Final evaluation. Decoding settings are chosen on DEV only; the test split is decoded once with them.

python -m isign_opt.evaluate --ckpt runs/main_full/best.pt --out reports/final_eval
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import MBartTokenizer

from isign_opt.common import EXP_ROOT, LANG, TOKENIZER_DIR
from isign_opt.data import BucketBatchSampler, Collate, FeatureCaptionDataset, FeatureStore, build_split
from isign_opt.metrics import rouge_l, translation_metrics
from isign_opt.model import FeatureSLT
from isign_opt.train import evaluate

DECODE_CANDIDATES = {
    "greedy": {"num_beams": 1, "gen_kw": {}},
    "beam4": {"num_beams": 4, "gen_kw": {}},
    "beam4_norepeat3": {"num_beams": 4, "gen_kw": {"no_repeat_ngram_size": 3}},
}


def loader(split, tok):
    recs, exc = build_split(split, "subset")
    ds = FeatureCaptionDataset(recs, FeatureStore(split), train=False)
    return DataLoader(ds, batch_sampler=BucketBatchSampler(ds.lengths, 32, shuffle=False), collate_fn=Collate(tok)), exc


def status(p, r):
    n = lambda s: " ".join(s.lower().strip(" .!?").split())
    if n(p) == n(r):
        return "exact"
    rl = rouge_l(p, r)
    return "partial" if rl >= 0.3 else ("weak" if rl > 0 else "miss")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="candidate checkpoints; the best on DEV is used for test")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out) if Path(a.out).is_absolute() else EXP_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    tok = MBartTokenizer.from_pretrained(TOKENIZER_DIR, src_lang=LANG, tgt_lang=LANG)
    dev_dl, dev_exc = loader("dev", tok)
    model = FeatureSLT(pretrained=False).to(dev)

    dev_scores, best_key = {}, None
    for cp in a.ckpts:
        cp_path = Path(cp) if Path(cp).is_absolute() else EXP_ROOT / cp
        ck = torch.load(cp_path, map_location="cpu")
        model.load_state_dict(ck["model"]); model.eval()
        for name, c in DECODE_CANDIDATES.items():
            res, *_ = evaluate(model, dev_dl, tok, dev, 0.1, num_beams=c["num_beams"], gen_kw=c["gen_kw"], shuffled_check=False)
            key = f"{cp_path.name}|epoch{ck['epoch']}|{name}"
            dev_scores[key] = res
            print(f"dev {key}: BLEU4 {res['bleu4']:.2f} BLEU1 {res['bleu1']:.2f} ROUGE-L {res['rougeL']:.2f} unique {res['unique_predictions']}", flush=True)
            if best_key is None or (res["bleu4"], res["rougeL"]) > (dev_scores[best_key]["bleu4"], dev_scores[best_key]["rougeL"]):
                best_key, ck_path, chosen = key, cp_path, name
    print(f"chosen on dev: {best_key}", flush=True)
    ck = torch.load(ck_path, map_location="cpu")
    model.load_state_dict(ck["model"]); model.eval()
    ls = 0.1
    c = DECODE_CANDIDATES[chosen]
    # store the selected weights under a stable name for inference / the app
    torch.save({"model": ck["model"], "args": ck.get("args"), "epoch": ck["epoch"], "dev_selection": best_key,
                "decoding": chosen}, out / "selected_model.pt")

    test_dl, test_exc = loader("test", tok)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
    res, uids, preds, refs = evaluate(model, test_dl, tok, dev, ls, num_beams=c["num_beams"], gen_kw=c["gen_kw"], shuffled_check=True)
    torch.cuda.synchronize(); el = time.time() - t0
    res["batched_seconds_per_video_incl_loss_passes"] = el / len(uids)
    res["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30

    # single-video latency (feature -> sentence), batch size 1
    ds = test_dl.dataset; lat = []
    with torch.no_grad():
        for i in range(50):
            b = Collate(tok)([ds[i]])
            torch.cuda.synchronize(); t = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model.generate(b["feats"].to(dev), b["lengths"], tok.lang_code_to_id[LANG], num_beams=c["num_beams"], **c["gen_kw"])
            torch.cuda.synchronize(); lat.append(time.time() - t)
    res["latency_ms_per_video_bs1_model_only"] = 1000 * sorted(lat)[len(lat) // 2]

    with (out / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["sample_id", "reference", "prediction", "rougeL", "status"])
        for u, r, p in zip(uids, refs, preds):
            w.writerow([u, r, p, f"{rouge_l(p, r):.3f}", status(p, r)])
    st = {}
    for u, r, p in zip(uids, refs, preds):
        st[status(p, r)] = st.get(status(p, r), 0) + 1
    from collections import Counter
    summary = {"checkpoint": str(ck_path), "checkpoint_epoch": ck["epoch"], "dev_selection_key": best_key,
               "selected_model_file": str(out / "selected_model.pt"),
               "decoding_selected_on_dev": chosen, "dev_decoding_comparison": dev_scores,
               "test_n": len(uids), "test_excluded_captions": test_exc, "dev_excluded_count": len(dev_exc),
               "test_metrics": res, "test_status_counts": st,
               "test_top_predictions": Counter(preds).most_common(10)}
    (out / "metrics.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1), flush=True)
    print("status counts", st)


if __name__ == "__main__":
    main()
