"""Write human-readable prediction logs (id / ref / pred) for train, dev and test with the selected model.

Output: <exp>/reports/prediction_logs/{train,dev,test}_predictions.log and all_predictions.log
Train videos were seen during training, so train predictions show fit, not generalisation.
"""
from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader
from transformers import MBartTokenizer

from isign_opt.common import EXP_ROOT, LANG, TOKENIZER_DIR
from isign_opt.data import BucketBatchSampler, Collate, FeatureCaptionDataset, FeatureStore, build_split
from isign_opt.evaluate import DECODE_CANDIDATES
from isign_opt.metrics import translation_metrics
from isign_opt.model import FeatureSLT

NOTES = {
    "train": "SEEN DURING TRAINING (gradient updates) - shows fit, not generalisation",
    "dev": "used only for checkpoint/decoding selection - not trained on",
    "test": "HELD-OUT, never used for training or selection",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test", "dev", "train"])
    ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()
    out = EXP_ROOT / "reports" / "prediction_logs"
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(EXP_ROOT / "reports" / "final_eval" / "selected_model.pt", map_location="cpu")
    dev = torch.device("cuda")
    tok = MBartTokenizer.from_pretrained(TOKENIZER_DIR, src_lang=LANG, tgt_lang=LANG)
    model = FeatureSLT(pretrained=False).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    c = DECODE_CANDIDATES[ck["decoding"]]
    start = tok.lang_code_to_id[LANG]

    for split in a.splits:
        recs, exc = build_split(split, "full" if split == "train" else "subset")
        ds = FeatureCaptionDataset(recs, FeatureStore(split), train=False)
        dl = DataLoader(ds, batch_sampler=BucketBatchSampler(ds.lengths, a.bs, shuffle=False), collate_fn=Collate(tok),
                        num_workers=2)
        preds, refs, t0, k = [], [], time.time(), 0
        with (out / f"{split}_predictions.log").open("w", encoding="utf-8") as f:
            f.write(f"# split: {split} ({NOTES[split]})\n# model: selected_model.pt (epoch {ck['epoch']}), decoding: {ck['decoding']}\n"
                    f"# samples: {len(ds)} (excluded empty/placeholder captions: {len(exc)})\n\n")
            with torch.no_grad():
                for b in dl:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        o = model.generate(b["feats"].to(dev), b["lengths"], start, num_beams=c["num_beams"], **c["gen_kw"])
                    for u, r, p in zip(b["uids"], b["texts"], tok.batch_decode(o, skip_special_tokens=True)):
                        k += 1; p = p.strip()
                        preds.append(p); refs.append(r)
                        f.write(f"[{split} {k}/{len(ds)}] id: {u}\nref:  {r}\npred: {p}\n\n")
                    f.flush()
                    if k % 5000 < a.bs:
                        print(f"{split}: {k}/{len(ds)} in {(time.time() - t0) / 60:.1f} min", flush=True)
            m = translation_metrics(preds, refs)
            f.write("# ---- summary ----\n" + "".join(f"# {key}: {v:.4f}\n" if isinstance(v, float) else f"# {key}: {v}\n" for key, v in m.items()))
        print(f"{split} done: {k} samples, BLEU-4 {m['bleu4']:.2f}, {(time.time() - t0) / 60:.1f} min", flush=True)

    with (out / "all_predictions.log").open("w", encoding="utf-8") as f:
        for split in ["train", "dev", "test"]:
            p = out / f"{split}_predictions.log"
            if p.exists():
                f.write(p.read_text(encoding="utf-8") + "\n\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
