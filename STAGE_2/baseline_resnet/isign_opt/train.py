"""Train the feature-level gloss-free SLT model on cached iSign features.

Checkpoint selection uses the dev split only; the test split is never loaded here.
Example: python -m isign_opt.train --train_source subset --epochs 15 --out runs/pilot_16k
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import MBartTokenizer

from isign_opt.common import EXP_ROOT, LANG, TOKENIZER_DIR
from isign_opt.data import BucketBatchSampler, Collate, FeatureCaptionDataset, FeatureStore, build_split
from isign_opt.metrics import translation_metrics
from isign_opt.model import FeatureSLT


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="run directory (relative paths are under the experiment root)")
    ap.add_argument("--train_source", default="subset", choices=["subset", "full"])
    ap.add_argument("--max_train", type=int, default=None, help="use only the first N training records (debug)")
    ap.add_argument("--dev_is_train", action="store_true", help="evaluate on the training records (overfit debug)")
    ap.add_argument("--dev_limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr_new", type=float, default=3e-4)
    ap.add_argument("--lr_pre", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--feat_dropout", type=float, default=0.1)
    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--no_pretrained", action="store_true", help="ablation: random mBART init")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=4, help="early stop after N evals without dev BLEU-4 gain")
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--init", default=None, help="initialise model weights from a checkpoint")
    ap.add_argument("--resume", action="store_true", help="resume from <out>/last.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    return ap.parse_args()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def ce_loss(logits, labels, mask, ls):
    return torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                             labels.masked_fill(mask == 0, -100).reshape(-1),
                                             ignore_index=-100, label_smoothing=ls)


@torch.no_grad()
def evaluate(model, loader, tok, device, ls, num_beams=1, gen_kw=None, shuffled_check=True):
    """Teacher-forced loss (paired and video-shuffled) plus generated translations."""
    model.eval()
    start = tok.lang_code_to_id[LANG]
    tot, tot_shuf, n, preds, refs, uids = 0.0, 0.0, 0, [], [], []
    for b in loader:
        x, lens = b["feats"].to(device), b["lengths"]
        y, m = b["labels"].to(device), b["label_mask"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tot += ce_loss(model(x, lens, y, m), y, m, ls).item() * len(lens)
            if shuffled_check and len(lens) > 1:
                perm = torch.roll(torch.arange(len(lens)), 1)
                tot_shuf += ce_loss(model(x[perm], lens[perm], y, m), y, m, ls).item() * len(lens)
            out = model.generate(x, lens, start, num_beams=num_beams, **(gen_kw or {}))
        preds += [p.strip() for p in tok.batch_decode(out, skip_special_tokens=True)]
        refs += b["texts"]; uids += b["uids"]; n += len(lens)
    res = translation_metrics(preds, refs)
    res["loss"] = tot / n
    if shuffled_check:
        res["loss_shuffled_video"] = tot_shuf / n
    return res, uids, preds, refs


def main():
    a = get_args()
    out = Path(a.out) if Path(a.out).is_absolute() else EXP_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    tok = MBartTokenizer.from_pretrained(TOKENIZER_DIR, src_lang=LANG, tgt_lang=LANG)

    train_rec, train_exc = build_split("train", a.train_source)
    if a.max_train:
        train_rec = dict(list(train_rec.items())[: a.max_train])
    if a.dev_is_train:
        dev_rec, dev_exc = dict(train_rec), {}
    else:
        dev_rec, dev_exc = build_split("dev", "subset")
    if a.dev_limit:
        dev_rec = dict(list(dev_rec.items())[: a.dev_limit])
    train_store = FeatureStore("train")
    dev_store = train_store if a.dev_is_train else FeatureStore("dev")
    train_ds = FeatureCaptionDataset(train_rec, train_store, train=not a.no_aug)
    dev_ds = FeatureCaptionDataset(dev_rec, dev_store, train=False)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    (out / "data_summary.json").write_text(json.dumps({
        "train_source": a.train_source, "train_n": len(train_ds), "train_excluded": len(train_exc),
        "dev_n": len(dev_ds), "dev_excluded": len(dev_exc), "dev_is_train": a.dev_is_train}, indent=1))
    print(f"train {len(train_ds)} (excluded {len(train_exc)}), dev {len(dev_ds)}", flush=True)

    collate = Collate(tok)
    sampler = BucketBatchSampler(train_ds.lengths, a.bs, shuffle=True, seed=a.seed)
    train_dl = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate, num_workers=a.workers,
                          pin_memory=True, persistent_workers=a.workers > 0)
    dev_dl = DataLoader(dev_ds, batch_sampler=BucketBatchSampler(dev_ds.lengths, 64, shuffle=False),
                        collate_fn=collate, num_workers=a.workers, pin_memory=True)

    model = FeatureSLT(a.dropout, a.feat_dropout, pretrained=not a.no_pretrained).to(device)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu")["model"])
    new_p = [p for n, p in model.named_parameters() if not n.startswith("mbart.")]
    pre_p = [p for n, p in model.named_parameters() if n.startswith("mbart.")]
    opt = torch.optim.AdamW([{"params": new_p, "lr": a.lr_new}, {"params": pre_p, "lr": a.lr_pre}], weight_decay=a.wd)
    total = a.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    start_epoch, best, bad, step = 0, -1.0, 0, 0
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_epoch, best, bad, step = ck["epoch"] + 1, ck["best"], ck["bad"], ck["step"]
        random.setstate(ck["py_rng"]); np.random.set_state(ck["np_rng"]); torch.set_rng_state(ck["torch_rng"])
        print(f"resumed at epoch {start_epoch}", flush=True)

    log = (out / "log.jsonl").open("a")
    for epoch in range(start_epoch, a.epochs):
        model.train(); sampler.set_epoch(epoch)
        t0, run, cnt, tdata, tgpu = time.time(), 0.0, 0, 0.0, 0.0
        tl = time.time()
        for b in train_dl:
            tdata += time.time() - tl; tg = time.time()
            x, lens = b["feats"].to(device, non_blocking=True), b["lengths"]
            y, m = b["labels"].to(device, non_blocking=True), b["label_mask"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = ce_loss(model(x, lens, y, m), y, m, a.label_smoothing)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch} step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step(); sched.step(); step += 1
            run += loss.item(); cnt += 1; tgpu += time.time() - tg
            if step % 200 == 0:
                print(f"ep {epoch} step {step} loss {run / cnt:.4f} gnorm {gn:.2f} lr {sched.get_last_lr()[1]:.2e} "
                      f"data {tdata / cnt * 1000:.0f}ms gpu {tgpu / cnt * 1000:.0f}ms per batch", flush=True)
            tl = time.time()
        rec = {"epoch": epoch, "step": step, "train_loss": run / cnt, "epoch_min": (time.time() - t0) / 60,
               "data_ms_per_batch": tdata / cnt * 1000, "gpu_ms_per_batch": tgpu / cnt * 1000,
               "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
        if (epoch + 1) % a.eval_every == 0 or epoch + 1 == a.epochs:
            te = time.time()
            res, uids, preds, refs = evaluate(model, dev_dl, tok, device, a.label_smoothing)
            rec.update({f"dev_{k}": v for k, v in res.items()}); rec["eval_min"] = (time.time() - te) / 60
            with (out / f"dev_predictions_epoch{epoch:02d}.jsonl").open("w", encoding="utf-8") as f:
                for u, p, r in zip(uids, preds, refs):
                    f.write(json.dumps({"uid": u, "ref": r, "pred": p}, ensure_ascii=False) + "\n")
            score = res["bleu4"] + 1e-3 * res["bleu1"]
            if score > best:
                best, bad = score, 0
                torch.save({"model": model.state_dict(), "args": vars(a), "epoch": epoch, "dev": res}, out / "best.pt")
                rec["saved_best"] = True
            else:
                bad += 1
            print(f"== epoch {epoch}: train {rec['train_loss']:.4f} dev loss {res['loss']:.4f} (shuffled {res['loss_shuffled_video']:.4f}) "
                  f"BLEU1 {res['bleu1']:.2f} BLEU4 {res['bleu4']:.2f} ROUGE-L {res['rougeL']:.2f} unique {res['unique_predictions']}/{res['n']} "
                  f"| {rec['epoch_min']:.1f}+{rec['eval_min']:.1f} min", flush=True)
            for i in range(3):
                print(f"   ref: {refs[i]}\n   pred: {preds[i]}", flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch,
                    "best": best, "bad": bad, "step": step, "py_rng": random.getstate(), "np_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state()}, out / "last.pt")
        if bad >= a.patience:
            print(f"early stopping at epoch {epoch}", flush=True)
            break
    print("done; best dev score", best, flush=True)


if __name__ == "__main__":
    main()
