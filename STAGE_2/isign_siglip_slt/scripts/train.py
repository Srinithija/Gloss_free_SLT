"""Train the SLT model on cached features; select by dev BLEU-4 (greedy, normalised).

Example (pilot):
  python scripts/train.py --config configs/pilot_hybrid.json
Config keys: see DEFAULTS. Every run writes runs/<name>/config.json, train_log.jsonl, best.pt (+ .sha256),
last.pt (+ .sha256), dev_predictions_best.tsv. Resumes from last.pt when present.
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slt.data import ClipDataset, LengthBucketSampler, load_records, make_collate  # noqa: E402
from slt.metrics import corpus_metrics  # noqa: E402
from slt.model import SLTModel, build_mbart  # noqa: E402
from slt.paths import PROJECT  # noqa: E402
from slt.shards import ShardStore  # noqa: E402
from slt.vocab import TrimmedVocab  # noqa: E402

DEFAULTS = {
    "name": "pilot_hybrid", "seed": 42,
    "train_manifest": None, "dev_manifest": "manifests/subset2k_dev.csv",
    "vis_cache_train": None, "vis_cache_dev": None, "lm_cache_train": None, "lm_cache_dev": None,
    "use_visual": True, "use_landmarks": True, "n_views": 1, "T": 64,
    "n_layers": 3, "dropout": 0.1, "feat_dropout": 0.1, "label_smoothing": 0.1,
    "lr_new": 3e-4, "lr_mbart": 1e-4, "weight_decay": 0.01, "warmup_steps": 500, "batch_size": 32,
    "epochs": 30, "patience": 5, "max_label_len": 64, "grad_clip": 1.0, "num_workers": 4,
    "vocab": "cache/vocab/trimmed_vocab_train.json", "max_train_steps": 0,
    "vis_dim": 768, "vis_kind": "shard", "keyframes": None, "frame_keep_min": 1.0,
}


def make_store(cfg, key):
    """key: vis_cache_train | vis_cache_dev | lm_cache_* -> a store, or None. A 'keyframe_view' visual store
    takes a list of roots holding all stored frames per clip and exposes the clip's keyframes only."""
    src = cfg.get(key)
    if not src:
        return None
    from slt.shards import KeyframeView
    if key.startswith("vis") and cfg["vis_kind"] == "keyframe_view":
        kf = json.loads((PROJECT / cfg["keyframes"]).read_text())
        return KeyframeView([str(PROJECT / r) if not Path(r).is_absolute() else r for r in src], kf)
    return ShardStore(PROJECT / src)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save_atomic(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
    digest = sha256(path)
    Path(str(path) + ".sha256").write_text(digest)
    torch.load(path, map_location="cpu", weights_only=False)  # verify it reloads
    return digest


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def uids_of(manifest):
    import csv
    return [r["uid"] for r in csv.DictReader(open(PROJECT / manifest, encoding="utf-8"))]


def load_tokenizer_and_mbart():
    from transformers import MBartForConditionalGeneration, MBartTokenizer
    rev = json.loads((PROJECT / "configs" / "model_revisions.json").read_text())["facebook/mbart-large-cc25"]
    tok = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", revision=rev, src_lang="en_XX", tgt_lang="en_XX")
    src = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25", revision=rev)
    return tok, src, rev


@torch.no_grad()
def predict(model, loader, vocab, dev, **gen):
    model.eval()
    hyps, refs, uids = [], [], []
    for b in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(b["vis"].to(dev) if b["vis"] is not None else None,
                                 b["lm"].to(dev) if b["lm"] is not None else None, b["mask"].to(dev), **gen)
        hyps += [vocab.decode(o[1:]) for o in out.tolist()]  # o[0] is the decoder start (en_XX)
        uids += b["uids"]
    return uids, hyps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = DEFAULTS | json.loads(Path(a.config).read_text())
    run = PROJECT / "runs" / cfg["name"]
    run.mkdir(parents=True, exist_ok=True)
    seed_all(cfg["seed"])
    dev = "cuda"

    tok, src, rev = load_tokenizer_and_mbart()
    vocab = TrimmedVocab.load(tok, PROJECT / cfg["vocab"])
    mbart, layer_ids = build_mbart(src, vocab, cfg["n_layers"], cfg["dropout"])
    del src
    model = SLTModel(mbart, cfg["use_visual"], cfg["use_landmarks"], vis_dim=cfg["vis_dim"],
                     feat_dropout=cfg["feat_dropout"]).to(dev)

    tr_recs, tr_exc = load_records("train", uids_of(cfg["train_manifest"]), drop_flagged=True)
    dv_recs, dv_exc = load_records("dev", uids_of(cfg["dev_manifest"]))
    st = lambda k: make_store(cfg, k)
    tr_ds = ClipDataset(tr_recs, vocab, st("vis_cache_train") if cfg["use_visual"] else None,
                        st("lm_cache_train") if cfg["use_landmarks"] else None, cfg["T"], True, cfg["n_views"],
                        cfg["max_label_len"], cfg["frame_keep_min"])
    dv_ds = ClipDataset(dv_recs, vocab, st("vis_cache_dev") if cfg["use_visual"] else None,
                        st("lm_cache_dev") if cfg["use_landmarks"] else None, cfg["T"], False, 1, cfg["max_label_len"])
    collate = make_collate(vocab)
    sampler = LengthBucketSampler([len(tr_recs[u]) for u in tr_ds.uids], cfg["batch_size"], cfg["seed"])
    tr_dl = DataLoader(tr_ds, batch_sampler=sampler, collate_fn=collate, num_workers=cfg["num_workers"],
                       persistent_workers=cfg["num_workers"] > 0)
    dv_dl = DataLoader(dv_ds, batch_size=64, shuffle=False, collate_fn=collate, num_workers=cfg["num_workers"])

    mb = [p for n, p in model.named_parameters() if n.startswith("mbart.")]
    new = [p for n, p in model.named_parameters() if not n.startswith("mbart.")]
    opt = torch.optim.AdamW([{"params": new, "lr": cfg["lr_new"]}, {"params": mb, "lr": cfg["lr_mbart"]}],
                            weight_decay=cfg["weight_decay"])
    total = cfg["max_train_steps"] or cfg["epochs"] * len(sampler)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / cfg["warmup_steps"]) *
                                              0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    info = {"config": cfg, "mbart_revision": rev, "pretrained_layers": layer_ids, "vocab_size": len(vocab),
            "n_train": len(tr_ds), "n_dev": len(dv_ds), "train_excluded": len(tr_exc), "dev_excluded": len(dv_exc),
            "params_total": sum(p.numel() for p in model.parameters()),
            "params_new": sum(p.numel() for p in new), "steps_per_epoch": len(sampler), "total_steps": total,
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name()}
    (run / "config.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)

    start_epoch, step, best, bad = 0, 0, -1.0, 0
    if (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_epoch, step, best, bad = ck["epoch"] + 1, ck["step"], ck["best"], ck["bad"]
        random.setstate(ck["rng"]["py"]); np.random.set_state(ck["rng"]["np"]); torch.set_rng_state(ck["rng"]["torch"])
        print(f"resumed from epoch {ck['epoch']}", flush=True)

    log = open(run / "train_log.jsonl", "a", encoding="utf-8")
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        sampler.set_epoch(epoch)
        t0, losses = time.time(), []
        for b in tr_dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = model(b["vis"].to(dev, non_blocking=True) if b["vis"] is not None else None,
                                b["lm"].to(dev, non_blocking=True) if b["lm"] is not None else None,
                                b["mask"].to(dev), b["labels"].to(dev), b["decoder_input_ids"].to(dev),
                                cfg["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            if not torch.isfinite(loss) or not torch.isfinite(gn):
                raise RuntimeError(f"non-finite loss/grad at step {step}")
            opt.step(); sched.step(); step += 1
            losses.append(loss.item())
            if len(losses) % 100 == 0:
                el = time.time() - t0
                print(json.dumps({"progress": f"epoch {epoch} step {len(losses)}/{len(sampler)}",
                                  "loss_last100": round(float(np.mean(losses[-100:])), 4),
                                  "s_per_step": round(el / len(losses), 3),
                                  "epoch_eta_min": round(el / len(losses) * (len(sampler) - len(losses)) / 60, 1),
                                  "lr_mbart": opt.param_groups[1]["lr"]}), flush=True)
            if cfg["max_train_steps"] and step >= cfg["max_train_steps"]:
                break
        train_s = time.time() - t0
        # dev: loss + greedy BLEU
        model.eval()
        dl_losses = []
        with torch.no_grad():
            for b in dv_dl:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    l, _ = model(b["vis"].to(dev) if b["vis"] is not None else None,
                                 b["lm"].to(dev) if b["lm"] is not None else None, b["mask"].to(dev),
                                 b["labels"].to(dev), b["decoder_input_ids"].to(dev), 0.0)
                dl_losses.append(l.item())
        t1 = time.time()
        uids, hyps = predict(model, dv_dl, vocab, dev, num_beams=1)
        m = corpus_metrics(hyps, [dv_recs[u] for u in uids])
        rec = {"epoch": epoch, "step": step, "train_loss": float(np.mean(losses)), "dev_loss_nols": float(np.mean(dl_losses)),
               "dev_bleu4": m["bleu4"], "dev_bleu1": m["bleu1"], "dev_rougeL": m["rougeL"], "dev_chrf": m["chrf"],
               "distinct": m["distinct_predictions"], "top_output": m["most_repeated_output"],
               "top_share": round(m["most_repeated_share"], 4), "train_min": round(train_s / 60, 2),
               "eval_min": round((time.time() - t1) / 60, 2),
               "peak_vram_GiB": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
               "lr_mbart": opt.param_groups[1]["lr"]}
        improved = m["bleu4"] > best
        if improved:
            best, bad = m["bleu4"], 0
            rec["best_sha256"] = save_atomic({"model": model.state_dict(), "config": info, "epoch": epoch,
                                              "dev": m}, run / "best.pt")
            with open(run / "dev_predictions_best.tsv", "w", encoding="utf-8") as f:
                f.write("uid\treference\tprediction\n")
                for u, h in zip(uids, hyps):
                    f.write(f"{u}\t{dv_recs[u]}\t{h}\n")
        else:
            bad += 1
        save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                     "epoch": epoch, "step": step, "best": best, "bad": bad, "config": info,
                     "rng": {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()}},
                    run / "last.pt")
        print(json.dumps(rec), file=log, flush=True)
        print(json.dumps(rec), flush=True)
        if bad >= cfg["patience"] or (cfg["max_train_steps"] and step >= cfg["max_train_steps"]):
            break
    print(json.dumps({"done": True, "best_dev_bleu4": best}), file=log, flush=True)


if __name__ == "__main__":
    main()
