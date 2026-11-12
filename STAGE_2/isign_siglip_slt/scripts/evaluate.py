"""Decode a trained checkpoint on dev (any number of times) or test (once; ledger-guarded).

  python scripts/evaluate.py --run runs/X --split dev --manifest manifests/subset2k_dev.csv \
      --vis_cache ... --lm_cache ... --beams 1 4 5 --length_penalty 1.0 --no_repeat 0 3
  python scripts/evaluate.py --run runs/X --split test ... --beams 4 --length_penalty 1.0 --no_repeat 3 --final
A test decode requires --final (exactly one decoding setting). A second test decode of the same checkpoint
hash is refused unless --repeat is given, and is then logged as a repeat evaluation.
"""
import argparse
import csv
import datetime
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import load_tokenizer_and_mbart, make_store, predict, sha256  # noqa: E402
from slt.data import ClipDataset, load_records, make_collate  # noqa: E402
from slt.metrics import corpus_metrics  # noqa: E402
from slt.model import SLTModel, build_mbart  # noqa: E402
from slt.paths import MANIFESTS, PROJECT, REPORTS  # noqa: E402
from slt.shards import ShardStore  # noqa: E402
from slt.text import normalize  # noqa: E402
from slt.vocab import TrimmedVocab  # noqa: E402

LEDGER = PROJECT / "TEST_LEDGER.csv"


def load_model(run, ckpt="best.pt"):
    ck = torch.load(Path(run) / ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]["config"]
    tok, src, _ = load_tokenizer_and_mbart()
    vocab = TrimmedVocab.load(tok, PROJECT / cfg["vocab"])
    mb, _ = build_mbart(src, vocab, cfg["n_layers"], cfg["dropout"])
    model = SLTModel(mb, cfg["use_visual"], cfg["use_landmarks"], vis_dim=cfg.get("vis_dim", 768),
                     feat_dropout=cfg["feat_dropout"])
    model.load_state_dict(ck["model"])
    return model.cuda().eval(), vocab, cfg, ck


def breakdown(uids, hyps, recs):
    train, _ = load_records("train")
    seen = {normalize(t) for t in train.values()}
    flagged = {r["uid"] for r in csv.DictReader(open(MANIFESTS / "flag_small_resolution.csv", encoding="utf-8"))}
    out = {}
    groups = {"seen_in_train": lambda u: normalize(recs[u]) in seen,
              "unseen_in_train": lambda u: normalize(recs[u]) not in seen,
              "without_flagged": lambda u: u not in flagged}
    for lo, hi in ((1, 3), (4, 8), (9, 15), (16, 25), (26, 999)):
        groups[f"ref_words_{lo}-{hi}"] = (lambda lo, hi: lambda u: lo <= len(normalize(recs[u]).split()) <= hi)(lo, hi)
    for name, f in groups.items():
        idx = [i for i, u in enumerate(uids) if f(u)]
        if idx:
            out[name] = corpus_metrics([hyps[i] for i in idx], [recs[uids[i]] for i in idx])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--vis_cache")
    ap.add_argument("--lm_cache")
    ap.add_argument("--beams", type=int, nargs="+", default=[1])
    ap.add_argument("--length_penalty", type=float, nargs="+", default=[1.0])
    ap.add_argument("--no_repeat", type=int, nargs="+", default=[0])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--repeat", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    run = Path(a.run)
    settings = list(itertools.product(a.beams, a.length_penalty, a.no_repeat))
    digest = sha256(run / a.ckpt)
    if a.split == "test":
        if not a.final or len(settings) != 1:
            raise SystemExit("test decoding needs --final and exactly one decoding setting fixed on dev")
        prior = [r for r in csv.DictReader(open(LEDGER, encoding="utf-8"))] if LEDGER.exists() else []
        if any(r["checkpoint_sha256"] == digest and r["kind"] == "final" for r in prior) and not a.repeat:
            raise SystemExit("this checkpoint already has a final test evaluation; use --repeat to log a repeat")

    model, vocab, cfg, ck = load_model(run, a.ckpt)
    uids_m = [r["uid"] for r in csv.DictReader(open(PROJECT / a.manifest, encoding="utf-8"))]
    recs, exc = load_records(a.split, uids_m)
    # --vis_cache / --lm_cache override the run's dev caches (e.g. point them at the test caches)
    ecfg = dict(cfg)
    if a.vis_cache:
        ecfg["vis_cache_eval"] = a.vis_cache.split(",") if cfg.get("vis_kind") == "keyframe_view" else a.vis_cache
    if a.lm_cache:
        ecfg["lm_cache_eval"] = a.lm_cache
    ds = ClipDataset(recs, vocab, make_store(ecfg, "vis_cache_eval") if cfg["use_visual"] else None,
                     make_store(ecfg, "lm_cache_eval") if cfg["use_landmarks"] else None, cfg["T"], False, 1,
                     cfg["max_label_len"])
    dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=make_collate(vocab), num_workers=4)
    out_dir = run / f"eval_{a.split}{('_' + a.tag) if a.tag else ''}"
    out_dir.mkdir(exist_ok=True)
    results = {}
    for nb, lp, nr in settings:
        key = f"beam{nb}_lp{lp}_nr{nr}"
        uids, hyps = predict(model, dl, vocab, "cuda", num_beams=nb, length_penalty=lp, no_repeat_ngram_size=nr)
        m = corpus_metrics(hyps, [recs[u] for u in uids], with_signature=True)
        m["breakdown"] = breakdown(uids, hyps, recs)
        m["excluded_captions"] = len(exc)
        results[key] = m
        with open(out_dir / f"predictions_{key}.tsv", "w", encoding="utf-8") as f:
            f.write("uid\treference\tprediction\n")
            for u, h in zip(uids, hyps):
                f.write(f"{u}\t{recs[u]}\t{h}\n")
        print(key, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()
                               if k not in ("breakdown",) and not k.startswith("signature")}, ensure_ascii=False), flush=True)
    (out_dir / "metrics.json").write_text(json.dumps({"checkpoint": str(run / a.ckpt), "sha256": digest,
                                                      "epoch": ck["epoch"], "results": results}, indent=1,
                                                     ensure_ascii=False))
    if a.split == "test":
        key, m = next(iter(results.items()))
        new = not LEDGER.exists()
        with open(LEDGER, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["timestamp", "kind", "system", "setting", "checkpoint_sha256", "n", "bleu1", "bleu4",
                            "rougeL", "chrf", "exact_match", "distinct_predictions", "note"])
            w.writerow([datetime.datetime.now().isoformat(timespec="seconds"), "repeat" if a.repeat else "final",
                        str(run), key, digest, m["n"], f'{m["bleu1"]:.2f}', f'{m["bleu4"]:.2f}', f'{m["rougeL"]:.2f}',
                        f'{m["chrf"]:.2f}', f'{m["exact_match"]:.2f}', m["distinct_predictions"], a.tag])


if __name__ == "__main__":
    main()
