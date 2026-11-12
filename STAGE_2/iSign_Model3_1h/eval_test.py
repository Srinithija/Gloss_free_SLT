"""Evaluate a run's best.pt once on the held-out test subset: test loss, greedy + beam-4 metrics,
inference time per video, prediction TSV, and a key-word recall analysis.
  python eval_test.py runs/m3_hybrid
"""
import json, sys, time
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "isign_siglip_slt"))
import numpy as np
import torch
import scripts.train as T
T.PROJECT = HERE
from torch.utils.data import DataLoader
from slt.data import ClipDataset, load_records, make_collate
from slt.metrics import corpus_metrics
from slt.model import SLTModel, build_mbart
from slt.shards import ShardStore
from slt.text import normalize
from slt.vocab import TrimmedVocab

run = HERE / sys.argv[1]
ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
cfg = ck["config"]["config"]
tok, src, _ = T.load_tokenizer_and_mbart()
vocab = TrimmedVocab.load(tok, Path(cfg["vocab"]))
mb, _ = build_mbart(src, vocab, cfg["n_layers"], cfg["dropout"]); del src
model = SLTModel(mb, cfg["use_visual"], cfg["use_landmarks"], feat_dropout=cfg["feat_dropout"])
model.load_state_dict(ck["model"]); model = model.cuda().eval()
recs, exc = load_records("test", T.uids_of(HERE / "manifests/test1k.csv"))
vis = ShardStore(HERE / "cache/siglip_test") if cfg["use_visual"] else None
lm = ShardStore(HERE / "cache/landmarks_test") if cfg["use_landmarks"] else None
ds = ClipDataset(recs, vocab, vis, lm, cfg["T"], False, 1, cfg["max_label_len"])
dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=make_collate(vocab))
dev = "cuda"
losses, ns = [], []
with torch.no_grad():
    for b in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l, _ = model(b["vis"].to(dev) if b["vis"] is not None else None, b["lm"].to(dev) if b["lm"] is not None else None,
                         b["mask"].to(dev), b["labels"].to(dev), b["decoder_input_ids"].to(dev), 0.0)
        losses.append(l.item()); ns.append(len(b["uids"]))
out = {"run": cfg["name"], "best_epoch": ck["epoch"], "dev_at_best": ck["dev"], "n_test": len(ds), "test_excluded": len(exc),
       "test_loss": float(np.average(losses, weights=ns))}
train_recs, _ = load_records("train", T.uids_of(cfg["train_manifest"]), drop_flagged=True)
seen = {normalize(t) for t in train_recs.values()}
train_words = Counter(w for t in train_recs.values() for w in normalize(t).split())
for name, gen in [("greedy", dict(num_beams=1)), ("beam4_nr3", dict(num_beams=4, no_repeat_ngram_size=3))]:
    torch.cuda.synchronize(); t0 = time.time()
    uids, hyps = T.predict(model, dl, vocab, dev, **gen)
    torch.cuda.synchronize(); el = time.time() - t0
    refs = [recs[u] for u in uids]
    m = corpus_metrics(hyps, refs, with_signature=True)
    m["infer_ms_per_video_batched64"] = 1000 * el / len(uids)
    # key-word analysis: content words (not in the 100 most frequent train words) predicted that occur in the reference
    stop = {w for w, _ in train_words.most_common(100)}
    tp = fp = fn = 0
    for h, r in zip(hyps, refs):
        H = set(normalize(h).split()) - stop; R = set(normalize(r).split()) - stop
        tp += len(H & R); fp += len(H - R); fn += len(R - H)
    m["content_word_precision"] = 100 * tp / max(1, tp + fp); m["content_word_recall"] = 100 * tp / max(1, tp + fn)
    m["test_refs_seen_in_train_subset"] = sum(normalize(r) in seen for r in refs)
    m["exact_match_count"] = sum(normalize(h) == normalize(r) for h, r in zip(hyps, refs))
    out[name] = m
    with open(run / f"test_predictions_{name}.tsv", "w", encoding="utf-8") as f:
        f.write("uid\treference\tprediction\tsent_rougeL\n")
        from rouge_score import rouge_scorer
        sc = rouge_scorer.RougeScorer(["rougeL"])
        for u, h, r in zip(uids, hyps, refs):
            f.write(f"{u}\t{r}\t{h}\t{sc.score(normalize(r), normalize(h))['rougeL'].fmeasure:.3f}\n")
(run / "test_metrics.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: (v if not isinstance(v, dict) else {a: v[a] for a in ("bleu1", "bleu4", "rougeL", "exact_match", "distinct_predictions", "most_repeated_share", "content_word_precision", "content_word_recall", "infer_ms_per_video_batched64") if a in v}) for k, v in out.items() if k != "dev_at_best"}, indent=1))
