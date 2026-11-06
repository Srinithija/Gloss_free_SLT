"""Correctness gate run before any long training. Writes <exp>/reports/correctness_tests.json.
Every check raises on failure; nothing is silently skipped."""
from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import MBartTokenizer

from isign_opt.common import (EXP_ROOT, FULL_LMDB, LANG, MEAN, STD, SUBSET_LMDB, TOKENIZER_DIR, FrameEncoder,
                              center_crop, load_labels)
from isign_opt.data import BucketBatchSampler, Collate, FeatureCaptionDataset, FeatureStore, build_split
from isign_opt.metrics import translation_metrics
from isign_opt.model import FeatureSLT, conv_output_lengths

R = {}


def check(name, cond, detail=""):
    R[name] = {"pass": bool(cond), "detail": detail}
    print(("PASS " if cond else "FAIL ") + name + (f" :: {detail}" if detail else ""), flush=True)
    if not cond:
        json.dump(R, open(EXP_ROOT / "reports" / "correctness_tests.json", "w"), indent=1)
        raise SystemExit(f"correctness check failed: {name}")


def main():
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    dev = torch.device("cuda")
    tok = MBartTokenizer.from_pretrained(TOKENIZER_DIR, src_lang=LANG, tgt_lang=LANG)
    store = FeatureStore("dev")
    recs, _ = build_split("dev", "subset")
    recs = {u: t for u, t in recs.items() if u in store}
    check("dev cache available", len(recs) >= 64, f"{len(recs)} cached dev records")

    # 1. labels/captions/features pairing
    labels = load_labels(SUBSET_LMDB / "labels" / "labels.dev")
    ds = FeatureCaptionDataset(recs, store, train=False)
    bad = [ds.uids[i] for i in range(len(ds)) if ds[i][2] != " ".join(labels[ds.uids[i]]["text"].split())]
    check("dataset captions match source labels", not bad, f"{len(ds)} checked")

    # 2. cache == direct extraction through the ORIGINAL PIL loader path (independent decoder)
    from PIL import Image
    import io, pickle, lmdb
    enc = FrameEncoder().to(dev)
    diffs = []
    for u in random.sample(ds.uids, 5):
        env = lmdb.open(str(FULL_LMDB / "dev" / u), readonly=True, lock=False)
        with env.begin() as t:
            keys = pickle.loads(t.get(pickle.dumps("keys", protocol=4)))
            fr = np.stack([np.array(Image.open(io.BytesIO(t.get(k))).convert("RGB")) for k in keys])
        env.close()
        direct = enc(torch.from_numpy(np.ascontiguousarray(center_crop(fr))).to(dev)).float().cpu()
        cached = torch.from_numpy(store.get(u)).float()
        check(f"cache length matches LMDB frames ({u})", direct.shape == cached.shape, str(tuple(cached.shape)))
        diffs.append(float(torch.nn.functional.cosine_similarity(direct, cached, dim=1).min()))
    check("cached features equal direct extraction (min cosine > 0.99)", min(diffs) > 0.99, f"min cosine {min(diffs):.5f}")

    # 3. bucketing preserves every sample exactly once and keeps uid/caption pairs
    s = BucketBatchSampler(ds.lengths, 8, shuffle=True, seed=1)
    flat = [i for b in s.batches() for i in b]
    check("bucket sampler covers each index once", sorted(flat) == list(range(len(ds))), f"{len(flat)} indices")
    coll = Collate(tok)
    b = coll([ds[i] for i in s.batches()[0]])
    check("collated batch keeps uid-caption pairs", all(recs[u] == t for u, t in zip(b["uids"], b["texts"])))
    dec = tok.batch_decode(b["labels"], skip_special_tokens=True)
    check("target tokens decode back to captions", sum(d.strip() == t for d, t in zip(dec, b["texts"])) >= len(dec) - 1,
          f"e.g. {b['texts'][0]!r} -> ids end {b['labels'][0][b['label_mask'][0].bool()][-2:].tolist()} (eos=2, en_XX={tok.lang_code_to_id[LANG]})")
    check("feature padding/lengths consistent", b["feats"].shape[1] == int((torch.ceil(b["lengths"] / 4.0) * 4 + 16).max()))

    # 4. model init repaired: pretrained embeddings, tied LM head, mapped layers
    model = FeatureSLT().to(dev)
    pre = torch.load(TOKENIZER_DIR / "pytorch_model.bin", map_location="cpu")
    m = model.mbart
    check("decoder embeddings are pretrained", torch.equal(m.model.decoder.embed_tokens.weight.cpu(), pre["model.shared.weight"]))
    check("LM head tied to pretrained embeddings", m.lm_head.weight.data_ptr() == m.model.shared.weight.data_ptr())
    check("decoder layer 1 = pretrained layer 5",
          torch.equal(m.model.decoder.layers[1].fc1.weight.cpu(), pre["model.decoder.layers.5.fc1.weight"]))
    check("encoder layer 2 = pretrained layer 11",
          torch.equal(m.model.encoder.layers[2].self_attn.k_proj.weight.cpu(), pre["model.encoder.layers.11.self_attn.k_proj.weight"]))
    del pre

    # 5. forward/backward, finite loss, gradients reach every component
    x, lens, y, mk = b["feats"].to(dev), b["lengths"], b["labels"].to(dev), b["label_mask"].to(dev)
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x, lens, y, mk)
        loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                                 y.masked_fill(mk == 0, -100).reshape(-1), ignore_index=-100)
    loss.backward()
    check("loss finite", torch.isfinite(loss).item(), f"{loss.item():.3f} (ln V = {np.log(len(tok)):.2f})")
    for comp in ["temporal", "proj", "mbart.model.encoder", "mbart.model.decoder"]:
        g = sum(p.grad.norm().item() ** 2 for n, p in model.named_parameters() if n.startswith(comp) and p.grad is not None) ** 0.5
        check(f"gradients reach {comp}", g > 0 and np.isfinite(g), f"grad norm {g:.4f}")
    model.zero_grad(set_to_none=True)

    # 6. tiny overfit (debug only, not generalisation evidence)
    small = dict(list(recs.items())[:16])
    sds = FeatureCaptionDataset(small, store, train=False)
    sb = coll([sds[i] for i in range(len(sds))])
    x, lens, y, mk = sb["feats"].to(dev), sb["lengths"], sb["labels"].to(dev), sb["label_mask"].to(dev)
    model = FeatureSLT(dropout=0.0, feat_dropout=0.0).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    losses = []
    for it in range(300):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(x, lens, y, mk)
            l = torch.nn.functional.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), y.masked_fill(mk == 0, -100).reshape(-1), ignore_index=-100)
        opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); losses.append(l.item())
    model.eval()
    start = tok.lang_code_to_id[LANG]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(x, lens, start, num_beams=1)
        perm = torch.roll(torch.arange(len(lens)), 1)
        lp = model(x, lens, y, mk); ls_ = model(x[perm], lens[perm], y, mk)
        ce = lambda q: torch.nn.functional.cross_entropy(q.float().reshape(-1, q.shape[-1]), y.masked_fill(mk == 0, -100).reshape(-1), ignore_index=-100).item()
        paired, shuffled = ce(lp), ce(ls_)
    preds = [p.strip() for p in tok.batch_decode(out, skip_special_tokens=True)]
    met = translation_metrics(preds, sb["texts"])
    check("tiny overfit: loss decreases", losses[-1] < 0.2 * losses[0], f"{losses[0]:.3f} -> {losses[-1]:.4f}")
    check("tiny overfit: predictions reproduce captions", met["bleu4"] > 80, f"BLEU-4 {met['bleu4']:.1f}, exact {met['exact_match']:.0f}%")
    check("predictions differ per video (no collapse)", met["unique_predictions"] >= 15, f"{met['unique_predictions']}/16 unique")
    check("visual input matters: shuffled-video loss >> paired", shuffled > paired + 1.0, f"paired {paired:.3f}, shuffled {shuffled:.3f}")
    R["tiny_overfit_examples"] = [{"ref": r, "pred": p} for r, p in list(zip(sb["texts"], preds))[:5]]

    # 7. checkpoint round trip reproduces generation
    with tempfile.TemporaryDirectory() as d:
        torch.save({"model": model.state_dict()}, Path(d) / "ck.pt")
        m2 = FeatureSLT(pretrained=False).to(dev)
        m2.load_state_dict(torch.load(Path(d) / "ck.pt", map_location="cpu")["model"])
        m2.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out2 = m2.generate(x, lens, start, num_beams=1)
    check("reloaded checkpoint reproduces generation", tok.batch_decode(out2, skip_special_tokens=True) == tok.batch_decode(out, skip_special_tokens=True))

    (EXP_ROOT / "reports").mkdir(exist_ok=True)
    json.dump(R, open(EXP_ROOT / "reports" / "correctness_tests.json", "w"), indent=1, ensure_ascii=False)
    print("ALL CORRECTNESS CHECKS PASSED")


if __name__ == "__main__":
    main()
