"""One-batch sanity check of Model 3: shapes at every stage, finite loss, gradient flow, parameter updates,
and the video-id -> caption mapping for a few clips."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "isign_siglip_slt"))
import torch
import scripts.train as T
T.PROJECT = HERE
from torch.utils.data import DataLoader
from slt.data import ClipDataset, load_records, make_collate
from slt.model import SLTModel, build_mbart
from slt.vocab import TrimmedVocab

cfg = T.DEFAULTS | json.loads((HERE / "configs/m3_hybrid.json").read_text())
T.seed_all(0)
tok, src, rev = T.load_tokenizer_and_mbart()
print("tokenizer src/tgt lang:", tok.src_lang, tok.tgt_lang)
vocab = TrimmedVocab.load(tok, Path(cfg["vocab"]))
mb, ids = build_mbart(src, vocab, cfg["n_layers"], cfg["dropout"]); del src
model = SLTModel(mb, True, True).cuda()
recs, exc = load_records("train", T.uids_of(cfg["train_manifest"]), drop_flagged=True)
ds = ClipDataset(recs, vocab, T.make_store(cfg, "vis_cache_train"), T.make_store(cfg, "lm_cache_train"), 32, True, 1, 64, 0.8)
b = next(iter(DataLoader(ds, batch_size=8, collate_fn=make_collate(vocab))))
for u in b["uids"][:4]:
    print("MAP", u, "->", recs[u], "| label round-trip:", vocab.decode([t for t in ds[ds.uids.index(u)]["labels"]]))
print("batch vis", tuple(b["vis"].shape), "lm", tuple(b["lm"].shape), "mask", tuple(b["mask"].shape),
      "labels", tuple(b["labels"].shape), "dec_in", tuple(b["decoder_input_ids"].shape),
      "vis finite", bool(torch.isfinite(b["vis"]).all()), "lm finite", bool(torch.isfinite(b["lm"]).all()),
      "vis std", float(b["vis"].float().std()))
shapes = {}
def hook(name):
    def f(m, i, o): shapes.setdefault(name, tuple((o if hasattr(o, "shape") else o[0]).shape))
    return f
for n in ["lm_proj", "conv", "bn", "pool", "proj", "mbart.model.encoder", "mbart.model.decoder", "mbart.lm_head"]:
    model.get_submodule(n).register_forward_hook(hook(n))
before = {n: p.detach().clone() for n, p in model.named_parameters()}
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
model.train()
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss, logits = model(b["vis"].cuda(), b["lm"].cuda(), b["mask"].cuda(), b["labels"].cuda(), b["decoder_input_ids"].cuda())
print("SHAPES", json.dumps(shapes))
print("loss", float(loss), "finite", bool(torch.isfinite(loss)), "nonzero", float(loss) > 0)
loss.backward()
gn = {g: float(sum(p.grad.norm() ** 2 for n, p in model.named_parameters() if n.startswith(g) and p.grad is not None) ** .5)
      for g in ["lm_proj", "conv", "bn", "proj", "mbart.model.encoder", "mbart.model.decoder", "mbart.model.shared"]}
print("GRAD NORMS", json.dumps(gn))
opt.step()
ch = sum((p.detach() != before[n]).any().item() for n, p in model.named_parameters())
print(f"params updated: {ch}/{len(before)} tensors; trainable params {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
      " (SigLIP is not part of the model: features are precomputed by the frozen encoder)")
model.eval()
out = model.generate(b["vis"][:2].cuda(), b["lm"][:2].cuda(), b["mask"][:2].cuda(), max_new_tokens=20)
print("generate ids shape", tuple(out.shape), "decoded:", [vocab.decode(o[1:]) for o in out.tolist()])
