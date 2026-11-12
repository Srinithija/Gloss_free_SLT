"""Key-word ("key-sign") recognition probe. iSign has no gloss annotations, so the targets are content words from
the English captions (top-K most frequent non-stopwords in the training captions), multi-label.
Input per clip: mean+max pooled frozen SigLIP (768) and landmark (204) features over the K=32 keyframes.
Model: 2-layer MLP, BCE, early stop on dev mean AUC. Reports test per-word AUC, P@5 vs. a frequency prior.
  python keyword_probe.py [visual|landmarks|hybrid]
"""
import csv, json, sys
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "isign_siglip_slt")); sys.path.insert(0, str(HERE))
import numpy as np, torch, torch.nn as nn


def roc_auc_score(y, p):
    """Mann-Whitney AUC with tie-averaged ranks (sklearn/scipy are not installed in this env)."""
    y = np.asarray(y).astype(bool); p = np.asarray(p, np.float64)
    o = np.argsort(p, kind='mergesort'); sp = p[o]; r = np.empty(len(p))
    _, first, cnt = np.unique(sp, return_index=True, return_counts=True)
    r[o] = np.repeat(first + (cnt + 1) / 2.0, cnt)  # tie-averaged 1-based ranks
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
import stores
from slt.data import load_records
from slt.landmarks import frame_features
from slt.text import normalize
from slt.shards import ShardStore

mode = sys.argv[1] if len(sys.argv) > 1 else "hybrid"
cfg = json.loads((HERE / "configs/m3_hybrid_5k.json").read_text())
S = HERE.parent / "isign_siglip_slt"
STOP = set("a an the is are was were be been being to of in on at for and or but with by from as that this it its it's he she they we you i his her their our your them him us me my not no do does did have has had will would can could shall should may might must so if then than there here what which who whom when where why how all any some also just very more most into over about after before up down out off again only own same such too s t don't".split())

def uids(p): return [r["uid"] for r in csv.DictReader(open(p))]
def pooled(vis, lm, U):
    X = []
    for u in U:
        parts = []
        if vis is not None:
            v = vis.get(u, "v0").astype(np.float32); parts += [v.mean(0), v.max(0)]
        if lm is not None:
            ex = lm.extra[u]
            f = frame_features(lm.get(u, "pose").astype(np.float32), lm.get(u, "lh").astype(np.float32), lm.get(u, "rh").astype(np.float32), ex["width"], ex["height"])
            parts += [f.mean(0), f.max(0), f.std(0)]
        X.append(np.concatenate(parts))
    return np.nan_to_num(np.stack(X))

uv, ul = mode in ("visual", "hybrid"), mode in ("landmarks", "hybrid")
splits = {"train": (HERE / "manifests/train5k.csv", cfg["vis_cache_train"], cfg["lm_cache_train"]),
          "dev": (HERE / "manifests/dev1k.csv", cfg["vis_cache_dev"], cfg["lm_cache_dev"]),
          "test": (HERE / "manifests/test1k.csv", str(HERE / "cache/siglip_test"), str(HERE / "cache/landmarks_test"))}
D = {}
for s, (man, vc, lc) in splits.items():
    recs, _ = load_records(s, uids(man), drop_flagged=(s == "train"))
    U = sorted(recs)
    vis = stores.build(vc, cfg["keyframes_json"]) if uv else None
    lm = stores.build(lc, cfg["keyframes_json"]) if ul else None
    D[s] = (pooled(vis, lm, U), [set(normalize(recs[u]).split()) for u in U])
cnt = Counter(w for ws in D["train"][1] for w in ws if w not in STOP and not w.isdigit())
vocab = [w for w, _ in cnt.most_common(300)]
Y = {s: np.array([[w in ws for w in vocab] for ws in D[s][1]], np.float32) for s in D}
mu, sd = D["train"][0].mean(0), D["train"][0].std(0) + 1e-5
X = {s: torch.tensor((D[s][0] - mu) / sd, dtype=torch.float32) for s in D}
torch.manual_seed(0)
net = nn.Sequential(nn.Dropout(0.2), nn.Linear(X["train"].shape[1], 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, len(vocab)))
opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
def auc(s):
    net.eval()
    with torch.no_grad(): P = torch.sigmoid(net(X[s])).numpy()
    a = [roc_auc_score(Y[s][:, j], P[:, j]) for j in range(len(vocab)) if 0 < Y[s][:, j].sum() < len(Y[s])]
    return float(np.mean(a)), P, len(a)
best, bad, log = -1, 0, []
yt = torch.tensor(Y["train"])
for ep in range(200):
    net.train()
    for idx in torch.randperm(len(yt)).split(128):
        loss = nn.functional.binary_cross_entropy_with_logits(net(X["train"][idx]), yt[idx]); opt.zero_grad(); loss.backward(); opt.step()
    a, _, _ = auc("dev"); log.append(round(a, 4))
    if a > best: best, bad, state = a, 0, {k: v.clone() for k, v in net.state_dict().items()}
    else:
        bad += 1
        if bad >= 10: break
net.load_state_dict(state)
ta, P, nw = auc("test")
prior = Y["train"].mean(0)
def p_at_k(P, k=5): return float(np.mean([Y["test"][i, np.argsort(-P[i])[:k]].mean() for i in range(len(P))]))
rng = np.random.default_rng(0)
shuf_auc = float(np.mean([roc_auc_score(Y["test"][:, j], P[rng.permutation(len(P)), j]) for j in range(len(vocab)) if 0 < Y["test"][:, j].sum() < len(P)]))
per = sorted([(vocab[j], roc_auc_score(Y["test"][:, j], P[:, j]), int(Y["test"][:, j].sum())) for j in range(len(vocab)) if Y["test"][:, j].sum() >= 10], key=lambda x: -x[1])
out = {"mode": mode, "n_train": len(yt), "n_dev": len(Y["dev"]), "n_test": len(P), "n_words": len(vocab), "input_dim": X["train"].shape[1],
       "dev_auc_best": best, "epochs": len(log), "test_mean_auc": ta, "test_words_scored": nw, "test_auc_shuffled_control": shuf_auc,
       "test_P@5_model": p_at_k(P), "test_P@5_frequency_prior": p_at_k(np.tile(prior, (len(P), 1))),
       "top_words_by_test_auc(>=10 test clips)": [(w, round(a, 3), n) for w, a, n in per[:15]],
       "bottom_words": [(w, round(a, 3), n) for w, a, n in per[-5:]]}
(HERE / "reports").mkdir(exist_ok=True)
(HERE / f"reports/keyword_probe_{mode}.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out))
