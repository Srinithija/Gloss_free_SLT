"""Read-only audit of iSign: clips, frames, resolution, corrupt clips, text statistics, split overlap.

Usage: python scripts/audit_data.py [--workers 12] [--limit N]
Writes only into isign_siglip_slt/manifests and reports/audit.
"""
import argparse
import csv
import gzip
import json
import os
import pickle
import random
import re
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2
import lmdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slt.paths import LMDB_ROOT, MANIFESTS, MBART_TRIMMED, REPORTS, SPLIT_NAMES, SPLITS, VIDEOS  # noqa: E402
from slt.text import exclusion_reason, normalize  # noqa: E402

OUT = REPORTS / "audit"


def scan_clip(args):
    split, uid = args
    r = {"uid": uid, "split": split, "lmdb_ok": 0, "n_stored": -1, "orig_frames_meta": -1, "lmdb_hw": "",
         "video_ok": 0, "video_w": -1, "video_h": -1, "video_fps": -1.0, "video_frames": -1, "error": ""}
    errs = []
    d = LMDB_ROOT / split / uid
    try:
        env = lmdb.open(str(d), readonly=True, lock=False, readahead=False, meminit=False)
        try:
            with env.begin() as t:
                proto = pickle.loads(t.get(b"protocol"))
                keys = pickle.loads(t.get(pickle.dumps("keys", protocol=proto)))
                meta = pickle.loads(t.get(b"metadata")) if t.get(b"metadata") else {}
                r["n_stored"] = len(keys)
                r["orig_frames_meta"] = meta.get("original_frame_count", -1)
                shapes = set()
                for k in (keys[0], keys[len(keys) // 2], keys[-1]):
                    im = cv2.imdecode(np.frombuffer(t.get(k), np.uint8), cv2.IMREAD_COLOR)
                    shapes.add(im.shape[:2])
                r["lmdb_hw"] = "|".join(f"{h}x{w}" for h, w in sorted(shapes))
                missing = sum(t.get(k) is None for k in keys)
                if missing:
                    errs.append(f"lmdb_missing_frames={missing}")
                else:
                    r["lmdb_ok"] = 1
        finally:
            env.close()
    except Exception as e:  # noqa: BLE001 - every failure is recorded, never dropped
        errs.append(f"lmdb:{type(e).__name__}:{e}"[:200])
    vp = VIDEOS / f"{uid}.mp4"
    try:
        cap = cv2.VideoCapture(str(vp))
        if cap.isOpened():
            r.update(video_w=int(cap.get(3)), video_h=int(cap.get(4)), video_fps=round(cap.get(5), 3),
                     video_frames=int(cap.get(7)))
            ok, _ = cap.read()
            r["video_ok"] = int(ok and r["video_frames"] > 0)
            if not ok:
                errs.append("video_first_frame_unreadable")
        else:
            errs.append("video_cannot_open" if vp.exists() else "video_missing")
        cap.release()
    except Exception as e:  # noqa: BLE001
        errs.append(f"video:{type(e).__name__}:{e}"[:200])
    r["error"] = ";".join(errs)
    return r


def pct(a, qs=(0, 50, 90, 99, 100)):
    a = np.asarray(a, dtype=float)
    return {f"p{q}": round(float(np.percentile(a, q)), 2) for q in qs} | {"mean": round(float(a.mean()), 2)}


def load_split_csv(split):
    with open(SPLITS / f"{split}.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def contact_sheets(rows_by_split, n_clips=30, cols=6, seed=0):
    """Save 3 sheets: random clips from LMDB (4 frames each), and the same clips from the original MP4."""
    rng = random.Random(seed)
    pool = [(s, r["uid"]) for s in SPLIT_NAMES for r in rows_by_split[s]]
    picks = rng.sample(pool, n_clips)
    tiles = []
    for split, uid in picks:
        env = lmdb.open(str(LMDB_ROOT / split / uid), readonly=True, lock=False)
        with env.begin() as t:
            proto = pickle.loads(t.get(b"protocol"))
            keys = pickle.loads(t.get(pickle.dumps("keys", protocol=proto)))
            sel = [keys[int(i)] for i in np.linspace(0, len(keys) - 1, 4)]
            ims = [cv2.imdecode(np.frombuffer(t.get(k), np.uint8), cv2.IMREAD_COLOR) for k in sel]
        env.close()
        strip = np.concatenate([cv2.resize(im, (160, 160)) for im in ims], 1)
        cv2.putText(strip, uid[:28], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(strip)
    per = 10
    paths = []
    for i in range(0, len(tiles), per):
        chunk = tiles[i:i + per]
        rows = [np.concatenate(chunk[j:j + 2], 1) if j + 1 < len(chunk) else
                np.concatenate([chunk[j], np.zeros_like(chunk[j])], 1) for j in range(0, len(chunk), 2)]
        p = OUT / f"contact_lmdb_{i // per + 1}.jpg"
        cv2.imwrite(str(p), np.concatenate(rows, 0))
        paths.append(p)
    # original-resolution middle frames, unscaled, for resolution inspection
    full = []
    for split, uid in picks[:12]:
        cap = cv2.VideoCapture(str(VIDEOS / f"{uid}.mp4"))
        n = int(cap.get(7))
        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ok, im = cap.read()
        cap.release()
        if ok:
            full.append(cv2.copyMakeBorder(cv2.resize(im, (300, int(300 * im.shape[0] / im.shape[1]))),
                                           2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    h = max(x.shape[0] for x in full)
    full = [cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 0, cv2.BORDER_CONSTANT) for x in full]
    rows = [np.concatenate(full[j:j + 4], 1) for j in range(0, len(full) - len(full) % 4, 4)]
    p = OUT / "contact_mp4_midframes.jpg"
    cv2.imwrite(str(p), np.concatenate(rows, 0))
    paths.append(p)
    return [str(x) for x in paths], [u for _, u in picks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    rows_by_split = {s: load_split_csv(s) for s in SPLIT_NAMES}
    summary = {"counts_csv": {s: len(v) for s, v in rows_by_split.items()}}

    # labels in the LMDB must match the split CSVs
    lab_check = {}
    for s in SPLIT_NAMES:
        with gzip.open(LMDB_ROOT / "labels" / f"labels.{s}", "rb") as f:
            lab = pickle.load(f)
        csv_map = {r["uid"]: r["text"] for r in rows_by_split[s]}
        lab_check[s] = {"n_labels": len(lab), "uid_sets_equal": set(lab) == set(csv_map),
                        "text_mismatches": sum(" ".join(str(lab[u]["text"]).split()) != " ".join(csv_map[u].split())
                                               for u in csv_map if u in lab)}
    summary["lmdb_labels_vs_csv"] = lab_check

    # --- per-clip scan
    jobs = [(s, r["uid"]) for s in SPLIT_NAMES for r in rows_by_split[s]]
    if a.limit:
        jobs = random.Random(0).sample(jobs, a.limit)
    res = []
    with Pool(a.workers) as p:
        for i, r in enumerate(p.imap_unordered(scan_clip, jobs, chunksize=64), 1):
            res.append(r)
            if i % 10000 == 0:
                print(f"scanned {i}/{len(jobs)} in {time.time() - t0:.0f}s", flush=True)
    res.sort(key=lambda r: (SPLIT_NAMES.index(r["split"]), r["uid"]))
    with open(MANIFESTS / "audit_clips.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)
    bad = [r for r in res if r["error"]]
    with open(MANIFESTS / "corrupt_clips.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["uid", "split", "error"])
        w.writeheader()
        w.writerows({k: r[k] for k in ("uid", "split", "error")} for r in bad)

    clip = {}
    for s in SPLIT_NAMES:
        rs = [r for r in res if r["split"] == s and r["video_ok"]]
        if not rs:
            continue
        dur = [r["video_frames"] / r["video_fps"] for r in rs if r["video_fps"] > 0]
        clip[s] = {"n_scanned": sum(r["split"] == s for r in res),
                   "video_frames": pct([r["video_frames"] for r in rs]),
                   "duration_s": pct(dur),
                   "lmdb_stored_frames": pct([r["n_stored"] for r in rs if r["n_stored"] > 0]),
                   "frac_stored_128": round(float(np.mean([r["n_stored"] == 128 for r in rs])), 4)}
    allok = [r for r in res if r["video_ok"]]
    summary["clips"] = clip
    summary["video_resolution_top"] = Counter(f'{r["video_w"]}x{r["video_h"]}' for r in allok).most_common(12)
    summary["video_resolution_min_side"] = pct([min(r["video_w"], r["video_h"]) for r in allok])
    summary["video_fps_top"] = Counter(r["video_fps"] for r in allok).most_common(8)
    summary["lmdb_frame_hw"] = Counter(r["lmdb_hw"] for r in res).most_common(5)
    summary["n_errors"] = len(bad)
    summary["error_kinds"] = Counter(e.split(":")[0].split("=")[0] for r in bad for e in r["error"].split(";")).most_common()
    summary["lmdb_bytes_total_GB"] = round(sum(
        (LMDB_ROOT / r["split"] / r["uid"] / "data.mdb").stat().st_size
        for r in res if (LMDB_ROOT / r["split"] / r["uid"] / "data.mdb").exists()) / 1e9, 2)
    summary["video_bytes_total_GB"] = round(sum(
        (VIDEOS / f'{r["uid"]}.mp4').stat().st_size for r in res if (VIDEOS / f'{r["uid"]}.mp4').exists()) / 1e9, 2)

    # --- text statistics (kept/excluded, lengths, tokens, vocabulary, duplicates)
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(MBART_TRIMMED / "sentencepiece.bpe.model"))
    text = {}
    norm_sets = {}
    raw_sets = {}
    for s in SPLIT_NAMES:
        texts = [r["text"] for r in rows_by_split[s]]
        excl = Counter(exclusion_reason(t) for t in texts)
        kept = [" ".join(t.split()) for t in texts if not exclusion_reason(t)]
        words = [len(t.split()) for t in kept]
        toks = [len(sp.encode(t)) for t in kept]
        nw = [normalize(t) for t in kept]
        vocab = Counter(w for t in nw for w in t.split())
        c = Counter(kept)
        text[s] = {"n": len(texts), "excluded": {k: v for k, v in excl.items() if k}, "kept": len(kept),
                   "words": pct(words), "sp_tokens": pct(toks),
                   "frac_tokens_gt_60": round(float(np.mean(np.array(toks) > 60)), 4),
                   "vocab_normalized": len(vocab), "vocab_singletons": sum(v == 1 for v in vocab.values()),
                   "unique_texts": len(c), "frac_unique": round(len(c) / len(kept), 4),
                   "top_texts": c.most_common(15),
                   "frac_starts_upper": round(float(np.mean([t[:1].isupper() for t in kept])), 4),
                   "frac_ends_punct": round(float(np.mean([t[-1:] in ".?!" for t in kept])), 4),
                   "frac_has_digit": round(float(np.mean([bool(re.search(r"\d", t)) for t in kept])), 4),
                   "frac_non_ascii": round(float(np.mean([any(ord(ch) > 127 for ch in t) for t in kept])), 4),
                   "frac_one_word": round(float(np.mean(np.array(words) == 1)), 4)}
        norm_sets[s] = nw
        raw_sets[s] = kept
        if s == "train":
            train_vocab = vocab
    for s in ("dev", "test"):
        tr_raw, tr_norm = set(raw_sets["train"]), set(norm_sets["train"])
        toks = [w for t in norm_sets[s] for w in t.split()]
        text[s]["overlap_with_train"] = {
            "verbatim_frac": round(float(np.mean([t in tr_raw for t in raw_sets[s]])), 4),
            "normalized_frac": round(float(np.mean([t in tr_norm for t in norm_sets[s]])), 4),
            "normalized_count": int(sum(t in tr_norm for t in norm_sets[s])),
            "word_oov_rate_vs_train": round(float(np.mean([w not in train_vocab for w in toks])), 4)}
    summary["text"] = text

    # source-video / channel structure (no signer field exists in the CSV)
    srcs = {s: Counter(r["source_video_id"] for r in rows_by_split[s]) for s in SPLIT_NAMES}
    summary["source_videos"] = {s: {"n": len(c), "clips_per_source": pct(list(c.values())),
                                    "id_style_youtube11": sum(len(k) == 11 for k in c),
                                    "id_style_hex12": sum(bool(re.fullmatch(r"[0-9a-f]{12}", k)) for k in c)}
                                for s, c in srcs.items()}
    summary["source_overlap_train_test"] = len(set(srcs["train"]) & set(srcs["test"]))

    sheets, picks = contact_sheets(rows_by_split)
    summary["contact_sheets"] = sheets
    summary["contact_sheet_uids"] = picks
    summary["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n_errors", "error_kinds", "runtime_s")}, indent=1))


if __name__ == "__main__":
    main()
