"""Build terminal-style training/testing logs from the saved run files only (nothing is re-run or estimated).
Output: logs/project_logs/{training_history.txt, test_results.txt, training_*.png}
"""
import csv
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
ROOT = H.parent
sys.path.insert(0, str(ROOT / "isign_siglip_slt"))
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from slt.metrics import corpus_metrics  # noqa: E402
from slt.text import normalize  # noqa: E402

OUT = H / "logs" / "project_logs"
OUT.mkdir(parents=True, exist_ok=True)
OLD = Path(r"D:\iSign_Experiments\GFSLT\optimized_v2_20261023")
RULE = "-" * 96
FONT = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 26)


def clip(s, n):
    return s if len(s) <= n else s[:n - 1] + "…"


def png(lines, path, mark=None):
    """Dark terminal-style image; header lines blue, lines containing `mark` green."""
    w = max(len(l) for l in lines)
    cw, ch = FONT.getlength("M"), 34
    im = Image.new("RGB", (int(cw * w) + 48, ch * len(lines) + 48), (30, 30, 30))
    d = ImageDraw.Draw(im)
    for i, l in enumerate(lines):
        col = (212, 212, 212)
        if l.startswith(("TRAINING", "TEST")):
            col = (120, 190, 255)
        if mark and mark in l:
            col = (130, 220, 150)
        d.text((24, 24 + i * ch), l, font=FONT, fill=col)
    im.save(path)


def history(title, setup, rows, best, best_tag):
    L = [RULE, f"TRAINING HISTORY - {title}", setup, RULE,
         "epoch | train loss | val loss | BLEU-1 | BLEU-4 | ROUGE-L | sec/epoch", RULE]
    for r in rows:
        tag = f"  <- {best_tag}" if r["epoch"] == best else ""
        L.append(f"{r['epoch']:5d} | {r['train']:10.4f} | {r['val']:8.4f} | {r['b1']:6.2f} | "
                 f"{r['b4']:6.2f} | {r['rl']:7.2f} | {r['sec']:6.0f}{tag}")
    L.append(RULE)
    return L


def test_block(where, model, dec, rows, m):
    """Scores only: this repository must not contain dataset captions. Counts are checked against the stored metrics."""
    assert len(rows) == m["n"], (len(rows), m["n"])
    hdr = [RULE, f"TEST, {where} - {model}, {dec}",
           f"TEST : BLEU-1 {m['bleu1']:.2f} | BLEU-4 {m['bleu4']:.2f} | "
           f"ROUGE-L {m['rougeL']:.2f} | distinct outputs {m['distinct_predictions']}", RULE]

    return hdr


hist_all, test_all = [], []

# This experiment and the pilot share one log format.
RUNS = [("m3_hybrid_5k", "Model 3 (SigLIP + MediaPipe), 4,921 train clips"),
        ("visual_only_5k", "Visual-only SigLIP baseline, 4,921 train clips"),
        ("m3_hybrid", "Model 3 (SigLIP + MediaPipe), 2,000 train clips"),
        ("visual_only", "Visual-only SigLIP baseline, 2,000 train clips")]
PILOT = [("pilot_hybrid", "Pilot: Model 3, T=64, 3 views"), ("pilot_visual", "Pilot: visual-only, T=64, 3 views"),
         ("pilot_landmarks", "Pilot: landmarks-only, T=64"), ("pilot_visual_noaug", "Pilot: visual-only, clean view"),
         ("pilot_visual_T96", "Pilot: visual-only, T=96")]
ALL = [(H / "runs" / k, t, k) for k, t in RUNS]
ALL += [(ROOT / "isign_siglip_slt" / "runs" / k, t + ", 3,000 train clips (no test run)", k) for k, t in PILOT]
for d, title, key in ALL:
    rows = [json.loads(l) for l in open(d / "train_log.jsonl", encoding="utf-8") if '"epoch"' in l]
    best = max(rows, key=lambda x: x["dev_bleu4"])["epoch"]
    P = list(csv.DictReader(open(d / "dev_predictions_best.tsv", encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))
    assert len(P) == json.loads((d / "config.json").read_text())["n_dev"], (key, len(P))
    R = [dict(epoch=x["epoch"], train=x["train_loss"], val=x["dev_loss_nols"], b1=x["dev_bleu1"], b4=x["dev_bleu4"],
              rl=x["dev_rougeL"], sec=60 * x["train_min"]) for x in rows]
    L = history(title, f"(validation = {len(P)} dev clips, greedy decoding, every epoch; epochs start at 0)", R, best, "best")
    hist_all += L + [""]
    png(L, OUT / f"training_{key}.png", "<- ")

# ResNet baseline: predictions were saved for every epoch, so metrics are re-scored per epoch.
R = []
for l in open(OLD / "runs/main_full/log.jsonl", encoding="utf-8"):
    x = json.loads(l)
    ps = [json.loads(p) for p in open(OLD / f"runs/main_full/dev_predictions_epoch{x['epoch']:02d}.jsonl", encoding="utf-8")]
    m = corpus_metrics([p["pred"] for p in ps], [p["ref"] for p in ps])
    R.append(dict(epoch=x["epoch"], train=x["train_loss"], val=x["dev_loss"], b1=m["bleu1"], b4=m["bleu4"], rl=m["rougeL"],
                  sec=60 * x["epoch_min"]))
L = history("ResNet-18 + TemporalConv + mBART baseline, about 101k train clips (earlier work, 23 Oct)",
            f"(validation = {m['n']} dev clips, greedy decoding, every epoch; metrics re-scored from saved predictions)",
            R, 7, "selected, tested with beam-4")
hist_all += L + [""]
png(L, OUT / "training_resnet_baseline.png", "<- ")

for key, title in RUNS:
    d = H / "runs" / key
    M = json.loads((d / "test_metrics.json").read_text(encoding="utf-8"))
    for dec, dec_name in (("greedy", "greedy"), ("beam4_nr3", "beam-4, no-repeat-3")):
        n = sum(1 for _ in open(d / f"test_predictions_{dec}.tsv", encoding="utf-8")) - 1
        test_all += test_block("test1k (993 held-out clips)", title, dec_name, [None] * n, M[dec])[1:3] + [""]

bs = json.loads((ROOT / "isign_siglip_slt/reports/baseline_rescored.json").read_text(encoding="utf-8"))["test_all"]
n = sum(1 for _ in csv.DictReader(open(OLD / "reports/final_eval/test_predictions.csv", encoding="utf-8")))
test_all += test_block("subset of 1,979 clips (different clips from test1k)", "ResNet-18 baseline, about 101k train clips",
                       "beam-4, no-repeat-3", [None] * n, bs)[1:3] + [""]

(OUT / "training_history.txt").write_text("\n".join(hist_all), encoding="utf-8")
(OUT / "test_results.txt").write_text("\n".join(test_all), encoding="utf-8")
print("written to", OUT)
