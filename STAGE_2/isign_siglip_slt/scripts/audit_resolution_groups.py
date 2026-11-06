"""Read-only: resolution groups from manifests/audit_clips.csv and a mid-frame contact sheet per group."""
import csv, random, sys
from collections import Counter
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slt.paths import MANIFESTS, REPORTS, VIDEOS

rows = list(csv.DictReader(open(MANIFESTS / "audit_clips.csv", encoding="utf-8")))
def group(r):
    w, h = int(r["video_w"]), int(r["video_h"])
    if (w, h) == (300, 300): return "300x300"
    if min(w, h) < 256: return "small(<256)"
    return "large(>=256, not 300sq)"
g = Counter(group(r) for r in rows)
print(g)
for name in g:
    sub = [r for r in rows if group(r) == name]
    print(name, "splits", Counter(r["split"] for r in sub), "sources", len({r["uid"].rsplit("-", 1)[0] for r in sub}),
          "sizes", Counter(f'{r["video_w"]}x{r["video_h"]}' for r in sub).most_common(6))
    if name == "300x300": continue
    picks = random.Random(1).sample(sub, min(16, len(sub)))
    tiles = []
    for r in picks:
        cap = cv2.VideoCapture(str(VIDEOS / f'{r["uid"]}.mp4')); n = int(cap.get(7))
        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2); ok, im = cap.read(); cap.release()
        im = cv2.resize(im, (240, int(240 * im.shape[0] / im.shape[1])))
        im = cv2.copyMakeBorder(im, 0, max(0, 300 - im.shape[0]), 0, 0, cv2.BORDER_CONSTANT)[:300]
        cv2.putText(im, f'{r["video_w"]}x{r["video_h"]}', (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(im)
    while len(tiles) % 8: tiles.append(np.zeros_like(tiles[0]))
    sheet = np.concatenate([np.concatenate(tiles[i:i + 8], 1) for i in range(0, len(tiles), 8)], 0)
    out = REPORTS / "audit" / f"contact_{name.split('(')[0]}.jpg"
    cv2.imwrite(str(out), sheet); print("wrote", out)
