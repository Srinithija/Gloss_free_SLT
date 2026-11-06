"""Score a predictions TSV (uid, reference, prediction) with the common metrics + breakdowns; optionally log to the
test ledger. References are re-read from splits/v1, never trusted from the file.

  python scripts/score_predictions.py --tsv runs/baseline_resnet_old/test_full_predictions.tsv --split test \
      --system resnet_old_baseline --ledger final --note "..."
"""
import argparse
import csv
import datetime
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate import LEDGER, breakdown  # noqa: E402
from slt.data import load_records  # noqa: E402
from slt.metrics import corpus_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--ledger", choices=["none", "final", "repeat"], default="none")
    ap.add_argument("--checkpoint_sha256", default="")
    ap.add_argument("--setting", default="")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.tsv, encoding="utf-8"), delimiter="\t"))
    preds = {r["uid"]: r["prediction"] for r in rows}
    recs, exc = load_records(a.split, list(preds))
    uids = sorted(recs)
    hyps = [preds[u] for u in uids]
    m = corpus_metrics(hyps, [recs[u] for u in uids], with_signature=True)
    m["breakdown"] = breakdown(uids, hyps, recs)
    m["n_in_file"], m["excluded_captions"] = len(preds), len(exc)
    out = Path(a.tsv).with_suffix(".metrics.json")
    out.write_text(json.dumps(m, indent=1, ensure_ascii=False))
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()
                      if k != "breakdown" and not k.startswith("signature")}, ensure_ascii=False))
    if a.ledger != "none":
        new = not LEDGER.exists()
        with open(LEDGER, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["timestamp", "kind", "system", "setting", "checkpoint_sha256", "n", "bleu1", "bleu4",
                            "rougeL", "chrf", "exact_match", "distinct_predictions", "note"])
            w.writerow([datetime.datetime.now().isoformat(timespec="seconds"), a.ledger, a.system, a.setting,
                        a.checkpoint_sha256, m["n"], f'{m["bleu1"]:.2f}', f'{m["bleu4"]:.2f}', f'{m["rougeL"]:.2f}',
                        f'{m["chrf"]:.2f}', f'{m["exact_match"]:.2f}', m["distinct_predictions"], a.note])


if __name__ == "__main__":
    main()
