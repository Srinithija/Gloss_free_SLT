"""Re-score the ResNet baseline's SAVED predictions (no model run) with the new metric protocol.

Sources (read-only):
  test: <OLD_EXP>/reports/final_eval/test_predictions.csv  (selected model, beam4 no-repeat-3, the one test run)
  dev : <OLD_EXP>/reports/prediction_logs/dev_predictions.log (same model/decoding)
References are re-read from splits/v1 (not from the files) and checked against the saved references.
"""
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slt.data import load_records  # noqa: E402
from slt.metrics import corpus_metrics  # noqa: E402
from slt.paths import MANIFESTS, OLD_EXP, REPORTS  # noqa: E402


def read_dev_log(path):
    out, uid, ref = {}, None, None
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\[dev \d+/\d+\] id: (\S+)", line)
        if m:
            uid = m.group(1)
        elif line.startswith("ref:"):
            ref = line[4:].strip()
        elif line.startswith("pred:") and uid:
            out[uid] = (ref, line[5:].strip())
            uid = None
    return out


def main():
    flagged = {r["uid"] for r in csv.DictReader(open(MANIFESTS / "flag_small_resolution.csv", encoding="utf-8"))}
    res = {}
    test = {r["sample_id"]: (r["reference"], r["prediction"])
            for r in csv.DictReader(open(OLD_EXP / "reports/final_eval/test_predictions.csv", encoding="utf-8"))}
    dev = read_dev_log(OLD_EXP / "reports/prediction_logs/dev_predictions.log")
    for split, preds in (("dev", dev), ("test", test)):
        recs, _ = load_records(split, list(preds))
        missing = [u for u in preds if u not in recs]
        ref_mismatch = sum(" ".join(preds[u][0].split()) != recs[u] for u in recs)
        for tag, keep in (("all", lambda u: True), ("without_flagged", lambda u: u not in flagged)):
            us = [u for u in sorted(recs) if keep(u)]
            m = corpus_metrics([preds[u][1] for u in us], [recs[u] for u in us], with_signature=True)
            res[f"{split}_{tag}"] = m
        res[f"{split}_check"] = {"n_saved": len(preds), "not_in_kept_split": len(missing),
                                 "saved_ref_mismatch_vs_csv": ref_mismatch, "n_flagged": len(set(recs) & flagged)}
    out = REPORTS / "baseline_rescored.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for k, v in res.items():
        print(k, json.dumps({x: (round(y, 2) if isinstance(y, float) else y) for x, y in v.items()
                             if not str(x).startswith("signature")}, ensure_ascii=False))
    print("signature", res["test_all"]["signature_bleu4"])


if __name__ == "__main__":
    main()
