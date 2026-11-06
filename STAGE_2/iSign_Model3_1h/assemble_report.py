"""Assemble iSign_Model3_Experiment_Report.md from report_head.md, reports/tables.md, report_tail.md and run outputs."""
import csv, json, time, subprocess, sys
from pathlib import Path
H = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(H / "make_tables.py")], check=True, capture_output=True)
t = (H / "reports/tables.md").read_text(encoding="utf-8")
i = t.index("### Test results")
epoch_tables, test_tables = t[:i], t[i:]
SEL = "m3_hybrid_5k"
m = json.loads((H / f"runs/{SEL}/test_metrics.json").read_text())
R = list(csv.DictReader(open(H / f"runs/{SEL}/test_predictions_greedy.tsv", encoding="utf-8"), delimiter="\t"))
R.sort(key=lambda x: -float(x["sent_rougeL"]))
z = [x for x in R if float(x["sent_rougeL"]) == 0]
ex = ["Ground-truth captions are not reproduced here because iSign dataset captions must not be published."]
from collections import Counter
c = Counter(x["prediction"] for x in R).most_common(5)
ex.append(f"\n**Most frequent test outputs:** " + "; ".join(f'"{p}" ({n})' for p, n in c) + f". {len(z)} of {len(R)} test clips have sentence ROUGE-L 0.")
summ = ["| Experiment | Dataset | Samples | Exact Match | BLEU-4 | ROUGE-L | Remarks |", "|---|---|---:|---:|---:|---:|---|"]
for r, d in [("m3_hybrid_5k", "Model 3 hybrid, 4.9k train (selected)"), ("visual_only_5k", "Visual-only baseline, 4.9k train"),
             ("m3_hybrid", "Model 3 hybrid, 2k train"), ("visual_only", "Visual-only baseline, 2k train")]:
    p = H / f"runs/{r}/test_metrics.json"
    if p.exists():
        q = json.loads(p.read_text())
        g, b = q["greedy"], q["beam4_nr3"]
        summ.append(f"| {d} | iSign test1k | {q['n_test']} | {g['exact_match']:.1f}% | {g['bleu4']:.2f} (beam-4: {b['bleu4']:.2f}) | {g['rougeL']:.2f} | open-vocabulary generation; {g['distinct_predictions']} distinct outputs |")
summ.append("| *ResNet-18 gloss-free baseline (earlier work, 101k train)* | iSign test subset (different 1,979 clips) | 1979 | 0.1% | 0.76 | 8.16 | not the same test clips or training size; reference only |")
head = (H / "report_head.md").read_text(encoding="utf-8")
tail = (H / "report_tail.md").read_text(encoding="utf-8")
end = time.strftime("%H:%M")
rep = head + epoch_tables + tail
rep = rep.replace("{TABLES_TEST}", test_tables).replace("{EXAMPLES}", "\n".join(ex)).replace("{TABLE_ISIGN_SUMMARY}", "\n".join(summ))
rep = rep.replace("{END_TIME}", end).replace("{BEST_B4_GREEDY}", f"{m['greedy']['bleu4']:.2f}").replace("{BEST_B4_BEAM}", f"{m['beam4_nr3']['bleu4']:.2f}")
rep = rep.replace("{BEST_B1}", f"{m['greedy']['bleu1']:.2f}").replace("{BEST_RL}", f"{m['greedy']['rougeL']:.2f}")
import re; left = re.findall(r"{[A-Z_0-9]+}", rep); assert not left, left
(H / "iSign_Model3_Experiment_Report.md").write_text(rep, encoding="utf-8")
print("written", len(rep), "chars")
