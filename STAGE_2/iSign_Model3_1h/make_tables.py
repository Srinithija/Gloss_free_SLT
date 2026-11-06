"""Build the report's result tables from runs/*/train_log.jsonl, runs/*/test_metrics.json and reports/keyword_probe_*.json."""
import csv, json
from pathlib import Path
H = Path(__file__).resolve().parent
RUNS = [("m3_hybrid", "Model 3 hybrid (SigLIP+MediaPipe), 2k train"), ("visual_only", "Visual-only SigLIP baseline, 2k train"),
        ("m3_hybrid_5k", "Model 3 hybrid (SigLIP+MediaPipe), 4.9k train"), ("visual_only_5k", "Visual-only SigLIP baseline, 4.9k train")]
out = []
for r, desc in RUNS:
    p = H / "runs" / r / "train_log.jsonl"
    if not p.exists():
        continue
    rows = [json.loads(l) for l in open(p) if '"epoch"' in l]
    best = max(rows, key=lambda x: x["dev_bleu4"])["epoch"]
    out.append(f"\n#### {desc} (`runs/{r}`)\n\n| Epoch | Training Loss | Validation Loss | Exact Match | BLEU-1 | BLEU-4 | ROUGE-L | Distinct outputs / top share | Time (train+eval) |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    em = {}
    for x in rows:
        out.append(f"| {x['epoch']}{' **(best)**' if x['epoch']==best else ''} | {x['train_loss']:.3f} | {x['dev_loss_nols']:.3f} | {'0.0%' if x['epoch']==best else 'not logged'} | {x['dev_bleu1']:.2f} | {x['dev_bleu4']:.2f} | {x['dev_rougeL']:.2f} | {x['distinct']} / {100*x['top_share']:.0f}% | {60*(x['train_min']+x['eval_min']):.0f} s |")
    out.append(f"\nPeak VRAM {max(x['peak_vram_GiB'] for x in rows):.2f} GiB. Train loss uses label smoothing 0.1; validation loss has none.")
out.append("\nExact match per epoch was not logged by the training loop. It is known only at the selected epoch (0.0% in every run; see `dev_at_best` in `test_metrics.json`).\n")
out.append("\n### Test results (held-out `test1k`, 993 scorable clips)\n\n| Experiment | Dataset | Samples | Decoding | Exact Match | BLEU-1 | BLEU-4 | ROUGE-L | Test loss | Distinct outputs (top share) | Content-word P / R (%) | ms/video | Remarks |\n|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
for r, desc in RUNS:
    p = H / "runs" / r / "test_metrics.json"
    if not p.exists():
        continue
    m = json.loads(p.read_text())
    for dec in ("greedy", "beam4_nr3"):
        x = m[dec]
        out.append(f"| {desc} | iSign test1k | {m['n_test']} | {dec} | {x['exact_match']:.1f}% | {x['bleu1']:.2f} | {x['bleu4']:.2f} | {x['rougeL']:.2f} | {m['test_loss']:.3f} | {x['distinct_predictions']} ({100*x['most_repeated_share']:.0f}%) | {x['content_word_precision']:.1f} / {x['content_word_recall']:.2f} | {x['infer_ms_per_video_batched64']:.1f} | best epoch {m['best_epoch']} (dev BLEU-4 {m['dev_at_best']['bleu4']:.2f}) |")
out.append("\n### Key-word recognition probe (test1k; 300 most frequent content words; chance AUC = 0.5)\n\n| Input features | Train / dev / test clips | Mean per-word AUC (test) | Shuffled-input control AUC | P@5 model | P@5 frequency prior | Best-recognised words (AUC, test clips) |\n|---|---:|---:|---:|---:|---:|---|")
for mode in ("hybrid", "landmarks", "visual"):
    p = H / "reports" / f"keyword_probe_{mode}.json"
    if p.exists():
        k = json.loads(p.read_text())
        tw = ", ".join(f"{w} ({a:.2f}, {n})" for w, a, n in k["top_words_by_test_auc(>=10 test clips)"][:8])
        out.append(f"| {mode} | {k['n_train']} / {k['n_dev']} / {k['n_test']} | **{k['test_mean_auc']:.3f}** | {k['test_auc_shuffled_control']:.3f} | {k['test_P@5_model']:.3f} | {k['test_P@5_frequency_prior']:.3f} | {tw} |")
(H / "reports" / "tables.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
