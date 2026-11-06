"""Terminal progress (always flushed), per-epoch CSV/JSONL logs, example predictions and prediction CSVs."""
import csv
import json
import os

import torch

from slt_vit import paths


def say(*parts):
    print(*parts, flush=True)


def fmt_time(seconds):
    s = int(max(seconds, 0))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f'{h}h{m:02d}m{s:02d}s' if h else f'{m}m{s:02d}s'


def gpu_line():
    if not torch.cuda.is_available():
        return 'GPU: none (CPU)'
    p = torch.cuda.get_device_properties(0)
    line = (f'GPU: {p.name} | allocated {torch.cuda.memory_allocated() / 2 ** 30:.2f} GB, peak '
            f'{torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB of {p.total_memory / 2 ** 30:.1f} GB')
    try:
        import subprocess
        temp = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=5).stdout.strip()
        line += f' | {temp} C' if temp else ''
    except Exception:  # noqa: BLE001 - temperature is optional information
        pass
    return line


class RunLog:
    """``log.jsonl`` (one flat record per epoch) is the source of truth; ``metrics.csv`` is regenerated from it."""

    def __init__(self, exp):
        self.jsonl, self.csv = os.path.join(exp, 'log.jsonl'), os.path.join(exp, 'metrics.csv')
        self.records = []
        if os.path.isfile(self.jsonl):
            with open(self.jsonl, encoding='utf-8') as f:
                self.records = [json.loads(line) for line in f if line.strip()]

    def truncate_from(self, epoch):
        self.records = [r for r in self.records if r['epoch'] < epoch]
        self._write()

    def add(self, record):
        self.records.append(record)
        self._write()

    def _write(self):
        with open(paths.writable(self.jsonl), 'w', encoding='utf-8') as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in self.records)
        cols = []
        for r in self.records:
            cols += [k for k in r if k not in cols and not isinstance(r[k], dict)]
        with open(paths.writable(self.csv), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            w.writeheader()
            w.writerows(self.records)


def print_examples(rows, epoch, source, limit=8):
    """First limit/2 correct and first limit/2 wrong examples in manifest order (a fixed rule)."""
    right = [r for r in rows if r['exact']][:limit // 2]
    wrong = [r for r in rows if not r['exact']][:limit - len(right)]
    say(f'  sample predictions (epoch {epoch}, {source}):')
    for r in right + wrong:
        say(f"    [{'YES' if r['exact'] else 'NO '}] {r['id']} (signer {r['signer']})\n          ref : {r['reference']}\n          pred: {r['prediction']}")


def print_errors(summary):
    say(f"  errors: {summary['wrong']} wrong | prefix-only outputs {summary['incomplete_prefix_outputs']} | empty {summary['empty_outputs']}")
    say('  most repeated predictions: ' + '; '.join(f'"{t}" x{n}' for t, n in summary['most_repeated_predictions']))
    for (ref, pred), n in summary['most_common_substitutions'][:3]:
        say(f'  substitution x{n}: "{ref}" -> "{pred}"')


def write_predictions(path, rows, checkpoint, epoch, metrics):
    cols = ['sample_id', 'signer', 'frame_dir', 'reference', 'prediction', 'exact_match', 'sentence_bleu', 'rouge_l',
            'checkpoint', 'epoch', 'set_exact_match', 'set_bleu1', 'set_bleu4', 'set_rouge_l']
    with open(paths.writable(path), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r['id'], r['signer'], r['frame_dir'], r['reference'], r['prediction'], 'YES' if r['exact'] else 'NO',
                        f"{r['sentence_bleu']:.2f}", f"{r['rouge_l']:.2f}", checkpoint, epoch, f"{metrics['exact_match']:.4f}",
                        f"{metrics['bleu1']:.2f}", f"{metrics['bleu4']:.2f}", f"{metrics['rouge_l']:.2f}"])
