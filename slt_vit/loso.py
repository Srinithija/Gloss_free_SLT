"""Leave-one-signer-out diagnostic for stage-1 configs (signers 1-6 only; the test signer 7 is never used).

    python -m slt_vit.loso --configs configs/siglip_b16_stage1.yaml configs/clip_b16_stage1.yaml --folds 3 4 5
    python -m slt_vit.loso --stage 2 --epochs 24 --configs configs/siglip_b16_stage2.yaml         --init-template outputs/loso/siglip_b16_stage1_h{fold}/best.pt   # each fold starts from its own stage 1

For each held-out signer, each config trains on the other five of signers 1-6 for a fixed number of epochs and
is scored once after the last epoch, so no epoch is chosen with the held-out signer. A paired bootstrap over
all held-out clips compares each config with the first one.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import sacrebleu
import yaml

import slt_vit  # noqa: F401
from slt_vit import paths
from slt_vit.logs import say
from slt_vit.metrics import translation_metrics

POOL = [1, 2, 3, 4, 5, 6]


def bootstrap(base, other, n=2000, seed=0):
    ids = sorted(base)
    refs = [base[i]['reference'] for i in ids]
    bh, oh = [base[i]['prediction'] for i in ids], [other[i]['prediction'] for i in ids]
    be, oe = np.array([h == r for h, r in zip(bh, refs)], float), np.array([h == r for h, r in zip(oh, refs)], float)
    rng = np.random.default_rng(seed)
    dx, db = [], []
    for _ in range(n):
        s = rng.integers(0, len(ids), len(ids))
        dx.append(oe[s].mean() - be[s].mean())
        db.append(sacrebleu.corpus_bleu([oh[i] for i in s], [[refs[i] for i in s]]).score
                  - sacrebleu.corpus_bleu([bh[i] for i in s], [[refs[i] for i in s]]).score)
    return [float(np.percentile(dx, 2.5)), float(np.percentile(dx, 97.5))], [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+', required=True)
    ap.add_argument('--folds', type=int, nargs='+', default=[3, 4, 5])
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--stage', type=int, default=1, choices=[1, 2])
    ap.add_argument('--init-template', default='', help='stage 2: stage-1 checkpoint per fold, with {fold}')
    args = ap.parse_args()
    if 7 in args.folds:
        raise SystemExit('signer 7 is the held-out test signer')
    if args.stage == 2 and not args.init_template:
        raise SystemExit('stage 2 needs --init-template (the stage-1 checkpoint of each fold)')
    root = paths.writable(os.path.join(paths.OUTPUTS, 'loso'))
    os.makedirs(root, exist_ok=True)
    preds = {}
    for cfg_path in args.configs:
        base = yaml.safe_load(open(cfg_path, encoding='utf-8'))['name']
        preds[base] = {}
        for fold in args.folds:
            name = f'loso/{base}_h{fold}'
            exp = os.path.join(paths.OUTPUTS, name)
            if not os.path.isfile(os.path.join(exp, 'status.json')):
                train = [s for s in POOL if s != fold]
                extra = [f"init={args.init_template.replace('{fold}', str(fold))}"] if args.init_template else []
                cmd = [sys.executable, os.path.join(paths.PROJECT, f'train_stage{args.stage}.py'), '--config', cfg_path, '--set', *extra,
                       f'name={name}', f'train_signers={train}', f'val_signers=[{fold}]', f'epochs={args.epochs}',
                       f'eval_every={args.epochs}', 'patience=99', 'deteriorate_k=99', 'overfit_k=99', 'examples=2', 'log_every=100']
                say(f'=== {base}, held-out signer {fold}')
                with open(exp + '.console.log', 'w', encoding='utf-8') as log:
                    if subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=paths.PROJECT):
                        raise SystemExit(f'{name} failed; see {exp}.console.log')
            with open(os.path.join(exp, 'val_predictions_best.json'), encoding='utf-8') as f:
                for r in json.load(f):
                    preds[base][r['id']] = r
    lines = [f'Leave-one-signer-out ({args.epochs} epochs, scored once at the end; held-out signers {args.folds})', '',
             '| config | ' + ' | '.join(f'signer {f}' for f in args.folds) + ' | pooled exact | pooled BLEU-4 | ROUGE-L |',
             '|---|' + '---|' * (len(args.folds) + 3)]
    summary = {}
    for base, p in preds.items():
        cells = []
        for f in args.folds:
            rs = [r for r in p.values() if r['signer'] == f]
            cells.append(f"{sum(r['exact'] for r in rs)}/{len(rs)}")
        m = translation_metrics([r['prediction'] for r in p.values()], [r['reference'] for r in p.values()])
        summary[base] = m
        lines.append(f"| {base} | " + ' | '.join(cells) + f" | {m['exact_match']:.1%} | {m['bleu4']:.1f} | {m['rouge_l']:.1f} |")
    first = list(preds)[0]
    for base in list(preds)[1:]:
        ci_x, ci_b = bootstrap(preds[first], preds[base])
        d = summary[base]['exact_match'] - summary[first]['exact_match']
        lines.append(f"\n{base} vs {first}: exact {100 * d:+.1f} pts (95% CI {100 * ci_x[0]:+.1f} to {100 * ci_x[1]:+.1f}), "
                     f"BLEU-4 {summary[base]['bleu4'] - summary[first]['bleu4']:+.1f} (CI {ci_b[0]:+.1f} to {ci_b[1]:+.1f})")
    text = '\n'.join(lines)
    say(text)
    with open(os.path.join(root, f'summary_stage{args.stage}.md'), 'w', encoding='utf-8') as f:
        f.write(text + '\n')


if __name__ == '__main__':
    main()
