"""Show predictions with confidence scores for a fixed, already evaluated setting (display only).

    env\\python.exe experiments\\prediction_improvement_v1\\show_predictions.py --set test --setting E3
    env\\python.exe experiments\\prediction_improvement_v1\\show_predictions.py --set val  --setting D3

Confidence = softmax over the 97 training sentences of the setting's combined score (the weighted average of the
models' sentence log-probabilities, alpha 0). It is the model's relative belief among the 97 candidates, not a
calibrated probability of being right. The predictions must equal the saved ones in results/; nothing new is
selected, and no ledger row is written (these clips were already evaluated with this exact setting).
Writes results/scores/<set>_<setting>.csv.
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from pi_exp import common, engine, scoring, views  # noqa: E402

SETTINGS = {'D3': ('D3_ens_candidates_w0.5_trim5', 'D_combinations.json', 'test_repeat/R1'),
            'E3': ('E3_ens3_candidates_trim5', 'E_hybrid.json', 'test_repeat/R2')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', choices=['val', 'test'], required=True)
    ap.add_argument('--setting', choices=list(SETTINGS), default='E3')
    args = ap.parse_args()
    name, cfg_file, test_dir = SETTINGS[args.setting]
    setting = json.load(open(os.path.join(common.EXP, 'configs', cfg_file)))[name]
    setting['members'] = [tuple(m) for m in setting['members']]
    saved_dir = os.path.join(common.RESULTS, test_dir if args.set == 'test' else 'val', name)
    saved = {r['id']: r['prediction'] for r in json.load(open(os.path.join(saved_dir, 'predictions.json'), encoding='utf-8'))}
    cs = engine.ClipSet(args.set, [k for k, _ in setting['members']], allow_test=args.set == 'test')
    rows, mismatch = [], 0
    for e in cs.entries:
        vs = views.STRATEGIES[setting['strategy']]
        total = sum(w * torch.stack([scoring.normalize(cs.summed(k, e, v), cs.lengths.cpu(), setting['alpha']) for v in vs]).mean(0)
                    for k, w in setting['members']) / sum(w for _, w in setting['members'])
        p = torch.softmax(total.float(), 0)
        order = torch.argsort(p, descending=True).tolist()
        pred = cs.cand_texts[order[0]]
        mismatch += pred != saved[e['id']]
        ref_i = cs.cand_texts.index(e['text'])
        rows.append({'clip': e['id'], 'signer': e['signer'], 'reference': e['text'], 'prediction': pred,
                     'correct': 'YES' if pred == e['text'] else 'NO', 'confidence_%': round(100 * float(p[order[0]]), 1),
                     'reference_rank': order.index(ref_i) + 1, 'reference_confidence_%': round(100 * float(p[ref_i]), 1),
                     '2nd_choice': cs.cand_texts[order[1]], '2nd_%': round(100 * float(p[order[1]]), 1),
                     '3rd_choice': cs.cand_texts[order[2]], '3rd_%': round(100 * float(p[order[2]]), 1)})
    if mismatch:
        raise SystemExit(f'{mismatch} predictions differ from the saved results; not writing')
    out = os.path.join(common.out_dir('results', 'scores'), f'{args.set}_{args.setting}.csv')
    with open(out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{args.setting} on {'signer 7 (test)' if args.set == 'test' else 'signer 6 (validation)'}: "
          f"{sum(r['correct'] == 'YES' for r in rows)}/{len(rows)} correct (same predictions as the saved results)\n")
    print(f"{'OK':3} {'conf':>6}  {'real sentence':36} {'prediction':36} 2nd choice")
    for r in rows:
        print(f"{'✓' if r['correct'] == 'YES' else '✗':3} {r['confidence_%']:5.1f}%  {r['reference'][:36]:36} {r['prediction'][:36]:36} "
              f"{r['2nd_choice'][:30]} ({r['2nd_%']}%)")
    ok = [r['confidence_%'] for r in rows if r['correct'] == 'YES']
    bad = [r['confidence_%'] for r in rows if r['correct'] == 'NO']
    print(f"\nmean confidence: correct {sum(ok) / max(len(ok), 1):.1f}%, wrong {sum(bad) / max(len(bad), 1):.1f}%")
    for lo in (80, 50):
        sel = [r for r in rows if r['confidence_%'] >= lo]
        print(f"confidence >= {lo}%: {len(sel)} clips, {sum(r['correct'] == 'YES' for r in sel)} correct")
    print(f'\nwritten {out}')


if __name__ == '__main__':
    main()
