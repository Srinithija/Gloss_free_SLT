"""Inference-only experiments A-C (and the planned combination D) on signer 6 or on the leave-one-signer-out signers.

    env\\python.exe experiments\\prediction_improvement_v1\\run_inference.py --set val
    env\\python.exe experiments\\prediction_improvement_v1\\run_inference.py --set loso --settings <names...>

--set val   signer 6 with the selected checkpoints (siglip_s2 = outputs/siglip_b16_stage2/best.pt, clip_s1 = CLIP stage 1)
--set loso  signers 3, 4, 5, each with the fold models trained without that signer; results pooled over 282 clips
Each setting writes results/<set>/<setting>/{config.json, metrics.json, predictions.csv, predictions.json}.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pi_exp import common, engine, views  # noqa: E402
from slt_vit.loso import bootstrap  # noqa: E402
from slt_vit.metrics import translation_metrics  # noqa: E402

S, C = 'siglip_s2', 'clip_s1'


def grid():
    g = {'baseline_greedy': dict(members=[(S, 1.0)], strategy='uniform', mode='greedy', alpha=0.0)}
    for a in (0.0, 0.5, 1.0):                                               # A: candidate restriction
        g[f'A_candidates_alpha{a}'] = dict(members=[(S, 1.0)], strategy='uniform', mode='candidates', alpha=a)
    for st in views.STRATEGIES:                                              # B: temporal sampling (unconstrained greedy)
        if st != 'uniform':
            g[f'B_greedy_{st.replace("/", "")}'] = dict(members=[(S, 1.0)], strategy=st, mode='greedy', alpha=0.0)
    g['C_clip_alone_greedy'] = dict(members=[(C, 1.0)], strategy='uniform', mode='greedy', alpha=0.0)
    for w in (0.5, 0.6, 0.7, 0.8, 0.9):                                      # C: CLIP + SigLIP, unconstrained greedy
        g[f'C_ens_greedy_w{w}'] = dict(members=[(S, w), (C, round(1 - w, 2))], strategy='uniform', mode='greedy', alpha=0.0)
    # C with candidate-sequence scoring, at the alpha selected in A (0.0: tie with 0.5 on exact match, higher BLEU-4)
    g['C_clip_alone_candidates'] = dict(members=[(C, 1.0)], strategy='uniform', mode='candidates', alpha=0.0)
    for w in (0.5, 0.6, 0.7, 0.8, 0.9):
        g[f'C_ens_candidates_w{w}'] = dict(members=[(S, w), (C, round(1 - w, 2))], strategy='uniform', mode='candidates', alpha=0.0)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', choices=['val', 'loso'], required=True)
    ap.add_argument('--settings', nargs='*', help='names from the grid (default: all)')
    ap.add_argument('--extra', help='JSON file with additional named settings (planned combinations)')
    args = ap.parse_args()
    g = grid()
    if args.extra:
        g.update(json.load(open(args.extra)))
    names = args.settings or list(g)
    members = sorted({k for n in names for k, _ in g[n]['members']})
    sets = ['val'] if args.set == 'val' else ['h3', 'h4', 'h5']
    results = {}
    for s in sets:
        cs = engine.ClipSet(s, members)
        for n in names:
            m, rows = engine.evaluate(cs, n, g[n])
            results.setdefault(n, []).extend(rows)
            print(f'{s:4} {n:32} {engine.describe(m)}', flush=True)
    if args.set == 'loso':
        summary = {}
        for n, rows in results.items():
            m = translation_metrics([r['prediction'] for r in rows], [r['reference'] for r in rows])
            summary[n] = m
            d = common.out_dir('results', 'loso_pooled', n)
            json.dump({'metrics': m, 'per_signer': {s: sum(r['exact'] for r in rows if r['signer'] == int(s[1])) for s in sets}},
                      open(os.path.join(d, 'metrics.json'), 'w'), indent=1, default=str)
            json.dump(rows, open(os.path.join(d, 'predictions.json'), 'w'), indent=1, default=str, ensure_ascii=False)
            print(f'pooled {n:32} {engine.describe(m)}', flush=True)
    base = {r['id']: r for r in results['baseline_greedy']} if 'baseline_greedy' in results else None
    if base:
        print('\npaired bootstrap vs baseline_greedy (2000 resamples over clips), exact-match difference in points:')
        lines = {}
        for n, rows in results.items():
            if n == 'baseline_greedy':
                continue
            other = {r['id']: r for r in rows}
            cx, cb = bootstrap(base, other)
            d = sum(r['exact'] for r in rows) - sum(r['exact'] for r in base.values())
            lines[n] = {'clips_gained': d, 'exact_ci_points': [100 * cx[0], 100 * cx[1]], 'bleu4_ci': cb}
            print(f'  {n:32} {d:+3d} clips, 95% CI [{100 * cx[0]:+.1f}, {100 * cx[1]:+.1f}] pts, BLEU-4 CI [{cb[0]:+.1f}, {cb[1]:+.1f}]')
        json.dump(lines, open(os.path.join(common.out_dir('results', 'val' if args.set == 'val' else 'loso_pooled'),
                                            'bootstrap_vs_baseline.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
