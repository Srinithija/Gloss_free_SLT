"""Repeat evaluation on signer 7 for the two settings fixed in reports/SELECTION_RULES.md (R1, R2). Run once each.

    env\\python.exe experiments\\prediction_improvement_v1\\tools\\repeat_test.py R1
    env\\python.exe experiments\\prediction_improvement_v1\\tools\\repeat_test.py R2

R1 = D3_ens_candidates_w0.5_trim5 (best inference-only setting on the existing checkpoints, chosen on signer 6)
R2 = E3_ens3_candidates_trim5     (best setting overall on signer 6, uses the new SigLIP+landmark model)
Results go to results/test_repeat/<R>/ and one row in results/REPEAT_TEST_LEDGER.csv. The original
outputs/TEST_EVALUATION_LEDGER.csv and the initial test files are not touched. A setting already in the repeat
ledger is refused. The results are reported and are not used to change any choice.
"""
import csv
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pi_exp import common, engine  # noqa: E402
from slt_vit import paths  # noqa: E402
from slt_vit.loso import bootstrap  # noqa: E402

PLAN = {'R1': ('D3_ens_candidates_w0.5_trim5', 'D_combinations.json'), 'R2': ('E3_ens3_candidates_trim5', 'E_hybrid.json'),
        # added 2026-10-27 at the user's request (Model 3 alone, for the report; see SELECTION_RULES.md)
        'R3a': ('R3a_model3_alone_greedy', 'R3_model3_alone.json'),
        'R3b': ('R3b_model3_alone_candidates_trim5', 'R3_model3_alone.json'),
        # added 2026-10-28 at the user's request (Model 2 alone, for the report)
        'R4a': ('R4a_model2_alone_greedy', 'R4_model2_alone.json'),
        'R4b': ('R4b_model2_alone_candidates_trim5', 'R4_model2_alone.json'),
        # added 2026-10-28 at the user's request (Model 1 alone with sentence scoring = D1 setting, for the report)
        'R5': ('R5_model1_alone_candidates_trim5', 'R5_model1_alone.json')}
LEDGER = os.path.join(common.RESULTS, 'REPEAT_TEST_LEDGER.csv')
INITIAL = os.path.join(paths.OUTPUTS, 'siglip_b16_stage2', 'final_test_beam1.json')


def main():
    tag = sys.argv[1]
    name, cfg_file = PLAN[tag]
    if os.path.isfile(LEDGER) and any(r['setting'] == name for r in csv.DictReader(open(LEDGER, encoding='utf-8'))):
        raise SystemExit(f'{name} was already evaluated on signer 7; see {LEDGER}')
    setting = json.load(open(os.path.join(common.EXP, 'configs', cfg_file)))[name]
    setting['members'] = [tuple(m) for m in setting['members']]
    cs = engine.ClipSet('test', [k for k, _ in setting['members']], allow_test=True)
    cs.name = os.path.join('test_repeat', tag)
    m, rows = engine.evaluate(cs, name, setting)
    initial = {r['id']: r for r in json.load(open(INITIAL, encoding='utf-8'))['predictions']}
    new = {r['id']: r for r in rows}
    if set(initial) != set(new):
        raise RuntimeError('test clips differ from the initial evaluation')
    cx, cb = bootstrap(initial, new)
    init_correct = sum(r['exact'] for r in initial.values())
    new_file = not os.path.isfile(LEDGER)
    with open(paths.writable(LEDGER), 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(['timestamp', 'evaluation', 'setting', 'checkpoints', 'correct', 'n', 'exact_match', 'bleu4', 'rouge_l',
                        'initial_correct', 'exact_diff_ci_low_pts', 'exact_diff_ci_high_pts', 'note'])
        w.writerow([datetime.datetime.now().isoformat(timespec='seconds'), f'repeat test evaluation {tag}', name,
                    '; '.join(f"{k}={common.translator(v).info['sha256'][:12]}" for k, v in cs.keys.items()), m['correct'], m['n'],
                    f"{m['exact_match']:.4f}", f"{m['bleu4']:.2f}", f"{m['rouge_l']:.2f}", init_correct, f'{100 * cx[0]:.1f}', f'{100 * cx[1]:.1f}',
                    'repeat evaluation; setting fixed on signers 1-6 before this run; initial result 23/91 stays the official first test'])
    print(f"{tag} {name}: signer 7 {engine.describe(m)}")
    print(f"initial test {init_correct}/91; paired difference 95% CI [{100 * cx[0]:+.1f}, {100 * cx[1]:+.1f}] pts, BLEU-4 CI [{cb[0]:+.1f}, {cb[1]:+.1f}]")


if __name__ == '__main__':
    main()
