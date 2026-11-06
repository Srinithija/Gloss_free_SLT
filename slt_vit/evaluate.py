"""Evaluate a checkpoint on one split through the inference path (canonical frames -> model).

    python -m slt_vit.evaluate --ckpt outputs/<exp>/best.pt --split val [--beams 1]
    python -m slt_vit.evaluate --ckpt outputs/<exp>/best.pt --split test --final

Writes <exp>/eval_<split>_beam<N>.json and .csv (final_test_... for the test split). The test split needs
--final, runs once per checkpoint and beam setting, and is recorded in outputs/TEST_EVALUATION_LEDGER.csv.
"""
import argparse
import csv
import datetime
import json
import os

import numpy as np
import torch

import slt_vit  # noqa: F401
from slt_vit import logs, paths
from slt_vit.dataset import cache_name, load_manifest, select
from slt_vit.frames import sample_indices, to_unit_float
from slt_vit.infer import Translator
from slt_vit.logs import say
from slt_vit.metrics import error_summary, per_sample, translation_metrics
from slt_vit.text_model import LANG
from slt_vit.train import token_loss


@torch.no_grad()
def teacher_forced_loss(tr, frames, target):
    idx = sample_indices(len(frames), tr.cfg['num_frames'])
    cuda = tr.device.type == 'cuda'
    with torch.autocast('cuda', enabled=cuda, dtype=torch.float16):
        feats = tr.model.frame_features(to_unit_float(frames[idx], tr.device).unsqueeze(0)).float()
        dec_in = torch.tensor([[LANG] + target[:-1]], device=tr.device)
        logits = tr.model(feats, dec_in, torch.ones_like(dec_in))
    return token_loss(logits, torch.tensor([target], device=tr.device), 0.0, 'sum').item(), len(target)


def ledger_rows():
    if not os.path.isfile(paths.TEST_LEDGER):
        return []
    with open(paths.TEST_LEDGER, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def record_test(tr, beams, m, note):
    new = not os.path.isfile(paths.TEST_LEDGER)
    os.makedirs(paths.OUTPUTS, exist_ok=True)
    with open(paths.writable(paths.TEST_LEDGER), 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['timestamp', 'checkpoint', 'sha256', 'beams', 'exact_match', 'correct', 'n', 'bleu4', 'rouge_l', 'note'])
        w.writerow([datetime.datetime.now().isoformat(timespec='seconds'), os.path.relpath(tr.info['checkpoint'], paths.PROJECT),
                    tr.info['sha256'], beams, f"{m['exact_match']:.4f}", m['correct'], m['n'], f"{m['bleu4']:.2f}", f"{m['rouge_l']:.2f}", note])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    ap.add_argument('--beams', type=int, default=1)
    ap.add_argument('--signers', nargs='*', type=int, help='evaluate these signers instead of a split (never signer 7)')
    ap.add_argument('--final', action='store_true', help='required for the held-out test split')
    ap.add_argument('--out')
    args = ap.parse_args()
    tr = Translator(args.ckpt)
    if args.split == 'test' and not args.signers:
        if not args.final:
            raise SystemExit('the test split is held out: pass --final once the checkpoint and decoding are chosen on validation')
        if any(r['sha256'] == tr.info['sha256'] and r['beams'] == str(args.beams) for r in ledger_rows()):
            raise SystemExit('this checkpoint was already evaluated on the test split with these beams; see ' + paths.TEST_LEDGER)
    exp = os.path.dirname(tr.info['checkpoint'])
    stem = f"final_test_beam{args.beams}" if args.split == 'test' else f"eval_{args.split}{'_signers' + ''.join(map(str, args.signers)) if args.signers else ''}_beam{args.beams}"
    out = paths.writable(args.out or os.path.join(exp, stem + '.json'))
    if os.path.exists(out):
        raise SystemExit(f'{out} exists; pass --out')
    manifest = load_manifest()
    entries = select(manifest, args.split, args.signers)
    say(f"evaluating {tr.info['checkpoint']} ({tr.info['encoder']}, epoch {tr.info['epoch_number']}, sha256 {tr.info['sha256'][:12]}) "
        f"on {len(entries)} clips of {'signers ' + str(args.signers) if args.signers else args.split}, beams={args.beams}")
    if tr.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    hyps, refs, loss_sum, n_tok, latency, no_eos = [], [], 0.0, 0, [], 0
    for i, e in enumerate(entries):
        frames = np.load(os.path.join(paths.FRAME_CACHE, cache_name(e)))
        r = tr.translate_frames(frames, args.beams)
        if i:
            latency.append(r['model_seconds'])
        no_eos += not r['stopped_at_eos']
        l, n = teacher_forced_loss(tr, frames, tr.tok.encode(e['text']))
        loss_sum, n_tok = loss_sum + l, n_tok + n
        hyps.append(r['text'])
        refs.append(e['text'])
    rows = [{'id': e['id'], 'signer': e['signer'], 'frame_dir': e['frame_dir'], 'reference': ref, 'prediction': h, **s}
            for e, ref, h, s in zip(entries, refs, hyps, per_sample(hyps, refs))]
    m = translation_metrics(hyps, refs)
    per_signer = {}
    for s in sorted({r['signer'] for r in rows}):
        rs = [r for r in rows if r['signer'] == s]
        per_signer[s] = translation_metrics([r['prediction'] for r in rs], [r['reference'] for r in rs])
    m.update(loss=loss_sum / n_tok, split=args.split, signers=args.signers, beams=args.beams, checkpoint=tr.info['checkpoint'],
             checkpoint_sha256=tr.info['sha256'], epoch_number=tr.info['epoch_number'], no_eos=no_eos,
             latency_model_s_mean=float(np.mean(latency)) if latency else None,
             latency_model_s_p95=float(np.percentile(latency, 95)) if latency else None,
             peak_gpu_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30 if tr.device.type == 'cuda' else None)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'metrics': m, 'per_signer': per_signer, 'errors': error_summary(rows), 'predictions': rows}, f, indent=1, ensure_ascii=False, default=str)
    logs.write_predictions(os.path.splitext(out)[0] + '.csv', rows, tr.info['checkpoint'], tr.info['epoch_number'], m)
    if args.split == 'test' and not args.signers:
        record_test(tr, args.beams, m, 'final evaluation of the selected checkpoint')
    say(f"\n{args.split}: exact {m['correct']}/{m['n']} = {m['exact_match']:.1%} | BLEU-1 {m['bleu1']:.2f} BLEU-4 {m['bleu4']:.2f} "
        f"ROUGE-L {m['rouge_l']:.2f} | loss {m['loss']:.3f} | distinct predictions {m['unique_predictions']} (most repeated {m['most_common_count']}x)")
    if len(per_signer) > 1:
        say('per signer: ' + ', '.join(f"{s}: {v['correct']}/{v['n']}" for s, v in per_signer.items()))
    logs.print_errors(error_summary(rows))
    say(f'written {out} and {os.path.splitext(out)[0]}.csv')


if __name__ == '__main__':
    main()
