"""Training core for both stages.

    python train_stage1.py --config configs/siglip_b16_stage1.yaml [--resume] [--set key=value ...]
    python train_stage2.py --config configs/siglip_b16_stage2.yaml [--resume]

Stage 1 trains the temporal encoder and mBART on cached frozen ViT features. Stage 2 starts from a stage-1
checkpoint and also fine-tunes the last vision blocks on augmented frames. Only the validation split (or the
explicit leave-one-signer-out signers) is used; the held-out test signer is never loaded here.

Stopping rules (counted in validation evaluations): patience (no new best BLEU-4, ties -> lower loss),
deterioration (last k evaluations all more than a margin below the best), overfitting (last k validation
losses above a ratio of the lowest), and fatal stops on non-finite loss or persistent non-finite gradients.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

import slt_vit  # noqa: F401
from slt_vit import checkpoints, logs, paths
from slt_vit.augment import augment_clip
from slt_vit.dataset import FeatureClips, FrameClips, load_manifest, select
from slt_vit.frames import to_unit_float
from slt_vit.logs import say
from slt_vit.metrics import error_summary, per_sample, translation_metrics
from slt_vit.model import SignTranslatorViT
from slt_vit.text_model import LANG, PAD, TrimmedTokenizer

DEFAULTS = dict(
    name='', stage=1, encoder='siglip_b16_224', init='',
    num_frames=32, clip_mean=True, temporal_hidden=1024, encoder_layers=3, decoder_layers=3, dropout=0.1,
    batch_size=16, accum_steps=1, epochs=100, lr_new=3e-4, lr_mbart=1e-4, lr_vision=0.0, vision_trainable_blocks=0,
    gradient_checkpointing=False, weight_decay=0.01, warmup_steps=200, label_smoothing=0.1, grad_clip=1.0,
    temporal_jitter=0.1, aug='base', eval_every=5, patience=8, deteriorate_k=4, deteriorate_margin=8.0,
    overfit_loss_ratio=1.3, overfit_k=3, seed=0, amp=True, max_new_tokens=20, log_every=0, examples=8,
    train_signers=[], val_signers=[], reference={}, verify_old_project=True,
)


class TrainingFailure(RuntimeError):
    pass


def collate(batch):
    xs, targets, idx = zip(*batch)
    length = max(len(t) for t in targets)
    labels = torch.full((len(targets), length), PAD, dtype=torch.long)
    dec_in = torch.full((len(targets), length), PAD, dtype=torch.long)
    for k, t in enumerate(targets):
        labels[k, :len(t)] = torch.tensor(t)
        dec_in[k, 0] = LANG
        dec_in[k, 1:len(t)] = torch.tensor(t[:-1])
    return torch.stack(xs), dec_in, labels, (labels != PAD).long(), list(idx)


def token_loss(logits, labels, smoothing, reduction='mean'):
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.masked_fill(labels.eq(PAD), -100).reshape(-1),
                           ignore_index=-100, label_smoothing=smoothing, reduction=reduction)


def visual_features(model, x, cfg, train, rng=None):
    """Stage 1: cached features. Stage 2: frames -> (augmentation) -> vision encoder."""
    dev = next(model.parameters()).device
    if cfg['stage'] == 1:
        return x.to(dev, non_blocking=True).float()
    clips = [to_unit_float(c, dev) for c in x]
    if train:
        clips = [augment_clip(c, rng, cfg['aug']) for c in clips]
    return model.frame_features(torch.stack(clips))


@torch.no_grad()
def evaluate(model, loader, tok, cfg):
    model.eval()
    dev = next(model.parameters()).device
    hyps, refs, rows, loss_sum, n_tok = [], [], [], 0.0, 0
    for x, dec_in, labels, mask, idx in loader:
        dec_in, labels, mask = dec_in.to(dev), labels.to(dev), mask.to(dev)
        with torch.autocast('cuda', enabled=cfg['amp']):
            feats = visual_features(model, x, cfg, train=False)
            logits = model(feats, dec_in, mask)
            out = model.generate(feats, max_new_tokens=cfg['max_new_tokens'])
        loss_sum += token_loss(logits, labels, 0.0, 'sum').item()
        n_tok += int(mask.sum())
        hyps += [tok.decode(o) for o in out.tolist()]
        for i in idx:
            e = loader.dataset.entries[i]
            refs.append(e['text'])
            rows.append({'id': e['id'], 'signer': e['signer'], 'frame_dir': e['frame_dir'], 'reference': e['text']})
    m = translation_metrics(hyps, refs)
    m['loss'] = loss_sum / n_tok
    for row, h, s in zip(rows, hyps, per_sample(hyps, refs)):
        row.update(prediction=h, **s)
    return m, rows


def param_groups(model, cfg):
    groups = []
    for name, module, lr in (('vision', model.vision, cfg['lr_vision']), ('new', model.temporal, cfg['lr_new']),
                             ('mbart', model.mbart, cfg['lr_mbart'])):
        params, seen = [], set()
        for p in module.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                params.append(p)
        if lr <= 0 or not params:
            for p in module.parameters():
                p.requires_grad_(False)
            continue
        groups.append({'name': name, 'params': [p for p in params if p.ndim > 1], 'lr': lr, 'weight_decay': cfg['weight_decay']})
        groups.append({'name': name + '_nodecay', 'params': [p for p in params if p.ndim <= 1], 'lr': lr, 'weight_decay': 0.0})
    return groups


def best_index(evals):
    return max(range(len(evals)), key=lambda i: (evals[i]['bleu4'], -evals[i]['loss']))


def stop_reason(evals, cfg):
    """Never stops on one noisy evaluation: each rule needs several evaluations in a row."""
    b = best_index(evals)
    if len(evals) - 1 - b >= cfg['patience']:
        return f'patience: no new best validation BLEU-4 in {len(evals) - 1 - b} evaluations'
    k = cfg['deteriorate_k']
    if len(evals) > k and all(e['bleu4'] < evals[b]['bleu4'] - cfg['deteriorate_margin'] for e in evals[-k:]):
        return f"deteriorating: last {k} evaluations more than {cfg['deteriorate_margin']} BLEU-4 below the best"
    k, low = cfg['overfit_k'], min(e['loss'] for e in evals)
    if len(evals) >= k and all(e['loss'] > cfg['overfit_loss_ratio'] * low for e in evals[-k:]):
        return f"overfitting: last {k} validation losses above {cfg['overfit_loss_ratio']} x the lowest ({low:.3f})"
    return None


def verify_old_project(cfg, when):
    if not cfg['verify_old_project']:
        return
    script = os.path.join(paths.PROJECT, 'tools', 'protect.py')
    r = subprocess.run([sys.executable, '-I', script, 'verify'], capture_output=True, text=True)
    say(f'[{when}] ' + (r.stdout.strip() or r.stderr.strip()))
    if r.returncode:
        raise TrainingFailure('the original sltbaselines project changed; stopping')


def load_config(path, overrides):
    with open(path, encoding='utf-8') as f:
        cfg = {**DEFAULTS, **(yaml.safe_load(f) or {})}
    for item in overrides:
        k, _, v = item.partition('=')
        if k not in DEFAULTS:
            raise SystemExit(f'unknown setting {k}')
        cfg[k] = yaml.safe_load(v)
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise SystemExit(f'unknown settings in {path}: {sorted(unknown)}')
    for k, default in DEFAULTS.items():   # YAML reads e.g. 1e-12 (no decimal point) as a string
        if isinstance(default, bool):
            if isinstance(cfg[k], str):
                cfg[k] = cfg[k].strip().lower() in ('1', 'true', 'yes')
        elif isinstance(default, (int, float)) and not isinstance(cfg[k], bool):
            try:
                cfg[k] = type(default)(float(cfg[k])) if isinstance(default, float) else int(cfg[k])
            except (TypeError, ValueError):
                raise SystemExit(f'setting {k}={cfg[k]!r} must be a number')
    if cfg['init'] and not os.path.isabs(cfg['init']):
        cfg['init'] = os.path.join(paths.PROJECT, cfg['init'])
    if not cfg['name']:
        raise SystemExit('the config needs a name (the experiment folder under outputs/)')
    return cfg


def reference_metrics(cfg):
    ref = cfg.get('reference') or {}
    if not ref.get('file') or not os.path.isfile(ref['file']):
        return None
    with open(ref['file'], encoding='utf-8') as f:
        d = json.load(f)
    return {'name': ref.get('name', 'reference'), **(d.get('metrics') or d.get('best_val') or {})}


def run(config_path, overrides, resume, stage):
    cfg = load_config(config_path, overrides)
    if cfg['stage'] != stage:
        raise SystemExit(f"{config_path} is a stage-{cfg['stage']} config; use train_stage{cfg['stage']}.py")
    exp = paths.writable(os.path.normpath(os.path.join(paths.OUTPUTS, cfg['name'])))
    last_path, best_path = os.path.join(exp, 'last.pt'), os.path.join(exp, 'best.pt')
    if resume and not os.path.isfile(last_path):
        raise SystemExit(f'--resume needs {last_path}')
    if not resume and (os.path.exists(last_path) or os.path.exists(best_path)):
        raise SystemExit(f'{exp} already has checkpoints; use --resume or a new experiment name')
    verify_old_project(cfg, 'start')
    os.makedirs(exp, exist_ok=True)
    manifest = load_manifest()
    train_entries = select(manifest, 'train', cfg['train_signers'])
    val_entries = select(manifest, 'val', cfg['val_signers'])
    torch.manual_seed(cfg['seed'])
    dev = torch.device('cuda')

    if resume:
        state = checkpoints.load(last_path)
        cfg, tok = {**DEFAULTS, **state['cfg']}, TrimmedTokenizer.from_dict(state['tokenizer'])
        model = SignTranslatorViT.from_checkpoint_state(state)
        provenance = state.get('provenance')
    elif cfg['init']:
        init = checkpoints.load(cfg['init'])
        for k in ('encoder', 'num_frames', 'temporal_hidden', 'encoder_layers', 'decoder_layers', 'clip_mean'):
            if init['cfg'][k] != cfg[k]:
                raise SystemExit(f"{k}={cfg[k]} differs from the --init checkpoint ({init['cfg'][k]})")
        tok = TrimmedTokenizer.from_dict(init['tokenizer'])
        cfg['vocab_size'] = len(tok)
        init_state = {**init, 'cfg': {**init['cfg'], 'dropout': cfg['dropout']}}
        model = SignTranslatorViT.from_checkpoint_state(init_state)
        model.cfg = cfg
        provenance = {**(init.get('provenance') or {}), 'initialized_from': cfg['init'], 'init_sha256': checkpoints.sha256(cfg['init'])}
        say(f"initialized from {cfg['init']} (epoch {init['epoch'] + 1})")
        del init, init_state
    else:
        tok = TrimmedTokenizer.build(sorted({e['text'] for e in train_entries}))
        cfg['vocab_size'] = len(tok)
        model, provenance = SignTranslatorViT.from_pretrained_parts(cfg, tok)
    with open(paths.writable(os.path.join(exp, 'config.json')), 'w') as f:
        json.dump(cfg, f, indent=1)

    model.vision.set_trainable(cfg['vision_trainable_blocks'] if cfg['stage'] == 2 else 0)
    if cfg['gradient_checkpointing']:
        model.vision.model.gradient_checkpointing_enable()
    model.to(dev)
    if cfg['stage'] == 1:
        fdir = os.path.join(paths.FEATURE_CACHE, cfg['encoder'])
        with open(os.path.join(fdir, 'meta.json')) as f:
            meta = json.load(f)
        if meta['provenance']['weights_sha256'] != provenance['vision']['weights_sha256']:
            raise TrainingFailure('cached features were made with different encoder weights')
        train_ds = FeatureClips(train_entries, [tok.encode(e['text']) for e in train_entries], fdir, cfg['num_frames'], True, cfg['seed'], cfg['temporal_jitter'])
        val_ds = FeatureClips(val_entries, [tok.encode(e['text']) for e in val_entries], fdir, cfg['num_frames'], False, cfg['seed'], 0.0)
    else:
        train_ds = FrameClips(train_entries, [tok.encode(e['text']) for e in train_entries], paths.FRAME_CACHE, cfg['num_frames'], True, cfg['seed'], cfg['temporal_jitter'])
        val_ds = FrameClips(val_entries, [tok.encode(e['text']) for e in val_entries], paths.FRAME_CACHE, cfg['num_frames'], False, cfg['seed'], 0.0)
    train_dl = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, collate_fn=collate, drop_last=True,
                          generator=torch.Generator().manual_seed(cfg['seed']))
    val_dl = DataLoader(val_ds, batch_size=max(1, cfg['batch_size']), shuffle=False, collate_fn=collate)
    opt = torch.optim.AdamW(param_groups(model, cfg), betas=(0.9, 0.98), eps=1e-8)
    steps_per_epoch = len(train_dl) // cfg['accum_steps']
    total_steps = cfg['epochs'] * steps_per_epoch
    sched = get_cosine_schedule_with_warmup(opt, cfg['warmup_steps'], total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['amp'])
    aug_rng = np.random.default_rng(cfg['seed'] + 1)
    log = logs.RunLog(exp)
    start_epoch, best, best_rows = 0, None, None
    if resume:
        opt.load_state_dict(state['optimizer'])
        sched.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        start_epoch, best = state['epoch'] + 1, state['best']
        train_ds.rng.bit_generator.state = state['data_rng']
        aug_rng.bit_generator.state = state['aug_rng']
        train_dl.generator.set_state(state['loader_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        del state
        log.truncate_from(start_epoch)
        with open(os.path.join(exp, 'val_predictions_best.json'), encoding='utf-8') as f:
            best_rows = json.load(f)
        say(f'resumed at epoch {start_epoch + 1}; best so far BLEU-4 {best["val"]["bleu4"]:.2f} at epoch {best["epoch"] + 1}')
    evals = [r['val'] for r in log.records if 'val' in r]
    ref = reference_metrics(cfg)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_batches = len(train_dl)
    every = cfg['log_every'] or max(1, n_batches // 4)
    spec = model.vision.spec
    say(f"\n=== experiment {cfg['name']} (stage {cfg['stage']})\nproject  {paths.PROJECT}\noutputs  {exp}")
    say(f"vision   {spec.hf_id}@{spec.revision[:10]} ({spec.family}, {spec.image_size}px, {spec.dim}-d)"
        f"{', weights sha256 ' + provenance['vision']['weights_sha256'][:12] if provenance and 'vision' in provenance else ''}")
    say(f"text     facebook/mbart-large-cc25 -> {cfg['encoder_layers']}+{cfg['decoder_layers']} layers, trimmed vocabulary {len(tok)} tokens, decoder start en_XX")
    say(f"data     train {len(train_ds)} clips (signers {sorted({e['signer'] for e in train_entries})}), "
        f"val {len(val_ds)} clips (signers {sorted({e['signer'] for e in val_entries})}); test signer never loaded")
    say(f"params   trainable {n_train / 1e6:.1f}M, frozen {n_frozen / 1e6:.1f}M | batch {cfg['batch_size']} x accum {cfg['accum_steps']} = "
        f"{cfg['batch_size'] * cfg['accum_steps']} | lr vision {cfg['lr_vision']} (last {cfg['vision_trainable_blocks']} blocks), new {cfg['lr_new']}, mbart {cfg['lr_mbart']}")
    say(f"input    {'cached features ' + os.path.join(paths.FEATURE_CACHE, cfg['encoder']) if cfg['stage'] == 1 else 'frames ' + paths.FRAME_CACHE + ', augmentation ' + cfg['aug']}"
        f" | {cfg['num_frames']} frames/clip | clip-mean {cfg['clip_mean']}")
    say(logs.gpu_line())
    if ref:
        say(f"reference ({ref['name']}): exact {ref['exact_match']:.1%}, BLEU-4 {ref['bleu4']:.2f}, ROUGE-L {ref['rouge_l']:.2f}")

    timer0, epoch_secs, eval_secs, reason, bad_grads = time.time(), [], [], None, 0
    for epoch in range(start_epoch, cfg['epochs']):
        model.train()
        if cfg['stage'] == 1 or cfg['vision_trainable_blocks'] <= 0:
            model.vision.eval()
        torch.cuda.reset_peak_memory_stats()
        t0, losses = time.time(), []
        opt.zero_grad(set_to_none=True)
        for step, (x, dec_in, labels, mask, _) in enumerate(train_dl, 1):
            dec_in, labels, mask = dec_in.to(dev), labels.to(dev), mask.to(dev)
            with torch.autocast('cuda', enabled=cfg['amp']):
                logits = model(visual_features(model, x, cfg, True, aug_rng), dec_in, mask)
            loss = token_loss(logits, labels, cfg['label_smoothing'])
            if not torch.isfinite(loss):
                raise TrainingFailure(f'non-finite training loss at epoch {epoch + 1}, batch {step}')
            scaler.scale(loss / cfg['accum_steps']).backward()
            losses.append(loss.item())
            if step % every == 0 or step == n_batches:
                say(f'  epoch {epoch + 1}/{cfg["epochs"]} batch {step}/{n_batches} loss {np.mean(losses[-every:]):.4f} ({time.time() - t0:.0f}s)')
            if step % cfg['accum_steps']:
                continue
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], cfg['grad_clip'])
            bad_grads = 0 if torch.isfinite(gnorm) else bad_grads + 1
            if bad_grads and (not cfg['amp'] or bad_grads >= 20):
                raise TrainingFailure(f'non-finite gradients for {bad_grads} optimizer steps at epoch {epoch + 1}')
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
        train_s = time.time() - t0
        epoch_secs.append(train_s)
        lrs = {g['name']: lr for g, lr in zip(opt.param_groups, sched.get_last_lr()) if not g['name'].endswith('_nodecay')}
        rec = {'epoch': epoch, 'epoch_number': epoch + 1, 'train_loss': float(np.mean(losses)), 'epoch_s': round(train_s, 1),
               'peak_mem_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), **{f'lr_{k}': v for k, v in lrs.items()}}
        is_eval = (epoch + 1) % cfg['eval_every'] == 0 or epoch + 1 == cfg['epochs']
        improved = False
        if is_eval:
            t1 = time.time()
            val, rows = evaluate(model, val_dl, tok, cfg)
            eval_secs.append(time.time() - t1)
            rec['val'] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in val.items()}
            rec.update({f'val_{k}': rec['val'][k] for k in ('loss', 'exact_match', 'correct', 'n', 'bleu1', 'bleu4', 'rouge_l', 'unique_predictions')})
            evals.append(rec['val'])
            if best_index(evals) == len(evals) - 1:
                improved = True
                best, best_rows = {'epoch': epoch, 'val': rec['val']}, rows
                digest = checkpoints.save_verified(best_path, model.checkpoint_state(tok, epoch=epoch, val=rec['val'], provenance=provenance))
                with open(paths.writable(os.path.join(exp, 'val_predictions_best.json')), 'w', encoding='utf-8') as f:
                    json.dump(rows, f, indent=1, ensure_ascii=False)
                checkpoints.record_best(exp, {'epoch_number': epoch + 1, 'checkpoint': best_path, 'sha256': digest,
                                              'val_bleu4': val['bleu4'], 'val_exact': val['exact_match'], 'val_loss': val['loss'], 'config': cfg})
                say(f'  saved best checkpoint -> {best_path} (sha256 {digest[:12]}, reload verified)')
            reason = stop_reason(evals, cfg)
        rec['best_epoch_number'] = best['epoch'] + 1 if best else None
        rec['evals_since_best'] = len(evals) - 1 - best_index(evals) if evals else None
        rec['elapsed_s'] = round(time.time() - timer0, 1)
        log.add(rec)

        left = cfg['epochs'] - epoch - 1
        evals_left = sum(1 for e in range(epoch + 1, cfg['epochs']) if (e + 1) % cfg['eval_every'] == 0 or e + 1 == cfg['epochs'])
        eta = left * np.mean(epoch_secs) + evals_left * (np.mean(eval_secs) if eval_secs else 0)
        say(f"\n[{cfg['name']}] epoch {epoch + 1}/{cfg['epochs']} | train loss {rec['train_loss']:.4f} | {train_s:.0f}s, "
            f"elapsed {logs.fmt_time(time.time() - timer0)}, est. remaining {logs.fmt_time(eta)} (if not stopped early)")
        say('  lr: ' + ', '.join(f'{k}={v:.2e}' for k, v in lrs.items()) + ' | ' + logs.gpu_line())
        if is_eval:
            v = rec['val']
            say(f"  VALIDATION: loss {v['loss']:.4f} | exact {v['correct']}/{v['n']} = {v['exact_match']:.1%} | BLEU-1 {v['bleu1']:.2f} "
                f"BLEU-4 {v['bleu4']:.2f} ROUGE-L {v['rouge_l']:.2f} | distinct predictions {v['unique_predictions']}")
            if ref:
                say(f"  vs {ref['name']}: exact {100 * (v['exact_match'] - ref['exact_match']):+.1f} pts, BLEU-4 {v['bleu4'] - ref['bleu4']:+.2f}, "
                    f"ROUGE-L {v['rouge_l'] - ref['rouge_l']:+.2f}")
            say(f"  best: epoch {best['epoch'] + 1} (BLEU-4 {best['val']['bleu4']:.2f}, exact {best['val']['exact_match']:.1%}); "
                f"patience {rec['evals_since_best']}/{cfg['patience']}{'  ** new best **' if improved else ''}")
            logs.print_examples(rows, epoch + 1, best_path if improved else 'weights of this epoch (not saved as best)', cfg['examples'])
            logs.print_errors(error_summary(rows))
        if is_eval or reason or epoch + 1 == cfg['epochs']:
            checkpoints.save_atomic(last_path, model.checkpoint_state(
                tok, epoch=epoch, best=best, provenance=provenance, optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                scaler=scaler.state_dict(), data_rng=train_ds.rng.bit_generator.state, aug_rng=aug_rng.bit_generator.state,
                loader_rng=train_dl.generator.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()))
            say(f'  saved resumable state -> {last_path}')
        if reason:
            say(f'\nEARLY STOP at epoch {epoch + 1}: {reason}')
            break

    status = {'status': 'early_stopped' if reason else 'completed', 'stop_reason': reason, 'elapsed_s': round(time.time() - timer0, 1)}
    if best:
        logs.write_predictions(os.path.join(exp, 'val_predictions_best.csv'), best_rows, best_path, best['epoch'] + 1, best['val'])
        status.update(best_epoch_number=best['epoch'] + 1, best_checkpoint=best_path, best_sha256=checkpoints.sha256(best_path), best_val=best['val'],
                      reference=ref)
        say(f"\nFINISHED {cfg['name']}: {status['status']}; best epoch {best['epoch'] + 1}: exact {best['val']['correct']}/{best['val']['n']} = "
            f"{best['val']['exact_match']:.1%}, BLEU-4 {best['val']['bleu4']:.2f}, ROUGE-L {best['val']['rouge_l']:.2f}")
        say(f"files: {best_path} | {os.path.join(exp, 'metrics.csv')} | {os.path.join(exp, 'val_predictions_best.csv')}")
    with open(paths.writable(os.path.join(exp, 'status.json')), 'w', encoding='utf-8') as f:
        json.dump(status, f, indent=1, default=str)
    verify_old_project(cfg, 'end')
    return status


def main(stage):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--set', nargs='*', default=[], help='override settings, e.g. --set epochs=10 name=pilot')
    args = ap.parse_args()
    try:
        run(args.config, args.set, args.resume, stage)
    except KeyboardInterrupt:
        say('\ninterrupted; continue with the same command plus --resume (the last saved state is kept)')
        raise SystemExit(130)
    except (TrainingFailure, RuntimeError, FloatingPointError) as e:
        traceback.print_exc()
        say(f'\nFATAL: {type(e).__name__}: {e}\nValid checkpoints were left untouched.')
        exp = os.path.join(paths.OUTPUTS, load_config(args.config, args.set)['name'])
        if os.path.isdir(exp):
            with open(paths.writable(os.path.join(exp, 'status.json')), 'w', encoding='utf-8') as f:
                json.dump({'status': 'failed', 'error': f'{type(e).__name__}: {e}'}, f, indent=1)
        raise SystemExit(2)
