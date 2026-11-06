"""Train translators on cached per-frame inputs: frozen SigLIP features, MediaPipe landmarks, or both (Phase 4/5).

    env\\python.exe experiments\\prediction_improvement_v1\\train_features.py --name V1_siglip_only --inputs siglip
    env\\python.exe experiments\\prediction_improvement_v1\\train_features.py --name L1_landmarks --inputs landmarks
    env\\python.exe experiments\\prediction_improvement_v1\\train_features.py --name H1_hybrid --inputs siglip+landmarks
    ... --set epochs=60 eval_every=60 patience=99 train_signers=[1,2,4,5,6] val_signers=[3]    (leave-one-signer-out)

Recipe = the project's stage 1 (configs/siglip_b16_stage1.yaml): 100 epochs, batch 16, AdamW, lr 3e-4 (new layers) /
1e-4 (mBART), 200 warm-up steps, cosine decay, label smoothing 0.1, temporal jitter 0.1, greedy validation every
5 epochs, best by BLEU-4 (ties -> lower loss), patience 8 evaluations, the same deterioration/overfit stops.
SigLIP input = cache/features/siglip_b16_224 (frozen pretrained SigLIP; a random one of the 8 augmented views
in training, the plain view for validation), clip-mean normalized as in the project. Landmarks =
landmark_features.clip_features on cache/landmarks/static_1440, projected by Linear(233, 256) + LayerNorm + ReLU.
The two are concatenated per frame and passed to the project's TemporalEncoder (clip-mean off: it is applied to the
SigLIP part only) and a compact mBART copied and verified from cc25, exactly as in the project.
Signer 7 is never loaded. Every run writes to runs/<name>/ and refuses to overwrite an existing run.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import GenerationConfig, MBartForConditionalGeneration, get_cosine_schedule_with_warmup  # noqa: E402

from pi_exp import common, landmark_features as lf  # noqa: E402
from slt_vit import checkpoints, logs, paths, text_model  # noqa: E402
from slt_vit.dataset import cache_name, load_manifest, select  # noqa: E402
from slt_vit.frames import sample_indices  # noqa: E402
from slt_vit.logs import say  # noqa: E402
from slt_vit.metrics import error_summary, per_sample, translation_metrics  # noqa: E402
from slt_vit.temporal_model import TemporalEncoder  # noqa: E402
from slt_vit.text_model import EOS, LANG, PAD, TrimmedTokenizer  # noqa: E402
from slt_vit.train import best_index, collate, stop_reason, token_loss  # noqa: E402

RUNS = os.path.join(common.EXP, 'runs')
VIS_DIR = os.path.join(paths.FEATURE_CACHE, 'siglip_b16_224')
LM_DIR = os.path.join(common.EXP, 'cache', 'landmarks', 'static_1440')
DEFAULTS = dict(name='', inputs='siglip', train_signers=[], val_signers=[], num_frames=32, clip_mean=True, temporal_hidden=1024,
                lm_proj=256, encoder_layers=3, decoder_layers=3, dropout=0.1, batch_size=16, epochs=100, lr_new=3e-4, lr_mbart=1e-4,
                weight_decay=0.01, warmup_steps=200, label_smoothing=0.1, grad_clip=1.0, temporal_jitter=0.1, eval_every=5, patience=8,
                deteriorate_k=4, deteriorate_margin=8.0, overfit_loss_ratio=1.3, overfit_k=3, seed=0, amp=True, max_new_tokens=20,
                lm_aug=True, time_warp=0.0, frame_drop=0.0, vis_views=True)


def sample(n, t, rng=None, jitter=0.0, warp=0.0, drop=0.0):
    """Training sampler = slt_vit.frames.sample_indices (+ optional speed warp and frame drop); deterministic without rng."""
    idx = sample_indices(n, t, rng, jitter)
    if rng is None:
        return idx
    if warp:                                    # monotone time warp: one part of the clip faster, the rest slower
        g = np.exp(rng.uniform(-np.log1p(warp), np.log1p(warp)))
        pos = (np.linspace(0, 1, t) ** g) * (idx[-1] - idx[0]) + idx[0]
        idx = np.clip(np.round(pos), 0, n - 1).astype(np.int64)
    if drop:                                    # a dropped frame repeats the previous sample
        for k in np.flatnonzero(rng.uniform(size=t) < drop):
            if k:
                idx[k] = idx[k - 1]
    return idx


class Clips(Dataset):
    def __init__(self, entries, targets, cfg, train):
        self.entries, self.targets, self.cfg, self.train = entries, targets, cfg, train
        self.rng = np.random.default_rng(cfg['seed'])
        self.use_vis, self.use_lm = 'siglip' in cfg['inputs'], 'landmarks' in cfg['inputs']
        self.lm = {}
        if self.use_lm:
            for e in entries:
                p = os.path.join(LM_DIR, cache_name(e)[:-4] + '.npz')
                d = dict(np.load(p))
                if len(d['present']) != len(e['frames']):
                    raise RuntimeError(f"landmark cache of {e['id']} has {len(d['present'])} frames, manifest {len(e['frames'])}")
                self.lm[e['id']] = lf.clip_features(d)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e, c = self.entries[i], self.cfg
        n = len(e['frames'])
        rng = self.rng if self.train else None
        idx = sample(n, c['num_frames'], rng, c['temporal_jitter'] if self.train else 0.0, c['time_warp'], c['frame_drop'])
        parts = []
        if self.use_vis:
            arr = np.load(os.path.join(VIS_DIR, cache_name(e)), mmap_mode='r')
            if arr.shape[1] != n:
                raise RuntimeError(f"feature cache of {e['id']} has {arr.shape[1]} frames, manifest {n}")
            view = int(self.rng.integers(arr.shape[0])) if self.train and c['vis_views'] else 0
            parts.append(arr[view, idx].astype(np.float32))
        if self.use_lm:
            x = self.lm[e['id']][idx]
            if self.train and c['lm_aug']:
                x = lf.augment(x, self.rng)
            parts.append(x)
        return torch.from_numpy(np.concatenate(parts, 1)), self.targets[i], i


class FusionTranslator(nn.Module):
    VIS = 768

    def __init__(self, cfg, mbart):
        super().__init__()
        self.cfg = cfg
        self.use_vis, self.use_lm = 'siglip' in cfg['inputs'], 'landmarks' in cfg['inputs']
        in_dim = (self.VIS if self.use_vis else 0) + (cfg['lm_proj'] if self.use_lm else 0)
        if self.use_lm:
            self.lm = nn.Sequential(nn.Linear(lf.DIM, cfg['lm_proj']), nn.LayerNorm(cfg['lm_proj']), nn.ReLU(inplace=True))
        self.temporal = TemporalEncoder(in_dim, cfg['temporal_hidden'], mbart.config.d_model, clip_mean=False, dropout=cfg['dropout'])
        self.mbart = mbart
        self.mbart.generation_config = GenerationConfig(decoder_start_token_id=LANG, eos_token_id=EOS, pad_token_id=PAD,
                                                        bos_token_id=0, forced_eos_token_id=EOS, num_beams=1)

    def encoder_inputs(self, x):
        parts = []
        if self.use_vis:
            v = x[..., :self.VIS]
            parts.append(v - v.mean(dim=1, keepdim=True) if self.cfg['clip_mean'] else v)
        if self.use_lm:
            parts.append(self.lm(x[..., -lf.DIM:]))
        return self.temporal(torch.cat(parts, -1))

    def forward(self, x, dec_in, mask):
        h = self.encoder_inputs(x)
        return self.mbart(inputs_embeds=h, attention_mask=torch.ones(h.shape[:2], dtype=torch.long, device=h.device),
                          decoder_input_ids=dec_in, decoder_attention_mask=mask, return_dict=True).logits

    @torch.no_grad()
    def generate(self, x, max_new_tokens=20):
        h = self.encoder_inputs(x)
        gen = GenerationConfig(decoder_start_token_id=LANG, eos_token_id=EOS, pad_token_id=PAD, bos_token_id=0, forced_eos_token_id=EOS,
                               num_beams=1, max_new_tokens=max_new_tokens)
        return self.mbart.generate(inputs_embeds=h, attention_mask=torch.ones(h.shape[:2], dtype=torch.long, device=h.device), generation_config=gen)

    @classmethod
    def from_state(cls, state):
        c = state['cfg']
        mbart = MBartForConditionalGeneration(text_model.mbart_config(c['vocab_size'], c['encoder_layers'], c['decoder_layers'], c['dropout']))
        m = cls(c, mbart)
        m.load_state_dict(state['model'], strict=True)
        return m


@torch.no_grad()
def evaluate(model, loader, tok, cfg):
    model.eval()
    dev = next(model.parameters()).device
    hyps, rows, loss_sum, n_tok = [], [], 0.0, 0
    for x, dec_in, labels, mask, idx in loader:
        x, dec_in, labels, mask = x.to(dev), dec_in.to(dev), labels.to(dev), mask.to(dev)
        with torch.autocast('cuda', enabled=cfg['amp']):
            logits = model(x, dec_in, mask)
            out = model.generate(x, cfg['max_new_tokens'])
        loss_sum += token_loss(logits, labels, 0.0, 'sum').item()
        n_tok += int(mask.sum())
        hyps += [tok.decode(o) for o in out.tolist()]
        rows += [{'id': loader.dataset.entries[i]['id'], 'signer': loader.dataset.entries[i]['signer'],
                  'frame_dir': loader.dataset.entries[i]['frame_dir'], 'reference': loader.dataset.entries[i]['text']} for i in idx]
    refs = [r['reference'] for r in rows]
    m = translation_metrics(hyps, refs)
    m['loss'] = loss_sum / n_tok
    for r, h, s in zip(rows, hyps, per_sample(hyps, refs)):
        r.update(prediction=h, **s)
    return m, rows


def load_cfg(args):
    cfg = dict(DEFAULTS, name=args.name, inputs=args.inputs)
    for item in args.set:
        k, _, v = item.partition('=')
        if k not in DEFAULTS:
            raise SystemExit(f'unknown setting {k}')
        cfg[k] = yaml.safe_load(v)
        if isinstance(DEFAULTS[k], float):
            cfg[k] = float(cfg[k])
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--inputs', required=True, choices=['siglip', 'landmarks', 'siglip+landmarks'])
    ap.add_argument('--set', nargs='*', default=[])
    args = ap.parse_args()
    cfg = load_cfg(args)
    run = paths.writable(os.path.normpath(os.path.join(RUNS, cfg['name'])))
    best_path = os.path.join(run, 'best.pt')
    if os.path.exists(best_path) or os.path.exists(os.path.join(run, 'status.json')):
        raise SystemExit(f'{run} already exists; choose a new --name')
    os.makedirs(run, exist_ok=True)
    m = load_manifest()
    train_e = select(m, 'train', cfg['train_signers'])
    val_e = select(m, 'val', cfg['val_signers'])
    torch.manual_seed(cfg['seed'])
    dev = torch.device('cuda')
    tok = TrimmedTokenizer.build(sorted({e['text'] for e in train_e}))
    ref_tok = checkpoints.load(os.path.join(paths.PROJECT, common.CKPTS['siglip_s2']))['tokenizer']
    if tok.old_ids != ref_tok['old_ids']:
        raise RuntimeError('tokenizer differs from the selected checkpoint (candidate scoring / ensembles need the same vocabulary)')
    cfg['vocab_size'] = len(tok)
    mbart = text_model.build_pretrained_mbart(tok.old_ids, cfg['encoder_layers'], cfg['decoder_layers'], cfg['dropout'])
    model = FusionTranslator(cfg, mbart).to(dev)
    if 'siglip' in cfg['inputs']:
        meta = json.load(open(os.path.join(VIS_DIR, 'meta.json')))
        cfg['siglip_features'] = {'dir': VIS_DIR, 'weights_sha256': meta['provenance']['weights_sha256'], 'views': meta['views']}
    json.dump(cfg, open(os.path.join(run, 'config.json'), 'w'), indent=1)
    train_ds = Clips(train_e, [tok.encode(e['text']) for e in train_e], cfg, True)
    val_ds = Clips(val_e, [tok.encode(e['text']) for e in val_e], cfg, False)
    train_dl = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, collate_fn=collate, drop_last=True,
                          generator=torch.Generator().manual_seed(cfg['seed']))
    val_dl = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, collate_fn=collate)
    new = [p for n, p in model.named_parameters() if not n.startswith('mbart.')]
    groups = []
    for name, params, lr in (('new', new, cfg['lr_new']), ('mbart', list(model.mbart.parameters()), cfg['lr_mbart'])):
        seen = {id(p): p for p in params}.values()
        groups += [{'name': name, 'params': [p for p in seen if p.ndim > 1], 'lr': lr, 'weight_decay': cfg['weight_decay']},
                   {'name': name + '_nodecay', 'params': [p for p in seen if p.ndim <= 1], 'lr': lr, 'weight_decay': 0.0}]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-8)
    sched = get_cosine_schedule_with_warmup(opt, cfg['warmup_steps'], cfg['epochs'] * len(train_dl))
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['amp'])
    log = logs.RunLog(run)
    say(f"=== {cfg['name']}: inputs {cfg['inputs']} | train {len(train_ds)} clips signers {sorted({e['signer'] for e in train_e})} | "
        f"val {len(val_ds)} clips signers {sorted({e['signer'] for e in val_e})} | test signer never loaded")
    say(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M | lm_aug {cfg['lm_aug']} time_warp {cfg['time_warp']} frame_drop {cfg['frame_drop']} | {logs.gpu_line()}")
    evals, best, best_rows, reason, t_start = [], None, None, None, time.time()
    for epoch in range(cfg['epochs']):
        model.train()
        torch.cuda.reset_peak_memory_stats()
        t0, losses = time.time(), []
        for x, dec_in, labels, mask, _ in train_dl:
            x, dec_in, labels, mask = x.to(dev), dec_in.to(dev), labels.to(dev), mask.to(dev)
            with torch.autocast('cuda', enabled=cfg['amp']):
                loss = token_loss(model(x, dec_in, mask), labels, cfg['label_smoothing'])
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite training loss at epoch {epoch + 1}')
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        rec = {'epoch': epoch, 'epoch_number': epoch + 1, 'train_loss': float(np.mean(losses)), 'epoch_s': round(time.time() - t0, 1),
               'peak_mem_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
        if (epoch + 1) % cfg['eval_every'] == 0 or epoch + 1 == cfg['epochs']:
            val, rows = evaluate(model, val_dl, tok, cfg)
            rec['val'] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in val.items()}
            rec.update({f'val_{k}': rec['val'][k] for k in ('loss', 'exact_match', 'correct', 'n', 'bleu4', 'rouge_l', 'unique_predictions')})
            evals.append(rec['val'])
            flag = ''
            if best_index(evals) == len(evals) - 1:
                best, best_rows = {'epoch': epoch, 'val': rec['val']}, rows
                digest = checkpoints.save_verified(best_path, {'model': model.state_dict(), 'cfg': cfg, 'tokenizer': tok.to_dict(),
                                                               'epoch': epoch, 'val': rec['val'], 'kind': 'FusionTranslator'})
                checkpoints.record_best(run, {'epoch_number': epoch + 1, 'checkpoint': best_path, 'sha256': digest, 'val_bleu4': val['bleu4'],
                                              'val_exact': val['exact_match'], 'val_loss': val['loss']})
                json.dump(rows, open(os.path.join(run, 'val_predictions_best.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
                flag = f'  ** new best, saved (sha256 {digest[:12]}, reload verified)'
            say(f"[{cfg['name']}] epoch {epoch + 1}: train loss {rec['train_loss']:.4f} ({rec['epoch_s']:.0f}s) | val loss {val['loss']:.4f} "
                f"exact {val['correct']}/{val['n']} = {val['exact_match']:.1%} BLEU-4 {val['bleu4']:.2f} ROUGE-L {val['rouge_l']:.2f}{flag}")
            reason = stop_reason(evals, cfg)
        log.add(rec)
        if reason:
            say(f'EARLY STOP at epoch {epoch + 1}: {reason}')
            break
    logs.write_predictions(os.path.join(run, 'val_predictions_best.csv'), best_rows, best_path, best['epoch'] + 1, best['val'])
    status = {'status': 'early_stopped' if reason else 'completed', 'stop_reason': reason, 'elapsed_s': round(time.time() - t_start, 1),
              'best_epoch_number': best['epoch'] + 1, 'best_val': best['val'], 'best_sha256': checkpoints.sha256(best_path),
              'peak_mem_gb': max(r['peak_mem_gb'] for r in log.records)}
    json.dump(status, open(os.path.join(run, 'status.json'), 'w'), indent=1, default=str)
    v = best['val']
    say(f"FINISHED {cfg['name']}: best epoch {best['epoch'] + 1}: exact {v['correct']}/{v['n']} = {v['exact_match']:.1%}, BLEU-4 {v['bleu4']:.2f}, "
        f"ROUGE-L {v['rouge_l']:.2f} | {status['elapsed_s'] / 60:.1f} min | peak GPU {status['peak_mem_gb']} GB")


if __name__ == '__main__':
    main()
