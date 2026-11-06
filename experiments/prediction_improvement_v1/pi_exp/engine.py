"""Evaluate inference settings (decoding mode, temporal views, checkpoint ensembles) on one clip set.

A setting is a dict:
  members  [(checkpoint key, weight), ...]   keys from common.CKPTS (on 'h3'..'h5' the fold models are used)
  strategy name in views.STRATEGIES           views of one member are averaged with equal weight
  mode     'greedy' (unconstrained, log-probs averaged over members x views at every step)
           'candidates' (argmax over the training sentences of the weighted, length-normalized log-probability)
  alpha    length normalization for 'candidates'

Encoder states and candidate log-probabilities are computed once per (checkpoint, clip, view) and reused by
every setting. All model computation uses the checkpoint's own preprocessing and CUDA autocast, as in
slt_vit.infer.Translator.
"""
import json
import os
import time

import numpy as np
import torch

from pi_exp import common, scoring, views
from slt_vit import logs
from slt_vit.metrics import error_summary, per_sample, translation_metrics


class ClipSet:
    def __init__(self, clip_set, member_keys, allow_test=False):
        self.name = clip_set
        self.entries = common.entries_for(clip_set, allow_test)
        self.keys = {k: common.member_key(k, clip_set) for k in member_keys}
        self.feats = {k: common.frame_features(real, self.entries) for k, real in self.keys.items()}
        tr = common.translator(next(iter(self.keys.values())))
        self.tok = tr.tok
        self.cand_texts, cand_ids = common.training_candidates(self.tok)
        self.dec_in, self.labels, self.lengths = scoring.pad_candidates(cand_ids, tr.device)
        self.device = tr.device
        self._h, self._summed = {}, {}

    def model(self, key):
        return common.translator(self.keys[key]).model

    def encoded(self, key, e, view):
        k = (key, e['id'], view)
        if k not in self._h:
            tr = common.translator(self.keys[key])
            f = self.feats[key][e['id']]
            idx = views.view_indices(len(f), tr.cfg['num_frames'], *view)
            x = torch.from_numpy(f[idx]).to(self.device).unsqueeze(0)
            with torch.no_grad(), torch.autocast('cuda', enabled=self.device.type == 'cuda' and tr.cfg['amp']):
                self._h[k] = scoring.encode(tr.model, x)
        return self._h[k]

    def summed(self, key, e, view):
        k = (key, e['id'], view)
        if k not in self._summed:
            h, mask = self.encoded(key, e, view)
            with torch.autocast('cuda', enabled=self.device.type == 'cuda'):
                self._summed[k] = scoring.candidate_logprobs(self.model(key), h, mask, self.dec_in, self.labels).cpu()
        return self._summed[k]


def predict(cs, setting, e):
    vs = views.STRATEGIES[setting['strategy']]
    if setting['mode'] == 'greedy':
        members = []
        for key, w in setting['members']:
            for v in vs:
                h, mask = cs.encoded(key, e, v)
                members.append((cs.model(key), h, mask, w / len(vs)))
        with torch.autocast('cuda', enabled=cs.device.type == 'cuda'):
            ids = scoring.ensemble_greedy(members)
        return cs.tok.decode(ids), {}
    total = None
    for key, w in setting['members']:
        s = torch.stack([scoring.normalize(cs.summed(key, e, v), cs.lengths.cpu(), setting['alpha']) for v in vs]).mean(0) * w
        total = s if total is None else total + s
    total = total / sum(w for _, w in setting['members'])
    order = torch.argsort(total, descending=True)
    ref_rank = cs.cand_texts.index(e['text']) if e['text'] in cs.cand_texts else None
    rank = int((order == ref_rank).nonzero()[0, 0]) + 1 if ref_rank is not None else None
    top = [(cs.cand_texts[i], round(float(total[i]), 4)) for i in order[:3].tolist()]
    return cs.cand_texts[int(order[0])], {'reference_rank': rank, 'top3': top, 'margin': round(float(total[order[0]] - total[order[1]]), 4)}


def evaluate(cs, name, setting, write=True):
    t0 = time.perf_counter()
    rows = []
    for e in cs.entries:
        text, extra = predict(cs, setting, e)
        rows.append({'id': e['id'], 'signer': e['signer'], 'frame_dir': e['frame_dir'], 'reference': e['text'], 'prediction': text, **extra})
    for r, s in zip(rows, per_sample([r['prediction'] for r in rows], [r['reference'] for r in rows])):
        r.update(s)
    m = translation_metrics([r['prediction'] for r in rows], [r['reference'] for r in rows])
    m['in_training_sentences'] = sum(r['prediction'] in cs.cand_texts for r in rows)
    if write:
        d = common.out_dir('results', cs.name, name)
        cfg = {'name': name, 'clip_set': cs.name, **setting, 'checkpoints': {k: {'path': common.CKPTS[v], 'sha256': common.translator(v).info['sha256']}
                                                                              for k, v in cs.keys.items() if k in dict(setting['members'])},
               'views': views.STRATEGIES[setting['strategy']], 'candidates': 'training split texts (signers 1-5)', 'n_candidates': len(cs.cand_texts)}
        json.dump(cfg, open(os.path.join(d, 'config.json'), 'w'), indent=1)
        json.dump({'metrics': m, 'errors': error_summary(rows), 'seconds_for_set': time.perf_counter() - t0}, open(os.path.join(d, 'metrics.json'), 'w'), indent=1, default=str)
        json.dump(rows, open(os.path.join(d, 'predictions.json'), 'w'), indent=1, ensure_ascii=False, default=str)
        logs.write_predictions(os.path.join(d, 'predictions.csv'), [{k: v for k, v in r.items() if k not in ('top3',)} for r in rows],
                               ', '.join(cfg['checkpoints'][k]['path'] for k in cfg['checkpoints']), '', m)
    return m, rows


def describe(m):
    return f"{m['correct']:3d}/{m['n']} = {m['exact_match']:6.1%} | BLEU-4 {m['bleu4']:5.1f} | ROUGE-L {m['rouge_l']:5.1f} | distinct {m['unique_predictions']:2d} | repeats {m['repetitive_outputs']}"
