"""Paths, checkpoint registry, clip sets and per-frame feature caching for prediction_improvement_v1.

Everything is written under experiments/prediction_improvement_v1. The existing checkpoints are only read.
"""
import json
import os
import sys

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT = os.path.dirname(os.path.dirname(EXP))
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import slt_vit  # noqa: E402,F401
from slt_vit import checkpoints, paths  # noqa: E402
from slt_vit.dataset import cache_name, load_manifest, select  # noqa: E402
from slt_vit.frames import to_unit_float  # noqa: E402
from slt_vit.infer import Translator  # noqa: E402

RESULTS = os.path.join(EXP, 'results')
FEATS = os.path.join(EXP, 'cache', 'frame_features')
CKPTS = {'siglip_s2': 'outputs/siglip_b16_stage2/best.pt', 'siglip_s1': 'outputs/siglip_b16_stage1/best.pt',
         'clip_s1': 'outputs/clip_b16_stage1/best.pt'}
for _f in (3, 4, 5):
    CKPTS[f'siglip_s2_h{_f}'] = f'outputs/loso/siglip_b16_stage2_h{_f}/best.pt'
    CKPTS[f'siglip_s1_h{_f}'] = f'outputs/loso/siglip_b16_stage1_h{_f}/best.pt'
    CKPTS[f'clip_s1_h{_f}'] = f'outputs/loso/clip_b16_stage1_h{_f}/best.pt'
LOSO_FOLDS = (3, 4, 5)
# Models trained in this experiment (train_features.py) on frozen SigLIP features and/or MediaPipe landmarks
FUSION = {'hybrid': 'H1_siglip_landmarks', 'landmarks': 'L1_landmarks_only', 'siglip_v1': 'V1_siglip_only'}
for _k, _run in FUSION.items():
    CKPTS[_k] = f'experiments/prediction_improvement_v1/runs/{_run}/best.pt'
    for _f in (3, 4, 5):
        CKPTS[f'{_k}_h{_f}'] = f'experiments/prediction_improvement_v1/runs/loso/{_run}_h{_f}/best.pt'


class FusionMember:
    """Same attributes as slt_vit.infer.Translator (model, cfg, tok, device, info) for a train_features.py checkpoint."""

    def __init__(self, path):
        import train_features
        from slt_vit.text_model import TrimmedTokenizer
        state = checkpoints.load(path)
        self.cfg = state['cfg']
        self.tok = TrimmedTokenizer.from_dict(state['tokenizer'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = train_features.FusionTranslator.from_state(state).to(self.device).eval()
        self.info = {'checkpoint': path, 'sha256': checkpoints.sha256(path), 'epoch_number': state['epoch'] + 1}

    def clip_inputs(self, entry):
        """Per-frame inputs [N, D] in the training layout: frozen SigLIP plain view, then landmark features."""
        import train_features
        from pi_exp import landmark_features
        parts = []
        if 'siglip' in self.cfg['inputs']:
            parts.append(np.load(os.path.join(train_features.VIS_DIR, cache_name(entry)), mmap_mode='r')[0].astype(np.float32))
        if 'landmarks' in self.cfg['inputs']:
            d = dict(np.load(os.path.join(train_features.LM_DIR, cache_name(entry)[:-4] + '.npz')))
            parts.append(landmark_features.clip_features(d))
        return np.concatenate(parts, 1)


def out_dir(*parts):
    p = paths.writable(os.path.join(EXP, *parts))
    os.makedirs(p, exist_ok=True)
    return p


def entries_for(clip_set, allow_test=False):
    """'val' (signer 6), 'h3'/'h4'/'h5' (that signer, for its leave-one-signer-out models), or 'test'."""
    m = load_manifest()
    if clip_set == 'val':
        return select(m, 'val')
    if clip_set in ('h3', 'h4', 'h5'):
        return select(m, signers=[int(clip_set[1])])
    if clip_set == 'train':
        return select(m, 'train')
    if clip_set == 'test':
        if not allow_test:
            raise PermissionError('signer 7 is only evaluated by tools/repeat_test.py')
        return select(m, 'test')
    raise ValueError(clip_set)


def member_key(key, clip_set):
    """Checkpoint key for a clip set: on 'h4', 'siglip_s2' means the fold model trained without signer 4."""
    return f'{key}_{clip_set}' if clip_set in ('h3', 'h4', 'h5') else key


_LOADED = {}


def translator(key):
    if key not in _LOADED:
        path = os.path.join(paths.PROJECT, CKPTS[key])
        _LOADED[key] = FusionMember(path) if key.split('_h')[0] in FUSION else Translator(path)
    return _LOADED[key]


@torch.no_grad()
def frame_features(key, entries):
    """Per-frame visual features of every frame of every clip, from the checkpoint's own vision tower
    (eval mode, CUDA float16 autocast as in slt_vit.infer). Cached per checkpoint; the cache records the
    checkpoint SHA-256 and is rebuilt if it does not match. Returns {clip id: float32 [N, D]}."""
    tr = translator(key)
    if isinstance(tr, FusionMember):       # inputs come from the frozen caches; nothing to compute
        return {e['id']: tr.clip_inputs(e) for e in entries}
    d = out_dir('cache', 'frame_features', key)
    meta_path = os.path.join(d, 'meta.json')
    meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
    if meta.get('sha256') != tr.info['sha256']:
        meta = {'sha256': tr.info['sha256'], 'checkpoint': CKPTS[key], 'precision': 'float16 autocast, stored float32'}
        for f in os.listdir(d):
            if f.endswith('.npy'):
                os.remove(os.path.join(d, f))
        json.dump(meta, open(meta_path, 'w'), indent=1)
    out = {}
    cuda = tr.device.type == 'cuda'
    for e in entries:
        p = os.path.join(d, cache_name(e))
        if os.path.isfile(p):
            out[e['id']] = np.load(p)
            continue
        x = to_unit_float(np.load(os.path.join(paths.FRAME_CACHE, cache_name(e))), tr.device)
        with torch.autocast('cuda', enabled=cuda, dtype=torch.float16):
            f = torch.cat([tr.model.vision(x[j:j + 32]) for j in range(0, len(x), 32)]).float()
        if not torch.isfinite(f).all():
            raise FloatingPointError(f"non-finite features for {e['id']}")
        out[e['id']] = f.cpu().numpy()
        np.save(p, out[e['id']])
    return out


def training_candidates(tok):
    """Candidate sentences: the normalized texts of the training split (signers 1-5) only."""
    texts = sorted({e['text'] for e in entries_for('train')})
    return texts, [tok.encode(t) for t in texts]


def sha256(path):
    return checkpoints.sha256(path)
