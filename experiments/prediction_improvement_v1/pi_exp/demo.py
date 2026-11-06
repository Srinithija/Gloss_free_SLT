"""Prediction with confidence scores for the demo app: D3 or E3 on one clip (canonical frames + optional landmarks).

The scores are exactly the experiment's: per model and view, the summed log-probability of each of the 97 training
sentences (alpha 0), averaged over models with the setting's weights. Confidence = softmax of that average.
For the hybrid member:
- dataset clips use the cached landmarks of all frames (as in the experiment);
- uploaded videos run MediaPipe on the sampled frames only (subprocess in env_mediapipe), so the per-clip body
  statistics come from those frames. That is a small approximation of the cached features.
"""
import json
import os
import subprocess
import tempfile
import time

import numpy as np
import torch

from pi_exp import common, landmark_features as lf, scoring, views
from slt_vit.frames import to_unit_float

SETTINGS = {'E3': ('E3_ens3_candidates_trim5', 'E_hybrid.json'), 'D3': ('D3_ens_candidates_w0.5_trim5', 'D_combinations.json')}
MP_PYTHON = os.path.join(common.PROJECT, 'env_mediapipe', 'python.exe')
MP_SCRIPT = os.path.join(common.EXP, 'landmarks', 'extract_frames.py')
TMP = os.path.join(common.EXP, 'demo_tmp')


def load_setting(tag):
    name, f = SETTINGS[tag]
    s = json.load(open(os.path.join(common.EXP, 'configs', f)))[name]
    s['members'] = [tuple(m) for m in s['members']]
    return name, s


class DemoPredictor:
    def __init__(self, tag):
        self.tag = tag
        self.name, self.setting = load_setting(tag)
        self.members = {k: common.translator(k) for k, _ in self.setting['members']}
        self.uses_landmarks = 'hybrid' in self.members
        self.frozen_vision = common.translator('siglip_s1').model.vision if self.uses_landmarks else None
        tr = next(iter(self.members.values()))
        self.device, self.tok = tr.device, tr.tok
        self.texts, ids = common.training_candidates(self.tok)
        self.dec_in, self.labels, self.lengths = scoring.pad_candidates(ids, self.device)
        (self.view,) = views.STRATEGIES[self.setting['strategy']]

    def sample(self, n):
        return views.view_indices(n, 32, *self.view)

    @torch.no_grad()
    def scores(self, frames, lm_sampled=None):
        """frames: canonical uint8 [N,224,224,3]; lm_sampled: landmark features [32, 233] at self.sample(N) (E3 only).
        Returns (probabilities over the 97 sentences as a numpy array, GPU model seconds)."""
        cuda = self.device.type == 'cuda'
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        idx = self.sample(len(frames))
        uniq, inv = np.unique(idx, return_inverse=True)
        x = to_unit_float(frames[uniq], self.device)
        inv = torch.as_tensor(inv, device=self.device)
        total = 0.0
        for key, w in self.setting['members']:
            tr = self.members[key]
            with torch.autocast('cuda', enabled=cuda, dtype=torch.float16):
                vis = (self.frozen_vision if key == 'hybrid' else tr.model.vision)(x).float()[inv]
            if key == 'hybrid':
                if lm_sampled is None:
                    raise ValueError('E3 needs landmarks')
                vis = torch.cat([vis, torch.as_tensor(lm_sampled, device=self.device)], 1)
            with torch.autocast('cuda', enabled=cuda):
                h, mask = scoring.encode(tr.model, vis.unsqueeze(0))
                s = scoring.candidate_logprobs(tr.model, h, mask, self.dec_in, self.labels)
            total = total + w * scoring.normalize(s.float(), self.lengths, self.setting['alpha'])
        total = total / sum(w for _, w in self.setting['members'])
        if cuda:
            torch.cuda.synchronize()
        return torch.softmax(total, 0).cpu().numpy(), time.perf_counter() - t0

    def landmarks_from_cache(self, entry, n):
        import train_features
        from slt_vit.dataset import cache_name
        d = dict(np.load(os.path.join(train_features.LM_DIR, cache_name(entry)[:-4] + '.npz')))
        return lf.clip_features(d)[self.sample(n)], d['present'][self.sample(n)]

    def landmarks_from_video(self, video_path, n):
        """Runs MediaPipe (env_mediapipe) on the sampled frames of the video. Returns (features [32,233], present [32,3], seconds)."""
        if not os.path.isfile(MP_PYTHON):
            raise RuntimeError(f'MediaPipe environment not found: {MP_PYTHON}')
        idx = self.sample(n)
        uniq, inv = np.unique(idx, return_inverse=True)
        os.makedirs(common.out_dir('demo_tmp'), exist_ok=True)
        fd, out = tempfile.mkstemp(suffix='.npz', dir=TMP)
        os.close(fd)
        t0 = time.perf_counter()
        try:
            r = subprocess.run([MP_PYTHON, '-I', MP_SCRIPT, '--video', video_path, '--indices', ','.join(map(str, uniq)),
                                '--expected-frames', str(n), '--out', out], capture_output=True, text=True,
                               env={k: v for k, v in os.environ.items() if k != 'PYTHONPATH'})
            if r.returncode:
                raise RuntimeError('MediaPipe failed: ' + (r.stderr.strip().splitlines() or ['unknown error'])[-1])
            d = dict(np.load(out))
        finally:
            if os.path.exists(out):
                os.remove(out)
        return lf.clip_features(d)[inv], d['present'][inv], time.perf_counter() - t0

    def top(self, probs, k=5):
        order = np.argsort(-probs)[:k]
        return [(self.texts[i], float(probs[i])) for i in order]
