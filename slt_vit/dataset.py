"""ISL-CSLRT sentence-level data: annotation parsing, integrity checks and the signer-independent split.

    python -m slt_vit.dataset            # build data/manifest.json and cross-check it

Labels come from ``corpus_csv_files/ISL_CSLRT_Corpus_frame_details.xlsx`` (one row per frame). Each sentence
folder holds numbered take folders; the take number is the signer. That was checked by eye across all 97
sentences in the earlier project (per-signer contact sheets), with one exception listed in SIGNER_OVERRIDES.
"""
import collections
import hashlib
import json
import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from slt_vit import paths

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')
SIGNER_OVERRIDES = {('do you need something', '4'): 3}   # folder 4 shows signer 3 (no folder 3 exists)
SPLIT_BY_SIGNER = {1: 'train', 2: 'train', 3: 'train', 4: 'train', 5: 'train', 6: 'val', 7: 'test'}
TEST_SIGNERS = {s for s, split in SPLIT_BY_SIGNER.items() if split == 'test'}


def natural_key(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]


def normalize_text(sentence):
    """Reference and prediction normalization used everywhere: lower case, single spaces."""
    return ' '.join(str(sentence).strip().lower().split())


def build_manifest(corpus=paths.DATASET):
    import pandas as pd
    xlsx = os.path.join(corpus, 'corpus_csv_files', 'ISL_CSLRT_Corpus_frame_details.xlsx')
    ann = pd.read_excel(xlsx)
    labels, ann_files = {}, collections.defaultdict(set)
    for sentence, path in zip(ann['Sentence'], ann['Frames path']):
        parts = str(path).replace('\\', '/').split('/')
        if parts[1] != 'Frames_Sentence_Level' or parts[2] != sentence:
            raise ValueError(f'unexpected annotation row: {sentence!r} {path!r}')
        labels.setdefault((parts[2], parts[3]), sentence)
        ann_files[(parts[2], parts[3])].add(parts[4].strip())   # a few file names carry a stray leading space

    root = os.path.join(corpus, 'Frames_Sentence_Level')
    takes, problems = [], []
    for sent in sorted(os.listdir(root)):
        for take in sorted(os.listdir(os.path.join(root, sent)), key=natural_key):
            folder = os.path.join(root, sent, take)
            if not os.path.isdir(folder):
                continue
            files = sorted([f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTENSIONS)], key=natural_key)
            key = (sent, take)
            if key not in labels:
                problems.append({'folder': f'{sent}/{take}', 'reason': 'no annotation rows'})
            elif {f.strip() for f in files} != ann_files[key]:
                problems.append({'folder': f'{sent}/{take}', 'reason': 'files differ from annotation rows'})
            elif len(files) < 2:
                problems.append({'folder': f'{sent}/{take}', 'reason': 'fewer than 2 frames'})
            else:
                takes.append((key, files))

    # Identical frames filed under two different sentences: at least one label is wrong, so exclude both.
    owners = collections.defaultdict(set)
    for key, files in takes:
        for f in files:
            with open(os.path.join(root, *key, f), 'rb') as fh:
                owners[hashlib.md5(fh.read()).hexdigest()].add(key)
    conflicted = {}
    for keys in owners.values():
        if len({labels[k] for k in keys}) > 1:
            for k in keys:
                conflicted[k] = sorted(f'{o[0]}/{o[1]}' for o in keys if o != k)

    entries, excluded = [], list(problems)
    for key, files in takes:
        if key in conflicted:
            excluded.append({'folder': f'{key[0]}/{key[1]}', 'reason': 'identical frames under another sentence: ' + ', '.join(conflicted[key])})
            continue
        signer = SIGNER_OVERRIDES.get(key, int(key[1]))
        entries.append({'id': f'{key[0]}__take{key[1]}', 'sentence': labels[key], 'text': normalize_text(labels[key]),
                        'signer': signer, 'split': SPLIT_BY_SIGNER[signer],
                        'frame_dir': os.path.join('Frames_Sentence_Level', *key).replace('\\', '/'), 'frames': files})
    return {'corpus': corpus.replace('\\', '/'), 'annotation_rows': len(ann), 'take_folders': len(takes) + len(problems),
            'split_by_signer': {str(k): v for k, v in SPLIT_BY_SIGNER.items()},
            'signer_overrides': [{'folder': f'{k[0]}/{k[1]}', 'signer': v} for k, v in SIGNER_OVERRIDES.items()],
            'entries': entries, 'excluded': excluded}


def summary(m):
    lines = [f"annotation rows {m['annotation_rows']}, take folders {m['take_folders']}, eligible {len(m['entries'])}, excluded {len(m['excluded'])}"]
    for split in ('train', 'val', 'test'):
        es = [e for e in m['entries'] if e['split'] == split]
        lines.append(f"{split:5}: {len(es)} clips, {len({e['text'] for e in es})} sentences, {sum(len(e['frames']) for e in es)} frames, "
                     f"signers {dict(sorted(collections.Counter(e['signer'] for e in es).items()))}")
    lines += [f"excluded {x['folder']}: {x['reason']}" for x in m['excluded']]
    return '\n'.join(lines)


def cross_check(m, old_path=paths.OLD_MANIFEST):
    """Compare with the earlier project's manifest (read-only): ids, labels, signers, splits and frame order."""
    if not os.path.isfile(old_path):
        return 'old manifest not found; no cross-check'
    with open(old_path, encoding='utf-8') as f:
        old = {e['id']: e for e in json.load(f)['entries']}
    new = {e['id']: e for e in m['entries']}
    diffs = [i for i in set(old) | set(new) if i not in old or i not in new
             or any(old[i][k] != new[i][k] for k in ('text', 'signer', 'split', 'frames'))]
    return f'cross-check with the earlier manifest: {len(new)} vs {len(old)} entries, {len(diffs)} differences' + (f' {sorted(diffs)[:5]}' if diffs else '')


def load_manifest(path=paths.MANIFEST):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def select(manifest, split=None, signers=None):
    """Entries of a manifest split, or of explicit signers (leave-one-signer-out). The test signer can only be
    selected through split='test'."""
    if signers:
        signers = {int(s) for s in signers}
        if signers & TEST_SIGNERS:
            raise ValueError('the held-out test signer cannot be used for training or validation')
        return [e for e in manifest['entries'] if e['signer'] in signers]
    return [e for e in manifest['entries'] if e['split'] == split]


def cache_name(entry):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', entry['id']) + '.npy'


class FeatureClips(Dataset):
    """Cached per-frame features [views, N, D] -> sampled [T, D]; a random view in training, view 0 otherwise."""

    def __init__(self, entries, targets, feature_dir, num_frames, train, seed, jitter):
        from slt_vit.frames import sample_indices
        self.sample_indices = sample_indices
        self.entries, self.targets, self.dir = entries, targets, feature_dir
        self.num_frames, self.train, self.jitter = num_frames, train, jitter
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        arr = np.load(os.path.join(self.dir, cache_name(self.entries[i])), mmap_mode='r')
        view = int(self.rng.integers(arr.shape[0])) if self.train else 0
        idx = self.sample_indices(arr.shape[1], self.num_frames, self.rng if self.train else None, self.jitter)
        return torch.from_numpy(arr[view, idx].astype(np.float32)), self.targets[i], i


class FrameClips(Dataset):
    """Cached canonical uint8 frames [N, S, S, 3] -> sampled [T, S, S, 3] (augmented later on the GPU)."""

    def __init__(self, entries, targets, frame_dir, num_frames, train, seed, jitter):
        from slt_vit.frames import sample_indices
        self.sample_indices = sample_indices
        self.entries, self.targets, self.dir = entries, targets, frame_dir
        self.num_frames, self.train, self.jitter = num_frames, train, jitter
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        arr = np.load(os.path.join(self.dir, cache_name(self.entries[i])), mmap_mode='r')
        idx = self.sample_indices(arr.shape[0], self.num_frames, self.rng if self.train else None, self.jitter)
        return torch.from_numpy(np.ascontiguousarray(arr[idx])), self.targets[i], i


def main():
    m = build_manifest()
    os.makedirs(paths.writable(paths.DATA), exist_ok=True)
    with open(paths.writable(paths.MANIFEST), 'w', encoding='utf-8') as f:
        json.dump(m, f, indent=1, ensure_ascii=False)
    print(summary(m))
    print(cross_check(m))
    print(f'written {paths.MANIFEST}')


if __name__ == '__main__':
    main()
