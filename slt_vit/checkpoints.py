"""Checkpoint safety: atomic writes, SHA-256, a reload check, and an append-only record of best-model updates."""
import datetime
import hashlib
import json
import os

import torch

from slt_vit import paths


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def save_atomic(path, state):
    """Write to ``path.tmp``, flush to disk, then rename. An interrupted or failed save never damages ``path``."""
    paths.writable(path)
    tmp = path + '.tmp'
    try:
        with open(tmp, 'wb') as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def save_verified(path, state, required=('model', 'cfg', 'tokenizer')):
    """Atomic save, then reload from disk and check the keys and tensor count; returns the file's SHA-256."""
    save_atomic(path, state)
    reloaded = torch.load(path, map_location='cpu', weights_only=False)
    missing = [k for k in required if k not in reloaded]
    if missing or len(reloaded['model']) != len(state['model']):
        raise RuntimeError(f'{path} did not reload correctly (missing {missing})')
    return sha256(path)


def record_best(exp_dir, entry):
    """Append one line per best-model update: epoch, metric, path, hash, config."""
    entry = {'time': datetime.datetime.now().isoformat(timespec='seconds'), **entry}
    with open(paths.writable(os.path.join(exp_dir, 'best_history.jsonl')), 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)
