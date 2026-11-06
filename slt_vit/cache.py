"""Build the project's caches (inside slt_vit_experiment/cache only).

    python -m slt_vit.cache frames                       # canonical 224x224 uint8 frames per clip
    python -m slt_vit.cache features --encoder siglip_b16_224 [--views 8]

Feature files hold [1 + views, N, 768] float16: view 0 is the clip as-is (used for validation), views 1..V use
``augment_clip(..., 'base')`` with seed 0. The encoder runs in eval mode under CUDA autocast, exactly as at
inference. Every file is checked for shape and finite values.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import slt_vit  # noqa: F401
from slt_vit import paths
from slt_vit.augment import augment_clip
from slt_vit.dataset import cache_name, load_manifest
from slt_vit.frames import load_frame_folder, to_unit_float
from slt_vit.logs import say
from slt_vit.vision_encoder import ENCODERS, load_pretrained

SIZE = 224


def build_frames(manifest):
    out = paths.writable(paths.FRAME_CACHE)
    os.makedirs(out, exist_ok=True)
    n_frames = sum(len(e['frames']) for e in manifest['entries'])
    say(f'frame cache -> {out}: {len(manifest["entries"])} clips, {n_frames} frames, about {n_frames * SIZE * SIZE * 3 / 2 ** 30:.1f} GB')
    t0 = time.time()
    for i, e in enumerate(manifest['entries']):
        path = os.path.join(out, cache_name(e))
        if os.path.isfile(path) and np.load(path, mmap_mode='r').shape == (len(e['frames']), SIZE, SIZE, 3):
            continue
        arr = load_frame_folder(os.path.join(manifest['corpus'], e['frame_dir']), e['frames'], SIZE)
        np.save(path, arr)
        if i % 100 == 0:
            say(f'  {i}/{len(manifest["entries"])} clips, {time.time() - t0:.0f}s')
    say(f'frame cache complete in {time.time() - t0:.0f}s')


@torch.no_grad()
def build_features(manifest, encoder_name, views, seed):
    spec = ENCODERS[encoder_name]
    out = paths.writable(os.path.join(paths.FEATURE_CACHE, encoder_name))
    os.makedirs(out, exist_ok=True)
    n_frames = sum(len(e['frames']) for e in manifest['entries'])
    say(f'feature cache -> {out}: {(1 + views) * n_frames} images, about {(1 + views) * n_frames * spec.dim * 2 / 2 ** 30:.2f} GB')
    encoder, prov = load_pretrained(encoder_name)
    say(f"verified {prov['tensors_verified']} pretrained tensors of {spec.hf_id}@{spec.revision[:10]} (sha256 {prov['weights_sha256'][:12]})")
    dev = torch.device('cuda')
    encoder = encoder.to(dev).eval()
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for i, e in enumerate(manifest['entries']):
        x = to_unit_float(np.load(os.path.join(paths.FRAME_CACHE, cache_name(e))), dev)
        feats = []
        for v in [x] + [augment_clip(x, rng, 'base') for _ in range(views)]:
            with torch.autocast('cuda', dtype=torch.float16):
                feats.append(torch.cat([encoder(v[j:j + 64]) for j in range(0, len(v), 64)]).float())
        arr = torch.stack(feats).cpu().numpy().astype(np.float16)
        if arr.shape != (1 + views, len(e['frames']), spec.dim) or not np.isfinite(arr).all():
            raise RuntimeError(f"bad features for {e['id']}: shape {arr.shape}, finite {np.isfinite(arr).all()}")
        np.save(os.path.join(out, cache_name(e)), arr)
        if i % 100 == 0:
            say(f'  {i}/{len(manifest["entries"])} clips, {time.time() - t0:.0f}s')
    with open(os.path.join(out, 'meta.json'), 'w') as f:
        json.dump({'encoder': encoder_name, 'provenance': prov, 'views': views, 'augment': 'base', 'seed': seed,
                   'precision': 'float16 autocast', 'image_size': SIZE}, f, indent=1)
    say(f'feature cache complete in {time.time() - t0:.0f}s')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['frames', 'features'])
    ap.add_argument('--encoder', choices=sorted(ENCODERS))
    ap.add_argument('--views', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    manifest = load_manifest()
    if args.what == 'frames':
        build_frames(manifest)
    else:
        build_features(manifest, args.encoder, args.views, args.seed)


if __name__ == '__main__':
    main()
