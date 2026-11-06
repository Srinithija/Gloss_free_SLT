"""MediaPipe Holistic landmark extraction for ISL-CSLRT frame folders (runs in env_mediapipe only).

    env_mediapipe\\python.exe -I experiments\\prediction_improvement_v1\\landmarks\\extract.py pilot
    env_mediapipe\\python.exe -I experiments\\prediction_improvement_v1\\landmarks\\extract.py all --mode static

Needs only numpy, OpenCV and mediapipe==0.10.14; it does not import torch or the slt_vit package. Clips come
from data/manifest.json, frames are read in the manifest's order (natural sort, the order used everywhere
else), and nothing about the label is read except the manifest id used to name the cache file.

Each frame is centre-cropped to at most 4:3 (as slt_vit.frames.canonicalize does, without padding/resizing)
and passed as RGB to a fresh Holistic instance per clip. Output per clip, cache/landmarks/<mode>/<id>.npz:
  pose   float32 [N,33,4]  x, y (fractions of the cropped image), z, visibility; NaN where not detected
  left   float32 [N,21,3]  MediaPipe's left_hand_landmarks (x, y, z); NaN where not detected
  right  float32 [N,21,3]  right_hand_landmarks
  present uint8  [N,3]     pose, left, right detected
  frame_index int32 [N]    position in the manifest frame list (0..N-1)
  image_hw int32 [2]       cropped image height, width (to restore the pixel aspect ratio)
The split of signer 7 (test) is extracted only with --include-test.
"""
import argparse
import json
import os
import re
import sys
import time

import cv2
import numpy as np

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT = os.path.dirname(os.path.dirname(EXP))
MANIFEST = os.path.join(PROJECT, 'data', 'manifest.json')
OUT = os.path.join(EXP, 'cache', 'landmarks')
MAX_ASPECT = 4 / 3


def cache_name(entry):     # same rule as slt_vit.dataset.cache_name
    return re.sub(r'[^A-Za-z0-9_-]+', '_', entry['id'])


def crop43(img):
    h, w = img.shape[:2]
    if w > MAX_ASPECT * h:
        cw = int(round(MAX_ASPECT * h))
        x0 = (w - cw) // 2
        img = img[:, x0:x0 + cw]
    return img


def to_array(lms, n, dims):
    if lms is None:
        return np.full((n, dims), np.nan, np.float32)
    if dims == 4:
        return np.array([[p.x, p.y, p.z, p.visibility] for p in lms.landmark], np.float32)
    return np.array([[p.x, p.y, p.z] for p in lms.landmark], np.float32)


def extract_clip(mp, corpus, entry, mode, max_side):
    folder = os.path.join(corpus, entry['frame_dir'])
    pose, left, right, present, hw = [], [], [], [], None
    with mp.solutions.holistic.Holistic(static_image_mode=(mode == 'static'), model_complexity=1,
                                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as hol:
        for f in entry['frames']:
            img = cv2.imread(os.path.join(folder, f))
            if img is None:
                raise ValueError(f'unreadable frame {folder}/{f}')
            img = crop43(img)
            s = max_side / max(img.shape[:2])
            if s < 1:
                img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
            hw = img.shape[:2]
            r = hol.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pose.append(to_array(r.pose_landmarks, 33, 4))
            left.append(to_array(r.left_hand_landmarks, 21, 3))
            right.append(to_array(r.right_hand_landmarks, 21, 3))
            present.append([r.pose_landmarks is not None, r.left_hand_landmarks is not None, r.right_hand_landmarks is not None])
    return {'pose': np.stack(pose), 'left': np.stack(left), 'right': np.stack(right), 'present': np.array(present, np.uint8),
            'frame_index': np.arange(len(entry['frames']), dtype=np.int32), 'image_hw': np.array(hw, np.int32)}


def pilot_entries(entries):
    """Two clips per signer 1-6 (shortest and longest of that signer), plus 'help me' and 'i am hungry' for each."""
    out = []
    for s in range(1, 7):
        es = sorted([e for e in entries if e['signer'] == s], key=lambda e: (len(e['frames']), e['id']))
        out += [es[0], es[-1]] + [e for e in es if e['text'] in ('help me', 'i am hungry')]
    seen, uniq = set(), []
    for e in out:
        if e['id'] not in seen:
            seen.add(e['id'])
            uniq.append(e)
    return uniq


def main():
    import mediapipe as mp
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['pilot', 'all'])
    ap.add_argument('--mode', choices=['static', 'track'], default='static')
    ap.add_argument('--max-side', type=int, default=1440)
    ap.add_argument('--include-test', action='store_true')
    args = ap.parse_args()
    m = json.load(open(MANIFEST, encoding='utf-8'))
    entries = [e for e in m['entries'] if e['signer'] != 7 or args.include_test]
    if args.what == 'pilot':
        entries = pilot_entries(entries)
    tag = f'{args.mode}_{args.max_side}' + ('_pilot' if args.what == 'pilot' else '')
    out = os.path.join(OUT, tag)
    os.makedirs(out, exist_ok=True)
    stats, t0, n_frames = [], time.time(), 0
    for i, e in enumerate(entries):
        p = os.path.join(out, cache_name(e) + '.npz')
        if os.path.isfile(p):
            d = dict(np.load(p))
        else:
            t1 = time.time()
            d = extract_clip(mp, m['corpus'], e, args.mode, args.max_side)
            np.savez_compressed(p + '.tmp.npz', **d)
            os.replace(p + '.tmp.npz', p)
            stats.append({'id': e['id'], 'signer': e['signer'], 'split': e['split'], 'frames': len(e['frames']),
                          'pose': float(d['present'][:, 0].mean()), 'left': float(d['present'][:, 1].mean()),
                          'right': float(d['present'][:, 2].mean()), 'any_hand': float(d['present'][:, 1:].max(1).mean()),
                          'seconds': round(time.time() - t1, 2)})
        n_frames += len(e['frames'])
        if i % 25 == 0:
            print(f'{i}/{len(entries)} clips, {n_frames} frames, {time.time() - t0:.0f}s', flush=True)
    with open(os.path.join(out, 'extraction_stats.json'), 'a', encoding='utf-8') as f:
        for s in stats:
            f.write(json.dumps(s) + '\n')
    json.dump({'mediapipe': mp.__version__, 'opencv': cv2.__version__, 'mode': args.mode, 'max_side': args.max_side, 'model_complexity': 1,
               'min_detection_confidence': 0.5, 'min_tracking_confidence': 0.5, 'clips': len(entries), 'python': sys.version.split()[0]},
              open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    print(f'done: {len(entries)} clips, {n_frames} frames in {time.time() - t0:.0f}s -> {out}')


if __name__ == '__main__':
    main()
