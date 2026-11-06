"""Per-frame landmark features from a cached MediaPipe Holistic clip (see landmarks/extract.py), numpy only.

Coordinates are first put in units of image height (x * W / H) so the pixel aspect ratio is kept, then made
body-relative per clip: origin = median mid-shoulder point over the clip, unit = median shoulder width.
Per-clip (not per-frame) statistics keep the body's own movement in the features. Missing hands are linearly
interpolated between detected frames (held at the nearest detection at the clip ends); a hand never detected in
the clip is all zeros. The raw detection flags are kept as features so the model can tell real from filled values.

Feature layout per frame (233 values):
  [0:14)    pose xy, body-relative, for POSE_IDX (nose, shoulders, elbows, wrists)
  [14:21)   pose visibility for POSE_IDX
  [21:63)   left hand xy, body-relative (21 x 2)
  [63:126)  left hand shape: xyz relative to the wrist, divided by the hand size (wrist -> middle-finger MCP)
  [126:168) right hand xy, body-relative
  [168:231) right hand shape
  [231:233) left / right hand detected in this frame (before interpolation)
"""
import numpy as np

POSE_IDX = [0, 11, 12, 13, 14, 15, 16]
DIM = 233
BODY_SLICES = [(0, 14), (21, 63), (126, 168)]     # body-relative xy blocks (augmented together)


def _interpolate(arr, present):
    """arr [N,K,C] with NaN rows where present is 0 -> filled copy (linear in time), or zeros if never present."""
    n = len(arr)
    idx = np.flatnonzero(present)
    if len(idx) == 0:
        return np.zeros_like(arr), False
    out = arr.copy()
    flat = out.reshape(n, -1)
    for j in range(flat.shape[1]):
        flat[:, j] = np.interp(np.arange(n), idx, flat[idx, j])
    return out, True


def clip_features(d):
    """d: dict with pose [N,33,4], left/right [N,21,3], present [N,3], image_hw [2] -> float32 [N, 233]."""
    h, w = (float(v) for v in d['image_hw'])
    ar = w / h
    pose = d['pose'].astype(np.float64).copy()
    pose[..., 0] *= ar
    n = len(pose)
    have_pose = d['present'][:, 0].astype(bool)
    if have_pose.any():
        sh = pose[have_pose][:, [11, 12], :2]
        center = np.median(sh.mean(1), axis=0)
        scale = float(np.median(np.linalg.norm(sh[:, 0] - sh[:, 1], axis=-1)))
    else:
        center, scale = np.array([ar / 2, 0.5]), 0.25
    scale = max(scale, 1e-3)
    pose_f, _ = _interpolate(pose[:, POSE_IDX], have_pose)
    feats = [((pose_f[..., :2] - center) / scale).reshape(n, -1), pose_f[..., 3].reshape(n, -1)]
    flags = []
    for key, col in (('left', 1), ('right', 2)):
        hand = d[key].astype(np.float64).copy()
        hand[..., 0] *= ar
        hand[..., 2] *= ar                                  # MediaPipe hand z is on the x scale
        det = d['present'][:, col].astype(bool)
        hand, ok = _interpolate(hand, det)
        if ok:
            body = (hand[..., :2] - center) / scale
            wrist = hand[:, :1]
            size = np.median(np.linalg.norm(hand[det][:, 9, :2] - hand[det][:, 0, :2], axis=-1))
            shape = (hand - wrist) / max(float(size), 1e-4)
        else:
            body, shape = np.zeros((n, 21, 2)), np.zeros((n, 21, 3))
        feats += [body.reshape(n, -1), shape.reshape(n, -1)]
        flags.append(det.astype(np.float64))
    out = np.concatenate(feats + [np.stack(flags, 1)], axis=1).astype(np.float32)
    assert out.shape == (n, DIM), out.shape
    return out


def augment(x, rng, scale=0.1, shift=0.1, rot_deg=5.0):
    """Training-only: one random similarity transform of the body-relative xy blocks for the whole clip.
    x: [T, 233] -> new array. Hand shape blocks and flags are unchanged (no mirroring: handedness matters)."""
    x = x.copy()
    s = rng.uniform(1 - scale, 1 + scale)
    t = rng.uniform(-shift, shift, 2)
    a = np.deg2rad(rng.uniform(-rot_deg, rot_deg))
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) * s
    for lo, hi in BODY_SLICES:
        xy = x[:, lo:hi].reshape(len(x), -1, 2)
        x[:, lo:hi] = (xy @ r.T + t).reshape(len(x), -1)
    return x
