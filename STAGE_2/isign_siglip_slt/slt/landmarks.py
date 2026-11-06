"""Per-frame landmark features from raw MediaPipe Holistic output (pure numpy, no mediapipe import).

Raw input per clip (as cached by the extractor; NaN rows mean "not detected"):
  pose [T, 33, 4]  x, y (image-normalised 0..1), z (x-scale), visibility
  lh   [T, 21, 3]  left hand  x, y, z (z relative to wrist depth, x-scale)
  rh   [T, 21, 3]  right hand
  width, height    pixel size of the frames MediaPipe saw

Output layout, FEATURE_DIM = 204 per frame (float32):
  [  0: 75]  upper-body pose, points 0..24 (legs 25..32 dropped), (x, y, z) body-relative
  [ 75:135]  left-hand shape: points 1..20 minus wrist, (x, y, z) / shoulder width
  [135:195]  right-hand shape: same
  [195:198]  left wrist (hand point 0) body-relative (x, y, z)
  [198:201]  right wrist body-relative
  [201:204]  detection flags: pose, left hand, right hand (1 = detected in this frame, 0 = filled)

Body-relative: coordinates are first converted to pixels (x*W, y*H, z*W) so non-square frames are
not distorted, then origin = the clip's median mid-shoulder point (pose 11/12) and
unit = the clip's median shoulder width. If no pose is ever found, origin = frame centre and
unit = 0.25 * frame width. Missing frames of each stream are linearly interpolated over time
between detected frames (nearest value held at the ends); a stream never detected is all zeros.
"""
import numpy as np

UPPER_POSE = 25
FEATURE_DIM = 75 + 60 + 60 + 3 + 3 + 3
L_SHOULDER, R_SHOULDER = 11, 12


def interpolate_missing(x, valid):
    """x: [T, ...] float, valid: [T] bool. Linear interpolation over T; ends hold the nearest valid frame."""
    x = np.array(x, dtype=np.float64, copy=True)
    if not valid.any():
        return np.zeros_like(x)
    t = np.arange(len(x))
    flat = x.reshape(len(x), -1)
    vi = np.flatnonzero(valid)
    for j in range(flat.shape[1]):
        flat[:, j] = np.interp(t, vi, flat[vi, j])
    return flat.reshape(x.shape)


def _to_pixels(a, w, h):
    a = a[..., :3].astype(np.float64).copy()
    a[..., 0] *= w
    a[..., 1] *= h
    a[..., 2] *= w
    return a


def frame_features(pose, lh, rh, width, height):
    T = len(pose)
    pose_ok = ~np.isnan(pose[:, L_SHOULDER, 0]) & ~np.isnan(pose[:, R_SHOULDER, 0])
    lh_ok = ~np.isnan(lh[:, 0, 0])
    rh_ok = ~np.isnan(rh[:, 0, 0])

    P = _to_pixels(np.nan_to_num(pose[:, :UPPER_POSE]), width, height)
    L = _to_pixels(np.nan_to_num(lh), width, height)
    R = _to_pixels(np.nan_to_num(rh), width, height)

    if pose_ok.any():
        mid = (P[pose_ok, L_SHOULDER, :2] + P[pose_ok, R_SHOULDER, :2]) / 2
        origin = np.median(mid, axis=0)
        sw = np.linalg.norm(P[pose_ok, L_SHOULDER, :2] - P[pose_ok, R_SHOULDER, :2], axis=1)
        unit = float(np.median(sw))
    else:
        origin = np.array([width / 2, height / 2])
        unit = 0.0
    if unit < 1e-6:
        unit = 0.25 * width
    origin3 = np.array([origin[0], origin[1], 0.0])

    P = interpolate_missing(P, pose_ok)
    L = interpolate_missing(L, lh_ok)
    R = interpolate_missing(R, rh_ok)

    def body(a, ok):
        return np.where(ok, (a - origin3) / unit, 0.0)

    any_p, any_l, any_r = pose_ok.any(), lh_ok.any(), rh_ok.any()
    out = np.zeros((T, FEATURE_DIM), np.float32)
    out[:, 0:75] = body(P, any_p).reshape(T, -1)
    out[:, 75:135] = ((L[:, 1:] - L[:, :1]) / unit * any_l).reshape(T, -1)
    out[:, 135:195] = ((R[:, 1:] - R[:, :1]) / unit * any_r).reshape(T, -1)
    out[:, 195:198] = body(L[:, 0], any_l)
    out[:, 198:201] = body(R[:, 0], any_r)
    out[:, 201] = pose_ok
    out[:, 202] = lh_ok
    out[:, 203] = rh_ok
    return out
