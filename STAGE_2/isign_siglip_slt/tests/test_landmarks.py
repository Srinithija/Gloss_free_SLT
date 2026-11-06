import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slt.landmarks import FEATURE_DIM, frame_features, interpolate_missing  # noqa: E402

W, H = 300, 300


def make_clip(T=10, seed=0):
    rng = np.random.default_rng(seed)
    pose = np.zeros((T, 33, 4))
    pose[..., :2] = rng.uniform(0.3, 0.7, (T, 33, 2))
    pose[:, 11, :2] = [0.6, 0.5]  # shoulder width 0.2 * 300 = 60 px, mid-shoulder (150, 150)
    pose[:, 12, :2] = [0.4, 0.5]
    lh = np.zeros((T, 21, 3))
    lh[..., :2] = rng.uniform(0.2, 0.8, (T, 21, 2))
    rh = lh.copy() + 0.01
    return pose, lh, rh


def test_shape_and_dtype():
    f = frame_features(*make_clip(), W, H)
    assert f.shape == (10, FEATURE_DIM) == (10, 204) and f.dtype == np.float32
    assert np.isfinite(f).all()


def test_body_relative_normalisation():
    pose, lh, rh = make_clip()
    f = frame_features(pose, lh, rh, W, H)
    p = f[:, :75].reshape(-1, 25, 3)
    np.testing.assert_allclose(p[:, 11, :2], [[0.5, 0.0]] * 10, atol=1e-6)  # (180-150)/60: half a shoulder width
    np.testing.assert_allclose(p[:, 12, :2], [[-0.5, 0.0]] * 10, atol=1e-6)


def test_invariant_to_translation_and_scale():
    pose, lh, rh = make_clip()
    f1 = frame_features(pose, lh, rh, W, H)
    s, d = 0.5, np.array([0.1, 0.2])
    pose2, lh2, rh2 = pose.copy(), lh.copy(), rh.copy()
    for a in (pose2, lh2, rh2):
        a[..., :2] = a[..., :2] * s + d
        a[..., 2] *= s
    f2 = frame_features(pose2, lh2, rh2, W, H)
    np.testing.assert_allclose(f1, f2, atol=1e-5)


def test_hand_shape_relative_to_wrist():
    pose, lh, rh = make_clip()
    f = frame_features(pose, lh, rh, W, H)
    exp = (lh[0, 1:, :2] - lh[0, 0, :2]) * W / 60.0
    np.testing.assert_allclose(f[0, 75:135].reshape(20, 3)[:, :2], exp, atol=1e-5)
    np.testing.assert_allclose(f[0, 195:197], (lh[0, 0, :2] * W - 150) / 60.0, atol=1e-5)


def test_missing_hand_interpolated_and_flagged():
    pose, lh, rh = make_clip()
    lh[3:6] = np.nan
    f = frame_features(pose, lh, rh, W, H)
    assert f[:, 202].tolist() == [1, 1, 1, 0, 0, 0, 1, 1, 1, 1]
    assert f[:, 203].all()
    full = frame_features(*make_clip(), W, H)
    # frame 4 is the linear midpoint of frames 2 and 6 for wrist position
    np.testing.assert_allclose(f[4, 195:198], (full[2, 195:198] + full[6, 195:198]) / 2, atol=1e-5)
    np.testing.assert_allclose(f[0, 195:198], full[0, 195:198], atol=1e-6)


def test_edges_hold_nearest():
    x = np.array([[np.nan], [2.0], [4.0], [np.nan]])
    valid = ~np.isnan(x[:, 0])
    np.testing.assert_allclose(interpolate_missing(np.nan_to_num(x), valid)[:, 0], [2, 2, 4, 4])


def test_hand_never_detected_is_zero():
    pose, lh, rh = make_clip()
    rh[:] = np.nan
    f = frame_features(pose, lh, rh, W, H)
    assert (f[:, 135:195] == 0).all() and (f[:, 198:201] == 0).all() and (f[:, 203] == 0).all()


def test_no_pose_fallback():
    pose, lh, rh = make_clip()
    pose[:] = np.nan
    f = frame_features(pose, lh, rh, W, H)
    assert np.isfinite(f).all() and (f[:, :75] == 0).all() and (f[:, 201] == 0).all()
    # unit falls back to 0.25 * width = 75 px, origin = frame centre
    np.testing.assert_allclose(f[0, 195:197], (lh[0, 0, :2] * W - 150) / 75.0, atol=1e-5)


def test_non_square_frames_use_pixels():
    pose, lh, rh = make_clip()
    f = frame_features(pose, lh, rh, 300, 600)
    # shoulders are horizontal, so width 60 px is unchanged; a y offset of 0.1 is 60 px = 1 unit
    pose2 = pose.copy()
    pose2[:, 0, 1] = 0.6
    g = frame_features(pose2, lh, rh, 300, 600)
    assert abs(g[0, 1] - 1.0) < 1e-5
    assert f.shape[1] == 204
