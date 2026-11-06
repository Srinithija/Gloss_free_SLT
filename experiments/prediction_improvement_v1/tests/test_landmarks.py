"""Tests for landmark caching and normalization (frame order, missing landmarks, shapes, normalization)."""
import json
import os
import sys
import unittest

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXP)
sys.path.insert(0, os.path.join(EXP, 'landmarks'))

import numpy as np  # noqa: E402

from pi_exp import landmark_features as lf  # noqa: E402


def synthetic(n=10, seed=0, hw=(1080, 1440)):
    rng = np.random.default_rng(seed)
    pose = rng.uniform(0.3, 0.7, (n, 33, 4)).astype(np.float32)
    pose[..., 3] = 0.9
    pose[:, 11, :2], pose[:, 12, :2] = [0.6, 0.4], [0.4, 0.4]       # shoulders (subject's left appears on image right)
    left = rng.uniform(0.3, 0.7, (n, 21, 3)).astype(np.float32)
    right = rng.uniform(0.3, 0.7, (n, 21, 3)).astype(np.float32)
    return {'pose': pose, 'left': left, 'right': right, 'present': np.ones((n, 3), np.uint8),
            'frame_index': np.arange(n, dtype=np.int32), 'image_hw': np.array(hw, np.int32)}


class TestFeatures(unittest.TestCase):
    def test_shape_and_finite(self):
        x = lf.clip_features(synthetic())
        self.assertEqual(x.shape, (10, lf.DIM))
        self.assertTrue(np.isfinite(x).all())

    def test_body_normalization(self):
        x = lf.clip_features(synthetic())
        sh = x[:, 2:6].reshape(-1, 2, 2)                    # POSE_IDX[1:3] = shoulders 11, 12
        np.testing.assert_allclose(sh.mean(1), 0, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(sh[:, 0] - sh[:, 1], axis=-1), 1, atol=1e-5)

    def test_translation_and_scale_invariance(self):
        a = synthetic()
        b = {k: v.copy() for k, v in a.items()}
        for key in ('pose', 'left', 'right'):
            b[key][..., :2] = b[key][..., :2] * 0.5 + 0.1        # signer smaller and elsewhere in the image
            if key != 'pose':
                b[key][..., 2] *= 0.5
        np.testing.assert_allclose(lf.clip_features(a), lf.clip_features(b), atol=1e-4)

    def test_aspect_ratio_is_respected(self):
        a = synthetic(hw=(1080, 1440))
        b = synthetic(hw=(1080, 1080))
        self.assertFalse(np.allclose(lf.clip_features(a), lf.clip_features(b)))

    def test_missing_hand_interpolated_and_flagged(self):
        d = synthetic()
        d['left'][3:6] = np.nan
        d['present'][3:6, 1] = 0
        x = lf.clip_features(d)
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_array_equal(x[:, 231], [1, 1, 1, 0, 0, 0, 1, 1, 1, 1])
        full = lf.clip_features(synthetic())
        np.testing.assert_allclose(x[4, 21:63], full[2, 21:63] + (full[6, 21:63] - full[2, 21:63]) * 2 / 4, atol=1e-4)

    def test_hand_never_detected_is_zero(self):
        d = synthetic()
        d['right'][:] = np.nan
        d['present'][:, 2] = 0
        x = lf.clip_features(d)
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue((x[:, 126:233 - 2] == 0).all() and (x[:, 232] == 0).all())

    def test_frame_order_is_preserved(self):
        d = synthetic()
        r = {k: (v[::-1].copy() if k not in ('image_hw',) else v) for k, v in d.items()}
        np.testing.assert_allclose(lf.clip_features(r), lf.clip_features(d)[::-1], atol=1e-5)

    def test_augment_changes_only_body_blocks(self):
        x = lf.clip_features(synthetic())
        y = lf.augment(x, np.random.default_rng(0))
        np.testing.assert_array_equal(x[:, 14:21], y[:, 14:21])
        np.testing.assert_array_equal(x[:, 63:126], y[:, 63:126])
        self.assertFalse(np.allclose(x[:, 0:14], y[:, 0:14]))


CACHE = os.path.join(EXP, 'cache', 'landmarks', 'static_1440')


@unittest.skipUnless(os.path.isdir(CACHE), 'landmark cache not built')
class TestCache(unittest.TestCase):
    def test_cached_clips_follow_manifest_order(self):
        from extract import MANIFEST, cache_name
        m = json.load(open(MANIFEST, encoding='utf-8'))
        checked = 0
        for e in m['entries']:
            p = os.path.join(CACHE, cache_name(e) + '.npz')
            if not os.path.isfile(p):
                continue
            d = np.load(p)
            self.assertEqual(len(d['present']), len(e['frames']))
            np.testing.assert_array_equal(d['frame_index'], np.arange(len(e['frames'])))
            self.assertEqual(d['pose'].shape, (len(e['frames']), 33, 4))
            self.assertTrue(np.isnan(d['left'][d['present'][:, 1] == 0]).all())
            self.assertTrue(np.isfinite(d['left'][d['present'][:, 1] == 1]).all())
            checked += 1
        self.assertGreater(checked, 0)


if __name__ == '__main__':
    unittest.main()
