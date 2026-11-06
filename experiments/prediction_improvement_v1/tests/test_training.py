"""Tests for train_features.py: the training sampler (speed warp, frame drop) and the fusion model."""
import os
import sys
import unittest

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXP)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import train_features as tf  # noqa: E402
from pi_exp import landmark_features as lf, scoring  # noqa: E402
from slt_vit import text_model  # noqa: E402
from slt_vit.frames import sample_indices  # noqa: E402


class TestSampler(unittest.TestCase):
    def test_deterministic_without_rng(self):
        for n in (6, 24, 147):
            np.testing.assert_array_equal(tf.sample(n, 32, None, 0.1, 0.4, 0.2), sample_indices(n, 32))

    def test_warp_and_drop_keep_order_and_range(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            n = int(rng.integers(6, 150))
            idx = tf.sample(n, 32, rng, 0.1, 0.4, 0.2)
            self.assertEqual(len(idx), 32)
            self.assertTrue((np.diff(idx) >= 0).all(), 'temporal order must never be reversed')
            self.assertTrue(idx.min() >= 0 and idx.max() <= n - 1)


def tiny_fusion(inputs):
    from transformers import MBartForConditionalGeneration
    cfg = dict(tf.DEFAULTS, inputs=inputs, temporal_hidden=32, lm_proj=16, vocab_size=10, encoder_layers=1, decoder_layers=1, dropout=0.0)
    torch.manual_seed(0)
    return tf.FusionTranslator(cfg, MBartForConditionalGeneration(text_model.mbart_config(10, 1, 1, 0.0))).eval()


class TestFusion(unittest.TestCase):
    def test_input_widths(self):
        for inputs, width in (('siglip', 768), ('landmarks', lf.DIM), ('siglip+landmarks', 768 + lf.DIM)):
            m = tiny_fusion(inputs)
            x = torch.randn(2, 32, width)
            self.assertEqual(m.encoder_inputs(x).shape, (2, 14, 1024))
            d = torch.tensor([[4, 5, 6], [4, 7, 1]])
            self.assertEqual(m(x, d, torch.ones_like(d)).shape, (2, 3, 10))

    def test_clip_mean_only_on_visual_part(self):
        m = tiny_fusion('siglip+landmarks')
        x = torch.randn(1, 32, 768 + lf.DIM)
        y = x.clone()
        y[..., :768] += 5.0                                   # constant offset on the visual part is removed
        torch.testing.assert_close(m.encoder_inputs(x), m.encoder_inputs(y), atol=1e-4, rtol=1e-4)
        z = x.clone()
        z[..., 768:] += 5.0                                   # but not on landmarks
        self.assertFalse(torch.allclose(m.encoder_inputs(x), m.encoder_inputs(z), atol=1e-3))

    def test_from_state_round_trip(self):
        m = tiny_fusion('landmarks')
        m2 = tf.FusionTranslator.from_state({'model': m.state_dict(), 'cfg': m.cfg})
        x = torch.randn(1, 32, lf.DIM)
        torch.testing.assert_close(m.encoder_inputs(x), m2.eval().encoder_inputs(x))
        self.assertEqual(m.generate(x, 5).tolist(), m2.generate(x, 5).tolist())


class TestScoringOnFusion(unittest.TestCase):
    def test_encode_matches_generate(self):
        m = tiny_fusion('siglip+landmarks')
        with torch.no_grad():
            m.mbart.lm_head.weight.add_(torch.randn_like(m.mbart.lm_head.weight))
        x = torch.randn(1, 32, 768 + lf.DIM)
        h, mask = scoring.encode(m, x)
        got = scoring.ensemble_greedy([(m, h, mask, 1.0)], max_new_tokens=5)
        ref = m.generate(x, 5)[0].tolist()
        self.assertEqual(got, ref[:len(got)])


if __name__ == '__main__':
    unittest.main()
