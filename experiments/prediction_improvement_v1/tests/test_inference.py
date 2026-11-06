"""Tests for the inference experiments. From slt_vit_experiment:
    env\\python.exe -m unittest discover -s experiments\\prediction_improvement_v1\\tests -t experiments\\prediction_improvement_v1 -v
"""
import os
import sys
import unittest

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXP)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from pi_exp import common, scoring, views  # noqa: E402
from slt_vit import paths, text_model, vision_encoder  # noqa: E402
from slt_vit.frames import sample_indices  # noqa: E402
from slt_vit.text_model import EOS, LANG  # noqa: E402
from slt_vit.train import DEFAULTS  # noqa: E402


def tiny_model(seed=0):
    from transformers import MBartForConditionalGeneration, SiglipVisionConfig
    from slt_vit.model import SignTranslatorViT
    vcfg = SiglipVisionConfig(hidden_size=768, intermediate_size=64, num_hidden_layers=1, num_attention_heads=12,
                              image_size=32, patch_size=16).to_dict()
    cfg = dict(DEFAULTS, encoder='siglip_b16_224', vocab_size=10, encoder_layers=1, decoder_layers=1, dropout=0.0,
               num_frames=12, amp=False, max_new_tokens=5, name='tiny')
    torch.manual_seed(seed)
    model = SignTranslatorViT(cfg, vision_encoder.build_from_config('siglip_b16_224', vcfg),
                              MBartForConditionalGeneration(text_model.mbart_config(10, 1, 1, 0.0)))
    with torch.no_grad():   # make decoding non-trivial: random output bias so EOS is not always first
        model.mbart.lm_head.weight.add_(torch.randn_like(model.mbart.lm_head.weight))
    return model.eval()


class TestScoring(unittest.TestCase):
    def test_single_member_greedy_equals_generate(self):
        for seed in range(4):
            model = tiny_model(seed)
            feats = torch.randn(1, 12, 768)
            ref = model.generate(feats, num_beams=1, max_new_tokens=5)[0].tolist()
            h, mask = scoring.encode(model, feats)
            got = scoring.ensemble_greedy([(model, h, mask, 1.0)], max_new_tokens=5)
            self.assertEqual(got, ref[:len(got)])
            self.assertTrue(all(t == 1 for t in ref[len(got):]), 'generate only pads after </s>')

    def test_candidate_logprob_equals_teacher_forced_sum(self):
        model = tiny_model(1)
        feats = torch.randn(1, 12, 768)
        cands = [[5, 6, EOS], [7, EOS], [5, 8, 9, 6, EOS]]
        dec_in, labels, lengths = scoring.pad_candidates(cands, 'cpu')
        h, mask = scoring.encode(model, feats)
        got = scoring.candidate_logprobs(model, h, mask, dec_in, labels)
        for k, c in enumerate(cands):
            d = torch.tensor([[LANG] + c[:-1]])
            with torch.no_grad():
                lp = F.log_softmax(model(feats, d, torch.ones_like(d)).float(), -1)[0]
            want = float(sum(lp[i, t] for i, t in enumerate(c)))
            self.assertLess(abs(float(got[k]) - want), 1e-5 * abs(want) + 1e-4)
        self.assertEqual(lengths.tolist(), [3, 2, 5])
        self.assertTrue(torch.allclose(scoring.normalize(got, lengths, 1.0), got / lengths))

    def test_two_identical_members_equal_one(self):
        model = tiny_model(2)
        h, mask = scoring.encode(model, torch.randn(1, 12, 768))
        self.assertEqual(scoring.ensemble_greedy([(model, h, mask, 1.0)]), scoring.ensemble_greedy([(model, h, mask, 0.3), (model, h, mask, 0.7)]))


class TestViews(unittest.TestCase):
    def test_uniform_is_the_baseline_sampler(self):
        for n in (6, 24, 32, 147):
            self.assertTrue(np.array_equal(views.view_indices(n, 32), sample_indices(n, 32)))

    def test_every_view_is_ordered_and_in_range(self):
        for name in views.STRATEGIES:
            for n in (6, 24, 147):
                for idx in views.strategy_indices(name, n, 32):
                    self.assertEqual(len(idx), 32)
                    self.assertTrue((np.diff(idx) >= 0).all() and idx.min() >= 0 and idx.max() <= n - 1)

    def test_views_differ_on_long_clips(self):
        a, b = views.strategy_indices('shift3', 147, 32)[:2]
        self.assertFalse(np.array_equal(a, b))


@unittest.skipUnless(os.path.isfile(paths.MANIFEST), 'manifest missing')
class TestData(unittest.TestCase):
    def test_candidates_are_training_texts_only(self):
        from slt_vit.dataset import load_manifest
        m = load_manifest()
        train = {e['text'] for e in m['entries'] if e['split'] == 'train'}
        texts = sorted({e['text'] for e in common.entries_for('train')})
        self.assertEqual(set(texts), train)
        self.assertTrue(all(e['signer'] in (1, 2, 3, 4, 5) for e in common.entries_for('train')))

    def test_test_signer_needs_explicit_permission(self):
        with self.assertRaises(PermissionError):
            common.entries_for('test')
        self.assertTrue(all(e['signer'] == 6 for e in common.entries_for('val')))
        self.assertTrue(all(e['signer'] == 4 for e in common.entries_for('h4')))


if __name__ == '__main__':
    unittest.main()
