"""Pipeline tests. Run from the project folder:  env\\python.exe -m unittest discover -s tests -v"""
import json
import os
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
import torch

import slt_vit  # noqa: F401
from slt_vit import checkpoints, paths, text_model, vision_encoder
from slt_vit.dataset import TEST_SIGNERS, natural_key, select
from slt_vit.frames import canonicalize, read_video, sample_indices, to_unit_float
from slt_vit.text_model import EOS, LANG, PAD, TrimmedTokenizer
from slt_vit.train import DEFAULTS, best_index, collate, evaluate, stop_reason, token_loss

HAVE_MANIFEST = os.path.isfile(paths.MANIFEST)


def tiny_tokenizer():
    pieces = ['<s>', '<pad>', '</s>', '<unk>', 'en_XX', '\u2581are', '\u2581you', '\u2581free', '\u2581to', 'day']
    return TrimmedTokenizer(list(range(len(pieces))), pieces)


def tiny_model(clip_mean=True):
    """Small random model with the real component classes (no downloads): 1-layer 768-d SigLIP tower on 32px input."""
    from transformers import MBartForConditionalGeneration, SiglipVisionConfig
    from slt_vit.model import SignTranslatorViT
    vcfg = SiglipVisionConfig(hidden_size=768, intermediate_size=64, num_hidden_layers=1, num_attention_heads=12,
                              image_size=32, patch_size=16).to_dict()
    cfg = dict(DEFAULTS, encoder='siglip_b16_224', vocab_size=10, encoder_layers=1, decoder_layers=1, dropout=0.0,
               clip_mean=clip_mean, num_frames=12, amp=False, max_new_tokens=5, name='tiny')
    torch.manual_seed(0)
    vision = vision_encoder.build_from_config('siglip_b16_224', vcfg)
    mbart = MBartForConditionalGeneration(text_model.mbart_config(10, 1, 1, 0.0))
    return SignTranslatorViT(cfg, vision, mbart), vcfg


class FrameTests(unittest.TestCase):
    def test_landscape_crop_and_letterbox_keep_proportions(self):
        img = np.zeros((1080, 1920, 3), np.uint8)
        img[:, :240] = (0, 0, 255)            # outside the 1440-wide centre crop
        img[:, 240:1680] = (0, 255, 0)
        out = canonicalize(img, 224)
        self.assertEqual(out.shape, (224, 224, 3))
        self.assertEqual(out[:, :, 0].max(), 0)
        self.assertEqual(tuple(out[112, 112]), (0, 255, 0))
        self.assertEqual(tuple(out[2, 112]), (0, 0, 0))

    def test_portrait_is_padded_not_stretched(self):
        out = canonicalize(np.full((320, 240, 3), 200, np.uint8), 224)
        self.assertEqual(tuple(out[112, 2]), (0, 0, 0))
        self.assertEqual(tuple(out[2, 112]), (200, 200, 200))

    def test_sampling_is_chronological_and_covers_the_clip(self):
        for n in (6, 30, 147):
            idx = sample_indices(n, 32)
            self.assertEqual((idx[0], idx[-1]), (0, n - 1))
            self.assertTrue(np.all(np.diff(idx) >= 0))
        rng = np.random.default_rng(0)
        for _ in range(100):
            idx = sample_indices(20, 32, rng, 0.1)
            self.assertTrue(idx.min() >= 0 and idx.max() <= 19 and np.all(np.diff(idx) >= 0))

    def test_natural_frame_order(self):
        self.assertEqual(sorted(['a (10).jpg', 'a (2).jpg', 'a (1).jpg'], key=natural_key), ['a (1).jpg', 'a (2).jpg', 'a (10).jpg'])

    def test_unreadable_and_unsupported_videos_raise_clear_errors(self):
        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, 'x.mp4')
            open(bad, 'wb').write(b'not a video')
            with self.assertRaisesRegex(ValueError, 'could not open|decodable'):
                read_video(bad, 224)
            txt = os.path.join(d, 'x.txt')
            open(txt, 'w').close()
            with self.assertRaisesRegex(ValueError, 'unsupported file type'):
                read_video(txt, 224)
            with self.assertRaisesRegex(ValueError, 'not found'):
                read_video(os.path.join(d, 'missing.mp4'), 224)


@unittest.skipUnless(HAVE_MANIFEST, 'run python -m slt_vit.dataset first')
class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = json.load(open(paths.MANIFEST, encoding='utf-8'))

    def test_counts_and_signer_independent_split(self):
        es = self.m['entries']
        self.assertEqual(self.m['take_folders'], 663)
        self.assertEqual(len(es), 659)
        self.assertEqual({s: sum(e['split'] == s for e in es) for s in ('train', 'val', 'test')}, {'train': 474, 'val': 94, 'test': 91})
        signers = {s: {e['signer'] for e in es if e['split'] == s} for s in ('train', 'val', 'test')}
        self.assertEqual(signers, {'train': {1, 2, 3, 4, 5}, 'val': {6}, 'test': {7}})
        self.assertEqual(len({e['id'] for e in es}), len(es))

    def test_every_frame_exists_and_is_in_natural_order(self):
        for e in self.m['entries']:
            self.assertEqual(e['frames'], sorted(e['frames'], key=natural_key))
            folder = os.path.join(self.m['corpus'], e['frame_dir'])
            for f in e['frames']:
                self.assertTrue(os.path.isfile(os.path.join(folder, f)), os.path.join(folder, f))

    def test_test_signer_cannot_leak_into_training_or_validation(self):
        with self.assertRaises(ValueError):
            select(self.m, signers=[1, 7])
        self.assertTrue(all(e['signer'] not in TEST_SIGNERS for e in select(self.m, 'train') + select(self.m, 'val')))


class TokenizerTests(unittest.TestCase):
    def test_decode_stops_at_eos_and_skips_language_token(self):
        self.assertEqual(tiny_tokenizer().decode([LANG, 5, 6, 7, 8, 9, EOS, 5]), 'are you free today')

    def test_decoder_inputs_are_labels_shifted_right_after_en_xx(self):
        _, dec_in, labels, mask, _ = collate([(torch.zeros(1), [5, 6, EOS], 0), (torch.zeros(1), [7, EOS], 1)])
        self.assertEqual(labels.tolist(), [[5, 6, EOS], [7, EOS, PAD]])
        self.assertEqual(dec_in.tolist(), [[LANG, 5, 6], [LANG, 7, PAD]])
        self.assertEqual(mask.tolist(), [[1, 1, 1], [1, 1, 0]])

    @unittest.skipUnless(HAVE_MANIFEST, 'needs the manifest')
    def test_trimmed_ids_map_to_cc25_ids_and_rows(self):
        cc25 = text_model.cc25_tokenizer()
        texts = sorted({e['text'] for e in json.load(open(paths.MANIFEST, encoding='utf-8'))['entries']})
        tok = TrimmedTokenizer.build(texts, cc25)
        self.assertEqual(tok.old_ids[:5], [0, 1, 2, 3, 250004])
        for t in texts:
            ids = tok.encode(t)
            self.assertEqual(ids[-1], EOS)
            self.assertEqual([tok.old_ids[i] for i in ids[:-1]], cc25(t, add_special_tokens=False)['input_ids'])
            self.assertEqual(tok.decode(ids), t)


class WeightTransferTests(unittest.TestCase):
    def test_compact_mbart_equals_cc25(self):
        try:
            paths.cc25_file('model.safetensors')
        except FileNotFoundError:
            self.skipTest('mbart-large-cc25 not in the local Hugging Face cache')
        old_ids = [0, 1, 2, 3, 250004, 621, 1556]
        model = text_model.build_pretrained_mbart(old_ids, 2, 2, 0.1)
        self.assertGreater(text_model.verify_mbart_transfer(model, old_ids), 50)
        with torch.no_grad():
            model.model.decoder.layers[1].fc1.weight[0, 0] += 1.0
        with self.assertRaisesRegex(RuntimeError, 'differs from cc25'):
            text_model.verify_mbart_transfer(model, old_ids)

    def test_vision_weights_verified_against_the_file(self):
        spec = vision_encoder.ENCODERS['siglip_b16_224']
        snap = os.path.join(paths.HF_CACHE, 'models--google--siglip-base-patch16-224', 'snapshots', spec.revision, spec.weights_file)
        if not os.path.isfile(snap):
            self.skipTest('SigLIP weights not downloaded yet')
        enc, prov = vision_encoder.load_pretrained('siglip_b16_224')
        self.assertEqual(prov['tensors_verified'], len(enc.model.state_dict()))
        real = vision_encoder._raw_state

        def tampered(path):
            raw = real(path)
            k = next(k for k in raw if k.startswith('vision_model.encoder.layers.0'))
            raw[k] = raw[k] + 1
            return raw
        with mock.patch.object(vision_encoder, '_raw_state', tampered):
            with self.assertRaisesRegex(RuntimeError, 'differ'):
                vision_encoder.load_pretrained('siglip_b16_224')

    def test_our_preprocessing_matches_the_official_image_processors(self):
        from transformers import AutoImageProcessor
        img = canonicalize(np.random.default_rng(0).integers(0, 255, (1080, 1920, 3), dtype=np.uint8), 224)
        compared = []
        for name, spec in vision_encoder.ENCODERS.items():
            folder = os.path.join(paths.HF_CACHE, 'models--' + spec.hf_id.replace('/', '--'), 'snapshots', spec.revision)
            if not os.path.isfile(os.path.join(folder, 'preprocessor_config.json')):
                continue
            ref = AutoImageProcessor.from_pretrained(folder)(images=img, return_tensors='pt')['pixel_values']
            enc = vision_encoder.VisionEncoder(name, torch.nn.Identity())
            ours = (to_unit_float(img[None]) - enc.mean) / enc.std
            self.assertEqual(tuple(ours.shape), tuple(ref.shape), name)
            self.assertLess(float((ours - ref).abs().max()), 0.02, name)   # bicubic no-op resize rounding only
            compared.append(name)
        if not compared:
            self.skipTest('no preprocessor_config.json downloaded yet')
        self.assertEqual(sorted(compared), sorted(vision_encoder.ENCODERS))


class ModelTests(unittest.TestCase):
    def test_shapes_finite_loss_and_gradients_everywhere(self):
        model, _ = tiny_model()
        model.vision.set_trainable(1)
        frames = torch.rand(2, 12, 3, 32, 32)
        _, dec_in, labels, mask, _ = collate([(torch.zeros(1), [5, 6, EOS], 0), (torch.zeros(1), [7, EOS], 1)])
        feats = model.frame_features(frames)
        self.assertEqual(tuple(feats.shape), (2, 12, 768))
        logits = model(feats, dec_in, mask)
        self.assertEqual(tuple(logits.shape), (2, 3, 10))
        loss = token_loss(logits, labels, 0.1)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, module in (('vision', model.vision), ('temporal', model.temporal), ('mbart', model.mbart)):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads), name)
        model.eval()
        out = model.generate(feats, max_new_tokens=5)
        self.assertTrue((out[:, 0] == LANG).all())
        self.assertLessEqual(out.shape[1], 6)

    def test_greedy_generation_is_really_greedy(self):
        """generate(num_beams=1) must equal a hand-written argmax loop (guards against inherited beam settings)."""
        model, _ = tiny_model()
        model.eval()
        torch.manual_seed(1)
        feats = torch.randn(1, 12, 768)
        with torch.no_grad():
            out = model.generate(feats, num_beams=1, max_new_tokens=5)[0].tolist()
            seq = [LANG]
            for _ in range(5):
                dec = torch.tensor([seq])
                nxt = int(model(feats, dec, torch.ones_like(dec))[0, -1].argmax())
                seq.append(nxt)
                if nxt == EOS:
                    break
        if seq[-1] != EOS and len(seq) == 6:
            seq[-1] = EOS            # forced_eos_token_id puts </s> in the last allowed position
        self.assertEqual(out, seq)
        self.assertEqual(model.mbart.generation_config.num_beams, 1)

    def test_clip_mean_removes_what_is_constant_over_a_clip(self):
        model, _ = tiny_model(clip_mean=True)
        model.eval()
        x = torch.randn(1, 12, 768)
        with torch.no_grad():
            self.assertTrue(torch.allclose(model.temporal(x), model.temporal(x + 3.0), atol=1e-4))

    def test_frozen_vision_gets_no_gradient(self):
        model, _ = tiny_model()
        model.vision.set_trainable(0)
        self.assertFalse(any(p.requires_grad for p in model.vision.parameters()))


class CheckpointTests(unittest.TestCase):
    def test_verified_save_reload_rebuilds_an_identical_model(self):
        from slt_vit.model import SignTranslatorViT
        model, vcfg = tiny_model()
        model.eval()
        with tempfile.TemporaryDirectory(dir=paths.PROJECT) as d:
            path = os.path.join(d, 'best.pt')
            state = model.checkpoint_state(tiny_tokenizer(), epoch=0, val={})
            digest = checkpoints.save_verified(path, state)
            self.assertEqual(digest, checkpoints.sha256(path))
            restored = SignTranslatorViT.from_checkpoint_state(checkpoints.load(path)).eval()
            x = torch.randn(1, 12, 768)
            with torch.no_grad():
                self.assertTrue(torch.equal(model.temporal(x), restored.temporal(x)))

    def test_failed_save_keeps_the_previous_file(self):
        with tempfile.TemporaryDirectory(dir=paths.PROJECT) as d:
            path = os.path.join(d, 'best.pt')
            checkpoints.save_atomic(path, {'v': 1})
            with mock.patch('slt_vit.checkpoints.torch.save', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    checkpoints.save_atomic(path, {'v': 2})
            self.assertEqual(torch.load(path)['v'], 1)
            self.assertFalse(os.path.exists(path + '.tmp'))

    def test_writes_outside_the_project_are_refused(self):
        with self.assertRaises(PermissionError):
            paths.writable(r'C:\Users\SRINITHIJA\Desktop\Sign_processing\sltbaselines\outputs\x.pt')
        with self.assertRaises(PermissionError):
            checkpoints.save_atomic(os.path.join(paths.DATASET, 'x.pt'), {})


class StopRuleTests(unittest.TestCase):
    def ev(self, bleus, losses=None):
        return [{'bleu4': b, 'loss': l} for b, l in zip(bleus, losses or [1.0] * len(bleus))]

    def test_rules(self):
        cfg = dict(DEFAULTS, patience=3, deteriorate_k=3, deteriorate_margin=8.0, overfit_k=3, overfit_loss_ratio=1.3)
        self.assertEqual(best_index(self.ev([10, 30, 30], [1, 0.9, 0.8])), 2)
        self.assertIsNone(stop_reason(self.ev([30, 31, 12]), cfg))                  # one noisy evaluation
        self.assertIn('patience', stop_reason(self.ev([30, 29, 29, 28]), cfg))
        self.assertIn('deteriorating', stop_reason(self.ev([40, 30, 29, 28], [1, 1, 1, 1]), dict(cfg, patience=50)))
        self.assertIn('overfitting', stop_reason(self.ev([40, 40, 40, 40], [0.6, 0.9, 0.95, 1.0]), dict(cfg, patience=50, deteriorate_k=50)))


class ConfigTests(unittest.TestCase):
    def test_numbers_written_without_a_decimal_point_are_numbers(self):
        from slt_vit.train import load_config
        with tempfile.TemporaryDirectory(dir=paths.PROJECT) as d:
            path = os.path.join(d, 'c.yaml')
            open(path, 'w').write('name: x\nlr_new: 1e-4\nepochs: "7"\n')
            cfg = load_config(path, ['lr_mbart=1e-12', 'amp=false', 'train_signers=[1, 2]'])
        self.assertEqual((cfg['lr_new'], cfg['lr_mbart'], cfg['epochs'], cfg['amp'], cfg['train_signers']), (1e-4, 1e-12, 7, False, [1, 2]))
        with self.assertRaises(SystemExit):
            with tempfile.TemporaryDirectory(dir=paths.PROJECT) as d:
                path = os.path.join(d, 'c.yaml')
                open(path, 'w').write('name: x\nlr_new: fast\n')
                load_config(path, [])


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from slt_vit.infer import Translator
        model, _ = tiny_model()
        cls.tmp = tempfile.mkdtemp(dir=paths.PROJECT)
        ckpt = os.path.join(cls.tmp, 'tiny.pt')
        checkpoints.save_verified(ckpt, model.checkpoint_state(tiny_tokenizer(), epoch=0, val={}))
        cls.video = os.path.join(cls.tmp, 'clip.mp4')
        w = cv2.VideoWriter(cls.video, cv2.VideoWriter_fourcc(*'mp4v'), 25, (320, 180))
        rng = np.random.default_rng(0)
        for _ in range(30):
            w.write(rng.integers(0, 255, (180, 320, 3), dtype=np.uint8))
        w.release()
        cls.tr = Translator(ckpt, device='cpu')
        cls.tr.size = 32                      # the tiny test tower takes 32px input

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_single_video_inference_terminates(self):
        r = self.tr.translate_video(self.video)
        self.assertEqual(r['video']['decoded_frames'], 30)
        self.assertEqual(len(r['sampled_indices']), 12)
        self.assertEqual(r['token_ids'][0], LANG)
        self.assertLessEqual(len(r['token_ids']), 6)

    def test_app_reports_bad_uploads(self):
        from slt_vit.app import run_translation
        data = open(self.video, 'rb').read()
        self.assertIsNotNone(run_translation(self.tr, 'clip.mp4', data)[0])
        self.assertIn('Could not translate', run_translation(self.tr, 'clip.mp4', b'garbage')[1])
        self.assertIn('unsupported file type', run_translation(self.tr, 'notes.txt', data)[1])

    def test_batch_evaluation(self):
        class Split(torch.utils.data.Dataset):
            entries = [{'id': f'c{i}', 'signer': 6, 'frame_dir': 'x', 'text': 'are you free today'} for i in range(3)]

            def __len__(self):
                return 3

            def __getitem__(self, i):
                return torch.randn(12, 768), [5, 6, 7, 8, 9, EOS], i
        loader = torch.utils.data.DataLoader(Split(), batch_size=2, collate_fn=collate)
        m, rows = evaluate(self.tr.model, loader, self.tr.tok, dict(self.tr.cfg, stage=1, amp=False))
        self.assertEqual((m['n'], len(rows)), (3, 3))
        self.assertTrue(np.isfinite(m['loss']))


if __name__ == '__main__':
    unittest.main()
