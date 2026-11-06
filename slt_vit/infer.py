"""Translate ISL videos or frame folders with a trained checkpoint of this project.

    python -m slt_vit.infer --ckpt outputs/<experiment>/best.pt video.mp4 [more videos or frame folders]

Preprocessing is exactly the training/evaluation path: canonical 224x224 frames (crop to <=4:3, pad, resize),
32 frames sampled uniformly over the clip, the encoder's own normalization, CUDA autocast as when caching.
"""
import argparse
import os
import time

import torch

import slt_vit  # noqa: F401
from slt_vit import checkpoints
from slt_vit.dataset import IMG_EXTENSIONS, natural_key
from slt_vit.frames import load_frame_folder, read_video, sample_indices, to_unit_float
from slt_vit.model import SignTranslatorViT
from slt_vit.text_model import EOS, TrimmedTokenizer


class Translator:
    def __init__(self, ckpt_path, device=None):
        state = checkpoints.load(ckpt_path)
        self.cfg = state['cfg']
        self.tok = TrimmedTokenizer.from_dict(state['tokenizer'])
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = SignTranslatorViT.from_checkpoint_state(state).to(self.device).eval()
        self.size = self.model.vision.spec.image_size
        self.info = {'checkpoint': os.path.abspath(ckpt_path), 'sha256': checkpoints.sha256(ckpt_path),
                     'epoch_number': state['epoch'] + 1, 'val': state.get('val'), 'encoder': self.cfg['encoder'],
                     'hf_id': self.model.vision.spec.hf_id}

    @torch.no_grad()
    def translate_frames(self, frames, num_beams=1):
        """frames: canonical uint8 [N,S,S,3] in temporal order."""
        t0 = time.perf_counter()
        idx = sample_indices(len(frames), self.cfg['num_frames'])
        x = to_unit_float(frames[idx], self.device).unsqueeze(0)
        cuda = self.device.type == 'cuda'
        with torch.autocast('cuda', enabled=cuda, dtype=torch.float16):
            feats = self.model.frame_features(x).float()
        if not torch.isfinite(feats).all():
            raise FloatingPointError('non-finite visual features')
        with torch.autocast('cuda', enabled=cuda and self.cfg['amp']):
            ids = self.model.generate(feats, num_beams=num_beams, max_new_tokens=self.cfg['max_new_tokens'])[0].tolist()
        if cuda:
            torch.cuda.synchronize()
        return {'text': self.tok.decode(ids), 'token_ids': ids, 'sampled_indices': idx.tolist(),
                'model_seconds': time.perf_counter() - t0, 'stopped_at_eos': EOS in ids[1:]}

    def translate_video(self, path, num_beams=1):
        t0 = time.perf_counter()
        frames, info = read_video(path, self.size)
        r = self.translate_frames(frames, num_beams)
        r.update(video=info, total_seconds=time.perf_counter() - t0, frames=frames)
        return r

    def translate_folder(self, folder, num_beams=1):
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTENSIONS)], key=natural_key)
        if len(files) < 2:
            raise ValueError(f'{folder} has {len(files)} image(s); at least 2 are needed')
        return self.translate_frames(load_frame_folder(folder, files, self.size), num_beams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--beams', type=int, default=1)
    ap.add_argument('inputs', nargs='+')
    args = ap.parse_args()
    tr = Translator(args.ckpt)
    print(f"loaded {tr.info['checkpoint']} ({tr.info['encoder']}, epoch {tr.info['epoch_number']}, sha256 {tr.info['sha256'][:12]})", flush=True)
    for p in args.inputs:
        try:
            r = tr.translate_folder(p, args.beams) if os.path.isdir(p) else tr.translate_video(p, args.beams)
            print(f"{p}\t{r['text']}\t({r['model_seconds']:.2f}s model)", flush=True)
        except (ValueError, FloatingPointError) as e:
            print(f'{p}\tERROR: {e}', flush=True)


if __name__ == '__main__':
    main()
