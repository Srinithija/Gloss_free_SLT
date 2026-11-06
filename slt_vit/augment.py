"""Clip-level augmentation: one random transform applied to every frame of a clip.

'base'   crop 75-100 %, rotation +-8 deg, brightness/contrast/saturation +-25 %
'strong' the same with +-40 % colour, hue shifts and 25 % greyscale, to weaken clothing/skin-colour cues
No horizontal flip: mirroring swaps the signer's dominant hand.
"""
import math

import torchvision.transforms.functional as TF

STRENGTH = {'base': dict(jitter=0.25, hue=0.0, grey=0.0), 'strong': dict(jitter=0.4, hue=0.1, grey=0.25)}


def augment_clip(x, rng, strength='base'):
    """x: float [T,3,H,W] in [0,1]; rng: numpy Generator."""
    p = STRENGTH[strength]
    _, _, h, w = x.shape
    scale, ratio = rng.uniform(0.75, 1.0), math.exp(rng.uniform(math.log(0.85), math.log(1.15)))
    ch, cw = min(h, int(round(h * math.sqrt(scale / ratio)))), min(w, int(round(w * math.sqrt(scale * ratio))))
    top, left = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
    x = TF.resized_crop(x, top, left, ch, cw, [h, w], antialias=True)
    x = TF.rotate(x, float(rng.uniform(-8, 8)))
    j = p['jitter']
    x = TF.adjust_brightness(x, float(rng.uniform(1 - j, 1 + j)))
    x = TF.adjust_contrast(x, float(rng.uniform(1 - j, 1 + j)))
    x = TF.adjust_saturation(x, float(rng.uniform(1 - j, 1 + j)))
    if p['hue']:
        x = TF.adjust_hue(x, float(rng.uniform(-p['hue'], p['hue'])))
    if p['grey'] and rng.uniform() < p['grey']:
        x = TF.rgb_to_grayscale(x, num_output_channels=3)
    return x.clamp(0, 1)
