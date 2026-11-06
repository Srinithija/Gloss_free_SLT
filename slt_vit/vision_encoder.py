"""Pretrained ViT vision encoders (SigLIP, CLIP) behind one interface, with verified weight loading.

``load_pretrained(name)`` downloads a pinned revision into the project cache, loads only the vision tower,
then re-reads the raw weight file and checks that every tensor of the loaded tower equals the file. Missing,
mismatched or freshly initialized weights raise an error; there is no fallback to random weights.
"""
import hashlib
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from slt_vit import paths


@dataclass(frozen=True)
class EncoderSpec:
    hf_id: str
    revision: str
    family: str          # 'siglip' or 'clip'
    weights_file: str
    image_size: int
    dim: int
    mean: tuple          # from the checkpoint's preprocessor_config.json
    std: tuple


ENCODERS = {
    'siglip_b16_224': EncoderSpec('google/siglip-base-patch16-224', '7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed', 'siglip',
                                  'model.safetensors', 224, 768, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    'clip_b16_224': EncoderSpec('openai/clip-vit-base-patch16', '57c216476eefef5ab752ec549e440a49ae4ae5f3', 'clip',
                                'pytorch_model.bin', 224, 768, (0.48145466, 0.4578275, 0.40821073),
                                (0.26862954, 0.26130258, 0.27577711)),
}


def _model_class(family):
    from transformers import CLIPVisionModel, SiglipVisionModel
    return {'siglip': SiglipVisionModel, 'clip': CLIPVisionModel}[family]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def _raw_state(path):
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location='cpu', weights_only=True)


class VisionEncoder(nn.Module):
    """frames [N,3,S,S] in [0,1] -> pooled embeddings [N,dim]. Normalization uses the checkpoint's own statistics."""

    def __init__(self, name, model):
        super().__init__()
        self.name, self.spec = name, ENCODERS[name]
        self.model = model
        self.register_buffer('mean', torch.tensor(self.spec.mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor(self.spec.std).view(1, 3, 1, 1), persistent=False)

    def forward(self, x01):
        x = (x01 - self.mean) / self.std
        return self.model(pixel_values=x).pooler_output

    def blocks(self):
        return self.model.vision_model.encoder.layers

    def set_trainable(self, last_blocks):
        """Freeze everything, then unfreeze the last ``last_blocks`` transformer blocks and the output norm/head."""
        for p in self.model.parameters():
            p.requires_grad_(False)
        if last_blocks <= 0:
            return
        for block in list(self.blocks())[-last_blocks:]:
            for p in block.parameters():
                p.requires_grad_(True)
        vm = self.model.vision_model
        for attr in ('post_layernorm', 'head'):
            if getattr(vm, attr, None) is not None:
                for p in getattr(vm, attr).parameters():
                    p.requires_grad_(True)


def load_pretrained(name, cache_dir=paths.HF_CACHE):
    """Pretrained vision tower + a provenance record. Raises if anything was not loaded from the checkpoint."""
    from huggingface_hub import hf_hub_download
    spec = ENCODERS[name]
    paths.writable(cache_dir)
    model, info = _model_class(spec.family).from_pretrained(spec.hf_id, revision=spec.revision, cache_dir=cache_dir,
                                                             output_loading_info=True)
    if info['missing_keys'] or info['mismatched_keys'] or info.get('error_msgs'):
        raise RuntimeError(f'{spec.hf_id}: weights not fully loaded: missing={info["missing_keys"]} '
                           f'mismatched={info["mismatched_keys"]} errors={info.get("error_msgs")}')
    weights = hf_hub_download(spec.hf_id, spec.weights_file, revision=spec.revision, cache_dir=cache_dir)
    raw = _raw_state(weights)
    loaded = model.state_dict()
    absent = [k for k in loaded if k not in raw]
    differ = [k for k in loaded if k in raw and not torch.equal(loaded[k].cpu(), raw[k].to(loaded[k].dtype))]
    if absent or differ:
        raise RuntimeError(f'{spec.hf_id}: {len(absent)} tensors absent from the file, {len(differ)} differ from it '
                           f'(e.g. {(absent + differ)[:3]})')
    cfg = model.config
    if cfg.image_size != spec.image_size or cfg.hidden_size != spec.dim:
        raise RuntimeError(f'{spec.hf_id}: config {cfg.image_size}px/{cfg.hidden_size}d does not match the registry')
    provenance = {**asdict(spec), 'weights_sha256': _sha256(weights), 'tensors_verified': len(loaded),
                  'parameters': sum(p.numel() for p in model.parameters()), 'layers': cfg.num_hidden_layers,
                  'patch_size': cfg.patch_size, 'unexpected_keys_ignored': len(info['unexpected_keys'])}
    return VisionEncoder(name, model), provenance


def build_from_config(name, config_dict):
    """Rebuild the architecture for loading a trained checkpoint (weights come from that checkpoint, strictly)."""
    spec = ENCODERS[name]
    cls = _model_class(spec.family)
    return VisionEncoder(name, cls(cls.config_class.from_dict(config_dict)))
