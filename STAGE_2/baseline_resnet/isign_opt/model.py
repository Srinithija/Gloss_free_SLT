"""Feature-level gloss-free SLT model: TemporalConv -> projection -> 3+3-layer mBART.

Repair over the earlier setup: the earlier `mytran` checkpoint gave the decoder random token
embeddings, a random LM head and random layers. Here every mBART tensor comes from the
pretrained `MBart_trimmed` (mBART-cc25 trimmed to the iSign vocabulary, verified row-consistent
with the official model): embeddings/LM head (tied), positional embeddings, layer norms, and
encoder/decoder layers 0, 5 and 11 (evenly spaced, as in shrink-and-fine-tune distillation).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import MBartConfig, MBartForConditionalGeneration

from isign_opt.common import FEATURE_DIM, TOKENIZER_DIR
from models.models import TemporalConv, V_encoder

PRETRAINED_LAYERS = (0, 5, 11)
LEFT_PAD = 8


def build_mbart(dropout: float = 0.1, pretrained: bool = True) -> MBartForConditionalGeneration:
    cfg = MBartConfig.from_pretrained(TOKENIZER_DIR)
    cfg.encoder_layers = cfg.decoder_layers = cfg.num_hidden_layers = len(PRETRAINED_LAYERS)
    cfg.dropout = dropout
    cfg.tie_word_embeddings = True
    model = MBartForConditionalGeneration(cfg)
    if pretrained:
        src = torch.load(TOKENIZER_DIR / "pytorch_model.bin", map_location="cpu")
        new = {}
        for k, v in src.items():
            parts = k.split(".")
            if len(parts) > 3 and parts[2] == "layers":  # model.{encoder,decoder}.layers.N.*
                n = int(parts[3])
                if n not in PRETRAINED_LAYERS:
                    continue
                parts[3] = str(PRETRAINED_LAYERS.index(n))
            new[".".join(parts)] = v
        missing, unexpected = model.load_state_dict(new, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"mBART init mismatch: missing={missing}, unexpected={unexpected}")
        model.tie_weights()
    return model


def conv_output_lengths(frame_lengths: torch.Tensor) -> torch.Tensor:
    padded = torch.ceil(frame_lengths / 4.0) * 4 + 2 * LEFT_PAD
    return (((padded - 4) // 2) - 4) // 2


class FeatureSLT(nn.Module):
    def __init__(self, dropout: float = 0.1, feat_dropout: float = 0.1, pretrained: bool = True):
        super().__init__()
        self.temporal = TemporalConv(input_size=FEATURE_DIM, hidden_size=1024, conv_type=2)
        self.proj = V_encoder(emb_size=1024, feature_size=1024, config={"model": {"sign_proj_normalization": "LN"}})
        self.feat_dropout = nn.Dropout(feat_dropout)
        self.mbart = build_mbart(dropout, pretrained)

    def encode_inputs(self, feats: torch.Tensor, frame_lengths: torch.Tensor):
        """feats: [B, T, 512] (already edge-padded by `pad_batch`); returns inputs_embeds, mask."""
        x = self.temporal(self.feat_dropout(feats.float()))
        x = self.proj(x)
        lens = conv_output_lengths(frame_lengths.float()).long().to(x.device)
        mask = (torch.arange(x.shape[1], device=x.device)[None] < lens[:, None]).long()
        return x, mask

    def forward(self, feats, frame_lengths, labels, label_mask):
        emb, mask = self.encode_inputs(feats, frame_lengths)
        out = self.mbart(inputs_embeds=emb, attention_mask=mask, labels=labels.masked_fill(label_mask == 0, -100),
                         decoder_attention_mask=label_mask, return_dict=True)
        return out.logits

    @torch.no_grad()
    def generate(self, feats, frame_lengths, decoder_start_token_id, num_beams=1, max_new_tokens=60, **kw):
        emb, mask = self.encode_inputs(feats, frame_lengths)
        return self.mbart.generate(inputs_embeds=emb, attention_mask=mask, decoder_start_token_id=decoder_start_token_id,
                                   num_beams=num_beams, max_new_tokens=max_new_tokens, **kw)


def pad_batch(feature_list):
    """Edge-pad each [L, 512] clip like the original collate (8 copies of the first frame on the left,
    last-frame copies up to ceil(L/4)*4+16), then zero-pad to the batch maximum."""
    lens = torch.tensor([len(f) for f in feature_list])
    targets = (torch.ceil(lens / 4.0) * 4 + 2 * LEFT_PAD).long()
    out = torch.zeros(len(feature_list), int(targets.max()), feature_list[0].shape[1], dtype=feature_list[0].dtype)
    for i, f in enumerate(feature_list):
        L, T = len(f), int(targets[i])
        out[i, :LEFT_PAD] = f[0]
        out[i, LEFT_PAD:LEFT_PAD + L] = f
        out[i, LEFT_PAD + L:T] = f[-1]
    return out, lens
