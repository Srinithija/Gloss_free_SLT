"""Hybrid gloss-free SLT model ("Model 3"):

  visual [B,T,768] -> subtract mean over the clip's valid frames
  landmarks [B,T,204] -> Linear(204->256) + LayerNorm + ReLU
  concat -> [B,T,1024] (768 / 256 for the single-stream ablations)
  -> Conv1d(k5, ->1024) + masked BatchNorm + ReLU + MaxPool1d(2)  (pads zeroed; pooled mask = first of pair)
  -> Linear(1024->1024) + LayerNorm + ReLU
  -> mBART encoder (inputs_embeds, attention mask) -> mBART decoder -> trimmed vocabulary
mBART: mbart-large-cc25 config with N encoder + N decoder layers copied from evenly spaced pretrained layers;
embeddings, positions, layer norms and LM head copied; vocabulary rows restricted to the trimmed vocab.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationConfig, MBartConfig, MBartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


def pretrained_layer_ids(n, total=12):
    return [int(x) for x in np.round(np.linspace(0, total - 1, n))]


def build_mbart(src, vocab, n_layers, dropout=0.1):
    """src: a loaded full MBartForConditionalGeneration (cc25). Returns a new smaller model."""
    cfg = MBartConfig.from_dict(src.config.to_dict())
    cfg.encoder_layers = cfg.decoder_layers = cfg.num_hidden_layers = n_layers
    cfg.vocab_size = len(vocab)
    cfg.dropout = dropout
    cfg.pad_token_id, cfg.eos_token_id, cfg.bos_token_id = vocab.pad, vocab.eos, vocab.bos
    cfg.decoder_start_token_id, cfg.forced_eos_token_id = vocab.lang, vocab.eos
    cfg.tie_word_embeddings = True
    model = MBartForConditionalGeneration(cfg)
    ids = pretrained_layer_ids(n_layers)
    sd = src.state_dict()
    keep = torch.as_tensor(vocab.orig)
    new = {}
    for k, v in sd.items():
        p = k.split(".")
        if len(p) > 3 and p[2] == "layers":
            if int(p[3]) not in ids:
                continue
            p[3] = str(ids.index(int(p[3])))
        elif k in ("model.shared.weight", "model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight",
                   "lm_head.weight"):
            v = v[keep]
        elif k == "final_logits_bias":
            v = v[:, keep]
        new[".".join(p)] = v
    missing, unexpected = model.load_state_dict(new, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"mBART init mismatch: missing={missing} unexpected={unexpected}")
    model.tie_weights()
    model.generation_config = GenerationConfig(
        decoder_start_token_id=vocab.lang, eos_token_id=vocab.eos, pad_token_id=vocab.pad, bos_token_id=vocab.bos,
        forced_eos_token_id=vocab.eos, num_beams=1, do_sample=False, max_new_tokens=64)
    return model, ids


class MaskedBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm over valid time steps only (x [B,C,T], mask [B,T])."""

    def forward(self, x, mask):
        if not self.training:
            return super().forward(x)
        m = mask[:, None, :].to(x.dtype)
        n = m.sum().clamp_min(1)
        mean = (x * m).sum((0, 2)) / n
        var = (((x - mean[None, :, None]) ** 2) * m).sum((0, 2)) / n
        with torch.no_grad():
            self.running_mean.lerp_(mean.float(), self.momentum)
            self.running_var.lerp_((var * n / (n - 1).clamp_min(1)).float(), self.momentum)
            self.num_batches_tracked += 1
        y = (x - mean[None, :, None]) / torch.sqrt(var[None, :, None] + self.eps)
        return y * self.weight[None, :, None] + self.bias[None, :, None]


class SLTModel(nn.Module):
    def __init__(self, mbart, use_visual=True, use_landmarks=True, vis_dim=768, lm_dim=204, lm_hidden=256,
                 d_model=1024, feat_dropout=0.1):
        super().__init__()
        assert use_visual or use_landmarks
        self.use_visual, self.use_landmarks = use_visual, use_landmarks
        in_dim = (vis_dim if use_visual else 0) + (lm_hidden if use_landmarks else 0)
        if use_landmarks:
            self.lm_proj = nn.Sequential(nn.Linear(lm_dim, lm_hidden), nn.LayerNorm(lm_hidden), nn.ReLU())
        self.feat_drop = nn.Dropout(feat_dropout)
        self.conv = nn.Conv1d(in_dim, d_model, kernel_size=5, padding=2)
        self.bn = MaskedBatchNorm1d(d_model)
        self.pool = nn.MaxPool1d(2, ceil_mode=True)
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.mbart = mbart

    def encode_inputs(self, vis, lm, mask):
        """vis [B,T,768] | None, lm [B,T,204] | None, mask [B,T] bool -> (embeds [B,T',1024], mask [B,T'])."""
        fm = mask[..., None].float()
        parts = []
        if self.use_visual:
            v = vis.float()
            mean = (v * fm).sum(1, keepdim=True) / fm.sum(1, keepdim=True).clamp_min(1)
            parts.append(v - mean)
        if self.use_landmarks:
            parts.append(self.lm_proj(lm.float()))
        x = self.feat_drop(torch.cat(parts, -1)) * fm
        x = self.conv(x.transpose(1, 2))
        x = F.relu(self.bn(x, mask)) * mask[:, None, :].float()
        x = self.pool(x).transpose(1, 2)
        pmask = mask[:, ::2]
        x = self.proj(x) * pmask[..., None].float()
        return x, pmask

    def forward(self, vis, lm, mask, labels, decoder_input_ids, label_smoothing=0.1):
        emb, m = self.encode_inputs(vis, lm, mask)
        out = self.mbart(inputs_embeds=emb, attention_mask=m.long(), decoder_input_ids=decoder_input_ids,
                         decoder_attention_mask=(decoder_input_ids != self.mbart.config.pad_token_id).long(),
                         return_dict=True)
        loss = F.cross_entropy(out.logits.float().transpose(1, 2), labels, ignore_index=-100,
                               label_smoothing=label_smoothing)
        return loss, out.logits

    def encoder_outputs(self, vis, lm, mask):
        emb, m = self.encode_inputs(vis, lm, mask)
        enc = self.mbart.get_encoder()(inputs_embeds=emb, attention_mask=m.long(), return_dict=True)
        return BaseModelOutput(last_hidden_state=enc.last_hidden_state), m

    @torch.no_grad()
    def generate(self, vis, lm, mask, num_beams=1, length_penalty=1.0, no_repeat_ngram_size=0, max_new_tokens=64):
        enc, m = self.encoder_outputs(vis, lm, mask)
        gc = GenerationConfig(**self.mbart.generation_config.to_dict())
        gc.update(num_beams=num_beams, length_penalty=length_penalty, no_repeat_ngram_size=no_repeat_ngram_size,
                  max_new_tokens=max_new_tokens, do_sample=False, early_stopping=True if num_beams > 1 else False)
        assert gc.num_beams == num_beams and not gc.do_sample  # greedy really is beam 1
        return self.mbart.generate(encoder_outputs=enc, attention_mask=m.long(), generation_config=gc)
