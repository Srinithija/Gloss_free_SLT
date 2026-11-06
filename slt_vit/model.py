"""Integrated translator: ViT vision encoder -> temporal encoder -> compact pretrained mBART."""
import torch
import torch.nn as nn
from transformers import GenerationConfig, MBartForConditionalGeneration

from slt_vit import text_model, vision_encoder
from slt_vit.temporal_model import TemporalEncoder
from slt_vit.text_model import EOS, LANG, PAD


class SignTranslatorViT(nn.Module):
    def __init__(self, cfg, vision, mbart):
        super().__init__()
        self.cfg = cfg
        self.vision = vision
        self.temporal = TemporalEncoder(vision.spec.dim, cfg['temporal_hidden'], mbart.config.d_model,
                                        clip_mean=cfg['clip_mean'], dropout=cfg['dropout'])
        self.mbart = mbart
        # transformers fills generate() settings left at their default (e.g. num_beams=1) from the model's own
        # generation config, which cc25 sets to num_beams=5 / max_length=1024. Pin it so greedy means greedy.
        self.mbart.generation_config = GenerationConfig(decoder_start_token_id=LANG, eos_token_id=EOS, pad_token_id=PAD,
                                                        bos_token_id=0, forced_eos_token_id=EOS, num_beams=1)

    @classmethod
    def from_pretrained_parts(cls, cfg, tokenizer):
        """New model: verified pretrained vision encoder + verified cc25 transfer. Returns (model, provenance)."""
        vision, vision_prov = vision_encoder.load_pretrained(cfg['encoder'])
        mbart = text_model.build_pretrained_mbart(tokenizer.old_ids, cfg['encoder_layers'], cfg['decoder_layers'], cfg['dropout'])
        return cls(cfg, vision, mbart), {'vision': vision_prov, 'text': {'source': text_model.CC25,
                                                                         'weights_file': text_model.paths.cc25_file('model.safetensors'),
                                                                         'vocab_size': len(tokenizer)}}

    @classmethod
    def from_checkpoint_state(cls, state):
        """Architecture rebuilt from the configs stored in a checkpoint; all weights then loaded strictly from it."""
        cfg = state['cfg']
        vision = vision_encoder.build_from_config(cfg['encoder'], state['vision_config'])
        mbart = MBartForConditionalGeneration(text_model.mbart_config(cfg['vocab_size'], cfg['encoder_layers'],
                                                                      cfg['decoder_layers'], cfg['dropout']))
        model = cls(cfg, vision, mbart)
        model.load_state_dict(state['model'], strict=True)
        return model

    def frame_features(self, frames01):
        """frames [B,T,3,S,S] in [0,1] -> [B,T,D]."""
        b, t = frames01.shape[:2]
        return self.vision(frames01.flatten(0, 1)).view(b, t, -1)

    def forward(self, feats, decoder_input_ids, decoder_attention_mask):
        x = self.temporal(feats)
        mask = torch.ones(x.shape[:2], dtype=torch.long, device=x.device)
        return self.mbart(inputs_embeds=x, attention_mask=mask, decoder_input_ids=decoder_input_ids,
                          decoder_attention_mask=decoder_attention_mask, return_dict=True).logits

    @torch.no_grad()
    def generate(self, feats, num_beams=1, max_new_tokens=20):
        x = self.temporal(feats)
        mask = torch.ones(x.shape[:2], dtype=torch.long, device=x.device)
        gen = GenerationConfig(decoder_start_token_id=LANG, eos_token_id=EOS, pad_token_id=PAD, bos_token_id=0, forced_eos_token_id=EOS,
                               num_beams=num_beams, max_new_tokens=max_new_tokens, early_stopping=num_beams > 1)
        return self.mbart.generate(inputs_embeds=x, attention_mask=mask, generation_config=gen)

    def checkpoint_state(self, tokenizer, **extra):
        return {'model': self.state_dict(), 'cfg': self.cfg, 'tokenizer': tokenizer.to_dict(),
                'vision_config': self.vision.model.config.to_dict(), **extra}
