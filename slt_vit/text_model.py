"""mBART-cc25 language model with its vocabulary trimmed to the corpus, verified against the original weights.

The tokenizer is the original cc25 SentencePiece tokenizer. The trimmed model's embedding row ``i`` is the cc25
row ``old_ids[i]``; new ids 0-3 keep mBART's specials (<s>, <pad>, </s>, <unk>) and id 4 is ``en_XX``, the
decoder start token. Targets are ``tokens + [</s>]``. The first N encoder and decoder layers are copied from
cc25; every copied tensor is checked against the source file.
"""
import json

import torch
from transformers import MBartConfig, MBartForConditionalGeneration

from slt_vit import paths

CC25 = 'facebook/mbart-large-cc25'
BOS, PAD, EOS, UNK, LANG = 0, 1, 2, 3, 4
SPECIALS = ['<s>', '<pad>', '</s>', '<unk>', 'en_XX']
TIED = ('model.shared.weight', 'model.encoder.embed_tokens.weight', 'model.decoder.embed_tokens.weight', 'lm_head.weight')


def cc25_tokenizer():
    import os
    from transformers import MBartTokenizer
    snapshot = os.path.dirname(paths.cc25_file('sentencepiece.bpe.model'))
    return MBartTokenizer.from_pretrained(snapshot, src_lang='en_XX', tgt_lang='en_XX')


class TrimmedTokenizer:
    def __init__(self, old_ids, pieces):
        if pieces[:5] != SPECIALS or len(old_ids) != len(pieces):
            raise ValueError('vocabulary must start with <s> <pad> </s> <unk> en_XX')
        self.old_ids, self.pieces = list(old_ids), list(pieces)
        self.old_to_new = {o: n for n, o in enumerate(self.old_ids)}
        self._cc25 = None

    def __len__(self):
        return len(self.old_ids)

    @classmethod
    def build(cls, sentences, cc25=None):
        cc25 = cc25 or cc25_tokenizer()
        specials = [cc25.convert_tokens_to_ids(p) for p in SPECIALS]
        if specials != [0, 1, 2, 3, 250004]:
            raise ValueError(f'unexpected cc25 special ids {specials}')
        used = sorted({i for s in sentences for i in cc25(s, add_special_tokens=False)['input_ids']} - set(specials))
        if cc25.unk_token_id in used:
            raise ValueError('a sentence contains text the cc25 vocabulary cannot represent')
        tok = cls(specials + used, cc25.convert_ids_to_tokens(specials + used))
        tok._cc25 = cc25
        return tok

    def encode(self, text):
        if self._cc25 is None:
            self._cc25 = cc25_tokenizer()
        old = self._cc25(text, add_special_tokens=False)['input_ids']
        missing = [o for o in old if o not in self.old_to_new]
        if missing:
            raise KeyError(f'{text!r} needs tokens outside the trimmed vocabulary')
        return [self.old_to_new[o] for o in old] + [EOS]

    def decode(self, ids):
        out = []
        for i in ids:
            i = int(i)
            if i == EOS:
                break
            if i not in (BOS, PAD, LANG):
                out.append(self.pieces[i])
        return ''.join(out).replace('▁', ' ').strip()

    def to_dict(self):
        return {'base_tokenizer': CC25, 'old_ids': self.old_ids, 'pieces': self.pieces}

    @classmethod
    def from_dict(cls, d):
        return cls(d['old_ids'], d['pieces'])


def mbart_config(vocab_size, encoder_layers, decoder_layers, dropout):
    with open(paths.cc25_file('config.json'), encoding='utf-8') as f:
        cfg = MBartConfig.from_dict(json.load(f))
    cfg.vocab_size, cfg.encoder_layers, cfg.decoder_layers, cfg.num_hidden_layers = vocab_size, encoder_layers, decoder_layers, encoder_layers
    cfg.dropout = dropout
    cfg.pad_token_id, cfg.eos_token_id, cfg.bos_token_id, cfg.decoder_start_token_id, cfg.forced_eos_token_id = PAD, EOS, BOS, LANG, EOS
    return cfg


def build_pretrained_mbart(old_ids, encoder_layers, decoder_layers, dropout):
    """Compact mBART whose every tensor is copied from cc25 (first N layers, trimmed embedding rows)."""
    from safetensors.torch import load_file
    src = load_file(paths.cc25_file('model.safetensors'))
    model = MBartForConditionalGeneration(mbart_config(len(old_ids), encoder_layers, decoder_layers, dropout))
    idx = torch.tensor(old_ids)
    keep = {'encoder': encoder_layers, 'decoder': decoder_layers}
    state = {}
    for k, v in src.items():
        parts = k.split('.')
        if 'layers' in parts and int(parts[parts.index('layers') + 1]) >= keep[parts[parts.index('layers') - 1]]:
            continue
        if k in TIED:
            v = v[idx]
        elif k == 'final_logits_bias':
            v = v[:, idx]
        state[k] = v
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(missing) - set(TIED) or unexpected:
        raise RuntimeError(f'cc25 transfer incomplete: missing={missing} unexpected={unexpected}')
    verify_mbart_transfer(model, old_ids, src)
    return model


def verify_mbart_transfer(model, old_ids, src=None):
    """Every tensor of the compact model must equal its cc25 source; embeddings must be tied."""
    if src is None:
        from safetensors.torch import load_file
        src = load_file(paths.cc25_file('model.safetensors'))
    shared = model.model.shared.weight
    for m in (model.model.encoder.embed_tokens, model.model.decoder.embed_tokens, model.lm_head):
        if m.weight.data_ptr() != shared.data_ptr():
            raise RuntimeError('mBART embeddings are not tied')
    if not torch.equal(shared.detach().cpu(), src['model.shared.weight'][old_ids]):
        raise RuntimeError('trimmed embedding rows differ from cc25')
    checked = 1
    for k, v in model.state_dict().items():
        if k in TIED or k == 'final_logits_bias':
            continue
        if not torch.equal(v.detach().cpu(), src[k]):
            raise RuntimeError(f'{k} differs from cc25')
        checked += 1
    return checked
