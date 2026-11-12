"""mBART-cc25 vocabulary trimmed to the tokens of the training-split captions.

Target format used everywhere (explicit, no reliance on tokenizer post-processing defaults):
  labels            = sentencepiece ids of the caption + [</s>]
  decoder_input_ids = [en_XX] + labels[:-1]
  generation        : decoder_start_token_id = en_XX, eos = </s>
All ids below are in the trimmed space unless named *_orig.
"""
import json
from pathlib import Path

import numpy as np

SPECIAL = ["<s>", "<pad>", "</s>", "<unk>"]


class TrimmedVocab:
    def __init__(self, tokenizer, keep_orig_ids):
        self.tok = tokenizer
        self.orig = np.asarray(keep_orig_ids, dtype=np.int64)  # new id -> original id
        self.to_new = {int(o): n for n, o in enumerate(self.orig)}
        self.pad = self.to_new[tokenizer.pad_token_id]
        self.eos = self.to_new[tokenizer.eos_token_id]
        self.bos = self.to_new[tokenizer.bos_token_id]
        self.unk = self.to_new[tokenizer.unk_token_id]
        self.lang = self.to_new[tokenizer.convert_tokens_to_ids("en_XX")]

    def __len__(self):
        return len(self.orig)

    @classmethod
    def build(cls, tokenizer, train_texts):
        ids = set()
        for t in train_texts:
            ids.update(tokenizer(t, add_special_tokens=False)["input_ids"])
        special = [tokenizer.convert_tokens_to_ids(s) for s in SPECIAL] + [tokenizer.convert_tokens_to_ids("en_XX")]
        keep = special + sorted(ids - set(special))
        return cls(tokenizer, keep)

    def save(self, path):
        Path(path).write_text(json.dumps({"orig_ids": self.orig.tolist()}))

    @classmethod
    def load(cls, tokenizer, path):
        return cls(tokenizer, json.loads(Path(path).read_text())["orig_ids"])

    def encode(self, text, max_len=None):
        orig = self.tok(text, add_special_tokens=False)["input_ids"]
        ids = [self.to_new.get(o, self.unk) for o in orig]
        if max_len:
            ids = ids[: max_len - 1]
        return ids + [self.eos]

    def oov_rate(self, texts):
        orig = [o for t in texts for o in self.tok(t, add_special_tokens=False)["input_ids"]]
        return sum(o not in self.to_new for o in orig) / max(1, len(orig)), len(orig)

    def decode(self, ids):
        out = []
        for i in ids:
            i = int(i)
            if i == self.eos:
                break
            if i in (self.pad, self.bos, self.lang):
                continue
            out.append(int(self.orig[i]))
        return self.tok.decode(out, skip_special_tokens=True).strip()
