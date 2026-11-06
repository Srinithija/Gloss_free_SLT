"""Decoding on top of trained SignTranslatorViT checkpoints: candidate scoring and ensemble greedy decoding.

Both work on mBART encoder states, so several temporal views of a clip and several checkpoints (which share
one 199-token vocabulary, checked in reports/audit_facts.json) can be combined by averaging per-token
log-probabilities (a weighted product of experts over the same distribution).

Candidate scoring: score(c) = sum_t log p(c_t | c_<t, video) over the candidate's tokens *including* </s>,
divided by len(c) ** alpha (len counts </s>). alpha = 0 is the plain sequence log-probability, which favours
short sentences; alpha = 1 is the mean per-token log-probability.
"""
import torch
import torch.nn.functional as F

from slt_vit.text_model import EOS, LANG, PAD


def encode(model, feats):
    """Sampled per-frame inputs [B,T,D] -> (mBART encoder hidden states [B,L,d], attention mask [B,L]).
    Works for SignTranslatorViT (temporal encoder) and train_features.FusionTranslator (its own encoder_inputs)."""
    x = model.encoder_inputs(feats) if hasattr(model, 'encoder_inputs') else model.temporal(feats)
    mask = torch.ones(x.shape[:2], dtype=torch.long, device=x.device)
    h = model.mbart.model.encoder(inputs_embeds=x, attention_mask=mask, return_dict=True).last_hidden_state
    return h, mask


def _decoder_logprobs(model, h, mask, dec_in):
    # No decoder mask: the decoder is causal and padding only follows a candidate's last token, so it never
    # affects the positions that are scored (generate() passes no decoder mask either).
    out = model.mbart(encoder_outputs=(h,), attention_mask=mask, decoder_input_ids=dec_in, return_dict=True)
    return F.log_softmax(out.logits.float(), dim=-1)


def pad_candidates(cands, device):
    """Token-id lists (each ending in </s>) -> decoder inputs, labels [C,L] and lengths [C]."""
    length = max(len(c) for c in cands)
    labels = torch.full((len(cands), length), PAD, dtype=torch.long)
    dec_in = torch.full((len(cands), length), PAD, dtype=torch.long)
    for k, c in enumerate(cands):
        labels[k, :len(c)] = torch.tensor(c)
        dec_in[k, 0] = LANG
        dec_in[k, 1:len(c)] = torch.tensor(c[:-1])
    return dec_in.to(device), labels.to(device), torch.tensor([len(c) for c in cands], device=device)


@torch.no_grad()
def candidate_logprobs(model, h, mask, dec_in, labels):
    """Summed log-probability of every candidate for one encoded clip. h: [1,L,d] -> [C]."""
    c = dec_in.shape[0]
    lp = _decoder_logprobs(model, h.expand(c, -1, -1), mask.expand(c, -1), dec_in)
    tok = lp.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (tok * (labels != PAD)).sum(-1)


def normalize(summed, lengths, alpha):
    return summed / lengths.float() ** alpha


@torch.no_grad()
def ensemble_greedy(members, max_new_tokens=20):
    """Greedy decoding with log-probabilities averaged over members.

    members: list of (model, h [1,L,d], mask [1,L], weight). With one member this reproduces
    ``SignTranslatorViT.generate(num_beams=1)``: decoding starts at en_XX and the last allowed step is forced
    to </s> (transformers' forced_eos_token_id with max_length = 1 + max_new_tokens).
    """
    device = members[0][1].device
    ids = torch.tensor([[LANG]], device=device)
    total = sum(w for *_, w in members)
    for step in range(max_new_tokens):
        lp = sum(w * _decoder_logprobs(m, h, mask, ids)[:, -1] for m, h, mask, w in members) / total
        if step == max_new_tokens - 1:
            nxt = torch.tensor([[EOS]], device=device)
        else:
            nxt = lp.argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() == EOS:
            break
    return ids[0].tolist()
