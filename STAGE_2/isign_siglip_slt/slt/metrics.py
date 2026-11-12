"""Evaluation metrics on generated text. Identical code path for the ResNet baseline and every new model.

Both hypotheses and references go through slt.text.normalize (lowercase, punctuation stripped) first.
BLEU-1/BLEU-4/chrF: sacreBLEU corpus-level (signatures returned). ROUGE-L: rouge-score F-measure,
sentence-level, averaged x100. Exact match is for reference only.
"""
from collections import Counter

import numpy as np
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU, CHRF

from slt.text import normalize

_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def corpus_metrics(hyps, refs, with_signature=False):
    assert len(hyps) == len(refs) and len(hyps) > 0
    H = [normalize(h) for h in hyps]
    R = [normalize(r) for r in refs]
    b1, b4, ch = BLEU(max_ngram_order=1), BLEU(), CHRF()
    s1, s4, sc = b1.corpus_score(H, [R]), b4.corpus_score(H, [R]), ch.corpus_score(H, [R])
    rl = [_ROUGE.score(r, h)["rougeL"].fmeasure if r and h else float(r == h) for h, r in zip(H, R)]
    c = Counter(H)
    top, top_n = c.most_common(1)[0]
    m = {"n": len(H), "bleu1": s1.score, "bleu4": s4.score, "chrf": sc.score, "rougeL": 100 * float(np.mean(rl)),
         "exact_match": 100 * float(np.mean([h == r for h, r in zip(H, R)])),
         "distinct_predictions": len(c), "most_repeated_output": top, "most_repeated_count": top_n,
         "most_repeated_share": top_n / len(H), "avg_hyp_words": float(np.mean([len(h.split()) for h in H])),
         "avg_ref_words": float(np.mean([len(r.split()) for r in R]))}
    if with_signature:
        m["signature_bleu4"] = str(b4.get_signature())
        m["signature_bleu1"] = str(b1.get_signature())
        m["signature_chrf"] = str(ch.get_signature())
    return m


def sentence_bleu4(hyps, refs):
    """Per-sentence normalised BLEU-4 contribution stats for paired bootstrap (n-gram counts)."""
    b = BLEU()
    return [b._extract_corpus_statistics([normalize(h)], [[normalize(r)]])[0] for h, r in zip(hyps, refs)]


def paired_bootstrap(hyps_a, hyps_b, refs, n_resamples=1000, seed=0):
    """95% CI of BLEU-4 (normalised) for A, B and A-B over resampled sentence sets."""
    b = BLEU()
    sa, sb = np.array(sentence_bleu4(hyps_a, refs)), np.array(sentence_bleu4(hyps_b, refs))
    rng = np.random.default_rng(seed)
    n = len(refs)
    da, db = [], []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        da.append(b._compute_score_from_stats(sa[idx].sum(0).tolist()).score)
        db.append(b._compute_score_from_stats(sb[idx].sum(0).tolist()).score)
    da, db = np.array(da), np.array(db)
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    return {"A_ci": ci(da), "B_ci": ci(db), "diff_mean": float((da - db).mean()), "diff_ci": ci(da - db),
            "p_A_not_better": float(np.mean(da - db <= 0)), "n_resamples": n_resamples}
