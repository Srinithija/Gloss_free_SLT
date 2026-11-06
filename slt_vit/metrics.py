"""Translation metrics, identical in definition to the earlier ResNet18 project so numbers are comparable.

References and predictions are lower-cased with single spaces (``dataset.normalize_text``). Exact match compares
whole strings. BLEU is sacreBLEU with its default 13a tokenization (corpus BLEU-4 and BLEU-1; sentence BLEU
with sacreBLEU's default smoothing). ROUGE-L is the word-LCS F-score with beta = 1.2.
"""
import collections

import numpy as np
import sacrebleu


def _lcs(a, b):
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def rouge_l(hyp, ref, beta=1.2):
    h, r = hyp.split(), ref.split()
    lcs = _lcs(h, r) if h and r else 0
    if not lcs:
        return 0.0
    p, rec = lcs / len(h), lcs / len(r)
    return (1 + beta ** 2) * p * rec / (rec + beta ** 2 * p)


def has_repetition(text):
    t = text.split()
    bigrams = list(zip(t, t[1:]))
    return any(a == b for a, b in zip(t, t[1:])) or len(bigrams) != len(set(bigrams))


def per_sample(hyps, refs):
    return [{'exact': h == r, 'sentence_bleu': sacrebleu.sentence_bleu(h, [r]).score, 'rouge_l': 100 * rouge_l(h, r)}
            for h, r in zip(hyps, refs)]


def translation_metrics(hyps, refs):
    counts = collections.Counter(hyps)
    return {'n': len(hyps), 'correct': int(sum(h == r for h, r in zip(hyps, refs))),
            'exact_match': float(np.mean([h == r for h, r in zip(hyps, refs)])),
            'bleu4': sacrebleu.corpus_bleu(hyps, [refs]).score,
            'bleu1': sacrebleu.BLEU(max_ngram_order=1).corpus_score(hyps, [refs]).score,
            'rouge_l': 100 * float(np.mean([rouge_l(h, r) for h, r in zip(hyps, refs)])),
            'sentence_bleu_mean': float(np.mean([sacrebleu.sentence_bleu(h, [r]).score for h, r in zip(hyps, refs)])),
            'unique_predictions': len(counts), 'most_common_prediction': counts.most_common(1)[0][0],
            'most_common_count': counts.most_common(1)[0][1],
            'repetitive_outputs': sum(has_repetition(h) for h in hyps), 'empty_outputs': sum(not h.strip() for h in hyps)}


def error_summary(rows, top=5):
    wrong = [r for r in rows if not r['exact']]
    return {'wrong': len(wrong),
            'most_repeated_predictions': collections.Counter(r['prediction'] for r in rows).most_common(top),
            'most_common_substitutions': collections.Counter((r['reference'], r['prediction']) for r in wrong).most_common(top),
            'incomplete_prefix_outputs': sum(1 for r in wrong if r['prediction'] and r['reference'].startswith(r['prediction'])),
            'empty_outputs': sum(1 for r in rows if not r['prediction'].strip())}
