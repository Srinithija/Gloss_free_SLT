"""Translation metrics computed from actual predictions and references."""
from __future__ import annotations

from collections import Counter

from sacrebleu.metrics import BLEU


def _lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(pred: str, ref: str, beta: float = 1.2) -> float:
    """Sentence ROUGE-L F-score (coco-caption convention, beta=1.2) on lowercase whitespace tokens."""
    p, r = pred.lower().split(), ref.lower().split()
    if not p or not r:
        return 0.0
    l = _lcs(p, r)
    if l == 0:
        return 0.0
    prec, rec = l / len(p), l / len(r)
    return (1 + beta ** 2) * prec * rec / (rec + beta ** 2 * prec)


def has_repeated_ngram(text: str, n: int = 3) -> bool:
    w = text.split()
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return len(grams) != len(set(grams))


def translation_metrics(preds, refs) -> dict:
    assert len(preds) == len(refs) and preds
    m = {f"bleu{n}": BLEU(max_ngram_order=n).corpus_score(preds, [refs]).score for n in range(1, 5)}
    m["bleu4_lowercase"] = BLEU(lowercase=True).corpus_score(preds, [refs]).score
    m["rougeL"] = 100 * sum(rouge_l(p, r) for p, r in zip(preds, refs)) / len(preds)
    norm = lambda s: " ".join(s.lower().strip(" .!?").split())
    m["exact_match"] = 100 * sum(norm(p) == norm(r) for p, r in zip(preds, refs)) / len(preds)
    c = Counter(preds)
    m["n"] = len(preds)
    m["unique_predictions"] = len(c)
    m["unique_ratio"] = len(c) / len(preds)
    m["most_common_prediction_share"] = c.most_common(1)[0][1] / len(preds)
    m["repeated_trigram_rate"] = 100 * sum(has_repeated_ngram(p) for p in preds) / len(preds)
    m["avg_pred_words"] = sum(len(p.split()) for p in preds) / len(preds)
    m["avg_ref_words"] = sum(len(r.split()) for r in refs) / len(refs)
    return m
