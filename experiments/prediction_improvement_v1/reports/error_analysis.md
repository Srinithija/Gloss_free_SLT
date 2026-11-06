# Error analysis

Data is in `results/error_analysis.json` (`tools/error_analysis.py`). Per-clip predictions, with the reference's rank among
the 97 candidates, are in `results/<set>/<setting>/predictions.csv` and `.json`.

## How errors changed

| | Signer 6 | Signers 3–5 | Signer 7 |
|---|---|---|---|
| Baseline correct | 64/94 | 170/282 | 23/91 (initial) |
| D3 correct (fixed / broke vs baseline) | 73 (+11 / −2) | 190 (+37 / −17) | 43 (+22 / −2) |
| E3 correct (fixed / broke vs baseline) | 74 (+12 / −2) | 211 (+50 / −9) | 59 (+37 / −1) |
| Distinct predictions: baseline → E3 | 72 → 78 | 94 → 95 | 45 → 70 |
| Most repeated single output: baseline → E3 | 3 → 2 | 7 → 8 | 7 → 4 |
| Outputs that are not a corpus sentence: baseline → E3 | 1 → 0 | 0 → 0 | 1 → 0 |

E3 almost never breaks a clip the baseline got right. On signer 7 it broke one: "i got hurt" became "that is so kind of
you". It also removes the collapse onto a few frequent sentences that the baseline showed on signer 7: "how can i help
you" was output 7 times, and the broken output "how can i how can i help you" occurred once.

## Remaining errors of E3

Rank of the correct sentence among the 97 candidates, for the clips E3 gets wrong:

| Rank of the correct sentence | Signer 6 (20 errors) | Signers 3–5 (71) | Signer 7 (32) |
|---|---|---|---|
| 2nd | 13 | 23 | 16 |
| 3rd–5th | 4 | 19 | 5 |
| 6th or worse | 3 | 29 | 11 |

Half of the signer-7 errors are near misses: the correct sentence is the second choice.

## Confusion patterns

Signer 7, E3 (reference → prediction):

**Similar meaning or shared signs:**
- "do not worry" → "do not hurt me"
- "what happened" ↔ "what you want" (swapped both ways)
- "what are you doing" → "what do you do"
- "what do you think" → "what are you doing"
- "do not take it to the heart" → "do not be stubborn"
- "i enjoyed a lot" → "this place is beautiful", and the same swap occurs on signer 6

**Frequent "sink" sentences when the model is unsure:**
- "shall we go outside" ×3
- "serve the food" ×2
- "i am sitting in the class" ×2
- "that is so kind of you" ×2
- "bring water for me" ×2

**Far misses (rank 11–50):**
- "you are good", "you can do it", "you do anything, i do not care" and "i was stopped by some one" are ranked very
  low. Signer 7 probably signs these differently from all five training signers. More data would fix these, not
  decoding.

On signer 6, the remaining E3 errors are mostly sentences sharing a sign:
- "do not hurt me" ↔ "i got hurt"
- "i am tired" → "i am feeling bored"
- "how can i help you" → "can i help you"
- "can i help you" → "i can not help you there"

## Landmark quality and errors

The share of frames with a detected hand is the same for clips E3 gets right and wrong:
- signer 6: 0.71 vs 0.70;
- signer 7: 0.73 vs 0.77.

Hand-detection failures do not explain the remaining errors.

## Generalization limits

- **Closed set.** Every validation and test sentence also appears in training, so these numbers measure recognizing
  97 known sentences from new signers, not translating new sentences. Candidate restriction relies on this and would
  be wrong for open-vocabulary use.
- **Signer 6 is optimistic.** Accuracy there is 78.7%, against 74.8% on signers 3–5 and 64.8% on signer 7.
- **Small samples.** Signer 6 has only 94 clips, so differences of a few clips there (for example E3 vs D3) are not
  separable. The signers 3–5 check, with 282 clips, is what separates them.
- **Signer 7 tested more than once.** It has now been evaluated three times in this project: the initial test plus
  repeats R1 and R2. The repeats were fixed in advance and used no signer-7 information, but 64.8% is no longer an
  untouched test result.
