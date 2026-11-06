# 02 — ResNet baseline (existing results, re-scored)

> **Not signer-independent.** The split groups clips by source video, not by signer. The same few ISH News interpreters appear in train, dev and test.

## What the baseline is
The optimized GFSLT pipeline from 2026-10-23 (details in `00_inspection.md` §3–4):
- Frozen ImageNet ResNet-18 features from the 256 px LMDB frames.
- TemporalConv, then a 3+3-layer mBART initialised from `MBart_trimmed`.
- Trained on all 101,347 kept training clips. The selected checkpoint is epoch 7, decoded with beam 4 and no repeated 3-grams; both choices were made on dev.
- **It was not retrained here, and its code was not modified.**

## Re-scoring under the new protocol
- **Script:** `scripts/rescore_baseline.py`. **Log:** `logs/rescore_baseline.log`. **Numbers:** `reports/baseline_rescored.json`.
- **Inputs:** the saved predictions of the single test run (`final_eval/test_predictions.csv`) and the dev prediction log of the same model and decoding. **The model was not run again.**
- **References** are re-read from `splits/v1` and match the saved references exactly (0 mismatches).
- **Normalisation** (`slt/text.py:normalize`): lowercase, punctuation stripped. It is identical for every system.
- **sacreBLEU signature:** `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.3`. Inputs are already lowercased, so `case:mixed` has no effect.

| Set | n | BLEU-1 | BLEU-4 | chrF | ROUGE-L | Exact match | Distinct predictions | Most repeated output (count) |
|---|---|---|---|---|---|---|---|---|
| dev subset (all) | 1,984 | 8.33 | 0.59 | 16.91 | 7.44 | 0.10% | 1,013 | "lets try this" (70) |
| dev subset, without flagged | 1,956 | 8.36 | 0.60 | 16.93 | 7.48 | 0.10% | 1,005 | "lets try this" (70) |
| **test subset (all)** | 1,979 | **9.16** | **0.76** | 17.03 | 8.16 | 0.10% | 1,073 | "lets listen to the lesson" (93) |
| test subset, without flagged | 1,969 | 9.18 | 0.76 | 17.03 | 8.17 | 0.10% | 1,067 | "lets listen to the lesson" (93) |

**Compared with the original report** (case-sensitive, punctuation kept): test BLEU-1 12.64 → 9.16, and BLEU-4 0.74 → 0.76.
- The old BLEU-1 counted punctuation tokens such as "." as matches; normalisation removes them.
- ROUGE-L differs from the old 7.79 because the implementation changed: `rouge-score` F1 instead of the old β=1.2 LCS score.

## Earlier test decodes of this model (for the test ledger)
1. 2026-10-23: `final_eval` test run, BLEU-4 0.74 (case-sensitive). This is the official single run.
2. 2026-10-23: `dump_predictions.py` re-decode, BLEU-4 0.71. This was a **repeat evaluation**.
3. 2026-10-30: re-scoring of the saved predictions from run 1 (this report). **No new inference.**

## Still to do (step 9)
- Evaluate the baseline once on the **full** test split (12,627 kept clips), with its saved model and fixed decoding (beam 4, no repeated 3-grams).
- This needs ResNet-18 features for the full test split. They will be computed by a wrapper in this project that imports the old `isign_opt` code unchanged and writes only to `isign_siglip_slt/`.
- Estimated cost: about 11 min of feature caching (0.05 s per clip, measured earlier) plus about 13 min of beam decoding.
- The run will be logged in `TEST_LEDGER.csv`.
