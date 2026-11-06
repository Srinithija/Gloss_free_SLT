# Experiment comparison

All rows use the same evaluation: the same manifest, references, text normalization and metric code
(`slt_vit/metrics.py`).
- **Exact match:** whole-sentence equality.
- **BLEU:** corpus BLEU-4 from sacreBLEU 2.4.3 with its default 13a tokenization.
- **ROUGE-L:** word-LCS F-score with β = 1.2, as a percentage.
- **Paired tests:** 2000 bootstrap resamples over clips.
- **Machine-readable copy:** `results/summary.csv`.

**Columns:**
- **Signer 6:** validation, where every choice was made.
- **Signers 3–5:** each clip scored by the model trained without that signer (282 clips), a check on other unseen
  signers that was never used for tuning.
- **Signer 7:** the test signer. The baseline row is the initial evaluation; the other two are **repeat evaluations**,
  run once each after all choices were fixed.
- **Runtime/video:** measured end to end on the RTX 3050 6 GB laptop GPU over the 94 signer-6 clips, model time only
  (decoding the video from disk excluded).

## Main table

| Experiment | Signer 6 exact | BLEU-4 | ROUGE-L | Signers 3–5 exact | Signer 7 exact | Runtime/video |
|---|---:|---:|---:|---:|---:|---:|
| Original baseline (SigLIP stage 2, greedy) | 64/94 = 68.1% | 73.3 | 74.9 | 170/282 = 60.3% | 23/91 = 25.3% (initial) | 0.16 s |
| A: candidate restriction, alpha 0 | 66/94 = 70.2% | 76.1 | 77.1 | 179/282 = 63.5% | – | 0.14 s |
| B: temporal sampling, trim_5 (selected) | 68/94 = 72.3% | 76.9 | 78.0 | 171/282 = 60.6% | – | 0.15 s |
| B: three shifted views | 64/94 = 68.1% | 73.8 | 74.8 | 177/282 = 62.8% | – | 0.25 s |
| C: CLIP + SigLIP, w 0.6, greedy | 69/94 = 73.4% | 77.6 | 80.1 | 192/282 = 68.1% | – | 0.33 s |
| C: CLIP + SigLIP, w 0.5, candidates | 69/94 = 73.4% | 79.3 | 81.4 | 189/282 = 67.0% | – | 0.29 s |
| **D3: best validated inference-only combination** (C-candidates + A + trim_5) | **73/94 = 77.7%** | 83.7 | 83.8 | **190/282 = 67.4%** | **43/91 = 47.3%** (repeat R1) | 0.27 s |
| Hybrid SigLIP + MediaPipe alone (E0, greedy) | 73/94 = 77.7% | 82.9 | 84.6 | 178/282 = 63.1% | – | ≈2.4 s* |
| **E3: SigLIP s2 + CLIP + hybrid, candidates, trim_5 (best validated overall)** | **74/94 = 78.7%** | 83.9 | 84.1 | **211/282 = 74.8%** | **59/91 = 64.8%** (repeat R2) | ≈2.7 s* |

\* Most of the hybrid's time is MediaPipe on CPU, about 2.27 s per clip for its sampled frames (0.087 s/frame).
The GPU part is 0.13 s. `results/val/runtime_E3.json` gives the breakdown.

## Paired differences (exact match, points, 95% bootstrap interval)

| Comparison | Signer 6 | Signers 3–5 | Signer 7 |
|---|---|---|---|
| A vs baseline | +2.1 [+0.0, +5.3] | +3.2 [+0.7, +5.7] | – |
| B trim_5 vs baseline | +4.3 [−2.1, +11.7] | +0.4 [−3.5, +3.9] | – |
| C greedy w0.6 vs baseline | +5.3 [+0.0, +10.6] | +7.8 [+3.5, +12.1] | – |
| D3 vs baseline | +9.6 [+2.1, +17.0] | +7.1 [+2.1, +12.1] | +22.0 [+12.1, +31.9] |
| E3 vs baseline | +10.6 [+3.2, +19.1] | +14.5 [+9.2, +19.5] | +39.6 [+29.7, +50.5] |
| E3 vs D3 | +1.1 [−4.3, +6.4] | +7.4 [+3.9, +11.3] | +17.6 [+8.8, +27.5] |

On signer 6 alone, E3 and D3 cannot be separated. The advantage of the landmark model shows on the other unseen
signers and on signer 7.

## Training experiments

All use the same stage-1 recipe on cached inputs (`train_features.py`) and select the best epoch on signer 6.
"Signers 3–5" is a fixed 60-epoch retraining per fold, scored once.

| Run | Inputs | Signer 6 exact (best epoch) | BLEU-4 | Signers 3–5 | Train time | Peak GPU |
|---|---|---|---:|---|---:|---:|
| V1 control | frozen SigLIP | 62/94 = 66.0% (75) | 70.3 | 154/282 = 54.6% | 7.6 min | 1.78 GB |
| AUG1 | V1 + time warp 0.2, frame drop 0.1 | 60/94 = 63.8% (70) | 69.7 | not run (no gain on signer 6) | 5.7 min | 1.78 GB |
| AUG2 | V1 + time warp 0.4, frame drop 0.2 | 62/94 = 66.0% (70) | 66.3 | not run (no gain on signer 6) | 5.7 min | 1.78 GB |
| L1 | MediaPipe landmarks only | 64/94 = 68.1% (65) | 72.6 | 158/282 = 56.0% | 5.4 min | 1.74 GB |
| **H1** | frozen SigLIP + landmarks | **73/94 = 77.7%** (65) | **82.9** | **178/282 = 63.1%** | 6.1 min | 1.80 GB |

**Reading the training runs:**
- **Trainer parity:** the V1 control matches the project's own SigLIP stage-1 run (63/94 on signer 6, 153/282 on
  signers 3–5), so the training code is not a confound.
- **Landmarks alone** are as good as frozen SigLIP features (L1 vs V1 on signers 3–5: +1.4 points, interval −5.7 to
  +8.5).
- **Combined,** landmarks and SigLIP are clearly better than SigLIP alone: H1 vs V1 on signers 3–5 is +8.5 points,
  interval +2.8 to +14.5. On signers 3–5, the frozen hybrid also beats the fine-tuned SigLIP stage-2 model (178 vs
  170).
- **Speed and frame-drop augmentation** did not help on signer 6, so it was not adopted.

## Other measurements
- **Score comparability:** CLIP and SigLIP use the same vocabulary, and their reference NLL per token is 0.392 vs 0.393.
  Their top candidates agree on only 56/94 clips (`results/val/score_comparability.json`).
- **Live inference:** live end-to-end inference reproduces the cached-feature predictions on 94/94 clips for every
  timed setting (`results/val/runtime.json`).
- **MP4 input:** see `baseline_audit.md` §3. D3 gives the same prediction from the frame folder and from the matching
  MP4 frames for 21/21 signer-6 videos; the baseline does for 18/21.
