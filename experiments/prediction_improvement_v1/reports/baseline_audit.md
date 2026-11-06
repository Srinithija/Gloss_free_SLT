# Baseline audit (Phase 1)

Date: 2026-10-24. No model was retrained during this audit, and signer 7 was not evaluated.

Evidence files:
- `reports/audit_facts.json`
- `reports/audit_errors_and_duplicates.json`
- `results/val/baseline_existing_evaluator/`
- `results/mp4_consistency/`
- `results/val/score_comparability.json`

## 1. Verified baseline

| Item | Verified value | Source |
|---|---|---|
| Selected checkpoint | `outputs/siglip_b16_stage2/best.pt`, epoch 14, SHA-256 `e5ee1160…6e11d44` | `outputs/SELECTED_CHECKPOINT.txt`; hash recomputed in `protected_files_before.json` |
| Architecture | SigLIP ViT-B/16 (`google/siglip-base-patch16-224`, last 4 blocks fine-tuned) → clip-mean → Conv1d k5 + BN + ReLU + MaxPool2 → Linear + LN + ReLU (1024) → mBART-cc25 3+3 layers, 199-token vocabulary | checkpoint `cfg`, `slt_vit/model.py` |
| Validation, signer 6 | **64/94 = 68.1%**, BLEU-4 73.33, ROUGE-L 74.87, loss 0.370 | re-run with the existing evaluator into `results/val/baseline_existing_evaluator/`. The CSV is byte-identical to `outputs/siglip_b16_stage2/eval_val_beam1.csv` |
| Signers 3/4/5 held out (fold models) | 56/97 + 52/89 + 62/96 = **170/282 = 60.3%** | the fold checkpoints' stored `val`; re-run in `results/loso_pooled/baseline_greedy` gives the same 170/282 |
| Initial test, signer 7 | **23/91 = 25.3%**, BLEU-4 25.90, ROUGE-L 37.57 | `outputs/siglip_b16_stage2/final_test_beam1.csv` and the ledger (one row). Read, not re-run |
| CLIP comparison checkpoint | `outputs/clip_b16_stage1/best.pt`, frozen CLIP ViT-B/16, epoch 95, val 61/94 | checkpoint `cfg`/`val` |

All 9 checkpoints used here (SigLIP stage 1/2, CLIP stage 1, and their signer-3/4/5 fold versions) share **the same
tokenizer**: identical cc25 ids, built from the training sentences. They also share 32 frames, clip-mean, 3+3 mBART
layers, greedy decoding and `max_new_tokens` 20.

## 2. Pipeline, as implemented

| Stage | What the code does | Where |
|---|---|---|
| Frames | Each JPEG is centre-cropped to at most 4:3, padded to square with black, and resized to 224 with INTER_AREA (no stretching), then converted BGR→RGB, uint8. Signers 1–2 were recorded at 320×240; signers 3–7 at 1920×1080. | `slt_vit/frames.py: canonicalize` |
| Frame cache | `cache/frames224/<id>.npy`. It equals a fresh decode of the folder (checked on 10 validation clips, all identical). | `slt_vit/cache.py` |
| Temporal sampling | 32 indices `round(linspace(0, n-1, 32))` over the whole clip; deterministic at inference. **451 of 659 clips have fewer than 32 frames** (median 24, range 6–147), so frames are repeated. | `slt_vit/frames.py: sample_indices` |
| Normalization | Pixel values in [0,1], then SigLIP mean/std 0.5 (CLIP uses its own mean/std); vision tower under CUDA float16 autocast. | `slt_vit/vision_encoder.py` |
| Visual feature | The pooled 768-d output per frame. Clip-mean subtracts each clip's mean frame feature. | `slt_vit/model.py`, `temporal_model.py` |
| Decoder | `num_beams=1` (greedy), `max_new_tokens=20`, decoder start `en_XX`, `forced_eos_token_id=</s>` (the 20th token is forced to `</s>`), no length penalty. The model's generation config is pinned so cc25's `num_beams=5` cannot leak in. | `slt_vit/model.py: generate` |
| Output space | **Not restricted.** Any sequence over the 199 tokens is possible. On signer 6, 93/94 outputs are training sentences. On signer 7, at least one is not ("how can i how can i help you"). | `results/val/baseline_greedy/metrics.json` |
| Evaluation | `slt_vit/evaluate.py` calls `Translator.translate_frames`, the same function the app and `infer.py` use, so evaluation and inference share preprocessing, sampling and decoding. | `slt_vit/evaluate.py`, `infer.py` |

## 3. Consistency checks

- **Evaluator vs this experiment's engine.** The engine computes per-frame features once and samples them later.
  - Baseline predictions are identical on 94/94 clips.
  - End-to-end live inference reproduces the cached-feature predictions for every setting timed (94/94 each;
    `results/val/runtime.json`).
- **Frame folder vs MP4** (signer 6, the 21 MP4s that match a frame folder; `results/mp4_consistency/`):
  - **Coverage:** frame folders cover a median 97% of the video, with a median of 2 lead-in and 1 lead-out frames
    missing. The folders are a frame-rate subsample of the video (about 1 in 3 frames).
  - **Equivalent frames:** given the MP4 frames that match the folder frames, the baseline gives the same prediction
    for 18/21 clips. H.264 vs JPEG compression changes 3 outputs.
  - **Correct counts:** folder 9/21, full MP4 10/21, matched MP4 frames 8/21. The differences come from small
    appearance changes, not from a different code path.
  - **Data quirk:** one folder ("are you hiding something", signer 6) covers only frames 3–45 of a 106-frame video.
- **Pre-existing frame-cache hashes** are unchanged (`protected_files_before.json`).

## 4. Where the baseline's errors come from (signer 6)

For each clip, every training sentence was scored by the model's log-probability (`results/val/A_candidates_alpha0.0`).

| Of the 30 greedy errors | Clips |
|---|---|
| Correct sentence was the most likely training sentence; greedy decoding missed it (**decoding**) | 2 |
| Correct sentence ranked 2nd–3rd (**near miss in recognition**) | 11 |
| Correct sentence ranked below 3rd (**recognition**) | 17 |

The correct sentence is in the model's top 3 for 77/94 clips. **Almost all errors are visual recognition errors.**
Decoding changes alone can fix very few: candidate restriction fixes the 2 decoding errors. Gains have to come from
better visual evidence or from combining models.

## 5. Data separation and leakage

- **Split by signer** (train 1–5 / val 6 / test 7) from `data/manifest.json`, unchanged. It equals the earlier
  project's manifest (0 differences).
  - The 4 clips with byte-identical frames under different sentences are excluded.
  - Training never loads signer 7: `select()` refuses it, and this experiment's `entries_for('test')` raises unless
    the repeat-test tool asks for it.
- **Tokenizer and candidate list** come from training texts only. All 94 validation sentences and all 91 test sentences
  also occur in training, so this is a **closed-set, signer-independent** task.
- **Near-duplicates across splits** (cosine of clip-mean frozen SigLIP features):
  - Validation clips: at most 0.961 to any training clip (median 0.951).
  - Test clips: at most 0.897 (median 0.880).
  - Clips of the same signer in training: median nearest-neighbour similarity 0.992–0.995.
  - No validation or test clip is near-duplicate of a training clip.
  - The gap also shows that **signer 7 is visually much further from the training signers than signer 6 is**, which
    matches the low test score (25.3% vs 68.1%).

## 6. Issues and the planned experiments

**Issues found:**
1. The decoder can produce non-corpus sentences such as "how can i how can i help you". Candidate scoring removes this.
2. Most clips repeat frames to reach 32. Temporal views differ only on longer clips.
3. Errors are mostly recognition, and CLIP and SigLIP disagree on 38 of 94 top-1 candidates, which makes an
   ensemble promising.
4. The visual input sees hands at about 1/8 of a 224-px frame. Landmarks could add hand detail.
5. Signer 7 is out of distribution. Validation on signer 6 overestimates test accuracy, so signers 3–5 (fold models)
   are used as a second check.

**Planned experiments** (grids fixed in `SELECTION_RULES.md`):
- **A:** candidate scoring of the 97 training sentences by model log-probability, including `</s>`, with
  `len ** alpha` normalization, alpha ∈ {0, 0.5, 1}.
- **B:** temporal views, each 32 frames:
  - `uniform`, `shift_m1/3`, `shift_p1/3`, `trim_5`, `trim_10` (single views);
  - `shift3`, `trim3`, `mixed5` (multi-view, log-probabilities averaged at every decoding step or per candidate).
- **C:** CLIP + SigLIP, as a weighted average of per-token log-probabilities, w ∈ {0.5 … 0.9}, with greedy and
  candidate decoding. The scores are comparable:
  - both are softmaxes over the identical vocabulary;
  - reference NLL per token is 0.392 (SigLIP) vs 0.393 (CLIP);
  - the spread of candidate log-probabilities is 3.64 vs 3.58 (`results/val/score_comparability.json`).
- **D:** only the combinations of the individually selected A/B/C settings (`configs/D_combinations.json`).
- **Then:** MediaPipe pilot, landmark-only and hybrid models (same stage-1 recipe, with a control run), and training
  augmentation.
