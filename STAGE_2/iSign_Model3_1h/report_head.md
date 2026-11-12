# iSign — Model 3 (Frozen SigLIP + MediaPipe) Experiment Report

**Date:** 2026-10-30, one-hour run (14:42 → {END_TIME} IST)
**Experiment folder:** `E:\FINAL YEAR PROJECT\GLOSS FREE SLT\iSign_Model3_1h\` (configs, manifests, logs, runs, checkpoints, reports)
**Existing project reused (read-only):** `isign_siglip_slt\` (code, feature caches, environments). No original file, dataset, cache or checkpoint was modified. All new outputs are in the experiment folder.

> **Main result.** The full Model 3 pipeline is implemented and runs end to end on iSign:
> motion-aware keyframes → frozen SigLIP + MediaPipe → fusion → temporal conv → mBART → generated English.
>
> Within one hour, with 2k–4.9k training clips, it does **not** achieve sign-to-text translation:
> - Test BLEU-4 is 0.07–0.24 across the four models and both decodings, and exact match is 0%.
> - Outputs are fluent but generic English, mostly from a language prior.
>
> A separate **key-word probe** shows that the features carry a real but weak word-level signal: mean per-word AUC is 0.67 against 0.50 for shuffled inputs. Hybrid features beat either stream alone.

---

## A. Experiment overview

| Item | Value |
|---|---|
| Objective | Port ISL "Model 3" (frozen SigLIP ViT-B/16 + MediaPipe Holistic → temporal conv → mBART) to iSign and measure real sign-to-text learning |
| Dataset | iSign v1.1 (ISL ↔ English, sentence-level), group split `dataset\iSign\splits\v1` |
| Task actually trained | **Direct sign-to-text generation** (gloss-free, teacher-forced seq2seq), plus a **key-word recognition probe**. No sentence classification and no candidate list. |
| Hardware | Intel i7-13620H (10 cores, 16 threads), 15.7 GB RAM; NVIDIA RTX 4050 Laptop 6 GB (driver 592.82, CUDA available); E: 74 GB free at start |
| Software | Python venv `isign_siglip_slt\envs\main`: torch 2.5.1+cu124, transformers 4.46.3, sacrebleu 2.4.3, sentencepiece 0.2.0, numpy 1.26.4. MediaPipe venv: mediapipe 0.10.14, opencv-contrib 4.10.0. No packages were installed or upgraded. scikit-learn and scipy are absent, so AUC was implemented in numpy and unit-checked. |
| Pretrained models | `google/siglip-base-patch16-224` @ `7fd15f06` (frozen, 92.9M), `facebook/mbart-large-cc25` @ `f417e556` |

### Time line (actual)
| Time | Step |
|---|---|
| 14:42–14:48 | Inspection. Found the existing SigLIP+MediaPipe project, its caches and its audit. Built the test manifest. |
| 14:48–14:54 | Test-set SigLIP + MediaPipe extraction (background). Manifests, leakage check, sanity check. |
| 14:51–15:15 | Training: 4 runs (hybrid and visual-only × 2k / 4.9k training clips) |
| 15:00–15:03 | Test evaluation of the 2k runs; key-word probe (3 feature sets) |
| 15:15–{END_TIME} | 5k test evaluations, report |

---

## B. Dataset analysis

**Annotation format:** `splits/v1/{train,dev,test}.csv` with columns `uid, text, source_video_id`. There is one English sentence per clip, and the clip uid equals the video filename / LMDB key.
- **There are no gloss annotations in iSign.** Gloss recognition and gloss-to-text were therefore impossible, and no gloss labels were invented.

| Split | Clips | Kept captions (non-empty, non-placeholder) | Source videos | Distinct sentences | Word vocabulary |
|---|---:|---:|---:|---:|---:|
| train | 101,790 | 101,347 | 3,455 | 98,155 | 33,329 |
| dev | 12,724 | 12,653 | 429 | 12,445 | 11,396 |
| test | 12,723 | 12,627 | 428 | 12,416 | 11,768 |

- **Integrity:** previous audit of all 127,237 clips (`isign_siglip_slt/reports/01_data_audit.md`):
  - 0 corrupt or unreadable clips.
  - LMDB labels match the split CSVs exactly: same uid sets, 0 text mismatches.
  - 1,071 small-resolution clips are bad crops (torso, sleeve or background only). They are excluded from training only.
- **Video format:** 85% are 300×300 and 14% are about 650×720. fps is 29.97, 25, 29 or 30. Original frames per clip: median 183 and max 2,370, which is about 6.6 s median.
- **Sentence length (words, train):**
  - Mean 10.8, median 11, p90 18, max 59.
  - Distribution: 1–3 words 8.3%; 4–8 words 27.4%; 9–15 words 45.1%; 16–25 words 18.8%; 26 or more 0.5%.
- **Open-vocabulary task:** only **4.6% of test sentences occur verbatim in train**, so sentence classification or retrieval cannot work on iSign.
- **Duplicates:** 4,834 train clips share a sentence with another clip. These are mostly page numbers like "2" and "1".
- **Video-to-text mapping, verified programmatically:**
  - Each uid is looked up in the split CSV, and the target is encoded, then decoded back to the identical sentence (`logs/sanity_check.log`).
  - Example: clips `-3GZr4gY8Is--0` and `-3GZr4gY8Is--1` map to consecutive segments of the same news story (captions not reproduced).
  - LMDB frame *i* is pixel-identical to MP4 frame `selected_indices[i]` (previous alignment check on 20 clips).

### Subsets used in this one-hour experiment
| Subset | Clips | Source videos | How selected |
|---|---:|---:|---|
| `train2k` | 2,000 | **59** | All train clips that already had both SigLIP and MediaPipe features in `cache/full_k32` (the first 2,000 train uids in alphabetical order, hence few source videos) |
| `train5k` | 4,930 (4,921 after exclusions) | 1,685 | `train2k` ∪ the 3,000-clip random pilot train set (seed 0). Pilot caches are re-indexed to the same K=32 keyframes (see C). |
| `dev1k` | 1,000 (990 scorable) | 334 | Random (seed 0) from the existing 2k dev subset |
| `test1k` | 1,000 (993 scorable) | 323 | Random (seed 0) from the existing 2k test subset (`subset2k_test`). Features extracted in this run. |

- **Leakage check:** 0 source videos are shared between train/dev, train/test or dev/test (`logs/` and the manifest script).
- **Not signer-independent:** the same ISH News interpreters appear in every split.
- Test was decoded once per model, after model selection on dev.

---

## C. Preprocessing

### Intelligent frame selection (reused from the existing project: `isign_siglip_slt/scripts/select_keyframes.py`)
1. **Source:** LMDB frames (≤128 per clip, uniformly sub-sampled from the MP4 when longer), with `trim_5` removing 5 edge frames.
   - Corrupt or blank frames: none exist (audit: 0 corrupt clips; every frame key decodes).
2. **Motion signal:** m[t] = mean |gray64(t) − gray64(t−1)|, smoothed over 5 frames.
3. **Active span:** leading and trailing frames whose motion is below 30% of the clip's p90 motion are dropped (signer at rest). At least 50% of the range is always kept.
4. **K = 32 keyframes:** placed evenly in the cumulative weight 0.5·uniform + 0.5·motion. Frames are denser where the hands move, but static sign holds are still covered: frames are **not** dropped just for looking similar. Chronological order is kept.
5. **Training-time augmentation:** an ordered random 80–100% of the keyframes are kept.

| Subset | Original video frames | Stored LMDB frames | Mean active span | Selected frames | Retained of original | Retained of stored |
|---|---:|---:|---:|---:|---:|---:|
| train2k | 381,698 (190.8/clip) | 226,642 | 101.3 | 63,784 (31.9/clip) | 16.7% | 28.1% |
| dev1k | 212,187 (212.2/clip) | 116,311 | 104.6 | 31,979 | 15.1% | 27.5% |
| test1k | 204,901 (204.9/clip) | 117,465 | 105.6 | 31,955 | 15.6% | 27.2% |

**Keyframe selection time:** 378 s for all 127,237 clips, about 3 ms per clip, done once.

### SigLIP features (frozen)
- **Preprocessing:** 256×256 frame → bicubic 224×224 → mean/std 0.5 → SigLIP vision tower in `eval()`, `torch.no_grad()`, fp16 autocast, batches of 256 frames.
- **Output:** `pooler_output` **768-d**, cached as an fp16 memory map.
- **Test extraction in this run:** 31,955 frames in 2.9 min (181 frames/s while sharing the machine with MediaPipe), 0.54 GiB VRAM.

### MediaPipe Holistic landmarks
- **Model:** Holistic with `static_image_mode=True`, `model_complexity=1`, run on the **original-resolution MP4 frame** at each keyframe.
  - CPU: 14 worker processes, with a fresh Holistic instance per clip; this fixed a determinism bug found earlier.
- **Test extraction in this run:** 31,955 frames in 6.2 min (85–88 frames/s).
  - Detection rates: pose 99.3%; left hand 87.6–89.9%; right hand 84.0–87.5%.
- **Feature vector: 204-d per frame** (`isign_siglip_slt/slt/landmarks.py`):
  - Upper-body pose points 0–24 (x, y, z), body-relative: 75.
  - Left- and right-hand shape, 20 points relative to the wrist, scaled by shoulder width: 60 + 60.
  - Left and right wrist positions, body-relative: 3 + 3.
  - Detection flags for pose, left hand and right hand: 3.
- **Normalisation:** origin = the clip's median mid-shoulder point; unit = median shoulder width; pixel coordinates, so non-square video is not distorted.
- **Missing landmarks:** linearly interpolated over time; a hand that is never detected is all zeros, and its flag is set to 0.
- **Why 204 and not 233:** the two layouts were built separately and hold different features. Stage 1's 233 values (`experiments/prediction_improvement_v1/pi_exp/landmark_features.py`) are: 7 pose points as xy (14) and their visibility (7), each hand's 21 points as body-relative xy (42 + 42) and wrist-relative xyz shape (63 + 63), and 2 hand flags. Stage 2's 204 values are: 25 upper-body pose points as xyz (75), each hand's 20 wrist-relative points as xyz (60 + 60), each wrist's position (3 + 3), and 3 flags. The projection input was adapted to 204.

### Caching
- All features are computed once into sharded memory-maps and never recomputed during training.
- **Training:** 2k clips run at 8 s per epoch; 4.9k clips at 19 s per epoch.
- **Pilot re-indexing:** the pilot caches stored all trimmed frames. `stores.PositionsView` maps them to the same K=32 keyframe positions.
  - Verified on 297 clips present in both caches: landmarks are identical (max diff 0.0), and SigLIP differs only by fp16 rounding (≤0.008).

---

## D. Implemented architecture

```
Input video (MP4 / LMDB frames)
 └─ motion-aware keyframe selection (K=32, chronological)
     ├─ frozen SigLIP ViT-B/16 (no_grad, fp16) ── 768-d/frame ── subtract the clip's mean over valid frames
     └─ MediaPipe Holistic (static, original res) ── 204-d/frame ── Linear(204→256) + LayerNorm + ReLU
 └─ concat → 1024-d/frame (padding mask)                         [B, T≤32, 1024]
 └─ Conv1d(1024→1024, k=5, pad 2) + masked BatchNorm + ReLU + MaxPool1d(2)   [B, T/2, 1024]
 └─ Linear(1024→1024) + LayerNorm + ReLU   (= mBART d_model, so no extra projection is needed)
 └─ mBART-large-cc25 encoder (inputs_embeds + attention mask) → decoder → LM head
 └─ English text (decoder start = en_XX; greedy or beam-4 autoregressive generation)
```

**Shapes logged on the first forward pass** (`logs/sanity_check.log`, batch 8):
- Inputs: vis `[8,30,768]`, lm `[8,30,204]` → lm_proj `[8,30,256]`.
- Temporal encoder: conv `[8,1024,30]` → pool `[8,1024,15]` → proj `[8,15,1024]`.
- mBART: encoder `[8,15,1024]`, decoder `[8,22,1024]`, LM head `[8,22,19551]`.

**mBART connection:**
- **Tokenizer:** `MBartTokenizer` with `src_lang = tgt_lang = en_XX`.
- **Targets:** `caption </s>`; decoder input = `en_XX` + the shifted targets.
- **Padding:** pad positions are set to −100 and ignored by the cross-entropy loss. Label smoothing is 0.1 for training and 0 for validation and test loss.
- **Masks:** the encoder attention mask comes from the pooled frame mask; the decoder attention mask from non-pad tokens.

**Deviations from the ISL Model 3, all documented:**
1. **Landmark input is 204-d, not 233-d** (see C).
2. **mBART is reduced to 3 encoder + 3 decoder layers**, copied from pretrained cc25 layers 0, 5 and 11.
   - Embeddings, positions, layer norms and the LM head are copied too.
   - The vocabulary is trimmed to the 19,551 tokens used in the train captions (dev OOV 0.47%).
   - Reason: 6 GB VRAM and the time limit. The full 12+12 model (611M parameters) cannot be trained with AdamW on 6 GB at a useful batch size.
3. **Visual features are mean-centred per clip.** This removes the static studio and background component and is part of the existing implementation.

**Trainable versus frozen:**
- SigLIP is frozen by construction, because its features are precomputed.
- 116.66M parameters are trainable: landmark projection, conv, BatchNorm, projection and the 3+3 mBART.
- **Gradient flow verified:** non-zero gradient norms in lm_proj (14.9), conv (69.3), bn (1.9), proj (70.9), mBART encoder (69.9), decoder (63.7) and embeddings (27.5).
- After one AdamW step, 149 of 149 parameter tensors changed.
- **Initial loss:** 13.29, which is finite and non-zero, as expected for a random fusion head over 19.5k classes.

**Training setup:**
- **Optimiser:** AdamW; learning rate 3e-4 for new layers and 1e-4 for mBART; weight decay 0.01; linear warm-up (150 or 300 steps) then cosine decay.
- **Batches and precision:** batch 32 grouped by caption length; BF16 autocast; gradient clipping 1.0.
- **Checks:** a non-finite loss or gradient raises an error. There is no broad exception handling.
- **Selection:** by dev BLEU-4 (greedy), early-stopping patience 4.
- **Checkpoints:** written atomically with a SHA-256 hash and reload-verified.

---

## E. Training results (validation = `dev1k`, greedy decoding, metrics on lowercased, punctuation-stripped text)
