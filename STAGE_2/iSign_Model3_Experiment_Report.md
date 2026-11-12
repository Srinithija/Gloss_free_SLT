# iSign — Model 3 (Frozen SigLIP + MediaPipe) Experiment Report


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
| 15:15–15:26 | 5k test evaluations, report |

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

#### Model 3 hybrid (SigLIP+MediaPipe), 2k train (`runs/m3_hybrid`)

| Epoch | Training Loss | Validation Loss | Exact Match | BLEU-1 | BLEU-4 | ROUGE-L | Distinct outputs / top share | Time (train+eval) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7.943 | 6.549 | not logged | 0.74 | 0.01 | 1.44 | 53 / 10% | 116 s |
| 1 **(best)** | 6.259 | 5.635 | 0.0% | 7.61 | 0.18 | 9.31 | 41 / 22% | 27 s |
| 2 | 5.584 | 5.476 | not logged | 7.19 | 0.09 | 8.52 | 80 / 19% | 23 s |
| 3 | 5.062 | 5.417 | not logged | 9.28 | 0.15 | 8.17 | 147 / 15% | 26 s |
| 4 | 4.545 | 5.571 | not logged | 5.16 | 0.10 | 6.06 | 107 / 10% | 28 s |
| 5 | 4.143 | 5.772 | not logged | 5.24 | 0.10 | 6.65 | 245 / 12% | 25 s |

Peak VRAM 3.00 GiB. Train loss uses label smoothing 0.1; validation loss has none.

#### Visual-only SigLIP baseline, 2k train (`runs/visual_only`)

| Epoch | Training Loss | Validation Loss | Exact Match | BLEU-1 | BLEU-4 | ROUGE-L | Distinct outputs / top share | Time (train+eval) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7.999 | 6.689 | not logged | 0.69 | 0.01 | 1.34 | 21 / 29% | 40 s |
| 1 | 6.415 | 5.752 | not logged | 1.98 | 0.10 | 4.91 | 4 / 67% | 24 s |
| 2 **(best)** | 5.677 | 5.426 | 0.0% | 7.63 | 0.13 | 8.63 | 28 / 41% | 22 s |
| 3 | 5.081 | 5.393 | not logged | 8.71 | 0.07 | 8.22 | 55 / 14% | 24 s |
| 4 | 4.536 | 5.486 | not logged | 4.28 | 0.12 | 4.56 | 39 / 72% | 27 s |
| 5 | 4.107 | 5.798 | not logged | 2.05 | 0.05 | 3.34 | 18 / 43% | 22 s |
| 6 | 3.712 | 5.986 | not logged | 4.09 | 0.05 | 5.10 | 70 / 48% | 24 s |

Peak VRAM 2.98 GiB. Train loss uses label smoothing 0.1; validation loss has none.

#### Model 3 hybrid (SigLIP+MediaPipe), 4.9k train (`runs/m3_hybrid_5k`)

| Epoch | Training Loss | Validation Loss | Exact Match | BLEU-1 | BLEU-4 | ROUGE-L | Distinct outputs / top share | Time (train+eval) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7.342 | 5.657 | not logged | 0.00 | 0.00 | 7.23 | 6 / 97% | 46 s |
| 1 | 5.946 | 5.201 | not logged | 6.76 | 0.08 | 8.67 | 80 / 27% | 34 s |
| 2 | 5.470 | 5.049 | not logged | 5.68 | 0.11 | 5.96 | 108 / 53% | 38 s |
| 3 | 5.011 | 4.959 | not logged | 6.95 | 0.15 | 7.14 | 89 / 27% | 58 s |
| 4 | 4.670 | 5.009 | not logged | 6.17 | 0.08 | 7.38 | 116 / 27% | 35 s |
| 5 **(best)** | 4.356 | 5.025 | 0.0% | 8.29 | 0.27 | 8.31 | 187 / 10% | 36 s |
| 6 | 4.109 | 5.270 | not logged | 6.61 | 0.12 | 6.91 | 351 / 7% | 35 s |
| 7 | 3.892 | 5.330 | not logged | 5.38 | 0.12 | 7.44 | 291 / 16% | 35 s |
| 8 | 3.725 | 5.290 | not logged | 7.82 | 0.17 | 7.56 | 265 / 14% | 35 s |
| 9 | 3.590 | 5.346 | not logged | 7.97 | 0.10 | 7.10 | 401 / 10% | 37 s |

Peak VRAM 2.99 GiB. Train loss uses label smoothing 0.1; validation loss has none.

#### Visual-only SigLIP baseline, 4.9k train (`runs/visual_only_5k`)

| Epoch | Training Loss | Validation Loss | Exact Match | BLEU-1 | BLEU-4 | ROUGE-L | Distinct outputs / top share | Time (train+eval) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7.417 | 5.865 | not logged | 0.00 | 0.00 | 3.21 | 1 / 100% | 58 s |
| 1 | 5.961 | 5.129 | not logged | 5.30 | 0.06 | 9.05 | 11 / 68% | 37 s |
| 2 | 5.378 | 4.993 | not logged | 6.38 | 0.13 | 6.79 | 25 / 41% | 38 s |
| 3 **(best)** | 4.879 | 4.967 | 0.0% | 6.47 | 0.14 | 6.41 | 20 / 20% | 38 s |
| 4 | 4.483 | 5.056 | not logged | 6.17 | 0.09 | 8.12 | 21 / 44% | 38 s |
| 5 | 4.115 | 5.136 | not logged | 8.52 | 0.11 | 7.66 | 82 / 19% | 42 s |
| 6 | 3.806 | 5.384 | not logged | 8.78 | 0.11 | 7.70 | 126 / 39% | 42 s |
| 7 | 3.569 | 5.519 | not logged | 6.35 | 0.09 | 7.24 | 95 / 37% | 58 s |

Peak VRAM 2.97 GiB. Train loss uses label smoothing 0.1; validation loss has none.

\*Exact match per epoch was not logged by the training loop. It is known only at the selected epoch (0.0% in every run; see `dev_at_best` in `test_metrics.json`).



### Best checkpoint and why
- **Selected model:** `runs/m3_hybrid_5k/best.pt` (SHA-256 in `best.pt.sha256`), epoch 5.
  - Highest dev BLEU-4 of all runs: 0.27.
  - Most varied dev output: 187 distinct outputs, 10% top share.
  - Validation loss is close to its minimum (5.03 against 4.96 at epoch 3).
- **Overfitting:** every run overfits after 2–5 epochs. Train loss keeps falling while validation loss rises.
  - Training longer would not help at this data size; early stopping ended every run well before the deadline.
- **More data mattered:** going from 2k to 4.9k training clips lowered the best validation loss from 5.42 to 4.96 for the hybrid. That is the clearest learning signal in the experiment.
- **Hybrid versus visual-only at 4.9k clips (same data, same settings):**
  - Best validation loss: 4.96 hybrid against 4.97 visual-only. Test loss: 4.99 against 4.94.
  - **In the seq2seq model, the landmarks give no measurable loss advantage at this scale.**
  - The hybrid does give much more varied outputs: 181 against 23 distinct test outputs (greedy), and greedy BLEU-1 8.76 against 6.86. These differences are within the noise of a single seed.
- **Dev BLEU-4 is a weak selector:** at values of 0.1–0.3 it is very noisy, and differences of about 0.1 are within noise.

---

## F. Evaluation results
All rows are genuinely completed runs. Each model was decoded on test once per decoding setting, after selection on dev.
- **Metric code:** `isign_siglip_slt/slt/metrics.py`, the same code used for the ResNet baseline.
  - BLEU uses sacreBLEU, corpus-level, `13a` tokenisation, on lowercased, punctuation-stripped text. ROUGE-L is the sentence-level F-measure, averaged.
- **"Content-word P / R":** precision and recall of non-stop-words in the hypothesis against the reference, in %.
- **ms/video:** batched (64) GPU decoding time per clip. It excludes feature extraction, which costs about 0.18 s (SigLIP) and about 0.4 s (MediaPipe, 14 workers) per clip of 32 frames.

### Test results (held-out `test1k`, 993 scorable clips)

| Experiment | Dataset | Samples | Decoding | Exact Match | BLEU-1 | BLEU-4 | ROUGE-L | Test loss | Distinct outputs (top share) | Content-word P / R (%) | ms/video | Remarks |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Model 3 hybrid (SigLIP+MediaPipe), 2k train | iSign test1k | 993 | greedy | 0.0% | 8.11 | 0.17 | 9.51 | 5.661 | 40 (24%) | 0.6 / 0.14 | 4.1 | best epoch 1 (dev BLEU-4 0.18) |
| Model 3 hybrid (SigLIP+MediaPipe), 2k train | iSign test1k | 993 | beam4_nr3 | 0.0% | 3.61 | 0.20 | 4.46 | 5.661 | 28 (42%) | 0.3 / 0.05 | 36.2 | best epoch 1 (dev BLEU-4 0.18) |
| Visual-only SigLIP baseline, 2k train | iSign test1k | 993 | greedy | 0.0% | 8.09 | 0.23 | 8.90 | 5.407 | 33 (40%) | 0.1 / 0.04 | 2.4 | best epoch 2 (dev BLEU-4 0.13) |
| Visual-only SigLIP baseline, 2k train | iSign test1k | 993 | beam4_nr3 | 0.0% | 0.90 | 0.06 | 1.43 | 5.407 | 6 (60%) | 0.4 / 0.05 | 53.9 | best epoch 2 (dev BLEU-4 0.13) |
| Model 3 hybrid (SigLIP+MediaPipe), 4.9k train | iSign test1k | 993 | greedy | 0.0% | 8.76 | 0.13 | 8.58 | 4.993 | 181 (12%) | 0.7 / 0.42 | 4.5 | best epoch 5 (dev BLEU-4 0.27) |
| Model 3 hybrid (SigLIP+MediaPipe), 4.9k train | iSign test1k | 993 | beam4_nr3 | 0.0% | 6.97 | 0.24 | 6.73 | 4.993 | 93 (20%) | 0.8 / 0.40 | 51.2 | best epoch 5 (dev BLEU-4 0.27) |
| Visual-only SigLIP baseline, 4.9k train | iSign test1k | 993 | greedy | 0.0% | 6.86 | 0.07 | 6.89 | 4.936 | 23 (18%) | 0.8 / 0.35 | 3.2 | best epoch 3 (dev BLEU-4 0.14) |
| Visual-only SigLIP baseline, 4.9k train | iSign test1k | 993 | beam4_nr3 | 0.0% | 8.56 | 0.19 | 7.37 | 4.936 | 59 (17%) | 0.9 / 0.38 | 29.0 | best epoch 3 (dev BLEU-4 0.14) |

### Key-word recognition probe (test1k; 300 most frequent content words; chance AUC = 0.5)

| Input features | Train / dev / test clips | Mean per-word AUC (test) | Shuffled-input control AUC | P@5 model | P@5 frequency prior | Best-recognised words (AUC, test clips) |
|---|---:|---:|---:|---:|---:|---|
| hybrid | 4921 / 990 / 993 | **0.673** | 0.495 | 0.037 | 0.035 | page (0.93, 15), sign (0.88, 10), deaf (0.78, 11), day (0.76, 11), police (0.76, 19), time (0.76, 18), etc (0.75, 12), am (0.75, 10) |
| landmarks | 4921 / 990 / 993 | **0.661** | 0.505 | 0.034 | 0.035 | page (0.90, 15), am (0.82, 10), know (0.81, 11), sign (0.81, 10), found (0.79, 12), pm (0.78, 10), police (0.76, 19), years (0.76, 11) |
| visual | 4921 / 990 / 993 | **0.639** | 0.494 | 0.035 | 0.035 | page (0.95, 15), sign (0.85, 10), while (0.81, 13), government (0.77, 17), due (0.77, 14), deaf (0.77, 11), world (0.74, 14), time (0.72, 18) |

**Reference point on a different subset, not directly comparable:**
- The project's earlier ResNet-18 gloss-free baseline, trained on **101k** clips: test-subset (1,979 clips) BLEU-1 9.16, BLEU-4 0.76, ROUGE-L 8.16, exact match 0.10% (re-scored with the same metric code; `isign_siglip_slt/reports/02_baseline.md`).
- That model used 20× more training data, so a lower score from the one-hour Model 3 runs is expected. It says nothing yet about the architecture.

---

## G. Prediction examples (selected model `m3_hybrid_5k`, test1k, greedy)
**Exact matches:** none. 0 of 993 clips.

Ground-truth captions are not reproduced here because iSign dataset captions must not be published.

**Most frequent test outputs:** "The CPI(M) wants to remove the government." (121); "The CPI(M) wants to remove the government of the government." (89); "The name is a square." (68); "The police have been arrested." (52); "The CPI(M) wants to remove the media." (42). 394 of 993 test clips have sentence ROUGE-L 0.

**What the outputs show:**
- **The model has learned the video domain:**
  - news-studio clips → political and police sentences ("The CPI(M) wants to …", "The police have …");
  - children's-book or educational clips → "Let's write.", "The square is a square.", "The tiger is very big.".
- The "partially correct" rows are mostly domain and language-prior overlap ("Let's …", "police", "Rs … crore"). They are not word-level recognition of the signs.

---

## H. ISL vs iSign comparison

### Historical ISL results (a different dataset; reference only)
| Model (ISL, Kaggle) | Exact Match | BLEU-4 | ROUGE-L |
|---|---:|---:|---:|
| ResNet18 (original) | 9.9% | 12.6 | 23.2 |
| Model 1: fine-tuned SigLIP | 25.3% | 25.9 | 37.6 |
| Model 3 alone, greedy | 37.4% | 43.9 | 47.6 |
| Model 3 alone, 97 candidates | 45.1% | 47.6 | 53.9 |
| D3: Model 1 + Model 2 | 47.3% | 47.9 | 55.1 |
| E3: Model 1 + Model 2 + Model 3 | 64.8% | 67.4 | 70.3 |

### New iSign results (this experiment, test1k, greedy unless noted)
| Experiment | Dataset | Samples | Exact Match | BLEU-4 | ROUGE-L | Remarks |
|---|---|---:|---:|---:|---:|---|
| Model 3 hybrid, 4.9k train (selected) | iSign test1k | 993 | 0.0% | 0.13 (beam-4: 0.24) | 8.58 | open-vocabulary generation; 181 distinct outputs |
| Visual-only baseline, 4.9k train | iSign test1k | 993 | 0.0% | 0.07 (beam-4: 0.19) | 6.89 | open-vocabulary generation; 23 distinct outputs |
| Model 3 hybrid, 2k train | iSign test1k | 993 | 0.0% | 0.17 (beam-4: 0.20) | 9.51 | open-vocabulary generation; 40 distinct outputs |
| Visual-only baseline, 2k train | iSign test1k | 993 | 0.0% | 0.23 (beam-4: 0.06) | 8.90 | open-vocabulary generation; 33 distinct outputs |
| *ResNet-18 gloss-free baseline (earlier work, 101k train)* | iSign test subset (different 1,979 clips) | 1979 | 0.1% | 0.76 | 8.16 | not the same test clips or training size; reference only |

**Why the numbers are not comparable:**
- **ISL is effectively closed-set.** It has about 97 candidate sentences, each seen many times in training, which is why candidate re-ranking helped there.
- **iSign is open-vocabulary.** It has 98k distinct training sentences, and only 4.6% of test sentences appear in the full train set; just 8 of 993 test sentences appear in the 4.9k training subset.
- **Consequences:**
  - Exact match and candidate re-ranking are close to meaningless on iSign.
  - A 97-candidate restriction was deliberately **not** imposed.
  - The models were trained on 2–5% of iSign's training data for minutes.

**What transferred directly:**
- frozen SigLIP feature extraction;
- MediaPipe Holistic in static mode on the original-resolution frames;
- the landmark projection (Linear + LayerNorm + ReLU);
- 1024-d fusion;
- the Conv1d(k5) + BatchNorm + ReLU + MaxPool(2) + Linear + LayerNorm + ReLU temporal encoder;
- mBART-cc25 as the decoder with `en_XX`.

**What needed changes:**
- **Landmark dimensionality:** 204 instead of 233 (a separately designed layout; see C).
- **Frame budget:** K=32 motion-aware keyframes, because iSign clips are long (median 183 frames).
- **mBART size:** reduced to 3+3 layers with a trimmed vocabulary (6 GB GPU).
- **Visual features:** mean-centred per clip.
- **Data hygiene:** bad-crop clips excluded from training; a group split by source video.
- **Evaluation:** open-vocabulary generation metrics, plus a key-word probe instead of candidate sentences.

---

## I. Limitations and uncompleted work

1. **What was successfully implemented?**
   - The full Model 3 pipeline on iSign, end to end: keyframe selection → frozen SigLIP and MediaPipe features (cached) → fusion → temporal encoder → mBART encoder-decoder with teacher-forced cross-entropy → autoregressive greedy and beam generation → decoded English.
   - Verified shapes, masks, gradient flow and parameter updates.
   - A leakage-free train/dev/test split and test features extracted in this run.
   - Four trained models: hybrid and visual-only, each at 2k and 4.9k training clips.
   - A key-word recognition probe; saved checkpoints, logs and predictions.
2. **What actually learned during training?**
   - Train loss fell from 7.3–8.0 to 3.6–4.1, and validation loss improved to 4.96 (hybrid-5k).
   - The decoder learned **fluent English and the domain prior** (news versus children's educational content).
   - The visual and landmark encoders carry a **weak but real word-level signal**: key-word probe AUC 0.67 against 0.50 shuffled. The hybrid beats landmarks-only (0.66) and visual-only (0.64).
   - The generator does not yet convert that signal into correct content words.
3. **Sentence recognition or sign-to-text translation?**
   - The system is a genuine **direct sign-to-text generator**: there is no candidate list, and it produces 181 distinct outputs on test.
   - **In quality it does not translate yet.** Outputs are mostly generic, domain-conditioned sentences.
   - It is **not** sentence recognition, and it is **not** gloss recognition.
4. **Is the sign-to-word mapping correct?**
   - The **video-to-text mapping is correct**, verified programmatically and by inspecting consecutive story segments.
   - **No sign-to-word mapping is learned reliably:** content-word recall in generated text is under 0.5%.
   - At the probe level, some words are recognisable above chance: "sign", "deaf", "police", "government", "time" have AUC 0.73–0.88.
   - "page" (AUC 0.90–0.95) is almost certainly recognised from the visual **context** of story-book videos, not from a sign.
5. **Is mBART functioning properly?**
   - Mechanically yes. Pretrained weights load with no missing keys, the loss decreases, and generation starts from `en_XX` and stops at `</s>`.
   - Without beam search or repetition control, it sometimes loops ("registered registered …"). Beam-4 with no-repeat-3 removes the loops but makes outputs even more generic.
   - The tokenizer round-trip is exact.
6. **Can the model generate meaningful English sentences?**
   - It generates **grammatical, meaningful-looking English**, but the sentences are not faithful to the signed content.
7. **If only key signs are recognised, why?**
   - **There are no gloss labels.** The only supervision is free English sentences with about 98k distinct sentences and 33k word types, while this run had at most 4.9k clips.
   - That is far too little data for an alignment-free model to discover word-level correspondences.
   - The probe shows the features do separate some frequent words. The seq2seq decoder, trained for minutes, falls back on the language prior.
8. **Which components remain incomplete?**
   - Full-dataset training: landmark features exist for only about 4.9k training clips; the full extraction is estimated at 22–28 h of MediaPipe.
   - The full 12+12 mBART.
   - A landmarks-only seq2seq run.
   - Multi-seed runs and confidence intervals.
   - Full-test (12.6k clips) evaluation.
9. **What prevented further improvement?**
   - **Training-data volume:** 2–5% of iSign. Every run overfits within 2–5 epochs.
   - **Feature-extraction throughput:** MediaPipe runs at about 85 frames/s on the CPU, which is about 2.7 clips/s at K=32.
   - **The one-hour limit.**
10. **What caused the limitations?**
    - **Mainly the one-hour deadline,** through how many clips could be featurised, combined with **dataset limitations:** an open-vocabulary task with no gloss annotations.
    - **System specs played a role:** 6 GB VRAM forced the 3+3 mBART, and CPU-bound MediaPipe set the throughput.
    - **No implementation fault was found:** shapes, masks, the loss, gradients and the label mapping were all checked.
11. **What next?**
    1. **Run the planned full extraction** (K=32 MediaPipe for about 100k train clips, about 1 day in the background) and **train on all training clips.** The 2k → 4.9k gain in validation loss shows data is the main lever. The ResNet baseline on 101k clips also reached a higher BLEU-4 (0.76).
    2. **Add an auxiliary key-word loss.** A multi-label BCE over the top-N caption content words on the encoder output would push the encoder towards word-discriminative features. The probe shows this signal exists.
    3. **Pre-train the encoder–text alignment** contrastively, CLIP-style, against caption embeddings before seq2seq.
    4. **Use more frames for long clips** (K=48–64), plus SigLIP augmentation views.
    5. **Report on the full 12.6k test set with several seeds** and bootstrap confidence intervals.

---

## J. Final conclusion
- **Implementation:** Model 3 (frozen SigLIP ViT-B/16 + MediaPipe Holistic → 1024-d fusion → Conv1d temporal encoder → mBART-cc25) was ported to iSign and runs end to end.
  - It uses verified data mapping, leakage-free splits, motion-aware 32-keyframe sampling (about 16% of original frames kept), cached features, and a functioning teacher-forced and autoregressive mBART decoder.
- **Results:** in one hour, on 4.9k training clips, the best model (hybrid-5k) reaches test BLEU-4 0.13 greedy / 0.24 beam-4, BLEU-1 8.76, ROUGE-L 8.58 and 0% exact match on 993 held-out test clips.
  - Its outputs are fluent but domain-generic, so this is **not yet sign language translation**.
- **What did learn:**
  - More data clearly lowered the validation loss.
  - The fused features hold a measurable word-level signal (key-word AUC 0.67, chance 0.50), and the hybrid beats either stream alone in that probe.
  - In seq2seq loss, the hybrid and the visual-only model are tied at this scale. The ISL advantage of Model 3 is therefore only partly reproduced: in the probe and in output diversity, not in loss or BLEU.
- **Most promising next steps:** train on the full iSign training set once landmark extraction is complete, and add an auxiliary key-word or contrastive objective so the decoder is forced to use that signal.

### Artifacts
| What | Where (relative to `iSign_Model3_1h/`) |
|---|---|
| Selected checkpoint | `runs/m3_hybrid_5k/best.pt` (+ `.sha256`), with `last.pt` for resuming |
| Other checkpoints | `runs/{m3_hybrid,visual_only,visual_only_5k}/best.pt` |
| Per-epoch logs | `runs/*/train_log.jsonl`, `logs/train_*.log` |
| Test metrics | `runs/*/test_metrics.json` (per-clip predictions are kept locally) |
| Key-word probe | `reports/keyword_probe_{hybrid,visual,landmarks}.json`, `keyword_probe.py` |
| Sanity check | `logs/sanity_check.log`, `sanity_check.py` |
| Dataset and frame statistics | `logs/dataset_stats.log`, `logs/frame_stats.log` |
| Manifests | `manifests/{train2k,train5k,dev1k,test1k}.csv` |
| Test feature caches | `cache/siglip_test`, `cache/landmarks_test` |
| Code | `run_train.py` (wrapper around the unchanged `isign_siglip_slt/scripts/train.py`), `stores.py`, `eval_test.py`, `make_tables.py` |
