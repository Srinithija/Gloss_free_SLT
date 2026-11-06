# 03 — Pilot: features, landmarks and short training runs

> **Not signer-independent.** The split groups clips by source video. The same ISH News interpreters appear in train, dev and test.

**Pilot set:** `manifests/pilot_clips.csv`, 5,000 clips.
- **Train:** 3,000 random clips (seed 0, flagged clips excluded).
- **Dev:** the baseline's 2,000-clip dev subset (`manifests/subset2k_dev.csv`), including 29 flagged clips.
- **No test data was touched.**

## 1. Checks done before the pilot
| Check | Result | Evidence |
|---|---|---|
| MP4 frame `selected_indices[i]` equals LMDB frame `i` (20 clips: 12 at 300×300, 6 large, 2 small) | **Pixel-identical.** Mean absolute difference 0.000 at offset 0, against 4.7 at ±1 frame. | `reports/audit/alignment_check.json`, `logs/check_alignment.log` |
| Landmark feature unit tests (9) | pass | `logs/test_landmarks.log` |
| Pretrained models load and run on GPU | SigLIP `7fd15f06` (92.9M), CLIP `57c21647` (85.8M), mBART-cc25 `f417e556` (610.9M). No missing keys. | `reports/models_verified.json`, `configs/model_revisions.json` |
| mBART's default generation config | `num_beams=5`. The new model sets `num_beams=1` explicitly and asserts it at every decode. | `slt/model.py` |
| Trimmed vocabulary (train captions only) | 19,551 tokens. **Dev token OOV 0.47%.** Encode/decode round-trip is exact. | `cache/vocab/vocab_stats.json` |
| Model smoke test | Pretrained rows and layers copied. Padding cannot change a clip's encoding. All 3 variants run forward and backward. | `logs/smoke_model.log` |
| **Resume determinism**, SigLIP | Interrupted and resumed run vs uninterrupted run (different shard size): **bit-identical**, clean and augmented views | `logs/resume_test_siglip.log` |
| **Resume determinism**, MediaPipe | First version: 1 of 40 clips differed in a few hand frames, because a Holistic instance was shared across clips. **Fixed** with a fresh instance per clip (0.05 s). Then bit-identical across interrupt/resume and 4, 8 or 12 workers. | `logs/resume_test_landmarks.log` |

## 2. Frames per clip (T)
Measured on train at the LMDB's stored rate (`logs/frames_distribution.log`):
- Stored frames per clip: p10 79 and median 128. 71% of clips are at the 128 cap, so the trim_5 maximum is 116.
- **Covering 90% of clips without sub-sampling needs T = 116.** T = 64 covers 6.3% of clips and T = 96 covers 20%.

| T | Effective fps, p10 clip | Effective fps, median clip |
|---|---|---|
| 64 | 5.5 | 10.8 |
| 96 | 8.2 | 16.2 |

The pilot cached **all** trimmed positions (up to 116), so T is chosen at training time. The pilot compared T = 64 and T = 96 (§5).

## 3. Speed, memory and disk (measured)
| Step | Measured | Notes |
|---|---|---|
| SigLIP extraction | **295–356 frames/s** (mean about 320), peak VRAM **1.1 GiB** | 3,000 train clips × 3 views in 50.6 min; 2,000 dev clips × 1 view in 10.9 min. Shared the CPU with MediaPipe. |
| MediaPipe Holistic (12 workers, CPU) | **55–103 frames/s** per shard (mean 79 over 527k frames, 111 min) | Shared the CPU with SigLIP and training. Worker test: 8 workers 61 fps, 14 workers 68 fps. |
| Training, 3+3, batch 32, BF16 | 0.24 s/step; **3.1 GiB** peak VRAM | Projected about 12–13 min per epoch on 101k clips. Greedy evaluation of 2k dev clips takes 0.5–1.7 min. |
| Disk | SigLIP about 161 KB per clip per view; landmarks about 53 KB per clip; one 3+3 run about 1.7 GB (`best.pt` + `last.pt`) | Pilot total: caches 1.9 GB, runs 8.6 GB |

## 4. MediaPipe detection rate (per frame; `reports/pilot_detection_rates.json`)
| Group | Clips | Pose | Left hand | Right hand | Both hands | Either hand |
|---|---|---|---|---|---|---|
| 300×300, train | 2,566 | 99.7% | **90.2%** | **88.1%** | 81.8% | 96.5% |
| 300×300, dev | 1,674 | 99.2% | 89.5% | 87.8% | 81.3% | 96.1% |
| about 650×720, train | 434 | 97.9% | 87.9% | 87.5% | 81.6% | 93.8% |
| about 650×720, dev | 297 | 98.4% | 87.0% | 87.1% | 80.0% | 94.1% |
| small (flagged), dev | 29 | 31.4% | 11.7% | 11.7% | 7.6% | 15.7% |

- The 300×300 videos are **well above the 60% threshold**, so the hybrid model stays the main plan.
- The flagged small clips confirm the audit: they are bad crops.
- A missed hand also includes hands that are genuinely out of frame or at rest. The extractor cannot tell these cases apart.

## 5. Pilot training: 3,000 training clips, 1 seed, 3+3 mBART (dev = 2k subset, greedy decoding)
| Run | Best dev BLEU-4 (epoch) | BLEU-1 | chrF | Distinct outputs (of 1,972) | Top output share | Best dev loss | Shuffled − true dev loss |
|---|---|---|---|---|---|---|---|
| Visual only (SigLIP, 3 views, T64) | 0.21 (4) | 8.88 | 11.48 | 45 | 33% | 5.060 | +0.037 |
| **Hybrid (SigLIP + landmarks, T64)** | **0.26 (4)** | 9.25 | 13.17 | **431** | **6%** | 5.064 | **+0.084** |
| Landmarks only (T64) | 0.22 (2) | 7.66 | 12.24 | 128 | 29% | 5.101 | +0.015 |
| Visual only, clean view only (no augmentation) | 0.13 (3) | 4.68 | 11.83 | 156 | 12% | 5.086 | +0.014 |
| Visual only, T96 | 0.18 (2) | 9.98 | 16.40 | 36 | 85% | 5.102 | +0.008 |

**Reference point (not comparable):** the ResNet baseline on the same dev subset, trained on **101k clips**, reaches dev BLEU-4 0.59 after re-scoring (`02_baseline.md`).

**Interpretation, stated with its uncertainty.** This is 3k clips and one seed, so BLEU differences of about 0.1 are within noise. The pilot was meant to test that the pipeline works and what it costs, not quality.
- **Every run overfits after 3–4 epochs:** train loss keeps falling while dev loss rises. That is expected with 3k clips.
- **The hybrid model is the most input-dependent.** Its shuffled-features loss gap (+0.084) is the largest, 2.3× the visual-only gap. It also gives far more varied output: 431 distinct outputs with a 6% top share, against 45 and 33% for visual-only. Its best dev loss ties visual-only.
- **Augmented views look mildly useful:** better dev loss, BLEU and gap than the clean-only run.
- **T96 showed no benefit at this scale.** Dev loss was worse and outputs collapsed onto one sentence (85%). The pilot does **not** justify the extra cost of 96 frames.
- **The training pipeline works end to end:** masks, BF16, early stopping, atomic SHA-256 checkpoints that reload, and greedy decoding confirmed as beam 1.

## 6. Landmark feature layout (204 per frame; `slt/landmarks.py`)
| Slice | Content |
|---|---|
| 0:75 | Upper-body pose points 0–24 (face 0–10, shoulders, elbows, wrists, hand points, hips). Legs 25–32 are dropped. (x, y, z) are body-relative. |
| 75:135 | Left-hand shape: points 1–20 minus the wrist, divided by shoulder width, (x, y, z) |
| 135:195 | Right-hand shape: the same |
| 195:198 / 198:201 | Left / right wrist position, body-relative |
| 201:204 | Detection flags: pose, left hand, right hand (1 = detected in this frame) |

**Normalisation:**
- Pixel coordinates are used, so non-square frames are not distorted.
- Origin = the clip's median mid-shoulder point. Unit = the clip's median shoulder width.
- Missing frames are linearly interpolated over time; the nearest value is held at the ends.
- A hand that is never detected is all zeros.

**Compared with ISL-CSLRT's 233 dims:** the old code is **not on this machine** (a search for it found nothing), so I cannot give an exact field-by-field difference.
- From the brief, both keep the upper-body pose, hand shape relative to the wrist, wrist position relative to the body, interpolation and per-hand flags.
- Possible sources of the 29-dimension gap: whether pose visibility values were kept, whether the wrist point was kept inside the hand-shape block, 2D vs 3D coordinates, and the number of pose points.
- **Please share the old layout** if an exact comparison is needed.

## 7. Full-extraction options (`reports/full_projection.json`)
Rates used: SigLIP 320 fps; MediaPipe 79–100 fps. SigLIP runs on the GPU and MediaPipe on the CPU, so they run **in parallel**. Wall time ≈ the MediaPipe time.

| Option | Frames | SigLIP h | MediaPipe h | Feature disk |
|---|---|---|---|---|
| All trimmed (≤116), 2 aug views | 13.5M | 30.4 | 37.5–47.5 | 60.7 GB |
| All trimmed, 1 aug view | 13.5M | 21.0 | 37.5–47.5 | 44.2 GB |
| T96 positions, 1 aug view | 11.6M | 18.0 | 32–41 | 37.9 GB |
| **T64 positions, 1 aug view** | 8.1M | 12.5 | **22–28** | **26.3 GB** |
| T64 positions, 2 aug views | 8.1M | 18.1 | 22–28 | 36.2 GB |

**Disk budget:**
- E: has 75 GB free now; keeping 20 GB free leaves 55 GB.
- Checkpoints: about 1.7 GB per 3+3 run and about 2.5–3 GB per 6+6 run. About 8 runs means about 16–20 GB.
- With T64 and 1 aug view: 26 + 20 ≈ **46 GB, which fits**.
- Options with all trimmed frames plus 2 views do not fit.
