# 00 — Machine and existing-pipeline inspection

Date: 2026-10-30 · Read-only inspection. The only writes were this new directory, `isign_siglip_slt\`.

## 1. Hardware
| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop, **6,141 MiB VRAM**, WDDM, 40 W cap |
| Driver / CUDA | 592.82 / CUDA 13.1 (driver-level) |
| CPU | Intel i7-13620H, 10 cores / 16 threads |
| RAM | 15.7 GB |
| OS | Windows 11 Home 10.0.26200 |
| Free disk | C: 76.6 GB · D: 85.3 GB · **E: 92.7 GB** |

**Implications:**
- With 6 GB of VRAM, mBART-large must be trained on cached features with a shallow (3+3 / 6+6) model. That is the brief's design.
- End-to-end SigLIP fine-tuning (variant 6) will be tight on memory.
- 16 GB of RAM limits how many MediaPipe worker processes can run at once.

## 2. Python environments (none of them will be modified)
- `C:\Users\Revetha K\anaconda3\envs\sltbaselines` is the old pipeline's environment:
  - Python 3.9.25, torch 2.1.2+cu118, transformers 4.32.0, sacrebleu 2.2.0, lmdb 1.3.0, opencv 4.6.0
  - No `rouge_score`, no `mediapipe`.
- `GLOSS FREE SLT\.venv`: Python 3.13.0, purpose unknown. It will not be used.
- System Python 3.10 and 3.13 interpreters are available for building the new venvs.
- HF cache (`~\.cache\huggingface\hub`) already contains `facebook/mbart-large-cc25`. SigLIP and CLIP are not cached yet.

## 3. The old pipeline
- **Repository:** `sltbaselines\` (GFSLT-VLP fork). Key files:
  - `train_slt.py`, `train_vlp.py` (original end-to-end scripts)
  - `models/models.py`: `resnet`, `TemporalConv`, `TextCLIP`, `ImageCLIP`
  - `dataloader/`
  - `configs/isign/config1.yaml`
  - `scripts/prepare_isign_lmdb.py`, `scripts/create_isign_splits.py`, `scripts/create_isign_subset.py`
- **Optimized iSign re-implementation (2026-10-23):** `sltbaselines\isign_opt\`. Experiment root: `D:\iSign_Experiments\GFSLT\optimized_v2_20261023\`.
  - Frozen ImageNet ResNet-18 frame features: centre crop 224 of the 256 LMDB frames, 512-d, cached as fp16 sharded memmaps (12 GB).
  - Temporal stage: `TemporalConv` (K5-P2-K5-P2, 1024), then `V_encoder` projection with LayerNorm.
  - Text model: mBART with **3+3 layers**, initialised from `mbart_models\MBart_trimmed` layers 0/5/11, with tied embeddings and LM head.
  - `MBart_trimmed` is mBART-cc25 with the vocabulary trimmed to iSign. `en_XX` is id 19603 there and maps to official id 250004.
- **Training recipe:** AdamW (new layers 3e-4, mBART 1e-4), warmup 2k, cosine decay, label smoothing 0.1, BF16, batch 32, about 7.5 min per epoch on 101k clips.
- **Earlier defect, already documented:** the original `mytran` checkpoint used random decoder embeddings and LM head. The optimized run fixed this.
- **Metrics:** `isign_opt/metrics.py`.
  - BLEU-1..4: sacreBLEU 2.2.0, corpus-level, **case-sensitive, punctuation kept**.
  - `bleu4_lowercase`.
  - ROUGE-L: own LCS implementation, sentence-level F with β=1.2, averaged.
  - Exact match.
  - This differs from the new protocol (lowercase + strip punctuation). Section 6 explains how both will be scored identically.
- ⚠ **Generation default:** `MBart_trimmed/config.json` has `num_beams: 5`. Any `generate()` call without an explicit `num_beams=1` will silently run beam-5. The old code always passed `num_beams` explicitly. The new code must too, and must assert it.

## 4. Existing iSign results (ResNet baseline): they exist
From `D:\iSign_Experiments\GFSLT\optimized_v2_20261023\reports\FINAL_REPORT.md` and `final_eval\metrics.json`:

**Model**
- Selected model: epoch 7 of `runs\main_full` (trained on full train, 101,347 clips), saved as `final_eval\selected_model.pt`.
- Decoding: beam 4 with no repeated 3-grams, selected on dev.

**Dev (subset dev, 1,984 clips):** BLEU-4 0.62, case-sensitive.

**Test (subset test, 1,979 clips, evaluated once):**
| BLEU-1 | BLEU-4 | BLEU-4 lowercase | ROUGE-L (β=1.2) |
|---|---|---|---|
| 12.64 | 0.74 | 0.82 | 7.79 |

**Prediction files:**
- `reports\final_eval\test_predictions.csv`
- `reports\prediction_logs\{dev,test,train}_predictions.log`, with uid, reference and prediction.
- The prediction log shows a **second** test decode (BLEU-4 0.71), produced by `dump_predictions.py`. That is a repeat evaluation and will be listed as such in the ledger.

## 5. Data and LMDB (read-only)
**Raw data:** `dataset\iSign\`
- `iSign_v1.1.csv` (uid, text): 127,237 rows, sha256 `6299ed58…`
- 127,237 MP4s in `videos\iSign-videos_v1.1\`

**Original video resolution and fps** (12 random clips):
- Mostly **300×300** at 29.97 or 25 fps, some taller (for example 480×594).
- Clip length ranges from 60 to 582 frames.

**Split**
- **No official split is shipped with iSign v1.1.**
- Deterministic group split `dataset\iSign\splits\v1`:
  - seed 42, grouped by source video (4,312 groups), no group in more than one split
  - **101,790 train / 12,724 dev / 12,723 test**
- **Plan:** reuse this split unchanged as the project split. It is deterministic, documented in `metadata.json`, and is what the baseline used.

**Full LMDB:** `D:\iSign_LMDB\{train,dev,test}\<uid>\`, one LMDB environment per clip.
- **Keys:** `b"protocol"` holds the pickle protocol (4). `pickle.dumps("keys")` holds the ordered list of frame keys. Each frame key is `pickle.dumps(i)` for `i = 0..T-1`. `b"metadata"` holds a dict with `original_frame_count`, `stored_frame_count`, `sampling='uniform'`, `max_frames=128`, `jpeg_quality=80` and `selected_indices`.
- **Frames:** JPEG bytes (quality 80), **256×256**, which decode to BGR with OpenCV. The video was resized to 256×256, not cropped.
- **Up to 128 uniformly sampled frames**, with indices into the original video recorded.
- **Labels:** `D:\iSign_LMDB\labels\labels.{train,dev,test}`, a gzip-pickled dict mapping uid to `{name, text, length, original_length, …}`. Dev has 12,724 entries.
- **Size on disk:** a `du` did not finish within 2 minutes. To be measured in the audit.

**16k-subset LMDB:** `D:\iSign_LMDB_subset_16k` is **no longer on disk**. The baseline's dev/test subsets (2,000 uids each) can still be recovered exactly:
- from the shard JSONs of the ResNet cache (`cache\resnet18_frames\{dev,test}\shard_*.json`), and
- from `test_predictions.csv`.

## 6. Issues to decide (see the questions in the chat)
1. **Evaluation sets.** The baseline was scored on the 2k subset dev/test, not the full 12.7k dev/test.
2. **Metric protocol.** The baseline numbers above use case-sensitive BLEU. The new protocol needs lowercase with punctuation stripped. The existing baseline prediction files can be re-scored under the new protocol without running the model again.
3. **MediaPipe resolution.** The best source is the original 300×300 MP4s, which is only slightly better than the 256 LMDB frames. Hand-landmark quality at this size has to be measured in the pilot.
