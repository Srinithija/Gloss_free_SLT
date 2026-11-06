# 01 — iSign data audit (read-only)

**Scripts:** `scripts/audit_data.py`, `scripts/audit_resolution_groups.py`

**Logs:** `logs/audit_data.log` (685 s, 12 workers), `logs/audit_resolution_groups.log`

**Outputs:**
- `reports/audit/audit_summary.json`: all numbers below
- `manifests/audit_clips.csv`: one row per clip
- `manifests/corrupt_clips.csv`: empty, because there are no corrupt clips
- `manifests/flag_small_resolution.csv`: small-resolution clips, flagged for review
- Contact sheets: `reports/audit/contact_*.jpg`

What was checked for each of the 127,237 clips:
- The LMDB opens, and every frame key in its key list exists.
- The first, middle and last frames decode to 256×256.
- The MP4 opens, its header is read, and its first frame decodes.

## 1. Size and integrity
| Split | Clips | Source videos | Kept captions | Excluded (empty / placeholder) |
|---|---|---|---|---|
| train | 101,790 | 3,455 | 101,347 | 273 / 170 |
| dev | 12,724 | 429 | 12,653 | 44 / 27 |
| test | 12,723 | 428 | 12,627 | 74 / 22 |

- **Integrity:**
  - **0 unreadable or corrupt clips**, in either the LMDB or the MP4s.
  - The LMDB label files match the split CSVs exactly: the same uid sets and 0 text mismatches.
  - No source video is shared between train and test.
- **Exclusion rule** (same as the ResNet baseline, `slt/text.py:exclusion_reason`): an empty caption, a placeholder (`blank`, `dash`, `[music]`), or punctuation only. Excluded uids are logged per split, never dropped silently.
- **Storage:** LMDB 194.7 GB (D:), MP4s 58.6 GB (E:).

## 2. Frames, duration and fps
| | min | median | p90 | p99 | max |
|---|---|---|---|---|---|
| Original frames, train | 11 | 183 | 367 | 624 | 2,370 |
| Duration in seconds, train | 0.37 | 6.6 | 13.0 | 22.0 | 79.1 |
| Frames stored in the LMDB (cap 128) | 11 | 128 | 128 | 128 | 128 |

- **fps:** 29.97 for 65.7k clips, 25 for 57.4k, 29 for 3.8k, 30 for 0.3k, and 12 for 19 clips.
- **The 128-frame cap:** 71% of clips have more than 128 frames, so the LMDB already sub-samples them uniformly. Each clip's original frame indices are recorded in its `metadata.selected_indices`.
- Dev and test have the same distributions (see the JSON).

**Choice of T, made on train statistics only:**

| T | Median clip (6.6 s, after trim_5) | p90 clip (13 s) |
|---|---|---|
| 64 | about 10.8 fps | about 5.5 fps |
| 96 | about 16 fps, limited by the 128 stored frames | about 8 fps |

- Captions are at most 32 tokens at p99. After MaxPool(2), T=64 gives 32 encoder steps, which is enough.
- **Proposal:** T = 64 (trim_5, evenly spaced over the stored frames, padding mask for clips with fewer frames). Use T = 96 only if the pilot shows that MediaPipe's cost for T = 96 is acceptable, since MediaPipe is the bottleneck. The final T will be fixed at the pilot checkpoint.

## 3. Resolution and signer layout
| Group (original MP4) | Clips | Train / dev / test | Source videos | Typical size |
|---|---|---|---|---|
| 300×300 | 108,606 (85.4%) | 86,870 / 10,724 / 11,012 | — | 300×300 |
| Larger, not 300 square | 17,560 (13.8%) | 14,096 / 1,823 / 1,641 | 444 | about 650×720 |
| Small (min side < 256) | **1,071 (0.8%)** | 824 / 177 / 70 | 283 | about 160–240 × 190–280 |

Every LMDB frame is 256×256: the videos were resized, not cropped. Non-square sources are slightly squashed horizontally (about 0.92 aspect for 660×720).

**Signer layout (from viewing the contact sheets):**
- `contact_lmdb_{1,2,3}.jpg`: 30 random clips × 4 frames.
- `contact_mp4_midframes.jpg`: 12 clips at original resolution.
- `contact_large.jpg` and `contact_small.jpg`: 16 random clips from each group.

**Findings:**
- **The 300×300 and large groups are already signer-centred crops.**
  - One interpreter, upper body, face and both hands in frame. Mostly the ISH News studio, with a logo top-right and sometimes a subtitle bar.
  - **No picture-in-picture insets were seen**, so **no signer-crop step is needed** for these groups.
- **Exceptions seen in the 30 random clips:**
  - `9BB5MLMJZUE--19`: a crowd scene with no clear signer.
  - `Z6W_9iUhX4w--76`: a news layout with several people, interpreter in the centre.
  - MediaPipe Holistic follows a single person, so these may give landmarks for the wrong person.
- **The small group is mostly broken crops.** In 15 of 16 sampled clips the frame shows only a torso, a hand, a sleeve or background. The face is usually cut off, and sometimes there is no signer at all.
  - These are **not corrupt**: they decode fine. But the visual content does not match the caption.
  - They are listed in `manifests/flag_small_resolution.csv`.
  - **Proposal:** exclude them from **training** (logged). **Keep them in dev and test** so evaluation stays faithful to the split, and also report test results without them.
  - The pilot's MediaPipe pose-detection rate per clip will show whether other bad crops exist among the larger clips (one 310×304 example was seen).
- **Resolution for MediaPipe:**
  - 85% of clips are only 300×300. Hands are about 40–70 px wide in those frames.
  - Holistic re-crops each hand from the pose and upsamples it internally, so detection is plausible, but it has to be measured.
  - **Proposal:** run MediaPipe on the **original MP4 frames**, at 300 px or about 650×720, which is the best available. Use the same original frame indices as the visual features, taken from the LMDB `selected_indices`.
  - In the pilot, compare hand-detection rates between the MP4 frames and the 256 LMDB frames.

## 4. Text statistics (kept captions)
| | train | dev | test |
|---|---|---|---|
| Words: median / p90 / p99 / max | 10 / 18 / 24 / 59 | 10 / 18 / 24 / 66 | 10 / 18 / 24 / 57 |
| mBART sentencepiece tokens: median / p99 / max | 14 / 32 / 78 | 14 / 32 / 100 | 14 / 33 / 89 |
| Share of captions longer than 60 tokens | 0.01% | 0.04% | 0.04% |
| Vocabulary, normalised words | 33,329 (12,700 seen once) | 11,396 | 11,768 |
| Unique captions | 97.5% | 98.7% | 98.7% |
| Starts with a capital / ends with `.?!` | 77.6% / 75.7% | 78.2% / 75.9% | 79.4% / 75.7% |
| Contains a digit / non-ASCII / is one word | 20.2% / 2.9% / 2.9% | 21.0% / 2.6% / 3.0% | 19.9% / 3.2% / 3.2% |

- **The captions are subtitle segments, not always whole sentences.**
  - Fragments are common: some references are a single word or part of a sentence.
  - Many are numbers or letters from textbook-style videos, and these are among the most common train captions.
  - Casing and punctuation are inconsistent. That supports lowercase evaluation with punctuation stripped.
- **Evaluation normalisation** (`slt/text.py:normalize`):
  1. NFKC
  2. lowercase
  3. delete apostrophes
  4. turn all other Unicode punctuation into spaces
  5. collapse whitespace

  It is applied identically to hypotheses and references, for every system including the baseline.

## 5. Split overlap (how "open" the test set is)
| | dev | test |
|---|---|---|
| Captions found verbatim in train | 5.08% | 2.99% |
| Captions found in train after normalisation | 6.04% (764) | 3.92% (495) |
| Word OOV rate against the train vocabulary | 1.95% | 1.98% |

About 96% of test captions never occur in training, so this is genuinely open translation. The seen/unseen breakdown in evaluation will use the normalised match.

## 6. Signer and channel identity
- There is **no signer, channel or interpreter field.** The CSV has only uid and text.
- Source IDs: 3,371 / 414 / 415 YouTube-style 11-character IDs, plus 84 / 15 / 13 12-character hex IDs.
- The contact sheets show a **small pool of recurring ISH News interpreters** across many source videos. The split is grouped by source video, not by signer, so **the same signers appear in train and test**. Results are therefore not signer-independent.

## 7. Decisions proposed at this checkpoint
1. **Split:** use `splits/v1` as is.
2. **No signer crop.** Flag the 1,071 small-resolution clips: exclude them from training only, and report test results both with and without them.
3. **Frames:** T = 64 by default (or 96, decided by the pilot cost), with trim_5.
4. **MediaPipe** runs on the original MP4s at the LMDB's selected frame indices.
