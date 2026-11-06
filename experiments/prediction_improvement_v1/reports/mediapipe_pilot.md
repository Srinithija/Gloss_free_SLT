# MediaPipe pilot and landmark cache (Phase 3)

## Environment

MediaPipe runs in its **own** environment, `slt_vit_experiment\env_mediapipe`, so the main `env` is unchanged:
- Python 3.11, mediapipe 0.10.14, numpy 1.26.4, opencv-contrib-python 4.11.0.86 (pulled in by MediaPipe);
- matplotlib pinned to 3.9.2, because the newest one fails to import without `dateutil`;
- full list in `logs/env_mediapipe_freeze.txt`.

The extraction script imports neither torch nor `slt_vit`. The main environment only reads the `.npz` files.

## Extraction

| Setting | Value |
|---|---|
| Model | Holistic (pose 33, left hand 21, right hand 21), `model_complexity=1`, detection/tracking confidence 0.5 |
| Mode | static (each frame independently). Tracking mode gave the same detection rates in the pilot (any hand 76.5% vs 75.8%), and static mode does not depend on the frame-rate subsampling of the folders |
| Input | Each frame centre-cropped to at most 4:3, as the project does (no padding or resizing; 1440×1080 for signers 3–6, 320×240 for signers 1–2), RGB |
| Order | The manifest's frame list (natural sort, the order used for training and evaluation). `frame_index` = 0..N-1 is stored and tested |
| Labels | None are read. The manifest id is used only to name the file |
| Scope | Signers 1–6 (568 clips, 15,558 frames). Signer 7 is extracted only with `--include-test` |
| Time | 1350 s on CPU (0.087 s/frame), run in parallel with GPU work |
| Failures | 0 unreadable frames, 0 clips without pose |

Command:
```
env_mediapipe\python.exe -I experiments\prediction_improvement_v1\landmarks\extract.py all --mode static
```

## Cache format (`cache/landmarks/static_1440/<clip id>.npz`)

| Array | Shape, type | Content |
|---|---|---|
| `pose` | [N,33,4] float32 | x, y as fractions of the cropped image, z, visibility. NaN if no pose |
| `left`, `right` | [N,21,3] float32 | MediaPipe's left_hand / right_hand landmarks (subject's hands), x, y, z. NaN if not detected |
| `present` | [N,3] uint8 | pose, left hand, right hand detected |
| `frame_index` | [N] int32 | position in the manifest frame list |
| `image_hw` | [2] int32 | cropped image height and width (for the aspect ratio) |

`meta.json` records versions and settings. `extraction_stats.json` has one line per clip with detection rates and time.

## Detection results (signers 1–6, all frames)

| Signer | Clips | Pose | Left hand | Right hand | Any hand |
|---|---|---|---|---|---|
| 1 (320×240) | 96 | 100% | 79% | 59% | 87% |
| 2 (320×240) | 96 | 100% | 90% | 73% | 96% |
| 3 | 97 | 100% | 24% | 56% | 59% |
| 4 | 89 | 100% | 29% | 56% | 64% |
| 5 | 96 | 100% | 24% | 59% | 63% |
| 6 | 94 | 100% | 32% | 67% | 71% |
| **All** | 568 | **100%** | 41% | 61% | 71% |

The left hand is never detected in 125 clips; no clip lacks both hands throughout.

## Are the misses real misses? (pilot, 24 clips, 937 frames)

- **Overlays:** contact sheets are in `reports/mediapipe_overlays/static_1440_pilot/` (8 frames per clip, green = left
  hand, red = right hand, yellow = arms).
- **Resting hand:** a hand that is not signing rests on the lap at the bottom edge of the frame, often partly cut off.
  Most "missing left hand" frames are this case, and losing it costs little: the pose wrist and elbow still give the
  arm position.
- **Real misses:** a raised hand in fast motion is motion-blurred and is sometimes missed, for example signer 3 "i am
  hungry" frame 14 and signer 6 "help me" frame 20.
- **Signing-space miss rate:** where the pose wrist is above mid-torso with visibility ≥ 0.5, the hand is missed in
  13.6% of frames (left, 20/147) and 16.6% (right, 101/608).
  - The worst case is signer 1 at 320×240: 32/94 right-hand misses.
  - The 1080p signers are 2–14%.
- **Handedness:** labels are consistent across signers. `right` is the subject's right hand, drawn on the image's left.

**Verdict:** tracking is reliable enough to proceed with a cached-landmark model, with these precautions:
- missing hands are interpolated between detections;
- the raw detection flags are kept as features;
- the weaker low-resolution signers are noted as a limitation.

## Normalization (`pi_exp/landmark_features.py`, 233 values per frame)

- **Aspect ratio:** x and z are multiplied by W/H, so distances are in image heights.
- **Body-relative coordinates:** origin = the clip's median mid-shoulder point; unit = the clip's median shoulder width.
  These are per clip, so the body's own movement is kept.
- **Pose:** xy and visibility of the nose, shoulders, elbows and wrists.
- **Hands:**
  - body-relative xy;
  - hand shape: xyz relative to the wrist, divided by the median wrist→middle-finger-MCP length;
  - missing frames linearly interpolated; a hand never seen is all zeros;
  - 2 detection flags.
- **Training augmentation:** one random similarity transform (scale ±10%, shift ±0.1, rotation ±5°) of the body-relative
  blocks per clip. No mirroring, because handedness matters.

Tests (`tests/test_landmarks.py`, 9 passing) cover:
- shape and finiteness;
- body normalization: shoulders centred, unit width;
- translation/scale invariance and the aspect ratio;
- interpolation and flags for missing hands, and zeros for a hand never seen;
- frame order (reversal commutes);
- augmentation touching only the body blocks;
- the real cache matching the manifest's frame count and order, NaN exactly where not detected, and no signer-7 file.
