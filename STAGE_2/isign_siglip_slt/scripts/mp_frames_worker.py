"""Helper for infer.py, run in envs/mediapipe: MediaPipe Holistic on given original frame indices of one MP4.

  python scripts/mp_frames_worker.py VIDEO OUT.npz IDX [IDX ...]
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("GLOG_minloglevel", "2")
import mediapipe as mp  # noqa: E402

from slt.video import decode_video_frames  # noqa: E402


def lm(lms, n, vis):
    if lms is None:
        return np.full((n, 4 if vis else 3), np.nan, np.float32)
    return np.array([[p.x, p.y, p.z, p.visibility] if vis else [p.x, p.y, p.z] for p in lms.landmark], np.float32)


def main():
    video, out, idx = sys.argv[1], sys.argv[2], [int(i) for i in sys.argv[3:]]
    frames = decode_video_frames(None, idx, video_path=video)
    P, L, R = [], [], []
    with mp.solutions.holistic.Holistic(static_image_mode=True, model_complexity=1) as h:
        for f in frames:
            r = h.process(f)
            P.append(lm(r.pose_landmarks, 33, True))
            L.append(lm(r.left_hand_landmarks, 21, False))
            R.append(lm(r.right_hand_landmarks, 21, False))
    np.savez(out, pose=np.stack(P).astype(np.float16), lh=np.stack(L).astype(np.float16),
             rh=np.stack(R).astype(np.float16), width=frames.shape[2], height=frames.shape[1])


if __name__ == "__main__":
    main()
