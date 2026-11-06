"""Landmarks for selected frames of one video (runs in env_mediapipe; called by the demo as a subprocess).

    env_mediapipe\\python.exe -I extract_frames.py --video in.mp4 --indices 0,3,7 --expected-frames 120 --out out.npz

Frames are decoded in order with OpenCV, the requested ones are centre-cropped to at most 4:3 and resized to at most
1440 px (as in extract.py), and MediaPipe Holistic (static mode, the same settings as the cache) runs on each. The
output has the same arrays as the cache (pose, left, right, present, image_hw), one row per requested index, in
the order given. frame_index holds the video frame numbers.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract import crop43, to_array  # noqa: E402


def main():
    import mediapipe as mp
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--indices', required=True)
    ap.add_argument('--expected-frames', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-side', type=int, default=1440)
    args = ap.parse_args()
    want = [int(i) for i in args.indices.split(',')]
    need = set(want)
    cap = cv2.VideoCapture(args.video)
    frames, k = {}, 0
    while k <= max(want):
        ok, img = cap.read()
        if not ok:
            break
        if k in need:
            img = crop43(img)
            s = args.max_side / max(img.shape[:2])
            if s < 1:
                img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
            frames[k] = img
        k += 1
    cap.release()
    if len(frames) != len(need):
        sys.exit(f'decoded only {k} frames; expected {args.expected_frames} (frame {max(want)} needed)')
    pose, left, right, present = [], [], [], []
    with mp.solutions.holistic.Holistic(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5) as hol:
        for i in want:
            r = hol.process(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
            pose.append(to_array(r.pose_landmarks, 33, 4))
            left.append(to_array(r.left_hand_landmarks, 21, 3))
            right.append(to_array(r.right_hand_landmarks, 21, 3))
            present.append([r.pose_landmarks is not None, r.left_hand_landmarks is not None, r.right_hand_landmarks is not None])
    hw = frames[want[0]].shape[:2]
    np.savez(args.out, pose=np.stack(pose), left=np.stack(left), right=np.stack(right), present=np.array(present, np.uint8),
             frame_index=np.array(want, np.int32), image_hw=np.array(hw, np.int32))


if __name__ == '__main__':
    main()
