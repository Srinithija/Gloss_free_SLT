"""Deterministic temporal sampling strategies (always ``num_frames`` indices, in temporal order).

A view is (shift, trim_start, trim_end):
- shift moves every sample point by ``shift`` times the gap between neighbouring sample points
  (shift 0 = the baseline ``np.linspace(0, n-1, T)`` rounded, exactly ``slt_vit.frames.sample_indices``);
- trim_start / trim_end drop that fraction of the clip at each end before spreading the T points.
Indices are clipped to [0, n-1]. Short clips (most have fewer than 32 frames) repeat frames, so some views of
a short clip are identical to the baseline view.
"""
import numpy as np

STRATEGIES = {
    'uniform': [(0.0, 0.0, 0.0)],
    'shift_m1/3': [(-1 / 3, 0.0, 0.0)],
    'shift_p1/3': [(1 / 3, 0.0, 0.0)],
    'trim_5': [(0.0, 0.05, 0.05)],
    'trim_10': [(0.0, 0.10, 0.10)],
    'shift3': [(-1 / 3, 0.0, 0.0), (0.0, 0.0, 0.0), (1 / 3, 0.0, 0.0)],
    'trim3': [(0.0, 0.0, 0.0), (0.0, 0.05, 0.05), (0.0, 0.10, 0.10)],
    'mixed5': [(0.0, 0.0, 0.0), (-1 / 3, 0.0, 0.0), (1 / 3, 0.0, 0.0), (0.0, 0.10, 0.0), (0.0, 0.0, 0.10)],
}


def view_indices(n, num_frames, shift=0.0, trim_start=0.0, trim_end=0.0):
    start = trim_start * (n - 1)
    end = (n - 1) * (1 - trim_end)
    pos = np.linspace(start, end, num_frames)
    if shift:
        pos = pos + shift * (end - start) / max(num_frames - 1, 1)
    return np.clip(np.round(pos), 0, n - 1).astype(np.int64)


def strategy_indices(name, n, num_frames):
    return [view_indices(n, num_frames, *v) for v in STRATEGIES[name]]
