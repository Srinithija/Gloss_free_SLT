"""Browser preview of an uploaded video. Display only: the model always reads the original upload, never this copy.

Browsers cannot play some valid videos, such as MPEG-4 Part 2 (FMP4) in an .mp4 file. Those are re-encoded to VP8 WebM, the fastest
browser format the app's Python environment can write.
"""
import os
import tempfile

import cv2

PLAYABLE_CODECS = {'h264', 'avc1', 'vp80', 'vp90', 'vp09', 'av01'}
PLAYABLE_CONTAINERS = {'.mp4', '.m4v', '.webm'}
MAX_SIDE = 720
MAX_FRAMES = 2000


def _fourcc(cap):
    value = int(cap.get(cv2.CAP_PROP_FOURCC))
    return ''.join(chr((value >> 8 * i) & 255) for i in range(4)).lower()


def _transcode(cap, out_path):
    """Writes the frames of an open capture to VP8 WebM. Returns the number of frames written."""
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, MAX_SIDE / max(width, height, 1))
    size = (max(2, int(width * scale)) // 2 * 2, max(2, int(height * scale)) // 2 * 2)
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'VP80'), fps if fps and fps > 0 else 25, size)
    if not writer.isOpened():
        raise ValueError('the VP8 video writer is not available')
    count = 0
    try:
        while count < MAX_FRAMES:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
            count += 1
    finally:
        writer.release()
    return count


def browser_preview(data, name):
    """Returns (video bytes, MIME type) a browser can play, or (None, reason) if no preview could be made."""
    ext = os.path.splitext(name)[1].lower()
    mime = 'video/webm' if ext == '.webm' else 'video/mp4'
    with tempfile.TemporaryDirectory(prefix='isl_preview_') as tmp:
        src = os.path.join(tmp, 'source' + ext)
        with open(src, 'wb') as f:
            f.write(data)
        cap = cv2.VideoCapture(src)
        try:
            if not cap.isOpened():
                return None, 'the video could not be opened'
            if ext in PLAYABLE_CONTAINERS and _fourcc(cap) in PLAYABLE_CODECS:
                return data, mime
            out = os.path.join(tmp, 'preview.webm')
            try:
                count = _transcode(cap, out)
            except ValueError as e:
                return None, str(e)
        finally:
            cap.release()
        if not count:
            return None, 'no frames could be read from the video'
        with open(out, 'rb') as f:
            return f.read(), 'video/webm'
