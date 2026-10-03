"""On-demand file-backed incident media. No work occurs at import or during surveillance.

The public two-argument get_clip API deliberately hides the recording backend.
A future live-camera ring buffer can implement the same interface.
"""
from contextvars import ContextVar
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import config
from storage.repository import Repository

_reason = ContextVar('clip_unavailable_reason', default='Recording unavailable.')


def unavailable_reason():
    return _reason.get()


def ffmpeg_executable():
    executable = shutil.which('ffmpeg')
    if executable:
        return executable
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def clip_window(frame_no, fps, frame_count, pre, post, frame_base=1):
    if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
        raise ValueError('Recording has invalid FPS or duration.')
    index = frame_no - frame_base
    if index < 0 or index >= frame_count:
        raise ValueError('Incident has no valid source frame in this recording.')
    if not all(math.isfinite(v) and v >= 0 for v in (pre, post)) or pre + post <= 0:
        raise ValueError('Clip window must have a positive duration.')
    instant = index / fps
    return max(0.0, instant - pre), min(frame_count / fps, instant + post)


def get_clip(source_type, source_id):
    """Return a cached H.264 MP4 path, or None with unavailable_reason().

    Call only in response to a user request. Cache writes use unique temporary
    files and atomic replacement so concurrent sessions cannot serve partial MP4s.
    """
    temporary = None
    try:
        with Repository() as repository:
            source = repository.get_clip_source(source_type, source_id)
        raw_path = source['source_path']
        if not raw_path or '://' in raw_path:
            raise ValueError('No archived video file is available for this source.')
        path = Path(raw_path)
        if not path.is_absolute():
            path = Path(config.BASE_DIR) / path
        if not path.is_file():
            raise ValueError('Source video is missing or was deleted.')
        metadata = source['run_metadata']
        stat = path.stat()
        if (metadata.get('source_size', stat.st_size) != stat.st_size or
                metadata.get('source_mtime_ns', stat.st_mtime_ns) != stat.st_mtime_ns):
            raise ValueError('Source video was replaced after this run.')
        if source['frame_no'] is None:
            raise ValueError('This historical incident has no recorded frame number.')
        fingerprint = json.dumps([source_type, source_id, source['run_id'], source['frame_no'],
                                  str(path.resolve()), stat.st_size, stat.st_mtime_ns,
                                  metadata, config.CLIP_PRE_SECONDS, config.CLIP_POST_SECONDS,
                                  'h264-v1'], sort_keys=True)
        output = Path(config.CLIP_CACHE_DIR) / (hashlib.sha256(fingerprint.encode()).hexdigest() + '.mp4')
        if output.is_file() and output.stat().st_size > 0:
            return str(output)
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise ValueError('Source video cannot be opened; it may be corrupt.')
            fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        finally:
            cap.release()
        recorded_fps = metadata.get('source_fps', fps)
        if not math.isfinite(recorded_fps) or abs(recorded_fps - fps) > max(.01, fps * .001):
            raise ValueError('Recorded FPS does not match the source video.')
        start, end = clip_window(source['frame_no'], fps, count, config.CLIP_PRE_SECONDS,
                                 config.CLIP_POST_SECONDS, metadata.get('frame_number_base', 1))
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.mp4', delete=False) as file:
            temporary = Path(file.name)
        result = subprocess.run([
            ffmpeg_executable(), '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
            '-ss', f'{start:.9f}', '-i', str(path), '-t', f'{end-start:.9f}',
            '-map', '0:v:0', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', '-threads', '2', str(temporary)],
            capture_output=True, text=True, timeout=120,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode or temporary.stat().st_size == 0:
            raise ValueError('H.264 encoding failed: ' + (result.stderr.strip()[-300:] or 'empty output'))
        temporary.replace(output)
        return str(output)
    except Exception as exc:
        _reason.set(str(exc) or type(exc).__name__)
        return None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
