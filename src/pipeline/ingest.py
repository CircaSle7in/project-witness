"""Minimal v0.1 video ingest using OpenCV for frame extraction.

Extracts frames from video files at a specified FPS and returns them
as PNG-encoded bytes for downstream VLM processing.
"""

from __future__ import annotations

from pathlib import Path


def extract_frames(
    video_path: str,
    fps: float = 1.0,
) -> list[tuple[int, float, bytes]]:
    """Extract frames from a video file at the given FPS.

    Uses OpenCV for frame extraction. Each frame is returned as
    PNG-encoded bytes suitable for sending to a VLM API.

    Args:
        video_path: Path to the video file.
        fps: Frames per second to extract (default: 1.0).

    Returns:
        A list of (frame_number, timestamp_seconds, png_bytes) tuples.
        Returns an empty list if the video file is missing or unreadable.
    """
    path = Path(video_path)
    if not path.exists():
        return []

    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return []

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if video_fps <= 0 or total_frames <= 0:
            cap.release()
            return []

        frame_interval = max(1, int(video_fps / fps))
        results: list[tuple[int, float, bytes]] = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp_s = frame_idx / video_fps
                _, png_buf = cv2.imencode(".png", frame)
                results.append((frame_idx, timestamp_s, png_buf.tobytes()))

            frame_idx += 1

        cap.release()
        return results

    except ImportError:
        return []
    except Exception:
        return []
