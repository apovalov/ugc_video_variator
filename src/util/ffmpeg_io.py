"""Utilities for interacting with ffmpeg/ffprobe."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from .logging import get_logger

logger = get_logger(__name__)


class FFmpegError(RuntimeError):
    """Raised when ffmpeg/ffprobe fails."""


def run_command(args: List[str]) -> None:
    """Run a subprocess command, raising an informative error if it fails."""
    logger.debug("running command", extra={"event": "ffmpeg.exec", "cmd": " ".join(args)})
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise FFmpegError(
            "\n".join(
                [
                    f"Command failed ({result.returncode}): {' '.join(args)}",
                    f"stdout: {result.stdout.strip()}",
                    f"stderr: {result.stderr.strip()}",
                ]
            )
        )


def ffprobe(path: Path) -> Dict[str, Any]:
    """Return ffprobe metadata parsed as JSON."""
    args = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    logger.debug("probing video", extra={"event": "ffmpeg.probe", "path": str(path)})
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise FFmpegError(f"ffprobe failed: {result.stderr}")
    return json.loads(result.stdout)


def extract_frames(
    video_path: Path,
    *,
    num_frames: int,
    output_dir: Path,
) -> List[Path]:
    """Sample evenly spaced frames to PNG files and return their paths."""
    metadata = ffprobe(video_path)
    streams = metadata.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video_stream:
        raise FFmpegError("No video stream found in input.")
    duration = float(video_stream.get("duration") or metadata.get("format", {}).get("duration"))
    fps_num, _, fps_den = (video_stream.get("r_frame_rate") or "0/1").partition("/")
    fps = float(fps_num) / float(fps_den or "1")

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamps = _even_timestamps(duration, num_frames)

    frame_paths: List[Path] = []
    for idx, timestamp in enumerate(timestamps):
        frame_path = output_dir / f"ref_{idx:03d}.png"
        args = [
            "ffmpeg",
            "-y",
            "-accurate_seek",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            f"scale={video_stream.get('width')}:{video_stream.get('height')}",
            str(frame_path),
        ]
        run_command(args)
        frame_paths.append(frame_path)

    return frame_paths


def _even_timestamps(duration: float, count: int) -> List[float]:
    """Return evenly spaced timestamps across video duration."""
    if count <= 0 or math.isclose(duration, 0.0):
        return [0.0]
    step = duration / (count + 1)
    return [step * (i + 1) for i in range(count)]


def remux_audio(video_src: Path, audio_src: Path, output_path: Path) -> None:
    """Remux audio from audio_src into video_src without re-encoding."""
    args = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_src),
        "-i",
        str(audio_src),
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        str(output_path),
    ]
    run_command(args)


def ensure_properties(
    input_video: Path,
    reference_video: Path,
    output_path: Path,
    *,
    fps: float,
    width: int,
    height: int,
) -> None:
    """Force output video to match target fps/size using ffmpeg if needed."""
    args = [
        "ffmpeg",
        "-y",
        "-i",
        str(reference_video),
        "-r",
        f"{fps}",
        "-vf",
        f"scale={width}:{height}",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-an",
        str(output_path),
    ]
    run_command(args)


def extract_audio(input_video: Path, output_audio: Path) -> None:
    """Extract audio track to AAC file."""
    args = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-vn",
        "-acodec",
        "copy",
        str(output_audio),
    ]
    run_command(args)
