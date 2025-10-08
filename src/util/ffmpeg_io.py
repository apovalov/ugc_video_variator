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


def concat_with_crossfade(
    seg1: Path,
    seg2: Path,
    out: Path,
    *,
    fps: float,
    width: int,
    height: int,
    xfade_s: float = 0.3,
) -> None:
    """
    Creates a seamless transition between seg1 and seg2 with xfade/acrossfade,
    then encodes to H.264 yuv420p to standardize.
    """
    meta1 = ffprobe(seg1)
    meta2 = ffprobe(seg2)
    duration1 = float(meta1["format"].get("duration") or 0.0)
    duration2 = float(meta2["format"].get("duration") or 0.0)
    has_audio1 = any(s.get("codec_type") == "audio" for s in meta1.get("streams", []))
    has_audio2 = any(s.get("codec_type") == "audio" for s in meta2.get("streams", []))

    # Clamp crossfade duration to the available overlap while avoiding zero-length fades.
    effective_xfade = min(
        xfade_s,
        duration1 - 0.05 if duration1 > 0.05 else xfade_s,
        duration2 - 0.05 if duration2 > 0.05 else xfade_s,
    )
    if effective_xfade <= 0.05:
        effective_xfade = max(min(xfade_s, duration1, duration2), 0.05)
    offset = max(duration1 - effective_xfade, 0.0)

    fps_expr = f"fps={fps:.6f}," if fps > 0 else ""
    scale_expr = f"scale={width}:{height}"
    video_filter = (
        f"[0:v]{fps_expr}{scale_expr},format=yuv420p[v0];"
        f"[1:v]{fps_expr}{scale_expr},format=yuv420p[v1];"
        f"[v0][v1]xfade=transition=fade:duration={effective_xfade:.3f}:offset={offset:.3f}[vout]"
    )

    filter_parts = [video_filter]
    map_args = ["-map", "[vout]"]
    audio_args: List[str] = []

    if has_audio1 and has_audio2:
        audio_filter = (
            "[0:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[a0];"
            "[1:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[a1];"
            f"[a0][a1]acrossfade=d={effective_xfade:.3f}[aout]"
        )
        filter_parts.append(audio_filter)
        map_args.extend(["-map", "[aout]"])
        audio_args = ["-c:a", "aac", "-b:a", "192k"]
    else:
        audio_args = ["-an"]

    filter_complex = ";".join(filter_parts)

    args = [
        "ffmpeg",
        "-y",
        "-i",
        str(seg1),
        "-i",
        str(seg2),
        "-filter_complex",
        filter_complex,
        *map_args,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        *audio_args,
        str(out),
    ]
    logger.info(
        "Crossfading segments with ffmpeg",
        extra={"event": "ffmpeg.crossfade", "seg1": str(seg1), "seg2": str(seg2), "out": str(out)},
    )
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
