"""creative_variants CLI entrypoint."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

import typer

from .analysis import timeline
from .backends.kling import KlingBackend
from .render.text_transfer import apply_text_overlay_opencv
from .settings import Settings, get_settings
from .util import ffmpeg_io
from .util.logging import emit_event, get_logger

app = typer.Typer(add_completion=False)
logger = get_logger(__name__)


@dataclass
class PipelineContext:
    input_path: Path
    output_dir: Path
    frames_ref_count: int
    verbose: bool
    settings: Settings
    num_variants: int = 5


def instantiate_backend(settings: Settings, log_dir: Path | None = None) -> KlingBackend:
    """Instantiate the Kling backend with credentials sourced from settings."""
    return KlingBackend(
        access_key=settings.kling_access_key,
        secret_key=settings.kling_secret_key,
        log_dir=log_dir,
    )


def ensure_audio(original: Path, original_meta: dict, variant: Path) -> Path:
    """Ensure the variant video carries audio if the source contained it."""
    original_has_audio = any(s.get("codec_type") == "audio" for s in original_meta.get("streams", []))
    if not original_has_audio:
        return variant
    probe = ffmpeg_io.ffprobe(variant)
    has_audio = any(s.get("codec_type") == "audio" for s in probe.get("streams", []))
    if has_audio:
        return variant
    logger.warning("Variant missing audio; remuxing from original")
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / "audio.aac"
        video_path = Path(tmpdir) / "video.mp4"
        ffmpeg_io.extract_audio(original, audio_path)
        ffmpeg_io.remux_audio(variant, audio_path, video_path)
        Path(variant).write_bytes(video_path.read_bytes())
    return variant


def quality_guard(original_meta: dict, variant: Path, tolerance: float = 0.2) -> None:
    """Validate that the variant matches the source duration, FPS, and resolution."""
    target_duration = float(original_meta["format"].get("duration", 0.0))
    variant_meta = ffmpeg_io.ffprobe(variant)
    duration = float(variant_meta["format"].get("duration", 0.0))
    fps_target = _fps_from_stream(original_meta)
    fps_variant = _fps_from_stream(variant_meta)
    width_target, height_target = _size_from_stream(original_meta)
    width_variant, height_variant = _size_from_stream(variant_meta)
    if abs(duration - target_duration) > tolerance:
        raise RuntimeError(f"Duration mismatch: target {target_duration:.2f}s vs variant {duration:.2f}s")
    if abs(fps_target - fps_variant) > 0.1:
        raise RuntimeError(f"FPS mismatch: target {fps_target} vs variant {fps_variant}")
    if (width_target, height_target) != (width_variant, height_variant):
        raise RuntimeError(
            "Resolution mismatch: "
            f"target {width_target}x{height_target} vs variant {width_variant}x{height_variant}"
        )


def _fps_from_stream(meta: dict) -> float:
    """Extract frames-per-second value from ffprobe metadata."""
    video_stream = next(s for s in meta["streams"] if s.get("codec_type") == "video")
    num, _, den = (video_stream.get("r_frame_rate") or "0/1").partition("/")
    return float(num) / float(den or "1")


def _size_from_stream(meta: dict) -> tuple[int, int]:
    """Extract resolution tuple from ffprobe metadata."""
    video_stream = next(s for s in meta["streams"] if s.get("codec_type") == "video")
    return int(video_stream["width"]), int(video_stream["height"])


@app.command()
def creative_variants(
    input: Path = typer.Option(..., "--input", exists=True, dir_okay=False, readable=True),
    output_dir: Path = typer.Option(Path("./out"), "--output_dir", file_okay=False),
    num_variants: int = typer.Option(5, "--num_variants", min=1, max=10, help="Number of AI-generated appearance variants"),
    frames_ref_count: int = typer.Option(10, "--frames_ref_count", min=1, max=30),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging"),
) -> int:
    """Primary Typer command for generating appearance variants."""
    if verbose:
        os.environ["CREATIVE_VARIANTS_LOG_LEVEL"] = "DEBUG"
    settings = get_settings()
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx = PipelineContext(
        input_path=input,
        output_dir=output_dir,
        frames_ref_count=frames_ref_count,
        verbose=verbose,
        settings=settings,
        num_variants=num_variants,
    )

    try:
        return _run_pipeline(ctx)
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"Pipeline failed: {exc}", err=True, fg=typer.colors.RED)
        logger.exception("Pipeline error")
        return 1


def _run_pipeline(ctx: PipelineContext) -> int:
    """Execute the end-to-end variant generation workflow."""
    backend_impl = instantiate_backend(ctx.settings, log_dir=ctx.output_dir)
    original_meta = ffmpeg_io.ffprobe(ctx.input_path)

    analysis_result = timeline.analyse_video(
        ctx.input_path,
        frames=ctx.frames_ref_count,
        gemini_api_key=ctx.settings.gemini_api_key,
        use_video_upload=True,
        num_variants=ctx.num_variants,
    )

    variants_to_use = analysis_result.variants[: ctx.num_variants]

    if not variants_to_use:
        typer.secho(
            "No appearance variants were returned from analysis. Check Gemini configuration.",
            err=True,
            fg=typer.colors.RED,
        )
        return 1

    variants: List[dict] = []
    temp_dir = Path(tempfile.mkdtemp(prefix="creative_variants_backend_"))

    for idx, variant_desc in enumerate(variants_to_use, start=1):
        emit_event(logger, "persona.start", name=variant_desc.name, index=idx)
        generated = backend_impl.generate_variant(  # type: ignore[attr-defined]
            ctx.input_path,
            [Path(p) for p in analysis_result.refs],
            variant_desc.prompt,
            output_dir=Path(temp_dir),
        )
        apply_output = ctx.output_dir / f"{idx:02d}_{variant_desc.name}.mp4"
        apply_text_overlay_opencv(ctx.input_path, generated, apply_output)
        ensure_audio(ctx.input_path, original_meta, apply_output)
        quality_guard(original_meta, apply_output)
        variant_meta = ffmpeg_io.ffprobe(apply_output)
        variants.append(
            {
                "name": variant_desc.name,
                "path": str(apply_output),
                "duration_s": float(variant_meta["format"].get("duration", 0.0)),
                "changes_description": variant_desc.changes_description,
                "prompt": variant_desc.prompt,
            }
        )
        emit_event(logger, "persona.complete", name=variant_desc.name, index=idx)

    report = {
        "input": {
            "path": str(ctx.input_path),
            "duration_s": float(original_meta["format"].get("duration", 0.0)),
            "fps": _fps_from_stream(original_meta),
            "size": {
                "w": _size_from_stream(original_meta)[0],
                "h": _size_from_stream(original_meta)[1],
            },
        },
        "overlay": analysis_result.overlay.model_dump(),
        "variants": variants,
    }
    report_path = ctx.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))

    # Save full Gemini analysis separately if available
    if analysis_result.gemini_analysis:
        gemini_full_report = {
            "video_analysis": analysis_result.gemini_analysis,
            "ai_generated_variants": [
                {
                    "name": var.name,
                    "changes_description": var.changes_description,
                    "prompt": var.prompt
                }
                for var in analysis_result.variants
            ]
        }
        gemini_report_path = ctx.output_dir / "gemini_analysis.json"
        gemini_report_path.write_text(json.dumps(gemini_full_report, indent=2))
        logger.info(f"Gemini analysis saved to {gemini_report_path}")
    typer.secho(f"Variants created in {ctx.output_dir}", fg=typer.colors.GREEN)
    return 0


if __name__ == "__main__":
    sys.exit(app())
