from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from src.main import app
from src.analysis.timeline import AnalysisResult, OverlayInfo, VariantDescription, VideoSize

runner = CliRunner()


class DummyBackend:
    name = "dummy"

    def generate_variant(self, input_video: Path, reference_frames, appearance_prompt: str, *, output_dir: Path) -> Path:
        output = output_dir / f"dummy_{appearance_prompt.replace(' ', '_')}.mp4"
        output.write_bytes(Path(input_video).read_bytes())
        return output


def _ffprobe_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


@pytest.fixture()
def sample_video(tmp_path: Path) -> Path:
    path = tmp_path / "sample.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=720x1280:rate=30",
        "-t",
        "2",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


@pytest.fixture()
def fake_analysis(tmp_path: Path) -> AnalysisResult:
    frame = tmp_path / "ref_000.png"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=720x1280",
            "-frames:v",
            "1",
            str(frame),
        ],
        check=True,
        capture_output=True,
    )
    variants = [
        VariantDescription(
            name=f"variant_{idx}",
            prompt=f"make_variant_{idx}",
            changes_description=f"Change description {idx}",
        )
        for idx in range(1, 6)
    ]
    return AnalysisResult(
        fps=30.0,
        size=VideoSize(w=720, h=1280),
        duration_s=2.0,
        overlay=OverlayInfo(present=True, lines=["Sample Text"], bbox_norm=[0.1, 0.1, 0.8, 0.2]),
        refs=[str(frame)],
        variants=variants,
    )


def test_cli_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_video: Path, fake_analysis: AnalysisResult) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("KLING_ACCESS_KEY", "dummy-ak")
    monkeypatch.setenv("KLING_SECRET_KEY", "dummy-sk")

    monkeypatch.setattr("src.main.instantiate_backend", lambda settings, log_dir=None: DummyBackend())
    monkeypatch.setattr("src.analysis.timeline.analyse_video", lambda *args, **kwargs: fake_analysis)

    output_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "--input",
            str(sample_video),
            "--output_dir",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0, result.stdout

    mp4s = sorted(output_dir.glob("*.mp4"))
    assert len(mp4s) == 5

    target_duration = _ffprobe_duration(sample_video)
    for variant in mp4s:
        duration = _ffprobe_duration(variant)
        assert abs(duration - target_duration) < 0.3

    report = json.loads((output_dir / "report.json").read_text())
    assert len(report["variants"]) == 5
