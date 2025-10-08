"""Backend interfaces for appearance variant generation."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class VariantBackend(Protocol):
    name: str

    def generate_variant(
        self,
        input_video: Path,
        reference_frames: list[Path],
        appearance_prompt: str,
        *,
        output_dir: Path,
    ) -> Path:
        """Generate a variant video and return the resulting path."""


__all__ = ["VariantBackend"]
