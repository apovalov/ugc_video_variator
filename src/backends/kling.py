"""Kling backend wrapper for multi-image-to-video API."""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests
import jwt
from PIL import Image

from ..util import ffmpeg_io
from ..util.logging import get_logger
from .base import VariantBackend

logger = get_logger(__name__)


class KlingBackend(VariantBackend):
    """Call Kling's multi-image-to-video API to generate appearance variants."""

    name = "kling"

    # Official Kling AI API endpoints from documentation
    # Reference: https://app.klingai.com/global/dev/document-api/apiReference/model/multiImageToVideo
    BASE_URL = "https://api-singapore.klingai.com"
    CREATE_JOB_PATH = "/v1/videos/multi-image2video"  # Multi-image to video generation endpoint
    JOB_STATUS_PATH = "/v1/videos/multi-image2video/{job_id}"  # Status check endpoint

    def __init__(
        self,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        log_dir: Optional[Path] = None
    ) -> None:
        self.access_key = access_key or os.getenv("KLING_ACCESS_KEY")
        self.secret_key = secret_key or os.getenv("KLING_SECRET_KEY")
        if not self.access_key or not self.secret_key:
            raise ValueError(
                "Kling backend requires KLING_ACCESS_KEY and KLING_SECRET_KEY environment variables."
            )
        self._jwt_cache: Tuple[str, float] | None = None
        self.log_dir = log_dir  # Directory for storing request logs

    # pylint: disable=too-many-arguments
    def generate_variant(
        self,
        input_video: Path,
        reference_frames: List[Path],
        appearance_prompt: str,
        *,
        output_dir: Path,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        target_meta = ffmpeg_io.ffprobe(input_video)
        job_id = self._create_job(input_video, reference_frames, appearance_prompt)
        result_url = self._poll_job(job_id)
        output_path = output_dir / f"kling_{job_id}.mp4"
        self._download_file(result_url, output_path)
        self._harmonise_properties(input_video, output_path, target_meta)
        return output_path

    # pylint: enable=too-many-arguments

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._get_jwt()}"}

    def _get_jwt(self) -> str:
        """Generate JWT token according to Kling AI API specification.

        Reference: https://app.klingai.com/global/dev/document-api/apiReference/commonInfo
        """
        now = time.time()
        if self._jwt_cache:
            token, expiry = self._jwt_cache
            if now <= expiry - 30:
                return token

        headers = {
            "alg": "HS256",
            "typ": "JWT"
        }
        payload = {
            "iss": self.access_key,  # issuer = access key
            "exp": int(now + 1800),  # expires in 30 minutes
            "nbf": int(now - 5)  # not before: current time - 5s
        }
        token = jwt.encode(payload, self.secret_key, algorithm="HS256", headers=headers)
        self._jwt_cache = (token, payload["exp"])
        return token

    def _create_job(
        self,
        video: Path,
        frames: List[Path],
        prompt: str,
    ) -> str:
        """Create multi-image-to-video job with Kling API.

        Uses multi-image2video endpoint with reference frames encoded in base64.
        Reference: https://app.klingai.com/global/dev/document-api/apiReference/model/multiImageToVideo
        """
        # Get video metadata for duration
        probe = ffmpeg_io.ffprobe(video)
        video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
        duration = float(video_stream.get("duration") or probe["format"].get("duration", 5.0))

        # Select up to 4 reference frames (API limit)
        # Distribute evenly across available frames
        selected_frames = frames[:4] if len(frames) <= 4 else [
            frames[i * len(frames) // 4] for i in range(4)
        ]

        # Encode frames to base64 (optimize size if needed)
        image_list = []
        for idx, frame_path in enumerate(selected_frames, start=1):
            # Read and potentially compress image
            img = Image.open(frame_path)
            original_size = frame_path.stat().st_size

            # Resize if too large (Kling works well with smaller reference images)
            max_dimension = 1024  # Max width or height
            if img.width > max_dimension or img.height > max_dimension:
                ratio = min(max_dimension / img.width, max_dimension / img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                logger.debug(f"Resized frame {idx} from {frame_path.stat().st_size / 1024:.0f}KB")

            # Convert to JPEG with quality 85 for better compression
            buffer = io.BytesIO()
            img.convert("RGB").save(buffer, format="JPEG", quality=85, optimize=True)
            image_bytes = buffer.getvalue()
            compressed_size = len(image_bytes)

            image_data = base64.b64encode(image_bytes).decode("utf-8")
            image_list.append({"image": image_data})

            logger.debug(
                f"Frame {idx}: {original_size / 1024:.0f}KB → {compressed_size / 1024:.0f}KB "
                f"(base64: {len(image_data) / 1024:.0f}KB)"
            )

        # Prepare JSON payload for multi-image2video
        payload = {
            "model_name": "kling-v1-6",  # Latest model version
            "image_list": image_list,
            "prompt": prompt,
            "duration": str(int(min(duration, 10))),  # Kling supports 5 or 10 seconds
            "aspect_ratio": "9:16",  # Vertical video for social media
            "mode": "pro",  # "pro" for higher quality with 1080p resolution
        }

        # Log request data for debugging and audit (if log_dir is configured)
        if self.log_dir:
            self._log_request(prompt, selected_frames, payload)

        url = f"{self.BASE_URL}{self.CREATE_JOB_PATH}"
        logger.info(
            "Submitting Kling multi-image2video job",
            extra={
                "event": "kling.submit",
                "prompt": prompt[:100],
                "num_images": len(image_list),
                "duration": payload["duration"]
            }
        )

        # Calculate payload size for logging
        payload_size_mb = sys.getsizeof(json.dumps(payload)) / 1024 / 1024
        logger.info(f"Payload size: ~{payload_size_mb:.2f} MB")

        response = requests.post(
            url,
            headers={**self._headers(), "Content-Type": "application/json"},
            json=payload,
            timeout=300,  # 5 minutes for large payloads with base64 images
        )

        if response.status_code != 200:
            raise RuntimeError(f"Kling job creation failed: {response.status_code} {response.text}")

        result = response.json()

        # Handle different response formats
        if result.get("code") != 0 and result.get("code") is not None:
            raise RuntimeError(f"Kling API error: {result.get('message', 'Unknown error')}")

        job_id = result.get("data", {}).get("task_id") or result.get("task_id") or result.get("id")
        if not job_id:
            raise RuntimeError(f"Unexpected Kling create response: {result}")

        logger.info("Kling job created", extra={"event": "kling.created", "job_id": job_id})
        return str(job_id)

    def _log_request(
        self,
        prompt: str,
        selected_frames: List[Path],
        payload: dict,
    ) -> None:
        """Log Kling API request data for debugging and audit.

        Saves:
        - Reference frame images
        - Text prompt
        - Full request payload (without base64 data)
        """
        # Create kling_requests subdirectory with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        request_dir = self.log_dir / "kling_requests" / timestamp
        request_dir.mkdir(parents=True, exist_ok=True)

        # Save reference frames
        frames_dir = request_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        for idx, frame_path in enumerate(selected_frames, start=1):
            dest = frames_dir / f"frame_{idx:02d}{frame_path.suffix}"
            shutil.copy2(frame_path, dest)

        # Save prompt as text file
        prompt_file = request_dir / "prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")

        # Save full payload structure (replace base64 data with placeholders)
        payload_for_log = {
            "model_name": payload["model_name"],
            "image_list": [
                {
                    "image": f"<base64_data: {len(img['image'])} chars, ~{len(img['image']) // 1024} KB>"
                }
                for img in payload["image_list"]
            ],
            "prompt": payload["prompt"],
            "duration": payload["duration"],
            "aspect_ratio": payload["aspect_ratio"],
            "mode": payload["mode"],
        }

        # Add optional fields if present
        if "negative_prompt" in payload:
            payload_for_log["negative_prompt"] = payload["negative_prompt"]
        if "callback_url" in payload:
            payload_for_log["callback_url"] = payload["callback_url"]
        if "external_task_id" in payload:
            payload_for_log["external_task_id"] = payload["external_task_id"]

        payload_file = request_dir / "request_payload.json"
        payload_file.write_text(json.dumps(payload_for_log, indent=2), encoding="utf-8")

        logger.info(
            "Kling request logged",
            extra={
                "event": "kling.request_logged",
                "log_dir": str(request_dir),
                "num_frames": len(selected_frames)
            }
        )

    def _poll_job(self, job_id: str) -> str:
        """Poll Kling job status until completion."""
        url = f"{self.BASE_URL}{self.JOB_STATUS_PATH.format(job_id=job_id)}"

        for attempt in range(120):  # Increased timeout for video generation (10 minutes)
            response = requests.get(url, headers=self._headers(), timeout=30)
            if response.status_code != 200:
                raise RuntimeError(f"Failed to fetch Kling job: {response.status_code} {response.text}")

            result = response.json()

            # Handle response with data wrapper
            data = result.get("data", result)
            status = data.get("task_status") or data.get("status")

            logger.debug(f"Kling job {job_id} status: {status}", extra={"attempt": attempt + 1})

            if status in {"succeed", "completed", "success"}:
                # Try different possible fields for video URL
                video_url = (
                    data.get("task_result", {}).get("videos", [{}])[0].get("url")
                    or data.get("result", {}).get("video_url")
                    or data.get("video_url")
                    or data.get("works", [{}])[0].get("resource", {}).get("resource")
                )
                if video_url:
                    logger.info("Kling job completed", extra={"event": "kling.completed", "job_id": job_id})
                    return str(video_url)
                raise RuntimeError(f"Kling job completed without result URL: {result}")

            if status in {"failed", "error", "fail"}:
                error_msg = data.get("task_status_msg") or data.get("message") or "Unknown error"
                raise RuntimeError(f"Kling job failed: {error_msg}")

            # Job is still processing
            time.sleep(5)

        raise TimeoutError(f"Timed out waiting for Kling job {job_id} after 10 minutes")

    def _download_file(self, url: str, destination: Path) -> None:
        logger.info("Downloading Kling result", extra={"event": "kling.download"})
        with requests.get(url, headers=self._headers(), timeout=120, stream=True) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to download Kling asset: {resp.status_code} {resp.text}")
            with destination.open("wb") as handle:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        handle.write(chunk)

    def _harmonise_properties(self, original: Path, generated: Path, metadata: dict) -> None:
        video_stream = next(s for s in metadata["streams"] if s["codec_type"] == "video")
        fps_parts = video_stream.get("r_frame_rate", "0/1").split("/")
        fps = float(fps_parts[0]) / float(fps_parts[1]) if float(fps_parts[1]) else 0.0
        width = int(video_stream["width"])
        height = int(video_stream["height"])
        tmp_video = generated.with_suffix(".tmp.mp4")
        ffmpeg_io.ensure_properties(original, generated, tmp_video, fps=fps, width=width, height=height)
        tmp_video.replace(generated)
