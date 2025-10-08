"""Kling backend wrapper for multi-image-to-video API."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import random
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

    RETRY_STATUSES = {429, 500, 502, 503, 504}

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
        self._pending_external_task_id: Optional[str] = None

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
        extend_enabled = os.getenv("KLING_EXTEND_ENABLED", "0") == "1"
        target_meta = ffmpeg_io.ffprobe(input_video)
        target_duration = float(target_meta["format"].get("duration", 0.0))
        video_stream = next(s for s in target_meta["streams"] if s["codec_type"] == "video")
        fps_parts = (video_stream.get("r_frame_rate") or "0/1").split("/")
        fps = float(fps_parts[0]) / float(fps_parts[1]) if float(fps_parts[1]) else 0.0
        width = int(video_stream["width"])
        height = int(video_stream["height"])

        max_duration_env = self._safe_int(os.getenv("KLING_MAX_DURATION"), 10)
        base_max_duration = min(max_duration_env, 10)

        external_id = self._build_external_task_id(input_video, appearance_prompt, segment_index=0)
        job_id = self._create_job(
            input_video,
            reference_frames,
            appearance_prompt,
            external_task_id=external_id,
        )
        base_result = self._poll_job(job_id)
        result_url = str(base_result["url"])
        output_path = output_dir / f"kling_{job_id}.mp4"

        if not extend_enabled or target_duration <= base_max_duration + 1e-3:
            self._download_file(result_url, output_path)
            self._harmonise_properties(input_video, output_path, target_meta)
            return output_path

        logger.info(
            "Extend mode enabled for Kling job",
            extra={
                "event": "kling.extend.mode",
                "job_id": job_id,
                "target_duration": target_duration,
                "base_duration_limit": base_max_duration,
            },
        )

        base_path = output_dir / f"kling_{job_id}_base.mp4"
        self._download_file(result_url, base_path)
        final_clip_path = base_path
        current_duration = base_result.get("duration") or self._extract_duration(base_path)

        tolerance = 0.5
        if current_duration >= target_duration - tolerance:
            shutil.copy2(final_clip_path, output_path)
            self._harmonise_properties(input_video, output_path, target_meta)
            return output_path

        remaining = max(target_duration - current_duration, 0.0)
        extend_step = self._safe_int(os.getenv("KLING_EXTEND_STEP"), None)
        use_url_for_extend = os.getenv("KLING_EXTEND_USE_URL", "0") == "1"
        current_task_ref = job_id
        current_result_url = result_url
        current_video_id = base_result.get("video_id")
        segment_index = 1

        while remaining > tolerance:
            step_goal = remaining if extend_step is None else min(remaining, float(extend_step))
            requested_seconds = max(1, int(math.ceil(step_goal)))
            extend_external_id = self._build_external_task_id(input_video, appearance_prompt, segment_index)
            if extend_external_id:
                self._pending_external_task_id = extend_external_id

            logger.info(
                "Submitting extend segment",
                extra={
                    "event": "kling.extend.request",
                    "job_id": job_id,
                    "segment_index": segment_index,
                    "requested_seconds": requested_seconds,
                    "remaining": remaining,
                },
            )

            if use_url_for_extend:
                if not current_result_url:
                    raise RuntimeError("Extend mode configured to use video URL but none is available.")
                extend_task_id, poll_ref = self.extend_job(
                    task_id=None,
                    init_video_url=current_result_url,
                    video_id=current_video_id,
                    duration_s=requested_seconds,
                    prompt=appearance_prompt,
                )
            else:
                extend_task_id, poll_ref = self.extend_job(
                    task_id=current_task_ref,
                    init_video_url=None,
                    video_id=current_video_id,
                    duration_s=requested_seconds,
                    prompt=appearance_prompt,
                )

            segment_result = self._poll_job(poll_ref)
            segment_url = str(segment_result["url"])
            extend_path = output_dir / f"kling_{job_id}_extend_{segment_index}.mp4"
            self._download_file(segment_url, extend_path)
            extend_duration = segment_result.get("duration") or self._extract_duration(extend_path)

            if extend_duration >= target_duration - tolerance or extend_duration >= current_duration - 0.1:
                logger.info(
                    "Extend returned consolidated clip",
                    extra={
                        "event": "kling.extend.consolidated",
                        "segment_index": segment_index,
                        "duration": extend_duration,
                    },
                )
                final_clip_path = extend_path
                current_duration = extend_duration
            else:
                stitched_path = output_dir / f"kling_{job_id}_stitched_{segment_index}.mp4"
                logger.info(
                    "Stitching extend segment with crossfade",
                    extra={
                        "event": "kling.extend.stitch",
                        "segment_index": segment_index,
                        "extend_duration": extend_duration,
                        "current_duration": current_duration,
                    },
                )
                ffmpeg_io.concat_with_crossfade(
                    final_clip_path,
                    extend_path,
                    stitched_path,
                    fps=fps,
                    width=width,
                    height=height,
                )
                final_clip_path = stitched_path
                current_duration = self._extract_duration(stitched_path)

            remaining = max(target_duration - current_duration, 0.0)
            current_task_ref = extend_task_id
            current_result_url = segment_url
            current_video_id = segment_result.get("video_id") or current_video_id
            segment_index += 1

            if segment_index > 8 and remaining > tolerance:
                raise RuntimeError("Exceeded extend attempts without reaching target duration.")

        if final_clip_path != output_path:
            shutil.copy2(final_clip_path, output_path)
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
        external_task_id: str | None = None,
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
        model_name = os.getenv("KLING_MODEL_NAME", "kling-v1-6")
        aspect_ratio = os.getenv("KLING_ASPECT", "9:16")
        mode = os.getenv("KLING_MODE", "pro")
        max_duration = self._safe_int(os.getenv("KLING_MAX_DURATION"), 10)

        payload = {
            "model_name": model_name,
            "image_list": image_list,
            "prompt": prompt,
            "duration": str(int(min(duration, max_duration, 10))),
            "aspect_ratio": aspect_ratio,
            "mode": mode,
        }

        if external_task_id:
            payload["external_task_id"] = external_task_id

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

        response = self._request_with_retry(
            "post",
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

    def _poll_job(self, job_ref: str) -> dict[str, object]:
        """Poll Kling job status until completion and return structured metadata."""
        url = job_ref if job_ref.startswith("http") else f"{self.BASE_URL}{self.JOB_STATUS_PATH.format(job_id=job_ref)}"

        for attempt in range(120):  # Increased timeout for video generation (10 minutes)
            response = self._request_with_retry(
                "get",
                url,
                headers=self._headers(),
                timeout=30,
            )
            if response.status_code != 200:
                raise RuntimeError(f"Failed to fetch Kling job: {response.status_code} {response.text}")

            result = response.json()

            # Handle response with data wrapper
            data = result.get("data", result)
            status = data.get("task_status") or data.get("status")

            logger.debug(
                "Polling Kling job",
                extra={"event": "kling.poll", "job_ref": job_ref, "status": status, "attempt": attempt + 1},
            )

            if status in {"succeed", "completed", "success"}:
                videos = data.get("task_result", {}).get("videos", []) or []
                primary = videos[0] if videos else {}
                video_url = (
                    primary.get("url")
                    or data.get("result", {}).get("video_url")
                    or data.get("video_url")
                    or data.get("works", [{}])[0].get("resource", {}).get("resource")
                )
                if not video_url:
                    raise RuntimeError(f"Kling job completed without result URL: {result}")

                video_id = primary.get("id") or data.get("video_id")
                duration = self._safe_float(primary.get("duration"))
                job_data = {
                    "url": str(video_url),
                    "video_id": str(video_id) if video_id else None,
                    "duration": duration,
                    "raw": data,
                }
                logger.info(
                    "Kling job completed",
                    extra={"event": "kling.completed", "job_ref": job_ref, "video_id": video_id},
                )
                return job_data

            if status in {"failed", "error", "fail"}:
                error_msg = data.get("task_status_msg") or data.get("message") or "Unknown error"
                raise RuntimeError(f"Kling job failed: {error_msg}")

            # Job is still processing
            time.sleep(5)

        raise TimeoutError(f"Timed out waiting for Kling job {job_ref} after 10 minutes")

    def _download_file(self, url: str, destination: Path) -> None:
        logger.info("Downloading Kling result", extra={"event": "kling.download", "url": url})
        with self._request_with_retry(
            "get",
            url,
            headers=self._headers(),
            timeout=180,
            stream=True,
        ) as resp:
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

    def extend_job(
        self,
        *,
        task_id: str | None,
        init_video_url: str | None,
        video_id: str | None,
        duration_s: int,
        prompt: str,
    ) -> tuple[str, str]:
        """
        Create an extend task and return (extend_task_id, polling_url_or_task_id).

        Uses env KLING_EXTEND_PATH. Accepts either task_id or init_video_url depending on env
        KLING_EXTEND_USE_URL. Includes model_name/aspect/mode if required by provider (read from env).
        Adds optional 'external_task_id' if KLING_EXTERNAL_TASK_ID_PREFIX is set. When video_id is
        provided (recommended), it is attached for providers that require explicit parent asset ids.
        Implements robust error handling and returns a task identifier suitable for _poll_job().
        """
        extend_env = os.getenv("KLING_EXTEND_PATH")
        extend_path = extend_env or "/v1/videos/video-extend"
        cleaned_path = extend_path.rstrip("/")
        if cleaned_path.endswith("/v1/videos/extend"):
            extend_path = f"{cleaned_path[:-len('extend')]}video-extend"
            logger.info(
                "Normalising extend path to video-extend endpoint",
                extra={"event": "kling.extend.path_normalised", "original": extend_env, "normalised": extend_path},
            )
        url = extend_path if extend_path.startswith("http") else f"{self.BASE_URL}{extend_path}"

        model_name = os.getenv("KLING_MODEL_NAME", "kling-v1-6")
        aspect_ratio = os.getenv("KLING_ASPECT", "9:16")
        mode = os.getenv("KLING_MODE", "pro")
        use_url = os.getenv("KLING_EXTEND_USE_URL", "0") == "1"

        payload: dict[str, object] = {
            "model_name": model_name,
            "aspect_ratio": aspect_ratio,
            "mode": mode,
            "prompt": prompt,
            "duration": str(duration_s),
        }

        if use_url:
            if not init_video_url:
                raise ValueError("KLING_EXTEND_USE_URL=1 requires init_video_url.")
            payload["init_video_url"] = init_video_url
        else:
            if not task_id:
                raise ValueError("Extend mode requires task_id when KLING_EXTEND_USE_URL is not set.")
            payload["task_id"] = task_id

        if video_id:
            payload["video_id"] = video_id

        external_task_id = getattr(self, "_pending_external_task_id", None)
        self._pending_external_task_id = None
        if external_task_id:
            payload["external_task_id"] = external_task_id

        headers = {**self._headers(), "Content-Type": "application/json"}
        logger.info(
            "Submitting Kling extend job",
            extra={
                "event": "kling.extend.submit",
                "duration": duration_s,
                "use_url": use_url,
                "video_id": video_id,
            },
        )
        response = self._request_with_retry("post", url, headers=headers, json=payload, timeout=180)
        if response.status_code != 200:
            raise RuntimeError(f"Kling extend failed: {response.status_code} {response.text}")

        try:
            result = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError("Kling extend response was not valid JSON.") from exc

        data = result.get("data", result)
        extend_task_id = (
            data.get("task_id")
            or data.get("extend_task_id")
            or data.get("id")
        )
        poll_ref = (
            data.get("status_url")
            or data.get("polling_url")
            or data.get("task_id")
            or data.get("id")
        )

        if not extend_task_id or not poll_ref:
            raise RuntimeError(f"Unexpected Kling extend response: {result}")

        poll_ref_str = str(poll_ref)
        if not poll_ref_str.startswith("http"):
            status_base = extend_path if extend_path.startswith("http") else f"{self.BASE_URL}{extend_path}"
            poll_ref_str = f"{status_base.rstrip('/')}/{extend_task_id}"

        logger.info(
            "Kling extend task created",
            extra={
                "event": "kling.extend.created",
                "task_id": extend_task_id,
            },
        )
        return str(extend_task_id), poll_ref_str

    def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        expected_status: int = 200,
        max_attempts: int = 4,
        **kwargs,
    ) -> requests.Response:
        """Perform an HTTP request with exponential backoff on retryable failures."""
        delay = 1.0
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.request(method, url, **kwargs)
            except requests.RequestException as exc:
                last_error = exc
                if attempt == max_attempts:
                    raise RuntimeError(f"HTTP request to {url} failed") from exc
                time.sleep(delay + random.uniform(0, 0.25))
                delay *= 2
                continue

            if response.status_code == expected_status:
                return response

            if response.status_code in self.RETRY_STATUSES and attempt < max_attempts:
                response.close()
                time.sleep(delay + random.uniform(0, 0.25))
                delay *= 2
                continue

            text = self._safe_response_text(response)
            response.close()
            raise RuntimeError(f"HTTP {method.upper()} {url} failed: {response.status_code} {text}")

        raise RuntimeError(
            f"HTTP {method.upper()} {url} failed after {max_attempts} attempts"
        ) from last_error

    @staticmethod
    def _safe_response_text(response: requests.Response) -> str:
        try:
            return response.text.strip()
        except Exception:  # noqa: BLE001
            return "<unable to read response>"

    @staticmethod
    def _safe_int(value: str | None, default: Optional[int]) -> Optional[int]:
        if value is None or value == "":
            return default
        try:
            return int(value)
        except ValueError:
            return default

    @staticmethod
    def _safe_float(value: object) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _build_external_task_id(video: Path, prompt: str, segment_index: int) -> Optional[str]:
        prefix = os.getenv("KLING_EXTERNAL_TASK_ID_PREFIX")
        if not prefix:
            return None
        slug_source = f"{video.resolve()}::{prompt}::{segment_index}"
        digest = hashlib.sha1(slug_source.encode("utf-8")).hexdigest()[:10]
        return f"{prefix}-{digest}-{segment_index}"

    @staticmethod
    def _extract_duration(video: Path) -> float:
        meta = ffmpeg_io.ffprobe(video)
        return float(meta["format"].get("duration") or 0.0)
