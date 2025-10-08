"""Video analysis pipeline for reference frames and overlay metadata."""

from __future__ import annotations

import base64
import json
import os
import tempfile
from pathlib import Path
from typing import List, Optional

import requests
from pydantic import BaseModel, Field

from ..util import ffmpeg_io
from ..util.logging import get_logger

logger = get_logger(__name__)


class VideoSize(BaseModel):
    w: int = Field(..., description="Width in pixels")
    h: int = Field(..., description="Height in pixels")


class OverlayInfo(BaseModel):
    present: bool = False
    lines: List[str] = Field(default_factory=list)
    bbox_norm: Optional[List[float]] = None  # [x, y, w, h] normalized 0-1


class VariantDescription(BaseModel):
    name: str
    prompt: str
    changes_description: str


class AnalysisResult(BaseModel):
    fps: float
    size: VideoSize
    duration_s: float
    overlay: OverlayInfo
    refs: List[str]
    gemini_analysis: Optional[dict] = None  # Full Gemini response with scenes, lighting, etc.
    variants: List[VariantDescription] = Field(default_factory=list)  # AI-generated variant descriptions


class GeminiClient:
    """Thin wrapper around the Gemini multimodal API with robust JSON extraction."""

    generate_endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.5-pro:generateContent"
    )
    upload_endpoint = "https://generativelanguage.googleapis.com/upload/v1beta/files"
    files_endpoint = "https://generativelanguage.googleapis.com/v1beta/files"

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for video analysis.")
        self.api_key = api_key

    # ---------- helpers: response parsing ----------

    @staticmethod
    def _extract_json(payload: dict) -> dict:
        """
        Gemini sometimes returns multiple text parts and/or markdown fences.
        Concatenate text parts and coerce the first JSON object found.
        """
        parts = payload.get("candidates", [])[0].get("content", {}).get("parts", [])
        text = "\n".join(p.get("text", "") for p in parts if "text" in p).strip()
        return GeminiClient._coerce_to_json(text)

    @staticmethod
    def _coerce_to_json(text: str) -> dict:
        t = text.strip()
        if t.startswith("```json"):
            t = t[7:]
        if t.startswith("```"):
            t = t[3:]
        if t.endswith("```"):
            t = t[:-3]
        start = t.find("{")
        end = t.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError(f"Cannot locate JSON object in Gemini text: {text[:200]}...")
        body = t[start : end + 1]
        return json.loads(body)

    # ---------- file upload & polling ----------

    def upload_video(self, video_path: Path) -> dict[str, object]:
        """Upload video file to Gemini API and return file metadata."""
        logger.info("Uploading video to Gemini API", extra={"event": "gemini.upload.start"})

        mime_type = "video/mp4"
        display_name = video_path.name
        num_bytes = video_path.stat().st_size

        headers = {
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(num_bytes),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
        }

        payload = {"file": {"display_name": display_name}}
        params = {"key": self.api_key}

        init_response = requests.post(
            self.upload_endpoint,
            params=params,
            headers=headers,
            json=payload,
            timeout=30,
        )

        if not init_response.ok:
            raise RuntimeError(f"Failed to initiate upload: {init_response.status_code} {init_response.text}")

        upload_url = init_response.headers.get("X-Goog-Upload-URL")
        if not upload_url:
            raise RuntimeError("No upload URL returned from Gemini API")

        logger.info(
            "Uploading video data",
            extra={"event": "gemini.upload.data", "size_mb": round(num_bytes / 1024 / 1024, 2)},
        )

        upload_headers = {
            "Content-Length": str(num_bytes),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        }

        with video_path.open("rb") as video_file:
            upload_response = requests.post(
                upload_url,
                headers=upload_headers,
                data=video_file,
                timeout=300,  # 5 minutes for large videos
            )

        if not upload_response.ok:
            raise RuntimeError(f"Failed to upload video data: {upload_response.status_code} {upload_response.text}")

        file_info = upload_response.json()
        logger.info(
            "Video uploaded successfully",
            extra={"event": "gemini.upload.complete", "file": file_info.get("file", {}).get("name")},
        )
        return file_info.get("file", {})

    def wait_for_video_processing(self, file_name: str, timeout: int = 300) -> dict[str, object]:
        """Poll file status until processing is complete."""
        logger.info("Waiting for video processing", extra={"event": "gemini.processing.wait"})

        params = {"key": self.api_key}
        # file_name already contains "files/xxxxx" format, so use it as-is
        url = f"https://generativelanguage.googleapis.com/v1beta/{file_name}"

        import time
        start_time = time.time()

        while time.time() - start_time < timeout:
            response = requests.get(url, params=params, timeout=30)

            if not response.ok:
                raise RuntimeError(f"Failed to check file status: {response.status_code} {response.text}")

            file_info = response.json()
            state = file_info.get("state", "UNKNOWN")

            logger.debug("Video processing state: %s", state)

            if state == "ACTIVE":
                logger.info("Video processing complete", extra={"event": "gemini.processing.complete"})
                return file_info
            if state == "FAILED":
                raise RuntimeError(f"Video processing failed: {file_info.get('error', 'Unknown error')}")

            time.sleep(5)

        raise TimeoutError(f"Video processing timed out after {timeout} seconds")

    # ---------- analysis ----------

    def analyze_video(self, video_file_uri: string) -> dict[str, object]:  # type: ignore[name-defined]
        """Call Gemini for RICH scene-by-scene JSON (v2 schema) with overlay details."""
        prompt = (
            "You are a meticulous VIDEO & AUDIO SCENE ANNOTATOR. You will watch an entire vertical (9:16) UGC video that may contain static text overlays. "
            "Your PRIMARY goal is to extract WHAT happens in each scene: setting, actions, people, camera, lighting, and audio. "
            "Overlay text is SECONDARY but must be captured exactly.\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONE and only ONE JSON object (no extra text, no code fences), with the following schema and types:\n\n"
            "{\n"
            '  "video_summary": string,\n'
            '  "scenes": [\n'
            "    {\n"
            '      "start_s": number,\n'
            '      "end_s": number,\n'
            '      "setting": { "location": string, "background": string, "time_of_day": string|null },\n'
            '      "camera": { "framing": string, "movement": string|null, "composition_notes": string|null },\n'
            '      "lighting": { "type": string, "mood": string, "color_palette": [string] },\n'
            '      "actions": string,\n'
            '      "objects": [string],\n'
            '      "people": [\n'
            "        {\n"
            '          "role": string|null,\n'
            '          "approx_age": string,\n'
            '          "gender_presentation": string,\n'
            '          "hair": string,\n'
            '          "clothing": string,\n'
            '          "accessories": string|null,\n'
            '          "pose_gestures": string,\n'
            '          "facial_expression": string,\n'
            '          "movement": string|null\n'
            "        }\n"
            "      ],\n"
            '      "text_overlays": [\n'
            "        {\n"
            '          "text": string,\n'
            '          "bbox_norm": [number, number, number, number],\n'
            '          "style": {\n'
            '            "font_guess": string|null,\n'
            '            "fill_hex": string|null,\n'
            '            "stroke_hex": string|null,\n'
            '            "size_px_approx": number|null,\n'
            '            "align": "left"|"center"|"right"|null\n'
            "          },\n"
            '          "anim": string|null\n'
            "        }\n"
            "      ],\n"
            '      "audio": {\n'
            '        "music": string|null,\n'
            '        "sfx": string|null,\n'
            '        "speech_transcript": string|null,\n'
            '        "lang": string|null,\n'
            '        "voice_tone": string|null\n'
            "      }\n"
            "    }\n"
            "  ],\n"
            '  "overlay_present": boolean,\n'
            '  "overlay_lines": [string],\n'
            '  "overlay_bbox_norm": [number, number, number, number] | null,\n'
            '  "scene_breakdown_md": string\n'
            "}\n\n"
            "REQUIREMENTS:\n"
            "- Focus first on scene content (setting, actions, people, camera/lighting, audio). Overlay is captured faithfully but is not the main topic.\n"
            "- For each overlay, preserve exact text (casing/punctuation). If multiple overlays appear, list all in order.\n"
            "- IMPORTANT FOR overlay_lines: Split long text into NATURAL READING CHUNKS (phrases/clauses), not full sentences. "
            "Each line in overlay_lines should be a SHORT phrase that fits comfortably on one screen line (max 8-10 words). "
            "This helps with text wrapping and readability. For example, instead of one long sentence, break it into: "
            '["FIRST PHRASE HERE", "SECOND PHRASE CONTINUES", "FINAL PART OF THOUGHT"].\n'
            "- bbox_norm is [x,y,w,h] in 0..1 relative to the full frame. Choose the PRIMARY overlay region across scenes for overlay_bbox_norm "
            "(use the most frequent region; if uncertain, choose the area covering the longest on-screen duration).\n"
            '- scene_breakdown_md: produce a time-coded Markdown outline (as a string) like:\n'
            "  **0:00–0:07** — Setting..., People & Actions..., Camera/Lighting..., Text..., Audio...\n"
            "  **0:07–0:15** — ...\n"
            "- Be factual; if unknown, use null. Never add commentary outside the JSON. Do not identify real persons or brands unless they are explicitly visible in the video.\n"
            "- Output ONLY the JSON object. Do NOT wrap it in code fences.\n"
        )

        parts = [
            {"text": prompt},
            {
                "file_data": {
                    "mime_type": "video/mp4",
                    "file_uri": video_file_uri,
                }
            },
        ]

        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        payload = {"contents": [{"parts": parts}]}

        logger.info("Analyzing video with Gemini", extra={"event": "gemini.analyze.start"})

        response = requests.post(
            self.generate_endpoint,
            params=params,
            headers=headers,
            json=payload,
            timeout=120,  # Longer timeout for video analysis
        )

        if not response.ok:
            raise RuntimeError(f"Gemini API error: {response.status_code} {response.text}")

        result = self._extract_json(response.json())
        logger.info("Video analysis complete", extra={"event": "gemini.analyze.complete"})
        return result

    def analyze_frames(self, frames: List[Path]) -> dict[str, object]:
        """Fallback: multi-image analysis with the same v2 JSON schema as analyze_video."""
        logger.warning("Using fallback frame-by-frame analysis instead of video upload")

        prompt = (
            "You are a meticulous VIDEO & AUDIO SCENE ANNOTATOR. You will analyze a set of reference frames sampled from a vertical (9:16) UGC video (no full timeline available).\n"
            "Your PRIMARY goal is to infer WHAT happens overall and approximate scenes in chronological order (based on frame order). "
            "Overlay text is SECONDARY but must be captured exactly if visible.\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONE JSON object (no extra text, no code fences) with the SAME schema as the full-video mode:\n\n"
            "{\n"
            '  "video_summary": string,\n'
            '  "scenes": [\n'
            "    {\n"
            '      "start_s": number|null,\n'
            '      "end_s": number|null,\n'
            '      "setting": { "location": string, "background": string, "time_of_day": string|null },\n'
            '      "camera": { "framing": string, "movement": string|null, "composition_notes": string|null },\n'
            '      "lighting": { "type": string, "mood": string, "color_palette": [string] },\n'
            '      "actions": string,\n'
            '      "objects": [string],\n'
            '      "people": [\n'
            "        {\n"
            '          "role": string|null,\n'
            '          "approx_age": string,\n'
            '          "gender_presentation": string,\n'
            '          "hair": string,\n'
            '          "clothing": string,\n'
            '          "accessories": string|null,\n'
            '          "pose_gestures": string,\n'
            '          "facial_expression": string,\n'
            '          "movement": string|null\n'
            "        }\n"
            "      ],\n"
            '      "text_overlays": [\n'
            "        {\n"
            '          "text": string,\n'
            '          "bbox_norm": [number, number, number, number],\n'
            '          "style": {\n'
            '            "font_guess": string|null,\n'
            '            "fill_hex": string|null,\n'
            '            "stroke_hex": string|null,\n'
            '            "size_px_approx": number|null,\n'
            '            "align": "left"|"center"|"right"|null\n'
            "          },\n"
            '          "anim": string|null\n'
            "        }\n"
            "      ],\n"
            '      "audio": { "music": string|null, "sfx": string|null, "speech_transcript": string|null, "lang": string|null, "voice_tone": string|null }\n'
            "    }\n"
            "  ],\n"
            '  "overlay_present": boolean,\n'
            '  "overlay_lines": [string],\n'
            '  "overlay_bbox_norm": [number, number, number, number] | null,\n'
            '  "scene_breakdown_md": string\n'
            "}\n\n"
            "REQUIREMENTS:\n"
            "- Infer scene order from the order of frames; if exact timing is unknown, use null for start_s/end_s but keep scenes distinct.\n"
            "- Focus on scene content first (setting, actions, people, camera/lighting). Capture overlays exactly if present in frames.\n"
            "- IMPORTANT FOR overlay_lines: Split long text into NATURAL READING CHUNKS (phrases/clauses), not full sentences. "
            "Each line should be a SHORT phrase (max 8-10 words) for better text wrapping.\n"
            "- For overlay_bbox_norm, pick the most frequent/representative region among frames where text is visible.\n"
            "- Be factual; if unknown, use null. Output ONLY the JSON object (no code fences).\n"
        )
        parts = [{"text": prompt}]
        for frame in frames:
            encoded = base64.b64encode(frame.read_bytes()).decode("utf-8")
            parts.append({"inline_data": {"mime_type": "image/png", "data": encoded}})

        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        payload = {"contents": [{"parts": parts}]}

        response = requests.post(
            self.generate_endpoint,
            params=params,
            headers=headers,
            json=payload,
            timeout=60,
        )
        if not response.ok:
            raise RuntimeError(f"Gemini API error: {response.status_code} {response.text}")

        return self._extract_json(response.json())

    def generate_variants(self, video_analysis: dict, num_variants: int = 5) -> List[dict]:
        """Generate appearance variant descriptions based on the original video analysis.

        Args:
            video_analysis: The original Gemini video analysis with scenes, people, etc.
            num_variants: Number of variants to generate (default 5)

        Returns:
            List of variant descriptions with transformation prompts
        """
        prompt = f"""You analyzed a video and produced this detailed description:

{json.dumps(video_analysis, indent=2)}

Now generate {num_variants} CREATIVE APPEARANCE VARIANTS for the people in this video.

For each variant, provide:
1. A SHORT NAME (e.g., "business_professional", "casual_summer", "vintage_80s")
2. A DETAILED KLING-OPTIMIZED PROMPT that describes the transformed scene
3. A BRIEF DESCRIPTION of what changed

IMPORTANT REQUIREMENTS for the prompt:
- Start with the ORIGINAL scene description (setting, camera, lighting, objects, actions)
- Then specify the TRANSFORMATION: what changes in people's appearance (clothing, hair, accessories, style)
- Explicitly state what MUST STAY THE SAME: background, camera framing, lighting, overlay text position/content, timing, poses
- Use photorealistic, detailed descriptions
- Keep scene composition, actions, and emotional expressions identical

Return ONLY a JSON array with this schema (no markdown fences, no extra text before or after):
[
  {{
    "name": "variant_name",
    "changes_description": "Brief summary of appearance changes",
    "prompt": "Full detailed prompt for Kling API starting with scene description, then transformation, then constraints"
  }},
  ...
]

Generate {num_variants} diverse and creative variants (different styles: professional, casual, vintage, futuristic, cultural, etc.).
Output ONLY the JSON array. Do not include any explanatory text.
"""

        parts = [{"text": prompt}]
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        payload = {"contents": [{"parts": parts}]}

        logger.info("Generating appearance variants with Gemini", extra={"event": "gemini.variants.start"})

        response = requests.post(
            self.generate_endpoint,
            params=params,
            headers=headers,
            json=payload,
            timeout=60,
        )
        if not response.ok:
            raise RuntimeError(f"Gemini API error: {response.status_code} {response.text}")

        # Get raw response for debugging
        raw_response = response.json()

        # Extract text from Gemini response
        parts_data = raw_response.get("candidates", [])[0].get("content", {}).get("parts", [])
        raw_text = "\n".join(p.get("text", "") for p in parts_data if "text" in p).strip()

        logger.debug(f"Raw Gemini variants response (first 500 chars): {raw_text[:500]}")

        # Try to extract JSON array
        try:
            result = self._extract_json_array(raw_text)
        except Exception as e:
            logger.error(f"Failed to parse variants JSON. Raw text (first 1000 chars): {raw_text[:1000]}")
            raise RuntimeError(f"Failed to parse variants from Gemini: {e}") from e

        # Result should be a list
        if not isinstance(result, list):
            raise RuntimeError(f"Expected list of variants, got: {type(result)}")

        logger.info("Variants generated", extra={"event": "gemini.variants.complete", "count": len(result)})
        return result

    @staticmethod
    def _extract_json_array(text: str) -> list:
        """Extract JSON array from text that may contain markdown or extra content."""
        t = text.strip()

        # Remove markdown fences
        if t.startswith("```json"):
            t = t[7:]
        elif t.startswith("```"):
            t = t[3:]
        if t.endswith("```"):
            t = t[:-3]

        t = t.strip()

        # Find JSON array boundaries
        start = t.find("[")
        end = t.rfind("]")

        if start == -1 or end == -1 or end <= start:
            raise RuntimeError(f"Cannot locate JSON array in text: {text[:200]}...")

        body = t[start : end + 1]

        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            # Log the problematic JSON for debugging
            raise RuntimeError(f"Invalid JSON array: {e}. JSON snippet: {body[:200]}...") from e


def _generate_fallback_variants(analysis: Optional[dict], num_variants: int) -> List[VariantDescription]:
    """Generate generic fallback variants if Gemini AI generation fails.

    Args:
        analysis: The video analysis (may be None)
        num_variants: Number of variants to generate

    Returns:
        List of generic variant descriptions
    """
    # Extract scene description if available
    scene_context = ""
    if analysis and "scenes" in analysis and analysis["scenes"]:
        first_scene = analysis["scenes"][0]
        setting = first_scene.get("setting", {})
        camera = first_scene.get("camera", {})
        lighting = first_scene.get("lighting", {})

        scene_context = (
            f"Scene: {setting.get('location', 'indoor space')} with {setting.get('background', 'neutral background')}. "
            f"Camera: {camera.get('framing', 'medium shot')}, {camera.get('movement', 'static')}. "
            f"Lighting: {lighting.get('type', 'natural light')}, {lighting.get('mood', 'neutral')} mood. "
        )
    else:
        scene_context = "Scene with original setting, camera framing, and lighting. "

    # Generic style transformations
    fallback_styles = [
        {
            "name": "business_professional",
            "changes_description": "Professional business attire with neat styling",
            "style": "formal blazer, dress shirt, professional hairstyle"
        },
        {
            "name": "casual_relaxed",
            "changes_description": "Casual comfortable clothing and relaxed appearance",
            "style": "casual t-shirt or sweater, jeans, relaxed hair"
        },
        {
            "name": "athletic_sporty",
            "changes_description": "Athletic wear and active appearance",
            "style": "athletic hoodie or sportswear, sneakers, ponytail or cap"
        },
        {
            "name": "elegant_evening",
            "changes_description": "Elegant evening wear with sophisticated styling",
            "style": "elegant dress or suit, styled hair, refined accessories"
        },
        {
            "name": "street_urban",
            "changes_description": "Urban streetwear with contemporary style",
            "style": "streetwear jacket, trendy accessories, modern hairstyle"
        },
        {
            "name": "vintage_retro",
            "changes_description": "Vintage-inspired clothing and retro styling",
            "style": "vintage clothing style, classic hairstyle, retro accessories"
        },
        {
            "name": "bohemian_artistic",
            "changes_description": "Bohemian artistic style with creative flair",
            "style": "flowing bohemian clothing, artistic accessories, natural hair"
        },
    ]

    # Select requested number of variants
    selected_styles = fallback_styles[:num_variants]

    variants = []
    for style_data in selected_styles:
        prompt = (
            f"{scene_context}"
            f"Transform the person's appearance to {style_data['changes_description']}: {style_data['style']}. "
            f"KEEP IDENTICAL: background, camera framing and movement, lighting, all objects and their positions, "
            f"overlay text position and content, person's poses and timing, emotional expressions. "
            f"CHANGE ONLY: clothing, hairstyle, and accessories to match the style. "
            f"Photorealistic, detailed, natural-looking transformation."
        )

        variants.append(VariantDescription(
            name=style_data["name"],
            prompt=prompt,
            changes_description=style_data["changes_description"]
        ))

    logger.info(f"Generated {len(variants)} fallback variants", extra={"event": "analysis.fallback_variants"})
    return variants


def _ocr_fallback(frame: Path) -> tuple[List[str], Optional[List[float]]]:
    """Fallback OCR-based text detection when Gemini analysis fails."""
    try:
        import pytesseract  # type: ignore
        from PIL import Image
    except ImportError:
        logger.warning("pytesseract not available; skipping OCR fallback")
        return [], None

    image = Image.open(frame)
    width, height = image.size
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    lines: List[str] = []
    max_x: int | None = None
    max_y: int | None = None
    min_x: int | None = None
    min_y: int | None = None
    for idx, text in enumerate(data.get("text", [])):
        try:
            conf = int(float(data.get("conf", ["0"])[idx]))
        except Exception:
            conf = 0
        if not text.strip() or conf < 40:
            continue
        x = int(data["left"][idx])
        y = int(data["top"][idx])
        w = int(data["width"][idx])
        h = int(data["height"][idx])
        lines.append(text.strip())
        min_x = x if min_x is None else min(min_x, x)
        min_y = y if min_y is None else min(min_y, y)
        max_x = x + w if max_x is None else max(max_x, x + w)
        max_y = y + h if max_y is None else max(max_y, y + h)
    if not lines or min_x is None or min_y is None or max_x is None or max_y is None:
        return [], None
    raw_bbox = [
        min_x / width,
        min_y / height,
        (max_x - min_x) / width,
        (max_y - min_y) / height,
    ]
    bbox_norm = [max(0.0, min(1.0, float(v))) for v in raw_bbox]
    return lines, bbox_norm


def analyse_video(
    video_path: Path,
    *,
    frames: int = 10,
    gemini_api_key: Optional[str] = None,
    use_video_upload: bool = True,
    num_variants: int = 5,
) -> AnalysisResult:
    """Primary entrypoint for timeline analysis.

    Args:
        video_path: Path to the video file
        frames: Number of reference frames to extract (always extracted for backend generation)
        gemini_api_key: Gemini API key for video analysis
        use_video_upload: If True, upload full video to Gemini; if False, use frame-by-frame analysis
        num_variants: Number of appearance variants to generate (default 5)
    """
    gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
    logger.info("Starting analysis", extra={"event": "analysis.start", "path": str(video_path)})
    probe = ffmpeg_io.ffprobe(video_path)
    video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    fps_parts = video_stream.get("r_frame_rate", "0/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if float(fps_parts[1]) else 0.0
    duration = float(video_stream.get("duration") or probe["format"].get("duration", 0.0))
    width = int(video_stream["width"])
    height = int(video_stream["height"])

    # Always extract reference frames - needed for backend generation (Kling/Runway)
    ref_dir = Path(tempfile.mkdtemp(prefix="creative_variants_refs_"))
    frame_paths = ffmpeg_io.extract_frames(video_path, num_frames=frames, output_dir=ref_dir)

    overlay_lines: List[str] = []
    overlay_bbox_norm: Optional[List[float]] = None
    overlay_present = False

    analysis: Optional[dict] = None

    if gemini_api_key:
        try:
            gemini = GeminiClient(gemini_api_key)

            if use_video_upload:
                # Upload and analyze full video
                logger.info("Uploading full video to Gemini for analysis")
                uploaded_file = gemini.upload_video(video_path)
                file_name = uploaded_file.get("name")
                if not file_name:
                    raise RuntimeError("No file name returned from upload")

                processed_file = gemini.wait_for_video_processing(str(file_name))
                video_uri = processed_file.get("uri")
                if not video_uri:
                    raise RuntimeError("No URI returned for processed video")

                analysis = gemini.analyze_video(str(video_uri))
            else:
                # Fallback: Analyze extracted frames
                logger.info("Using frame-by-frame analysis (fallback mode)")
                analysis = gemini.analyze_frames(frame_paths)

            # Pull minimal overlay fields for backward-compat
            overlay_present = bool(analysis.get("overlay_present", False))
            overlay_lines = [str(line) for line in analysis.get("overlay_lines", []) if str(line).strip()]
            bbox_norm = analysis.get("overlay_bbox_norm")
            if isinstance(bbox_norm, list) and len(bbox_norm) == 4:
                overlay_bbox_norm = [float(x) for x in bbox_norm]

        except Exception as exc:  # noqa: BLE001
            logger.error("Gemini analysis failed: %s", exc)
            logger.warning("Falling back to OCR for text detection")
    else:
        logger.warning("GEMINI_API_KEY missing; skipping scene analysis")

    # If we have rich scenes, try to aggregate overlay info from them
    if analysis:
        try:
            # Collect overlay lines from per-scene text_overlays if top-level lines are empty
            if not overlay_lines:
                all_lines: List[str] = []
                for scene in analysis.get("scenes", []):
                    # Try both 'text_overlays' (new) and 'overlays' (old) for backward compatibility
                    overlays = scene.get("text_overlays", scene.get("overlays", []))
                    for ov in overlays:
                        txt = str(ov.get("text", "")).strip()
                        if txt:
                            for line in txt.splitlines():
                                if line.strip():
                                    all_lines.append(line.strip())
                # Deduplicate preserving order
                seen = set()
                overlay_lines = [x for x in all_lines if not (x in seen or seen.add(x))]
                if overlay_lines:
                    overlay_present = True

            # Choose primary bbox if not present at top-level
            if not overlay_bbox_norm:
                bb_top = analysis.get("overlay_bbox_norm")
                if isinstance(bb_top, list) and len(bb_top) == 4:
                    overlay_bbox_norm = [float(x) for x in bb_top]
                else:
                    # Majority bbox across scenes
                    buckets: dict[tuple[float, float, float, float], int] = {}

                    def _round4(bb: list[float]) -> tuple[float, float, float, float]:
                        return tuple(round(float(v), 4) for v in bb)

                    for scene in analysis.get("scenes", []):
                        # Try both 'text_overlays' (new) and 'overlays' (old) for backward compatibility
                        overlays = scene.get("text_overlays", scene.get("overlays", []))
                        for ov in overlays:
                            bb = ov.get("bbox_norm")
                            if isinstance(bb, list) and len(bb) == 4:
                                key = _round4(bb)
                                buckets[key] = buckets.get(key, 0) + 1
                    if buckets:
                        overlay_bbox_norm = list(max(buckets.items(), key=lambda kv: kv[1])[0])
                        overlay_present = True
        except Exception:  # noqa: BLE001
            logger.debug("rich-schema extraction failed; will rely on OCR fallback if needed")

    # OCR fallback if Gemini didn't find text or failed
    if (overlay_present and not overlay_bbox_norm) or (not overlay_lines and frame_paths):
        logger.info("Running OCR fallback for text detection")
        mid_frame = frame_paths[len(frame_paths) // 2]
        lines, bbox = _ocr_fallback(mid_frame)
        if lines and not overlay_lines:
            overlay_lines = lines
        if bbox and not overlay_bbox_norm:
            overlay_bbox_norm = bbox
            overlay_present = True

    overlay_present = overlay_present or bool(overlay_lines)

    overlay = OverlayInfo(
        present=overlay_present and bool(overlay_lines),
        lines=overlay_lines,
        bbox_norm=overlay_bbox_norm,
    )

    # Generate appearance variants using Gemini
    variants: List[VariantDescription] = []
    if analysis and gemini_api_key:
        try:
            gemini = GeminiClient(gemini_api_key)
            variants_data = gemini.generate_variants(analysis, num_variants=num_variants)
            for var_data in variants_data:
                variants.append(VariantDescription(
                    name=var_data.get("name", "unknown"),
                    prompt=var_data.get("prompt", ""),
                    changes_description=var_data.get("changes_description", "")
                ))
            logger.info(f"Generated {len(variants)} AI-powered appearance variants", extra={"event": "analysis.variants_generated"})
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to generate AI variants: {exc}")
            logger.info("Using fallback generic variants")
            # Fallback to generic variants based on analysis
            variants = _generate_fallback_variants(analysis, num_variants)

    result = AnalysisResult(
        fps=fps,
        size=VideoSize(w=width, h=height),
        duration_s=duration,
        overlay=overlay,
        refs=[str(path) for path in frame_paths],
        gemini_analysis=analysis,  # Store full Gemini response
        variants=variants,  # AI-generated variants
    )
    logger.info("Analysis complete", extra={"event": "analysis.complete"})
    return result
