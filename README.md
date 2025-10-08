# creative_variants

A production-leaning CLI that generates five appearance variants from a single UGC video while preserving scene composition, overlay text, and audio. The tool targets Kling for video-to-video generation and uses an OpenCV-based text transfer pipeline to preserve overlays.

## Prerequisites

- Python 3.11
- [uv](https://github.com/astral-sh/uv) (fast Python package manager)
- `ffmpeg`/`ffprobe` available on your `PATH`

Install uv if you do not have it yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

Synchronise the environment and dependencies:

```bash
uv sync
```

This creates (or refreshes) a `.venv/` and installs everything declared in `pyproject.toml`. To work inside the virtual environment manually:

```bash
source .venv/bin/activate
```

## Environment Variables

Copy the example file and fill in the required keys:

```bash
cp env.example .env
```

Set `GEMINI_API_KEY`, `KLING_ACCESS_KEY`, and `KLING_SECRET_KEY`. Optionally tweak `CREATIVE_VARIANTS_LOG_LEVEL` to override the default INFO logging level.

Additional Kling knobs:

- `KLING_EXTEND_ENABLED=1` turns on extend mode so the backend stitches clips beyond the base 10s window.
- `KLING_EXTEND_PATH` overrides the extend endpoint (default `/v1/videos/video-extend`), supporting aggregator gateways.
- `KLING_MODEL_NAME`, `KLING_ASPECT`, `KLING_MODE`, and `KLING_MAX_DURATION` refine the request payload.
- `KLING_EXTEND_STEP` (seconds) caps each extend call; otherwise the remaining duration is requested in one shot.
- `KLING_EXTEND_USE_URL=1` switches the extend payload to use `init_video_url` instead of `task_id`.
- `KLING_EXTERNAL_TASK_ID_PREFIX` seeds deterministic `external_task_id` values for idempotent retries.

Export them into your shell (for example via `source .env`) before running the CLI.

## Usage

### Main Pipeline

Run the Typer CLI through uv (which guarantees the synced environment is used):

```bash
uv run creative_variants --input path/to/video.mp4 --output_dir ./out
```

### Key options

- `--frames_ref_count 10` controls how many reference frames are sampled for Kling.
- `--num_variants 5` sets how many appearance prompts Gemini should create (and therefore how many videos Kling renders).
- `--verbose` enables DEBUG-level logging.

Outputs are written as `out/01_business_casual.mp4` ... `out/05_seasonal.mp4` plus `out/report.json` (and `out/gemini_analysis.json` when available).

## Backend Integration

### Kling

Kling authentication derives a short-lived JWT using the provided access/secret keys; every API call sends it via `Authorization: Bearer <token>`.

The backend generates an initial clip (10s max) and, when `KLING_EXTEND_ENABLED=1`, issues one or more extend jobs until the original duration is reached. Extend requests reuse the provider `video_id` alongside the latest `task_id`, can optionally continue from an `init_video_url`, and respect deterministic `external_task_id` prefixes for idempotent retries. When the provider only returns incremental segments, the pipeline downloads them and performs an ffmpeg crossfade (`xfade`/`acrossfade`) before the normal property harmonisation and OpenCV text overlay.

All endpoints can be overridden via environment variables so the same code works against Kling’s public API or aggregator proxies.

## Text Compositing

The pipeline always uses the OpenCV-based text transfer module to preserve on-screen text. It analyzes temporal variance to isolate static overlays, builds an alpha matte, and composites the text layer back onto each Kling-generated variant. No additional configuration or font management is required.

## Testing

```bash
uv run pytest
```

`tests/test_cli_smoke.py` wires the CLI end-to-end with mocked backends to ensure five outputs, consistent durations, and report generation.

## Video Analysis Pipeline

The tool uses Gemini 2.5 Pro to analyze videos for text overlays. Two modes are available:

### Full Video Upload (Recommended, Default)

```
┌──────────────┐
│ Input Video  │
└──────┬───────┘
       │
       ├─────────────────────┐
       │                     │
       ↓                     ↓
┌──────────────┐    ┌──────────────────┐
│ Extract      │    │ Upload Full Video│
│ Frames       │    │ to Gemini API    │
│ (for Kling)  │    └────────┬─────────┘
└──────────────┘             │
                             ↓
                    ┌─────────────────┐
                    │ Wait for        │
                    │ Processing      │
                    └────────┬────────┘
                             │
                             ↓
                    ┌─────────────────┐
                    │ Analyze Video   │
                    │ (Text, Bbox)    │
                    └────────┬────────┘
                             │
                             ↓
                    ┌─────────────────┐
                    │ Extract Overlay │
                    │ Metadata        │
                    └─────────────────┘
```

### Frame-by-Frame Analysis (Fallback)

Used automatically when full video upload is unavailable (e.g. missing credentials or API failure):

```
┌──────────────┐
│ Input Video  │
└──────┬───────┘
       │
       ↓
┌──────────────────┐
│ Extract Frames   │
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│ Analyze Frames   │
│ with Gemini      │
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│ Extract Overlay  │
│ Metadata         │
└──────────────────┘
```

## Development Notes

- The pipeline writes structured logs to stdout for each variant stage.
- Temporary reference frames are stored in OS temp directories; cleanups are handled by the OS.
- Full video upload to Gemini provides the most accurate text recognition; the frame-by-frame fallback is less precise but keeps the pipeline running offline.
- Gemini OCR fallback uses `pytesseract` when installed, otherwise logs a warning.
- Network calls implement minimal error handling with descriptive exceptions; consider extending with retries for production.

## Next Steps

- Surface extend-mode telemetry (segments stitched, retries) in the variant report.
- Expose crossfade configuration so operators can tune durations per provider.
- Improve overlay detection with dedicated OCR when Gemini is unavailable.
- Track cost/latency per variant in the report.
