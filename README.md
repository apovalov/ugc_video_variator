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
Endpoints in `src/backends/kling.py` are placeholders (`ENDPOINT_TODO`). Replace `BASE_URL`, `CREATE_JOB_PATH`, and `JOB_STATUS_PATH` with the official Kling API routes. The wrapper expects standard create/poll/download semantics with bearer authentication.

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

- Fill in Kling endpoint details and payload schema updates when they change.
- Harden API error handling with exponential backoff.
- Improve overlay detection with dedicated OCR when Gemini is unavailable.
- Track cost/latency per variant in the report.
