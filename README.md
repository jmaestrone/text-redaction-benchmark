# Text Redaction Benchmark

CLI-first workflows for offline text detection and redaction on images and videos.

This repository is separate from the face detection benchmark. The face benchmark can inform workflow shape, artifact hygiene, and reporting patterns, but this project has text-specific model choices, geometry, and privacy defaults.

## MVP Direction

- First target: offline text redaction for local images and videos.
- First detector: PaddleOCR text detection / DBNet-family models.
- Later baseline: RF-DETR text detection experiments.
- Default redaction: opaque solid masking, not blur.
- First video mode: process every frame offline.
- Later audit: OCR-after-redaction reporting.

## Local Artifacts

Generated and local-only artifacts are ignored by git:

- `input-videos/`
- `data/frames/`
- `data/predictions/`
- `runs/`
- `models/`
- `temp/`

Keep local research notes, plans, videos, model weights, predictions, redacted outputs, and run reports under ignored paths.

## CLI

```bash
uv run text-redact --help
uv run text-redact detect-text --help
uv run text-redact redact-image --help
uv run text-redact redact-video --help
```

The `detect-text` and `redact-image` commands are registered for PaddleOCR-oriented detection options and solid image masking options. The `redact-video` command runs the first artifact-writing workflow.

## PaddleOCR Detector

The first detector adapter targets PaddleOCR text detection with the default `PP-OCRv5_server_det` model. PaddleOCR is an optional dependency because model-backed local inference can download model weights and needs environment-specific PaddlePaddle setup.

```bash
uv sync --extra paddleocr
```

## Image Redaction

The first redaction implementation applies opaque solid masks to detected text polygons. Solid masking is the privacy-first default; blur and pixelation are intentionally deferred.

## Video Redaction

The first video workflow processes every frame offline, applies solid text masks, and writes local artifacts under `runs/redactions/<run-id>/`:

- `predictions/text_regions.jsonl`
- `redacted/<video-name>.mp4`
- `summary.json`
- `summary.csv`

The current video MVP writes video-only output. Audio preservation is deferred and reported as `audio_preserved: false` in the summary.

## Notebook

`notebooks/text_redaction_workflow.ipynb` is a safe wrapper around CLI commands. All `RUN_*` flags default to `False`; the notebook should not process videos, download models, or write artifacts unless a flag is explicitly enabled.
