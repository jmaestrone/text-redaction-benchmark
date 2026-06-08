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
```

The current CLI only registers the public command surface. Detector and redaction implementations will be added in later checkpoints.
