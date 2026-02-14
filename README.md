# TreeHacks

## Raspberry Pi / Computer-vision branch changes

- Audio output disabled for Raspberry Pi builds (audio feedback moved to a future feature).
- Added depth estimation (meters) per detection and a mathematical check whether an object lies in the user's walking path (`distance_m` + `in_path` in each detection).
- New / updated config keys:
  - `camera.horizontal_fov_deg`, `camera.vertical_fov_deg` — used for more accurate distance/offset math (defaults set for Pi camera v2).
  - `detection.path_width_m` — physical width (meters) treated as the user's walking corridor.

Usage note

- Detections returned by `process_detections(...)` now include `distance_m` (approx meters) and `in_path` (boolean).
- Audio-related packages have been removed from `requirements.txt`; audio is implemented as a no-op stub in `src/audio_feedback.py` to keep the API.
- Tested with `opencv-python==4.10.0` (recommended).

## Jetson / GPU acceleration ⚡

- The code now auto-selects `cuda` when available; set `model.device` in `config.yaml` to `auto` (default) or `cuda` to force GPU.
- Depth inference (ONNXRuntime) already prefers `CUDAExecutionProvider` when installed.
- Recommended Jetson steps:
  1. Install a Jetson-compatible PyTorch wheel (follow NVIDIA Jetson PyTorch instructions).
  2. Install JetPack / TensorRT and (optionally) `onnxruntime-gpu` for faster depth inference.
  3. Set `model.device: 'cuda'` in `config.yaml` or leave as `auto`.
  4. For maximum throughput export the YOLO model to TensorRT and enable `tensorrt.enabled: true` in `config.yaml`.

If you want, I can add an automated TensorRT-export helper and a `requirements-jetson.md` with exact package commands.

