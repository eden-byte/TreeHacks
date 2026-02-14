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

