# Mobile App (Legacy Expo Project)

This directory contains the previous iteration of the Atlas mobile application.
It is kept for reference and for clients still targeting the legacy build
pipeline.

- **Status**: maintenance only; new feature work happens in `../mobile/`.
- **Usage**: follow the legacy README in `docs/operations/mobile_legacy.md`
  (if still required) or migrate to the primary Expo project.
- **Note**: API clients inside this project do not automatically receive the new
  H3 grid risk responses. When backporting, mirror the updates made in
  `mobile/src/api/`.
