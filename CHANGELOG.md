# Changelog

All notable changes to **FaceMatch JS** will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-11-01

### Added
- **Automated Attendance Logging:** Automatic real-time check-in logging (`[Time, Date, Coworker Name, Status: Present]`) upon facial recognition ($\ge 70\%$ confidence).
- **Duplicate Check-In Protection:** Prevents duplicate attendance logging for the same coworker within the same day.
- **CSV Attendance Export:** One-click spreadsheet download (`facematch-attendance-YYYY-MM-DD.csv`) for Excel and Google Sheets integration.
- **Synthesized Web Audio Chime:** Two-tone subtle audio feedback upon successful biometric check-in without external audio assets.
- **LocalStorage Persistence:** Retains the day's attendance log through browser reloads.

## [1.1.0] - 2026-11-01

### Added
- **Live Webcam Mode:** Real-time continuous face detection, landmarks tracking, and profile matching directly from user's camera feed.
- **Dynamic Face Enrollment:** In-browser modal to enroll new profiles via file upload or camera snapshot without code modifications.
- **Dual Neural Detector Engine:** Switchable detection models between SSD MobileNet V1 (high accuracy) and Tiny Face Detector (ultra-fast for webcam).
- **Embeddings Import & Export:** Export registered 128-dimensional biometric vectors to JSON and reload without reprocessing raw photos.
- **Open-Source Governance:** Added `SECURITY.md`, `CODE_OF_CONDUCT.md`, GitHub issue forms, and pull request templates.

### Changed
- Refactored UI into a responsive, accessible dark glassmorphic design (`style.css`).
- Sanitized directory paths and labels to adhere to cross-platform URL encoding standards.
- Replaced absolute `/models` URI with relative `./models` to enable seamless GitHub Pages and subpath hosting.

### Fixed
- Fixed bug where undefined descriptors from failed face detections could pollute `FaceMatcher` Euclidean calculations.
- Fixed canvas coordinate misalignment on responsive window resizing.

## [1.0.0] - 2023-06-29
- Initial implementation of client-side face recognition using `face-api.js`.
