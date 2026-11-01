# FaceMatch JS

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Engine: TensorFlow.js](https://img.shields.io/badge/Engine-TensorFlow.js-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/js)
[![Library: face--api.js](https://img.shields.io/badge/Library-face--api.js-green.svg)](https://github.com/justadudewhohacks/face-api.js)
[![Version: 1.2.0](https://img.shields.io/badge/Version-1.2.0-informational.svg)](CHANGELOG.md)

Client-side facial recognition and automated biometric attendance terminal powered by TensorFlow.js and face-api.js.

---

## Live Demo

The production application is hosted on GitHub Pages:  
[https://krupalmalakiya.github.io/faceRecognition_JavaScript/](https://krupalmalakiya.github.io/faceRecognition_JavaScript/)

---

## Overview

FaceMatch JS is a browser-based biometric attendance terminal designed to replace physical paper-and-pen signature sheets in workplaces, classrooms, and labs.

Most proprietary attendance systems rely on cloud surveillance architectures where facial images are transmitted to external servers, creating privacy risks and compliance challenges. FaceMatch JS performs all detection, alignment, feature extraction, and vector matching locally within the browser's WebGL context. No images or biometric vectors ever leave the client machine.

---

## Features

- **Automated Check-in Logging:** Identifies enrolled faces at or above 70% confidence and records check-in timestamp, date, and status.
- **Duplicate Prevention:** Automatically ignores subsequent detections of the same individual on the same calendar day.
- **CSV Attendance Export:** Generates clean daily check-in spreadsheets compatible with Microsoft Excel and Google Sheets.
- **Persistent Local Storage:** Retains daily logs across browser reloads without requiring an external database.
- **Real-Time Video Feed:** Multi-face tracking over 30 FPS webcam streams with continuous FPS measurement.
- **In-Browser Profile Enrollment:** Allows enrolling new individuals directly via photo upload without code changes.
- **Biometric Vector Import/Export:** Serializes 128-dimensional facial descriptors to JSON for instant loading.
- **Dual Neural Detectors:** Supports SSD MobileNet V1 for high-precision photo matching and Tiny Face Detector for low-latency live video feeds.
- **68-Point Facial Landmarks:** Real-time mesh visualization mapping jawline, nose, eyes, and mouth contours.

---

## Architecture

```
                       [ Input: Static Image or Video Stream ]
                                         |
                  +----------------------+----------------------+
                  |                                             |
                  v                                             v
        [ SSD MobileNet V1 ]                         [ Tiny Face Detector ]
        (High Accuracy Photo Mode)                   (Low Latency Video Mode)
                  |                                             |
                  +----------------------+----------------------+
                                         |
                                         v
                      [ 68-Point Facial Landmark Network ]
                      (Face Alignment & Geometric Mesh)
                                         |
                                         v
                      [ ResNet-34 Face Recognition Net ]
                      (128-Dimensional Vector Descriptor)
                                         |
                                         v
                      [ Euclidean Distance FaceMatcher ]
                      (Adjustable Threshold: 0.10 - 0.80)
                                         |
                                         v
                      [ Canvas Overlay & Attendance Logger ]
                      (Bounding Box, Identity, Timestamp, CSV)
```

---

## Quick Start

### Node.js (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/KrupalMalakiya/faceRecognition_JavaScript.git
   cd faceRecognition_JavaScript
   ```

2. Start the local server:
   ```bash
   npm start
   ```
   *(or run `npx serve .`)*

3. Open your browser at `http://localhost:3000`.

### Python 3

```bash
python -m http.server 8080
```
Navigate to `http://localhost:8080`.

---

## Adding Custom Profiles

New individuals can be enrolled in two ways:

### Method A: In-Browser (No Code)
1. Open the web application.
2. Click **+ Enroll Face** in the Registered Profiles panel.
3. Enter the full name and upload a reference photo.
4. The system extracts the 128-dimensional descriptor and updates the matcher immediately.

### Method B: Directory Structure
1. Add a new directory under `./images/<Name>/`.
2. Add reference photos named `1.jpg`, `2.jpg`.
3. Register the folder in `script.js`:
   ```javascript
   const DEFAULT_PROFILES = [
     // ...
     { folder: "Jane_Doe", name: "Jane Doe", count: 2 },
   ];
   ```

---

## Model Specifications

| Model | Architecture | Task | Weight Size |
| :--- | :--- | :--- | :--- |
| **SSD MobileNet V1** | Single Shot MultiBox Detector + MobileNet | Face Detection (Photos) | ~5.6 MB |
| **Tiny Face Detector** | Depthwise Separable CNN | Face Detection (Webcam) | ~190 KB |
| **Face Landmark 68 Net** | Convolutional Landmark Regressor | 68-Point Facial Mesh | ~360 KB |
| **Face Recognition Net** | ResNet-34 Metric Learning | 128-d Biometric Embeddings | ~6.4 MB |

---

## Privacy & Security

- **Client-Side Execution:** Model weights execute in-browser via WebGL. Images, video frames, and biometric vectors are stored strictly in client memory.
- **Zero Telemetry:** The application contains no tracking scripts, remote analytics, cookies, or third-party network requests.

---

## Contributing

Pull requests, bug reports, and optimizations are welcome. Please consult [CONTRIBUTING.md](CONTRIBUTING.md) and adhere to the [Code of Conduct](CODE_OF_CONDUCT.md).

---

## License

This project is licensed under the [MIT License](LICENSE).
