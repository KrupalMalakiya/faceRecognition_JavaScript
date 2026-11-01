# FaceMatch JS &bull; Real-Time Client-Side Face Recognition &amp; Biometrics

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Engine: TensorFlow.js](https://img.shields.io/badge/Engine-TensorFlow.js-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/js)
[![Library: face--api.js](https://img.shields.io/badge/Library-face--api.js-green.svg)](https://github.com/justadudewhohacks/face-api.js)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Privacy First](https://img.shields.io/badge/Privacy-100%25%20Client--Side-blueviolet)](#-privacy--security-guarantee)
[![Version: 1.1.0](https://img.shields.io/badge/Version-1.1.0-informational.svg)](CHANGELOG.md)

> **High-performance client-side face recognition, live webcam video tracking, and 68-point facial landmark mesh localization powered by TensorFlow.js and face-api.js.**

---

## 🌐 Live Web Application

Try the production web app directly in your browser without local installation:  
👉 **[Launch FaceMatch JS Live Demo](https://krupalmalakiya.github.io/faceRecognition_JavaScript/)**

---

## ✨ Features & Capabilities

* **🔒 100% Client-Side Privacy:** All neural network inference, feature extraction, and Euclidean distance vector matching run strictly inside your browser's WebGL context. No images or biometric vectors are ever transmitted to any external server.
* **📹 Live Webcam Video Tracking:** Real-time multi-face tracking and recognition over 30 FPS video feeds with live FPS benchmarking.
* **➕ In-Browser Face Enrollment:** Dynamically enroll new face profiles directly from the UI without modifying source code.
* **💾 Biometric Vector Export / Import:** Export 128-dimensional facial embeddings to `.json` files and reload them instantly without re-processing raw images.
* **⚡ Dual Neural Detector Architecture:**
  * **SSD MobileNet V1:** High-precision Single Shot MultiBox Detector for static photos.
  * **Tiny Face Detector:** Ultra-lightweight (~190 KB) real-time detector optimized for live video feeds and low-power devices.
* **📐 68-Point Facial Landmark Mesh:** Pinpoints precise facial topography (eyes, eyebrows, nose bridge, lips, and jaw contours).
* **🎛️ Real-Time Distance Threshold Tuning:** Dynamically tune Euclidean distance thresholds (`0.10` to `0.80`) with zero-latency re-matching.
* **🧪 Pre-loaded Benchmark Samples:** Instant one-click testing with bundled multi-face group photos (`ALL.jpg` and `IMG_3641.jpg`).

---

## 🧠 Architecture & Neural Pipeline

```
                       ┌──────────────────────────────┐
                       │  Input Media (Photo / Video) │
                       └──────────────┬───────────────┘
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
        ┌─────────────────────┐               ┌─────────────────────┐
        │   SSD MobileNet V1  │               │ Tiny Face Detector  │
        │   (High Accuracy)   │               │ (Real-Time Webcam)  │
        └──────────┬──────────┘               └──────────┬──────────┘
                   └──────────────────┬──────────────────┘
                                      ▼
                      ┌────────────────────────────────┐
                      │ 68-Point Facial Landmark Net   │
                      │ (Facial Alignment & Mesh)      │
                      └───────────────┬────────────────┘
                                      ▼
                      ┌────────────────────────────────┐
                      │ Face Recognition ResNet-34 Net │
                      │ (128-Dimensional Vector Embed) │
                      └───────────────┬────────────────┘
                                      ▼
                      ┌────────────────────────────────┐
                      │ Euclidean Distance FaceMatcher │
                      │ (Distance Threshold: 0.10-0.80)│
                      └───────────────┬────────────────┘
                                      ▼
                      ┌────────────────────────────────┐
                      │ Synchronized Canvas Overlay    │
                      │ (Bounding Box, Name, Match %)  │
                      └────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option 1: Run with Node.js / NPX (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/KrupalMalakiya/faceRecognition_JavaScript.git
   cd faceRecognition_JavaScript
   ```

2. **Start the local server:**
   ```bash
   npm start
   ```
   *(or run `npx serve .`)*

3. **Open in browser:**  
   Navigate to `http://localhost:3000` (or the URL printed in terminal).

---

### Option 2: Python 3 HTTP Server

```bash
python -m http.server 8080
```
Open `http://localhost:8080` in your web browser.

---

### Option 3: VS Code Live Server Extension

1. Open this directory in **Visual Studio Code**.
2. Right-click `index.html` and select **"Open with Live Server"**.

---

## 📁 Repository Structure

```
faceRecognition_JavaScript/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml          # GitHub bug report form
│   │   └── feature_request.yml     # GitHub feature proposal form
│   ├── workflows/
│   │   └── deploy.yml              # Automated GitHub Pages CI/CD
│   └── PULL_REQUEST_TEMPLATE.md    # PR submission checklist
├── images/                         # Default labeled profile dataset
│   ├── Aditya/
│   ├── Anil/
│   ├── Archan/
│   ├── Gabriel/
│   ├── Krupal/
│   ├── Ritesh/
│   ├── Satish/
│   ├── Yash_Fullstack/
│   └── Yash_Hingu/
├── models/                         # Pre-trained TensorFlow.js neural weights
│   ├── ssd_mobilenetv1_model-*
│   ├── tiny_face_detector_model-*
│   ├── face_landmark_68_model-*
│   ├── face_recognition_model-*
│   └── mtcnn_model-*
├── Test/                           # Test group photos
│   ├── ALL.jpg
│   └── IMG_3641.jpg
├── CHANGELOG.md                    # Semantic version history
├── CODE_OF_CONDUCT.md              # Contributor Covenant v2.1
├── CONTRIBUTING.md                 # Contribution guidelines
├── LICENSE                         # MIT License
├── README.md                       # Documentation & architecture
├── face-api.min.js                 # face-api.js neural runtime
├── index.html                      # Application UI
├── package.json                    # Project configuration & npm metadata
├── script.js                       # Core application engine & webcam pipeline
├── SECURITY.md                     # Security & vulnerability reporting policy
└── style.css                       # Responsive dark glassmorphic styling
```

---

## 📦 Neural Models Specification

| Model | Underlying Architecture | Primary Function | Weight Size |
| :--- | :--- | :--- | :--- |
| **SSD MobileNet V1** | Single Shot MultiBox Detector + MobileNet | High-precision face bounding box detection | ~5.6 MB |
| **Tiny Face Detector** | Depthwise separable CNN (custom mobile) | Real-time video face detection | ~190 KB |
| **Face Landmark 68 Net** | Convolutional Landmark Localization | 68-point facial mesh mapping | ~360 KB |
| **Face Recognition Net** | ResNet-34 based deep metric embedding | 128-dimensional biometric descriptor | ~6.4 MB |

---

## 🔒 Privacy & Security Guarantee

* **Zero Cloud Data Transmission:** All face recognition, biometrics, and camera frames are processed exclusively inside your client machine's browser memory.
* **No Telemetry / No Analytics:** FaceMatch JS contains zero third-party trackers, analytics beacons, or remote logs.

---

## 🤝 Contributing

Contributions, bug reports, and optimizations are welcome! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) guide and adhere to our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## 📄 License

This software is released under the **MIT License** &bull; see the [LICENSE](LICENSE) file for complete details.

Developed with ❤️ by [Krupal Malakiya](https://github.com/KrupalMalakiya).
