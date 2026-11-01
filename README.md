# FaceMatch JS &bull; Client-Side AI Face Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![TensorFlow.js](https://img.shields.io/badge/Engine-TensorFlow.js-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/js)
[![face-api.js](https://img.shields.io/badge/Library-face--api.js-green.svg)](https://github.com/justadudewhohacks/face-api.js)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Privacy First](https://img.shields.io/badge/Privacy-100%25%20Client--Side-blueviolet)](#privacy--security)

> **Real-time, client-side face recognition and 68-point facial landmark detection web application powered by `face-api.js` and TensorFlow.js.**

---

## 🌐 Live Demo

Experience the live application directly in your browser without any installation:  
👉 **[Live Demo on GitHub Pages](https://krupalmalakiya.github.io/faceRecognition_JavaScript/)**

---

## ✨ Features

* **🔒 100% Client-Side Privacy:** All neural network inference, feature extraction, and Euclidean distance vector matching run locally in your browser memory. No images or biometric vectors are ever transmitted to an external server.
* **👥 Multi-Face Detection & Recognition:** Concurrently detects and identifies multiple faces in single group photographs.
* **📐 68-Point Facial Landmark Mesh:** Pinpoints detailed facial landmarks (eyes, eyebrows, nose bridge, lips, and jawline contours).
* **⚡ Real-Time Threshold Tuning:** Dynamic distance threshold slider with instant UI re-matching without needing to re-run neural network inference.
* **🧪 Pre-loaded Test Samples:** Instant one-click testing using bundled multi-face group photos (`ALL.jpg` and `IMG_3641.jpg`).
* **🎨 Modern UI / UX:** Built with a clean, responsive dark-mode interface, drag-and-drop file upload, live inference benchmarks, and status indicators.

---

## 🧠 Architecture & Neural Pipeline

```
  ┌────────────────┐
  │  Input Image   │ (User Upload / Sample Photo)
  └───────┬────────┘
          │
          ▼
  ┌─────────────────────────────────┐
  │ SSD MobileNet V1 Face Detector  │ -> Identifies face bounding boxes & confidence
  └───────┬─────────────────────────┘
          │
          ▼
  ┌─────────────────────────────────┐
  │ 68-Point Facial Landmark Net    │ -> Maps detailed facial landmarks & aligns face
  └───────┬─────────────────────────┘
          │
          ▼
  ┌─────────────────────────────────┐
  │ Face Recognition ResNet-34 Net  │ -> Extracts 128-dimensional biometric descriptor
  └───────┬─────────────────────────┘
          │
          ▼
  ┌─────────────────────────────────┐
  │ Euclidean Distance FaceMatcher  │ -> Computes distances against labeled embeddings
  └───────┬─────────────────────────┘
          │
          ▼
  ┌─────────────────────────────────┐
  │ Canvas Overlay & Classification │ -> Rendered bounding boxes, match % & metrics
  └─────────────────────────────────┘
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
   *(or use `npx serve .`)*

3. **Open in your browser:**  
   Navigate to `http://localhost:3000` (or the URL printed in your terminal).

---

### Option 2: Run with Python

If you have Python installed:

```bash
# Python 3
python -m http.server 8080
```
Open `http://localhost:8080` in your web browser.

---

### Option 3: VS Code Live Server

1. Open the project folder in **Visual Studio Code**.
2. Right-click `index.html` and click **"Open with Live Server"**.

---

## 📁 Repository Structure

```
faceRecognition_JavaScript/
├── .github/
│   └── workflows/
│       └── deploy.yml          # Automated GitHub Pages CI/CD workflow
├── images/                     # Labeled training images for each profile
│   ├── Aditya/
│   ├── Anil/
│   ├── Archan/
│   ├── Gabriel/
│   ├── Krupal/
│   ├── Ritesh/
│   ├── Satish/
│   ├── Yash_Fullstack/
│   └── Yash_Hingu/
├── models/                     # Pre-trained TensorFlow.js weight manifests & shards
│   ├── ssd_mobilenetv1_model-*
│   ├── face_landmark_68_model-*
│   ├── face_recognition_model-*
│   └── tiny_face_detector_model-*
├── Test/                       # Sample test group photos
│   ├── ALL.jpg
│   └── IMG_3641.jpg
├── CONTRIBUTING.md             # Contribution guidelines
├── LICENSE                     # MIT License
├── README.md                   # Project documentation
├── face-api.min.js             # face-api.js core bundle
├── index.html                  # Modern web application interface
├── package.json                # Project configuration and scripts
├── script.js                   # Application logic, inference & canvas rendering
└── style.css                   # Custom responsive dark-mode styling
```

---

## 🛠️ Adding Custom Faces to the Dataset

To add new people to the recognition system:

1. Create a new folder inside `./images/` with the person's name (e.g. `./images/John_Doe/`).
2. Place 2–3 clear frontal face images inside the folder, named `1.jpg`, `2.jpg`, etc.
3. Open `script.js` and add the profile to the `PROFILES` array:
   ```javascript
   const PROFILES = [
     // ... existing profiles
     { folder: "John_Doe", name: "John Doe", count: 2 },
   ];
   ```
4. Refresh the application—the engine will automatically load and compute their face descriptors on startup!

---

## 📦 Neural Models Used

| Model | Purpose | Architecture | Size |
| :--- | :--- | :--- | :--- |
| **SSD MobileNet V1** | Face Detection | Single Shot Multibox Detector with MobileNet backbone | ~5.6 MB |
| **Face Landmark 68 Net** | 68-Point Mesh | Deep convolutional landmark localization | ~360 KB |
| **Face Recognition Net** | Feature Extraction | ResNet-34 based deep metric learning model | ~6.4 MB |

---

## 🔒 Privacy & Security

* **Zero Cloud Dependency:** All computation takes place strictly on the client device using WebGL / WebAssembly via TensorFlow.js.
* **No Telemetry / No Tracking:** This project contains zero third-party tracking scripts, cookies, or remote analytics.

---

## 🤝 Contributing

Contributions, bug reports, and suggestions are welcome! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) guide to get started.

---

## 📄 License

This project is licensed under the **MIT License** &bull; see the [LICENSE](LICENSE) file for details.

Developed with ❤️ by [Krupal Malakiya](https://github.com/KrupalMalakiya).
