/**
 * FaceMatch JS (v1.1.0)
 * High-performance client-side face recognition, live webcam tracking, and 68-point landmark mesh detection.
 * Powered by face-api.js & TensorFlow.js
 *
 * Author: Krupal Malakiya
 * License: MIT
 */

// UI Elements - Navigation & Header
const statusPill = document.getElementById("statusPill");
const statusText = document.getElementById("statusText");
const tabPhoto = document.getElementById("tabPhoto");
const tabWebcam = document.getElementById("tabWebcam");
const panelPhoto = document.getElementById("panelPhoto");
const panelWebcam = document.getElementById("panelWebcam");

// UI Elements - Profiles & Actions
const profileCount = document.getElementById("profileCount");
const profileBadges = document.getElementById("profileBadges");
const btnOpenEnroll = document.getElementById("btnOpenEnroll");
const btnExportJson = document.getElementById("btnExportJson");
const btnImportJson = document.getElementById("btnImportJson");
const jsonInput = document.getElementById("jsonInput");

// UI Elements - Controls
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const btnSample1 = document.getElementById("btnSample1");
const btnSample2 = document.getElementById("btnSample2");
const btnToggleCamera = document.getElementById("btnToggleCamera");
const cameraBtnText = document.getElementById("cameraBtnText");
const modelSelector = document.getElementById("modelSelector");
const thresholdSlider = document.getElementById("thresholdSlider");
const thresholdValue = document.getElementById("thresholdValue");
const toggleLandmarks = document.getElementById("toggleLandmarks");
const toggleBoxes = document.getElementById("toggleBoxes");

// UI Elements - Viewport & Results
const viewerContainer = document.getElementById("viewerContainer");
const emptyState = document.getElementById("emptyState");
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingMessage = document.getElementById("loadingMessage");
const stageWrapper = document.getElementById("stageWrapper");

const statTotalFaces = document.getElementById("statTotalFaces");
const statMatched = document.getElementById("statMatched");
const statUnknown = document.getElementById("statUnknown");
const statSpeed = document.getElementById("statSpeed");
const statSpeedLabel = document.getElementById("statSpeedLabel");
const detectedList = document.getElementById("detectedList");

// UI Elements - Enrollment Modal & Toast
const enrollModal = document.getElementById("enrollModal");
const btnCloseEnroll = document.getElementById("btnCloseEnroll");
const btnCancelEnroll = document.getElementById("btnCancelEnroll");
const btnSubmitEnroll = document.getElementById("btnSubmitEnroll");
const enrollName = document.getElementById("enrollName");
const enrollFileInput = document.getElementById("enrollFileInput");
const toast = document.getElementById("toast");
const toastIcon = document.getElementById("toastIcon");
const toastMsg = document.getElementById("toastMsg");

// Profiles Dataset Configuration
const DEFAULT_PROFILES = [
  { folder: "Aditya", name: "Aditya", count: 2 },
  { folder: "Anil", name: "Anil", count: 2 },
  { folder: "Archan", name: "Archan", count: 2 },
  { folder: "Gabriel", name: "Gabriel", count: 2 },
  { folder: "Krupal", name: "Krupal", count: 2 },
  { folder: "Ritesh", name: "Ritesh", count: 2 },
  { folder: "Satish", name: "Satish", count: 2 },
  { folder: "Yash_Fullstack", name: "Yash Fullstack", count: 2 },
  { folder: "Yash_Hingu", name: "Yash Hingu", count: 2 },
];

// App State
let activeMode = "photo"; // "photo" | "webcam"
let labeledFaceDescriptors = [];
let faceMatcher = null;
let registeredNames = new Set();

let currentImage = null;
let currentCanvas = null;
let lastDetections = null;
let isReady = false;

// Webcam State
let webcamStream = null;
let webcamVideo = null;
let webcamAnimationId = null;
let isCameraRunning = false;
let lastFrameTime = 0;
let frameCount = 0;
let currentFps = 0;

// 1. Initialize Engine on DOM Ready
window.addEventListener("DOMContentLoaded", initEngine);

async function initEngine() {
  try {
    updateStatus("loading", "Loading Neural Models (SSD MobileNet, TinyFace, Landmarks)...");

    const MODEL_PATH = "./models";
    await Promise.all([
      faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_PATH),
      faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_PATH),
      faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_PATH),
      faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_PATH),
    ]);

    // Extract face descriptors for default dataset
    labeledFaceDescriptors = await loadInitialProfiles();

    // Initialize FaceMatcher with current threshold
    rebuildFaceMatcher();
    renderProfileBadges();

    isReady = true;
    updateStatus("ready", "System Ready for Recognition");

    // Enable interactive controls
    btnSample1.disabled = false;
    btnSample2.disabled = false;
    btnToggleCamera.disabled = false;

    setupEventListeners();
  } catch (error) {
    console.error("Initialization error:", error);
    updateStatus("loading", "Failed to load models. Check console.");
  }
}

function updateStatus(state, message) {
  statusPill.className = `status-pill ${state}`;
  statusText.textContent = message;
}

function rebuildFaceMatcher() {
  if (labeledFaceDescriptors.length > 0) {
    const threshold = parseFloat(thresholdSlider.value);
    faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, threshold);
  }
}

function renderProfileBadges() {
  profileCount.textContent = registeredNames.size;
  profileBadges.innerHTML = Array.from(registeredNames)
    .sort()
    .map((name) => `<span class="profile-tag">${escapeHtml(name)}</span>`)
    .join("");
}

// 2. Load Initial Labeled Images
async function loadInitialProfiles() {
  const descriptors = [];
  registeredNames.clear();

  for (let idx = 0; idx < DEFAULT_PROFILES.length; idx++) {
    const profile = DEFAULT_PROFILES[idx];
    updateStatus(
      "loading",
      `Computing embeddings: ${profile.name} (${idx + 1}/${DEFAULT_PROFILES.length})...`
    );

    const descriptions = [];

    for (let i = 1; i <= profile.count; i++) {
      try {
        const imgUrl = `./images/${profile.folder}/${i}.jpg`;
        const img = await faceapi.fetchImage(imgUrl);

        const detection = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();

        if (detection && detection.descriptor) {
          descriptions.push(detection.descriptor);
        }
      } catch (err) {
        console.warn(`[FaceMatch] Could not load image for ${profile.name} (${i}.jpg):`, err);
      }
    }

    if (descriptions.length > 0) {
      descriptors.push(new faceapi.LabeledFaceDescriptors(profile.name, descriptions));
      registeredNames.add(profile.name);
    }
  }

  return descriptors;
}

// 3. Get Selected Detector Options
function getDetectorOptions() {
  const selectedModel = modelSelector.value;
  if (selectedModel === "tinyFaceDetector") {
    return new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.5 });
  }
  return new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 });
}

// 4. Static Photo Processing Pipeline
async function processImageSource(src, isFile = false) {
  if (!isReady) {
    showToast("Neural engine is still initializing.", "warning");
    return;
  }

  showLoading("Analyzing facial embeddings...");

  try {
    const img = new Image();
    img.crossOrigin = "anonymous";

    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = () => reject(new Error("Unable to load the requested image."));
      img.src = src;
    });

    currentImage = img;

    // Reset stage
    stageWrapper.innerHTML = "";
    stageWrapper.style.display = "inline-block";
    emptyState.style.display = "none";

    // Append image & create synced canvas
    stageWrapper.appendChild(img);
    currentCanvas = faceapi.createCanvasFromMedia(img);
    stageWrapper.appendChild(currentCanvas);

    const displaySize = {
      width: img.naturalWidth || img.width,
      height: img.naturalHeight || img.height,
    };
    faceapi.matchDimensions(currentCanvas, displaySize);

    // Run inference with selected model
    const startTime = performance.now();
    const detections = await faceapi
      .detectAllFaces(img, getDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptors();
    const inferenceTime = Math.round(performance.now() - startTime);

    lastDetections = faceapi.resizeResults(detections, displaySize);
    statSpeedLabel.textContent = "Inference Time";
    renderDetections(inferenceTime);
  } catch (err) {
    console.error("Error processing image:", err);
    showToast("Failed to process image.", "error");
  } finally {
    hideLoading();
  }
}

// 5. Render Detections Overlay & Summary
function renderDetections(inferenceTime = null) {
  if (!currentCanvas || !lastDetections || !faceMatcher) return;

  const ctx = currentCanvas.getContext("2d");
  ctx.clearRect(0, 0, currentCanvas.width, currentCanvas.height);

  const results = lastDetections.map((d) => faceMatcher.findBestMatch(d.descriptor));

  let matchedCount = 0;
  let unknownCount = 0;
  const detectionSummaries = [];

  results.forEach((match, i) => {
    const box = lastDetections[i].detection.box;
    const isUnknown = match.label === "unknown";

    if (isUnknown) {
      unknownCount++;
    } else {
      matchedCount++;
    }

    const distance = match.distance;
    const confidencePct = Math.max(0, Math.min(100, Math.round((1 - distance) * 100)));
    const displayName = isUnknown ? "Unknown Face" : match.label;
    const badgeText = `${displayName} (${confidencePct}%)`;

    detectionSummaries.push({
      name: displayName,
      isUnknown,
      distance: distance.toFixed(2),
      confidence: confidencePct,
    });

    if (toggleBoxes.checked) {
      const boxColor = isUnknown ? "#ef4444" : "#10b981";
      const drawBox = new faceapi.draw.DrawBox(box, {
        label: badgeText,
        boxColor: boxColor,
        drawOptions: {
          lineWidth: 3,
          fontSize: 16,
          fontStyle: "600 Inter, sans-serif",
        },
      });
      drawBox.draw(currentCanvas);
    }
  });

  if (toggleLandmarks.checked) {
    faceapi.draw.drawFaceLandmarks(currentCanvas, lastDetections);
  }

  // Update Summary Dashboard
  statTotalFaces.textContent = lastDetections.length;
  statMatched.textContent = matchedCount;
  statUnknown.textContent = unknownCount;
  if (inferenceTime !== null) {
    statSpeed.textContent = `${inferenceTime}ms`;
  }

  renderResultsList(detectionSummaries);
}

function renderResultsList(summaries) {
  if (summaries.length === 0) {
    detectedList.innerHTML = `
      <p style="color: var(--text-muted); font-size: 0.82rem; text-align: center; padding: 1rem 0;">
        No faces detected in current frame.
      </p>
    `;
    return;
  }

  detectedList.innerHTML = summaries
    .map(
      (item) => `
      <div class="detected-item">
        <div class="detected-name">
          <span>${item.isUnknown ? "❓" : "✅"}</span>
          <span>${escapeHtml(item.name)}</span>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem;">
          <span class="detected-tag ${item.isUnknown ? "unknown" : "match"}">
            ${item.isUnknown ? "Unknown" : `${item.confidence}% Match`}
          </span>
          <span class="detected-score">dist: ${item.distance}</span>
        </div>
      </div>
    `
    )
    .join("");
}

// 6. Live Webcam Tracking Loop
async function startWebcam() {
  if (isCameraRunning) return;

  try {
    showLoading("Requesting webcam stream...");

    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false,
    });

    stageWrapper.innerHTML = "";
    stageWrapper.style.display = "inline-block";
    emptyState.style.display = "none";

    webcamVideo = document.createElement("video");
    webcamVideo.autoplay = true;
    webcamVideo.muted = true;
    webcamVideo.playsInline = true;
    webcamVideo.srcObject = webcamStream;

    await new Promise((resolve) => {
      webcamVideo.onloadedmetadata = () => {
        webcamVideo.play();
        resolve();
      };
    });

    stageWrapper.appendChild(webcamVideo);
    currentCanvas = faceapi.createCanvasFromMedia(webcamVideo);
    stageWrapper.appendChild(currentCanvas);

    const displaySize = { width: webcamVideo.videoWidth, height: webcamVideo.videoHeight };
    faceapi.matchDimensions(currentCanvas, displaySize);

    isCameraRunning = true;
    btnToggleCamera.classList.add("stopping");
    cameraBtnText.textContent = "Stop Webcam Stream";
    statSpeedLabel.textContent = "Live Stream FPS";
    hideLoading();

    // Automatically switch to TinyFaceDetector for smooth real-time video performance
    modelSelector.value = "tinyFaceDetector";

    runWebcamLoop(displaySize);
  } catch (err) {
    console.error("Webcam error:", err);
    hideLoading();
    showToast("Could not access webcam. Ensure permissions are granted.", "error");
  }
}

function stopWebcam() {
  if (!isCameraRunning) return;

  if (webcamAnimationId) {
    cancelAnimationFrame(webcamAnimationId);
    webcamAnimationId = null;
  }

  if (webcamStream) {
    webcamStream.getTracks().forEach((track) => track.stop());
    webcamStream = null;
  }

  isCameraRunning = false;
  btnToggleCamera.classList.remove("stopping");
  cameraBtnText.textContent = "Start Webcam Stream";

  stageWrapper.innerHTML = "";
  stageWrapper.style.display = "none";
  emptyState.style.display = "block";
  statSpeedLabel.textContent = "Inference Time";
  statSpeed.textContent = "0ms";
  detectedList.innerHTML = `
    <p style="color: var(--text-muted); font-size: 0.82rem; text-align: center; padding: 1rem 0;">
      Webcam stream stopped.
    </p>
  `;
}

async function runWebcamLoop(displaySize) {
  if (!isCameraRunning) return;

  const now = performance.now();
  frameCount++;
  if (now - lastFrameTime >= 1000) {
    currentFps = frameCount;
    frameCount = 0;
    lastFrameTime = now;
    statSpeed.textContent = `${currentFps} FPS`;
  }

  try {
    if (webcamVideo && !webcamVideo.paused && !webcamVideo.ended) {
      const detections = await faceapi
        .detectAllFaces(webcamVideo, getDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptors();

      lastDetections = faceapi.resizeResults(detections, displaySize);
      renderDetections();
    }
  } catch (err) {
    console.warn("Webcam inference frame dropped:", err);
  }

  webcamAnimationId = requestAnimationFrame(() => runWebcamLoop(displaySize));
}

// 7. Dynamic In-Browser Face Enrollment
async function handleEnrollSubmit() {
  const name = enrollName.value.trim();
  const file = enrollFileInput.files[0];

  if (!name) {
    showToast("Please enter a name for the profile.", "warning");
    return;
  }

  if (!file) {
    showToast("Please select a photo of the face.", "warning");
    return;
  }

  btnSubmitEnroll.disabled = true;
  btnSubmitEnroll.textContent = "Computing Embeddings...";

  try {
    const img = await faceapi.bufferToImage(file);
    const detection = await faceapi
      .detectSingleFace(img, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 }))
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (!detection || !detection.descriptor) {
      showToast("No clear frontal face detected in the photo. Please try another.", "error");
      btnSubmitEnroll.disabled = false;
      btnSubmitEnroll.textContent = "Compute & Enroll";
      return;
    }

    // Check if profile exists, append descriptor or create new
    const existing = labeledFaceDescriptors.find((l) => l.label === name);
    if (existing) {
      existing.descriptors.push(detection.descriptor);
    } else {
      labeledFaceDescriptors.push(new faceapi.LabeledFaceDescriptors(name, [detection.descriptor]));
      registeredNames.add(name);
    }

    rebuildFaceMatcher();
    renderProfileBadges();

    closeEnrollModal();
    showToast(`Profile "${name}" successfully enrolled!`, "success");
  } catch (err) {
    console.error("Enrollment error:", err);
    showToast("Failed to enroll face profile.", "error");
  } finally {
    btnSubmitEnroll.disabled = false;
    btnSubmitEnroll.textContent = "Compute & Enroll";
  }
}

// 8. Vector Embeddings Export & Import
function exportEmbeddingsToJson() {
  if (labeledFaceDescriptors.length === 0) {
    showToast("No profiles to export.", "warning");
    return;
  }

  const payload = {
    app: "FaceMatch JS",
    version: "1.1.0",
    exportedAt: new Date().toISOString(),
    profiles: labeledFaceDescriptors.map((ld) => ({
      name: ld.label,
      descriptors: ld.descriptors.map((d) => Array.from(d)),
    })),
  };

  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(payload, null, 2));
  const dlAnchor = document.createElement("a");
  dlAnchor.setAttribute("href", dataStr);
  dlAnchor.setAttribute("download", `facematch-embeddings-${Date.now()}.json`);
  document.body.appendChild(dlAnchor);
  dlAnchor.click();
  dlAnchor.remove();

  showToast("Biometric embeddings exported successfully!", "success");
}

function importEmbeddingsFromJson(e) {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (event) => {
    try {
      const data = JSON.parse(event.target.result);
      if (!data.profiles || !Array.isArray(data.profiles)) {
        throw new Error("Invalid embeddings format.");
      }

      let count = 0;
      data.profiles.forEach((p) => {
        if (p.name && Array.isArray(p.descriptors)) {
          const floatDescriptors = p.descriptors.map((arr) => new Float32Array(arr));
          const existing = labeledFaceDescriptors.find((l) => l.label === p.name);
          if (existing) {
            existing.descriptors.push(...floatDescriptors);
          } else {
            labeledFaceDescriptors.push(new faceapi.LabeledFaceDescriptors(p.name, floatDescriptors));
            registeredNames.add(p.name);
          }
          count++;
        }
      });

      rebuildFaceMatcher();
      renderProfileBadges();
      showToast(`Imported ${count} profiles from JSON!`, "success");
    } catch (err) {
      console.error("Import error:", err);
      showToast("Failed to parse JSON file.", "error");
    }
  };
  reader.readAsText(file);
}

// 9. Event Listeners & Mode Switching
function setupEventListeners() {
  // Mode Switcher Tabs
  tabPhoto.addEventListener("click", () => switchMode("photo"));
  tabWebcam.addEventListener("click", () => switchMode("webcam"));

  // Photo Dropzone & File Input
  dropzone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (ev) => processImageSource(ev.target.result, true);
      reader.readAsDataURL(file);
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = (ev) => processImageSource(ev.target.result, true);
      reader.readAsDataURL(file);
    }
  });

  // Instant Test Samples
  btnSample1.addEventListener("click", () => processImageSource("./Test/ALL.jpg"));
  btnSample2.addEventListener("click", () => processImageSource("./Test/IMG_3641.jpg"));

  // Webcam Toggle
  btnToggleCamera.addEventListener("click", () => {
    if (isCameraRunning) {
      stopWebcam();
    } else {
      startWebcam();
    }
  });

  // Threshold Slider
  thresholdSlider.addEventListener("input", (e) => {
    const newThreshold = parseFloat(e.target.value);
    thresholdValue.textContent = newThreshold.toFixed(2);
    rebuildFaceMatcher();
    if (!isCameraRunning) {
      renderDetections();
    }
  });

  // View Toggles & Model Selector
  toggleLandmarks.addEventListener("change", () => {
    if (!isCameraRunning) renderDetections();
  });
  toggleBoxes.addEventListener("change", () => {
    if (!isCameraRunning) renderDetections();
  });

  // Enrollment Modal Handlers
  btnOpenEnroll.addEventListener("click", openEnrollModal);
  btnCloseEnroll.addEventListener("click", closeEnrollModal);
  btnCancelEnroll.addEventListener("click", closeEnrollModal);
  btnSubmitEnroll.addEventListener("click", handleEnrollSubmit);

  // JSON Export / Import
  btnExportJson.addEventListener("click", exportEmbeddingsToJson);
  btnImportJson.addEventListener("click", () => jsonInput.click());
  jsonInput.addEventListener("change", importEmbeddingsFromJson);
}

function switchMode(mode) {
  if (activeMode === mode) return;

  if (activeMode === "webcam" && isCameraRunning) {
    stopWebcam();
  }

  activeMode = mode;

  if (mode === "photo") {
    tabPhoto.classList.add("active");
    tabWebcam.classList.remove("active");
    panelPhoto.style.display = "block";
    panelWebcam.style.display = "none";
    modelSelector.value = "ssdMobilenetv1";
  } else {
    tabWebcam.classList.add("active");
    tabPhoto.classList.remove("active");
    panelPhoto.style.display = "none";
    panelWebcam.style.display = "block";
    modelSelector.value = "tinyFaceDetector";
  }

  // Reset stage
  stageWrapper.innerHTML = "";
  stageWrapper.style.display = "none";
  emptyState.style.display = "block";
}

// Modal & Toast Helpers
function openEnrollModal() {
  enrollName.value = "";
  enrollFileInput.value = "";
  enrollModal.classList.add("open");
}

function closeEnrollModal() {
  enrollModal.classList.remove("open");
}

function showToast(msg, type = "info") {
  toastMsg.textContent = msg;
  toastIcon.textContent = type === "success" ? "✅" : type === "error" ? "❌" : type === "warning" ? "⚠️" : "ℹ️";
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 4000);
}

function showLoading(msg) {
  loadingMessage.textContent = msg;
  loadingOverlay.classList.remove("hidden");
}

function hideLoading() {
  loadingOverlay.classList.add("hidden");
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
