/**
 * FaceMatch JS - Real-Time Client-Side Face Recognition & Landmark Detection
 * Powered by face-api.js & TensorFlow.js
 * Author: Krupal Malakiya
 * License: MIT
 */

// UI Elements
const statusPill = document.getElementById("statusPill");
const statusText = document.getElementById("statusText");
const profileCount = document.getElementById("profileCount");
const profileBadges = document.getElementById("profileBadges");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const btnSample1 = document.getElementById("btnSample1");
const btnSample2 = document.getElementById("btnSample2");
const thresholdSlider = document.getElementById("thresholdSlider");
const thresholdValue = document.getElementById("thresholdValue");
const toggleLandmarks = document.getElementById("toggleLandmarks");
const toggleBoxes = document.getElementById("toggleBoxes");

const viewerContainer = document.getElementById("viewerContainer");
const emptyState = document.getElementById("emptyState");
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingMessage = document.getElementById("loadingMessage");
const stageWrapper = document.getElementById("stageWrapper");

const statTotalFaces = document.getElementById("statTotalFaces");
const statMatched = document.getElementById("statMatched");
const statUnknown = document.getElementById("statUnknown");
const statTime = document.getElementById("statTime");
const detectedList = document.getElementById("detectedList");

// Profiles Dataset Configuration
const PROFILES = [
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
let labeledFaceDescriptors = [];
let faceMatcher = null;
let currentImage = null;
let currentCanvas = null;
let lastDetections = null;
let isReady = false;

// 1. Initialize Engine on DOM Ready
window.addEventListener("DOMContentLoaded", initEngine);

async function initEngine() {
  try {
    updateStatus("loading", "Loading Neural Network Models...");

    // Load pre-trained models from relative path for universal compatibility (local & GitHub Pages)
    const MODEL_PATH = "./models";
    await Promise.all([
      faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_PATH),
      faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_PATH),
      faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_PATH),
    ]);

    // Render registered profile badges dynamically
    renderProfileBadges();

    // Extract face descriptors for known profiles
    labeledFaceDescriptors = await loadAllProfiles();

    // Initialize FaceMatcher with default threshold (0.50)
    const initialThreshold = parseFloat(thresholdSlider.value);
    faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, initialThreshold);

    isReady = true;
    updateStatus("ready", "System Ready for Recognition");

    // Enable interactive controls
    btnSample1.disabled = false;
    btnSample2.disabled = false;
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

function renderProfileBadges() {
  profileCount.textContent = PROFILES.length;
  profileBadges.innerHTML = PROFILES.map(
    (p) => `<span class="profile-tag">${escapeHtml(p.name)}</span>`
  ).join("");
}

// 2. Load Labeled Descriptors from Images
async function loadAllProfiles() {
  const labeledDescriptors = [];

  for (let idx = 0; idx < PROFILES.length; idx++) {
    const profile = PROFILES[idx];
    updateStatus(
      "loading",
      `Computing embeddings: ${profile.name} (${idx + 1}/${PROFILES.length})...`
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
        } else {
          console.warn(`[FaceMatch] Face not detected in ${imgUrl}`);
        }
      } catch (err) {
        console.warn(`[FaceMatch] Could not load image for ${profile.name} (${i}.jpg):`, err);
      }
    }

    if (descriptions.length > 0) {
      labeledDescriptors.push(new faceapi.LabeledFaceDescriptors(profile.name, descriptions));
    }
  }

  return labeledDescriptors;
}

// 3. Image Processing & Face Recognition Pipeline
async function processImageSource(src, isFile = false) {
  if (!isReady) {
    alert("Neural engine is still initializing. Please wait a moment.");
    return;
  }

  showLoading("Analyzing facial features...");

  try {
    const img = new Image();
    img.crossOrigin = "anonymous";

    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = () => reject(new Error("Unable to load the requested image."));
      if (isFile) {
        img.src = src;
      } else {
        img.src = src;
      }
    });

    currentImage = img;

    // Clear previous render
    stageWrapper.innerHTML = "";
    stageWrapper.style.display = "inline-block";
    emptyState.style.display = "none";

    // Append image to DOM
    stageWrapper.appendChild(img);

    // Create synchronized overlay canvas
    currentCanvas = faceapi.createCanvasFromMedia(img);
    stageWrapper.appendChild(currentCanvas);

    // Set canvas dimensions to image natural dimensions
    const displaySize = {
      width: img.naturalWidth || img.width,
      height: img.naturalHeight || img.height,
    };
    faceapi.matchDimensions(currentCanvas, displaySize);

    // Execute Inference & Benchmark
    const startTime = performance.now();
    const detections = await faceapi
      .detectAllFaces(img)
      .withFaceLandmarks()
      .withFaceDescriptors();
    const inferenceTime = Math.round(performance.now() - startTime);

    // Cache detections for real-time threshold slider updates
    lastDetections = faceapi.resizeResults(detections, displaySize);

    // Render Bounding Boxes & Update Results Card
    renderDetections(inferenceTime);
  } catch (err) {
    console.error("Error processing image:", err);
    alert("Error processing image. Check console for details.");
  } finally {
    hideLoading();
  }
}

// 4. Render Detections and Summary
function renderDetections(inferenceTime = null) {
  if (!currentCanvas || !lastDetections || !faceMatcher) return;

  const ctx = currentCanvas.getContext("2d");
  ctx.clearRect(0, 0, currentCanvas.width, currentCanvas.height);

  const results = lastDetections.map((d) => faceMatcher.findBestMatch(d.descriptor));

  let matchedCount = 0;
  let unknownCount = 0;
  const detectionSummaries = [];

  // Draw Boxes & Labels
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

  // Draw Facial Landmarks Mesh
  if (toggleLandmarks.checked) {
    faceapi.draw.drawFaceLandmarks(currentCanvas, lastDetections);
  }

  // Update Summary Dashboard
  statTotalFaces.textContent = lastDetections.length;
  statMatched.textContent = matchedCount;
  statUnknown.textContent = unknownCount;
  if (inferenceTime !== null) {
    statTime.textContent = `${inferenceTime}ms`;
  }

  renderResultsList(detectionSummaries);
}

function renderResultsList(summaries) {
  if (summaries.length === 0) {
    detectedList.innerHTML = `
      <p style="color: var(--text-muted); font-size: 0.82rem; text-align: center; padding: 1rem 0;">
        No faces detected in this image.
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

// 5. Setup UI Event Listeners
function setupEventListeners() {
  // File Dropzone & Input
  dropzone.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => processImageSource(event.target.result, true);
      reader.readAsDataURL(file);
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = (event) => processImageSource(event.target.result, true);
      reader.readAsDataURL(file);
    }
  });

  // Instant Sample Testing
  btnSample1.addEventListener("click", () => {
    processImageSource("./Test/ALL.jpg");
  });

  btnSample2.addEventListener("click", () => {
    processImageSource("./Test/IMG_3641.jpg");
  });

  // Threshold Slider (Instant re-matching without re-inference!)
  thresholdSlider.addEventListener("input", (e) => {
    const newThreshold = parseFloat(e.target.value);
    thresholdValue.textContent = newThreshold.toFixed(2);
    if (labeledFaceDescriptors.length > 0) {
      faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, newThreshold);
      renderDetections();
    }
  });

  // View Toggles
  toggleLandmarks.addEventListener("change", () => renderDetections());
  toggleBoxes.addEventListener("change", () => renderDetections());
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
