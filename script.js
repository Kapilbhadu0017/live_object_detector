// @ts-nocheck
import {
    ObjectDetector,
    FilesetResolver
} from "./mediapipe_wasm/vision_bundle.mjs";

// --- DOM Elements ---
const video = document.getElementById("webcam");
const canvas = document.getElementById("outputCanvas");
const canvasCtx = canvas.getContext("2d");

// Loaders & Overlays
const loadingContainer = document.getElementById("loadingContainer");
const loadingMessage = document.getElementById("loadingMessage");
const progressContainer = document.getElementById("progressContainer");
const progressBar = document.getElementById("progressBar");
const progressText = document.getElementById("progressText");

const liveView = document.getElementById("liveView");
const videoOverlay = document.getElementById("videoOverlay");
const overlaySpinner = videoOverlay.querySelector(".overlaySpinner");
const overlayMessage = document.getElementById("overlayMessage");
const overlayProgressContainer = document.getElementById("overlayProgressContainer");
const overlayProgressBar = document.getElementById("overlayProgressBar");
const overlayProgressText = document.getElementById("overlayProgressText");

const permissionOverlay = document.getElementById("permissionOverlay");
const permissionButton = document.getElementById("permissionButton");

// Stats & Header
const statusBadge = document.getElementById("statusBadge");
const fpsValue = document.getElementById("fpsValue");
const latencyValue = document.getElementById("latencyValue");
const objectCount = document.getElementById("objectCount");
const detectedTags = document.getElementById("detectedTags");
const detectionsCard = document.getElementById("detectionsCard");
const actionBar = document.getElementById("actionBar");

// Controls
const modelSelect = document.getElementById("modelSelect");
const cameraSelect = document.getElementById("cameraSelect");
const cameraSelectContainer = document.getElementById("cameraSelectContainer");
const maxResultsSlider = document.getElementById("maxResultsSlider");
const maxResultsValue = document.getElementById("maxResultsValue");
const thresholdSlider = document.getElementById("thresholdSlider");
const thresholdValue = document.getElementById("thresholdValue");
const flipButton = document.getElementById("flipButton");
const pauseButton = document.getElementById("pauseButton");
const pauseBtnText = document.getElementById("pauseBtnText");
const snapshotButton = document.getElementById("snapshotButton");
const toggleBoxesButton = document.getElementById("toggleBoxesButton");

// --- Global State ---
let objectDetector = null;
let isUpdatingDetector = false;
let isDetectionPaused = false;
let showBoundingBoxes = true;
let lastVideoTime = -1;
let lastTimestamp = 0;
let currentStream = null;
let isFlipped = false;
let videoDevices = [];
let isPredicting = false;
let listenersAdded = false;
let forceCpuDelegate = false;

// Performance metrics
let frameCount = 0;
let lastFpsUpdate = performance.now();
let currentFps = 0;
let lastInferenceTime = 0;

// Known model file sizes (in bytes) to calculate fallback download progress
const KNOWN_MODEL_SIZES = {
    "efficientdet_lite2.tflite": 12138859,
    "efficientdet_lite0.tflite": 7254339
};

// Model memory cache (Stores cloned Uint8Array buffers)
const modelCache = new Map();

function cloneBuffer(uint8Array) {
    if (!uint8Array || !uint8Array.byteLength) return null;
    const dst = new Uint8Array(uint8Array.byteLength);
    dst.set(uint8Array);
    return dst;
}

// Color palette generator for categories
const categoryColorMap = new Map();
const PRESET_COLORS = [
    "#38bdf8", // cyan
    "#34d399", // emerald
    "#a855f7", // purple
    "#fb923c", // orange
    "#f43f5e", // rose
    "#f59e0b", // amber
    "#818cf8", // indigo
    "#ec4899", // pink
    "#10b981", // green
    "#06b6d4"  // teal
];

function getCategoryColor(categoryName) {
    if (!categoryColorMap.has(categoryName)) {
        const colorIndex = categoryColorMap.size % PRESET_COLORS.length;
        categoryColorMap.set(categoryName, PRESET_COLORS[colorIndex]);
    }
    return categoryColorMap.get(categoryName);
}

// Start app when DOM is ready
document.addEventListener("DOMContentLoaded", setupApp);

async function setupApp() {
    try {
        await startWebcam();
        await populateCameraList();
        await createOrUpdateDetector();
        
        if (!listenersAdded) {
            addControlListeners();
            listenersAdded = true;
        }

        if (!isPredicting) {
            isPredicting = true;
            window.requestAnimationFrame(predictWebcam);
        }
    } catch (error) {
        if (error.name === "NotAllowedError" || error.name === "PermissionDeniedError") {
            showPermissionError();
        } else {
            handleSetupError(error);
        }
    }
}

function showPermissionError() {
    loadingContainer.classList.add("hidden");
    liveView.classList.remove("hidden");
    permissionOverlay.classList.remove("hidden");
    statusBadge.textContent = "Error";
    statusBadge.className = "badge";
}

permissionButton.addEventListener("click", async () => {
    permissionOverlay.classList.add("hidden");
    loadingContainer.classList.remove("hidden");
    loadingMessage.textContent = "Waiting for camera permission...";
    await setupApp(); 
});

async function populateCameraList() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        videoDevices = devices.filter(device => device.kind === 'videoinput');
        
        if (videoDevices.length > 0) {
            cameraSelect.innerHTML = '';
            videoDevices.forEach((device, index) => {
                const option = document.createElement('option');
                option.value = device.deviceId;
                let label = device.label || `Camera ${index + 1}`;
                if (label.toLowerCase().includes('back') || label.toLowerCase().includes('rear')) {
                    label = 'Back Camera';
                } else if (label.toLowerCase().includes('front') || label.toLowerCase().includes('user')) {
                    label = 'Front Camera';
                }
                option.textContent = label;
                cameraSelect.appendChild(option);
            });

            if (videoDevices.length > 1) {
                cameraSelectContainer.style.display = 'flex';
            }
        }
    } catch (error) {
        console.error("Error enumerating devices:", error);
    }
}

async function downloadModelWithProgress(modelPath, progressCallback) {
    const response = await fetch(modelPath);
    if (!response.ok) {
        throw new Error(`Failed to fetch model: ${modelPath}. Server responded with ${response.status}`);
    }

    const reader = response.body.getReader();
    const totalSizeHeader = response.headers.get('Content-Length') || response.headers.get('content-length');
    let totalSize = totalSizeHeader ? parseInt(totalSizeHeader, 10) : 0;

    const filename = modelPath.split('/').pop();
    if ((!totalSize || isNaN(totalSize) || totalSize <= 0) && KNOWN_MODEL_SIZES[filename]) {
        totalSize = KNOWN_MODEL_SIZES[filename];
    }

    let downloadedSize = 0;
    let chunks = [];

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        chunks.push(value);
        downloadedSize += value.length;

        const downloadedMB = (downloadedSize / 1024 / 1024).toFixed(1);

        if (totalSize > 0) {
            const percentage = Math.min(100, Math.round((downloadedSize / totalSize) * 100));
            const totalMB = (totalSize / 1024 / 1024).toFixed(1);
            progressCallback(percentage, downloadedMB, totalMB);
        } else {
            progressCallback(0, downloadedMB, null);
        }
    }

    const modelBuffer = new Uint8Array(downloadedSize);
    let offset = 0;
    for (const chunk of chunks) {
        modelBuffer.set(chunk, offset);
        offset += chunk.length;
    }

    const finalMB = (downloadedSize / 1024 / 1024).toFixed(1);
    progressCallback(100, finalMB, finalMB);
    
    return modelBuffer;
}

function createProgressCallback(isInitialLoad) {
    const pBar = isInitialLoad ? progressBar : overlayProgressBar;
    const pText = isInitialLoad ? progressText : overlayProgressText;
    const pContainer = isInitialLoad ? progressContainer : overlayProgressContainer;

    pContainer.classList.remove('hidden');

    return (percentage, downloadedMB, totalMB) => {
        if (percentage > 0) {
            pBar.style.width = `${percentage}%`;
        } else {
            pBar.style.width = '30%';
        }

        if (totalMB) {
            pText.textContent = `Downloading AI Model... ${downloadedMB} MB / ${totalMB} MB`;
        } else {
            pText.textContent = `Downloading AI Model... ${downloadedMB} MB`;
        }
    };
}

async function createOrUpdateDetector() {
    if (isUpdatingDetector) return;
    isUpdatingDetector = true;

    const isInitialLoad = !objectDetector;
    
    if (isInitialLoad) {
        loadingContainer.classList.remove("hidden");
        loadingMessage.textContent = "Loading MediaPipe libraries...";
    } else {
        videoOverlay.classList.remove("hidden");
        overlaySpinner.classList.remove("hidden");
        overlayMessage.textContent = "Switching AI model...";
    }

    try {
        const vision = await FilesetResolver.forVisionTasks('./mediapipe_wasm');

        const modelPath = modelSelect.value;
        const maxResults = parseInt(maxResultsSlider.value, 10);
        const scoreThreshold = parseFloat(thresholdSlider.value);

        let modelBuffer;
        if (modelCache.has(modelPath)) {
            const cached = modelCache.get(modelPath);
            const cloned = cloneBuffer(cached);
            if (cloned) {
                modelBuffer = cloned;
                const msgElement = isInitialLoad ? loadingMessage : overlayMessage;
                msgElement.textContent = "Loading model from memory cache...";
            } else {
                modelCache.delete(modelPath);
            }
        }

        if (!modelBuffer) {
            const progressCallback = createProgressCallback(isInitialLoad);
            const downloaded = await downloadModelWithProgress(modelPath, progressCallback);
            modelBuffer = downloaded;
            modelCache.set(modelPath, cloneBuffer(downloaded));
        }

        const loadingMsgElement = isInitialLoad ? loadingMessage : overlayMessage;
        loadingMsgElement.textContent = "Initializing AI detector...";

        if (objectDetector) {
            const oldDetector = objectDetector;
            objectDetector = null; // Prevent predict loop from referencing closed object
            try { oldDetector.close(); } catch (e) {}
        }

        let preferredDelegate = forceCpuDelegate ? "CPU" : "GPU";
        let detectorOptions = {
            baseOptions: {
                modelAssetBuffer: modelBuffer,
                delegate: preferredDelegate
            },
            runningMode: "VIDEO",
            maxResults: maxResults,
            scoreThreshold: scoreThreshold
        };

        try {
            objectDetector = await ObjectDetector.createFromOptions(vision, detectorOptions);
        } catch (gpuError) {
            console.warn("GPU Delegate creation failed or crashed, falling back to CPU:", gpuError);
            forceCpuDelegate = true;
            detectorOptions.baseOptions.delegate = "CPU";
            objectDetector = await ObjectDetector.createFromOptions(vision, detectorOptions);
        }

        if (isInitialLoad) {
            loadingContainer.classList.add("hidden");
            liveView.classList.remove("hidden");
            actionBar.classList.remove("hidden");
            detectionsCard.classList.remove("hidden");
        } else {
            videoOverlay.classList.add("hidden");
        }
        statusBadge.textContent = "Active";
        statusBadge.className = "badge active";

    } catch (error) {
        handleSetupError(error);
    } finally {
        isUpdatingDetector = false;
    }
}

async function startWebcam() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }

    const deviceId = cameraSelect.value;
    const isMobile = /Android|iPhone|iPad|iPod|Windows Phone/i.test(navigator.userAgent) || (navigator.maxTouchPoints && navigator.maxTouchPoints > 2);

    let constraints;
    if (deviceId) {
        constraints = { video: { deviceId: { exact: deviceId } } };
    } else if (isMobile) {
        constraints = { video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } } };
    } else {
        constraints = { video: { width: { ideal: 1280 }, height: { ideal: 720 } } };
    }

    try {
        currentStream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (firstErr) {
        console.warn("Primary camera constraints failed, attempting fallback:", firstErr);
        try {
            currentStream = await navigator.mediaDevices.getUserMedia({ video: true });
        } catch (fallbackErr) {
            console.error("Camera access failed completely:", fallbackErr);
            throw fallbackErr;
        }
    }

    video.srcObject = currentStream;

    // Explicit play call required on mobile/iOS browsers
    try {
        await video.play();
    } catch (playErr) {
        console.warn("Video play warning:", playErr);
    }

    autoFlipCamera();
    permissionOverlay.classList.add("hidden");
}

function autoFlipCamera() {
    let facingMode = 'user'; 
    if (currentStream && currentStream.getVideoTracks().length > 0) {
        const track = currentStream.getVideoTracks()[0];
        const settings = track.getSettings();
        if (settings.facingMode) {
            facingMode = settings.facingMode;
        } else if (track.label) {
            const label = track.label.toLowerCase();
            if (label.includes('back') || label.includes('rear') || label.includes('environment')) {
                facingMode = 'environment';
            }
        }
    }

    isFlipped = (facingMode === 'user');
    video.classList.toggle('flipped', isFlipped);
}

function addControlListeners() {
    maxResultsSlider.addEventListener("input", () => {
        maxResultsValue.textContent = maxResultsSlider.value;
    });

    thresholdSlider.addEventListener("input", () => {
        thresholdValue.textContent = `${Math.round(parseFloat(thresholdSlider.value) * 100)}%`;
    });

    let currentModel = modelSelect.value;
    modelSelect.addEventListener("change", () => {
        if (modelSelect.value !== currentModel) {
            currentModel = modelSelect.value;
            createOrUpdateDetector();
        }
    });

    let currentMaxResults = maxResultsSlider.value;
    maxResultsSlider.addEventListener("change", () => {
        if (maxResultsSlider.value !== currentMaxResults) {
            currentMaxResults = maxResultsSlider.value;
            createOrUpdateDetector();
        }
    });

    let currentThreshold = thresholdSlider.value;
    thresholdSlider.addEventListener("change", () => {
        if (thresholdSlider.value !== currentThreshold) {
            currentThreshold = thresholdSlider.value;
            createOrUpdateDetector();
        }
    });

    cameraSelect.addEventListener("change", startWebcam);

    flipButton.addEventListener("click", () => {
        isFlipped = !isFlipped;
        video.classList.toggle('flipped', isFlipped);
    });

    pauseButton.addEventListener("click", () => {
        isDetectionPaused = !isDetectionPaused;
        pauseBtnText.textContent = isDetectionPaused ? "Resume" : "Pause";
        pauseButton.classList.toggle("active", isDetectionPaused);
        if (!isDetectionPaused) {
            statusBadge.textContent = "Active";
            statusBadge.className = "badge active";
        } else {
            statusBadge.textContent = "Paused";
            statusBadge.className = "badge";
        }
    });

    toggleBoxesButton.addEventListener("click", () => {
        showBoundingBoxes = !showBoundingBoxes;
        toggleBoxesButton.innerHTML = `<span class="icon">👁</span> Boxes: ${showBoundingBoxes ? "On" : "Off"}`;
        toggleBoxesButton.classList.toggle("active", showBoundingBoxes);
        if (!showBoundingBoxes) {
            canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
        }
    });

    snapshotButton.addEventListener("click", captureSnapshot);
}

function captureSnapshot() {
    const tempCanvas = document.createElement("canvas");
    const w = video.videoWidth || 1280;
    const h = video.videoHeight || 720;
    tempCanvas.width = w;
    tempCanvas.height = h;
    const ctx = tempCanvas.getContext("2d");

    if (isFlipped) {
        ctx.translate(w, 0);
        ctx.scale(-1, 1);
    }
    ctx.drawImage(video, 0, 0, w, h);
    if (isFlipped) {
        ctx.setTransform(1, 0, 0, 1, 0, 0);
    }
    ctx.drawImage(canvas, 0, 0, w, h);

    const link = document.createElement("a");
    link.download = `detection-snapshot-${Date.now()}.png`;
    link.href = tempCanvas.toDataURL("image/png");
    link.click();
}

async function predictWebcam() {
    if (isDetectionPaused || isUpdatingDetector || !objectDetector) {
        window.requestAnimationFrame(predictWebcam);
        return;
    }

    if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
        if (video.currentTime !== lastVideoTime) {
            lastVideoTime = video.currentTime;

            // Dynamically update container aspect ratio to prevent stretch/crop mismatch
            if (video.videoWidth && video.videoHeight) {
                const aspect = video.videoWidth / video.videoHeight;
                const currentAspectStr = liveView.style.aspectRatio;
                let currentAspect = 0;
                if (currentAspectStr) {
                    const parts = currentAspectStr.split('/').map(p => parseFloat(p.trim()));
                    if (parts.length === 2 && parts[1] !== 0) {
                        currentAspect = parts[0] / parts[1];
                    }
                }
                if (Math.abs(aspect - currentAspect) > 0.01) {
                    liveView.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
                }
            }

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            const startTime = performance.now();

            try {
                // Strictly monotonic timestamp for MediaPipe WASM detector
                const now = performance.now();
                const timestamp = Math.max(now, lastTimestamp + 0.001);
                lastTimestamp = timestamp;

                const results = objectDetector.detectForVideo(video, timestamp);
                lastInferenceTime = Math.round(performance.now() - startTime);

                canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

                if (showBoundingBoxes && results && results.detections) {
                    for (const detection of results.detections) {
                        drawDetection(detection);
                    }
                }

                updateDetectionsSummary(results ? results.detections : []);
            } catch (err) {
                console.warn("Detection frame error:", err);
            }

            // Update FPS & Latency metrics
            frameCount++;
            const nowTime = performance.now();
            if (nowTime - lastFpsUpdate >= 1000) {
                currentFps = Math.round((frameCount * 1000) / (nowTime - lastFpsUpdate));
                frameCount = 0;
                lastFpsUpdate = nowTime;
                fpsValue.textContent = currentFps;
                latencyValue.textContent = `${lastInferenceTime} ms`;
            }
        }
    }

    window.requestAnimationFrame(predictWebcam);
}

function updateDetectionsSummary(detections) {
    objectCount.textContent = detections.length;

    if (!detections || detections.length === 0) {
        detectedTags.innerHTML = '<span class="no-objects">No objects detected</span>';
        return;
    }

    detectedTags.innerHTML = '';
    detections.forEach(detection => {
        const category = detection.categories[0];
        const color = getCategoryColor(category.categoryName);
        const tag = document.createElement("span");
        tag.className = "object-tag";
        tag.style.borderColor = color;
        tag.style.boxShadow = `0 2px 8px ${color}33`;
        tag.innerHTML = `<span style="color: ${color};">●</span> ${category.categoryName} <span class="tag-score">${Math.round(category.score * 100)}%</span>`;
        detectedTags.appendChild(tag);
    });
}

function drawDetection(detection) {
    const box = detection.boundingBox;
    const category = detection.categories[0];
    const color = getCategoryColor(category.categoryName);

    let x;
    if (isFlipped) {
        x = canvas.width - box.originX - box.width;
    } else {
        x = box.originX;
    }

    const y = box.originY;
    const w = box.width;
    const h = box.height;

    // --- 1. Draw Bounding Box with Rounded Corners ---
    canvasCtx.save();
    canvasCtx.strokeStyle = color;
    canvasCtx.lineWidth = Math.max(3, canvas.width * 0.004);
    canvasCtx.shadowColor = color;
    canvasCtx.shadowBlur = 8;
    
    // Draw box
    canvasCtx.beginPath();
    canvasCtx.rect(x, y, w, h);
    canvasCtx.stroke();
    canvasCtx.restore();

    // --- 2. Draw Label & Score Badge ---
    const scoreText = `${Math.round(category.score * 100)}%`;
    const labelText = `${category.categoryName} ${scoreText}`;

    const fontSize = Math.max(14, Math.round(canvas.width * 0.015));
    canvasCtx.font = `600 ${fontSize}px 'Inter', sans-serif`;
    const textMetrics = canvasCtx.measureText(labelText);
    const textWidth = textMetrics.width;
    const textHeight = fontSize * 1.5;

    const padX = 8;
    const padY = 4;
    const bgWidth = textWidth + (padX * 2);
    const bgHeight = textHeight + (padY * 2);

    let textBgX = x;
    // Keep badge within canvas boundaries
    if (textBgX + bgWidth > canvas.width) {
        textBgX = canvas.width - bgWidth - 4;
    }
    if (textBgX < 0) {
        textBgX = 4;
    }

    let textBgY = y - bgHeight - 4;
    if (textBgY < 0) { // Near top edge, place inside box
        textBgY = y + 4;
    }

    // Badge Background
    canvasCtx.save();
    canvasCtx.fillStyle = "rgba(11, 15, 25, 0.85)";
    canvasCtx.strokeStyle = color;
    canvasCtx.lineWidth = 1;
    
    canvasCtx.beginPath();
    canvasCtx.rect(textBgX, textBgY, bgWidth, bgHeight);
    canvasCtx.fill();
    canvasCtx.stroke();

    // Label Text
    canvasCtx.fillStyle = color;
    canvasCtx.fillText(labelText, textBgX + padX, textBgY + fontSize + padY - 2);
    canvasCtx.restore();
}

function handleSetupError(error) {
    console.error("Setup error:", error);
    const msg = error.message || "An unknown error occurred.";
    loadingMessage.textContent = `Error: ${msg}. Please refresh or check camera settings.`;
    loadingMessage.style.color = "#f43f5e";
    statusBadge.textContent = "Error";
    statusBadge.className = "badge";
    loadingContainer.classList.remove("hidden");
}