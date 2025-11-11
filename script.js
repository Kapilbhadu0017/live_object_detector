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

// Controls
const modelSelect = document.getElementById("modelSelect");
const cameraSelect = document.getElementById("cameraSelect");
const cameraSelectContainer = document.getElementById("cameraSelectContainer");
const maxResultsSlider = document.getElementById("maxResultsSlider");
const maxResultsValue = document.getElementById("maxResultsValue");
const thresholdSlider = document.getElementById("thresholdSlider");
const thresholdValue = document.getElementById("thresholdValue");
const flipButton = document.getElementById("flipButton");

// --- Global State ---
let objectDetector;
let lastVideoTime = -1;
let currentStream;
let isFlipped = false;
let videoDevices = [];

// --- Model Caching ---
// Cache downloaded models in memory
const modelCache = new Map();

/**
 * Main setup function. Waits for the DOM to be ready.
 */
document.addEventListener("DOMContentLoaded", setupApp);

async function setupApp() {
    try {
        // First, check for camera permissions
        await checkCameraPermissions();
        
        // Load the initial model and start the webcam
        await createOrUpdateDetector();
        await startWebcam();
        
        // Add event listeners for controls
        addControlListeners();

    } catch (error) {
        // This catches critical errors during *initial* setup
        if (error.name === "NotAllowedError" || error.name === "PermissionDeniedError") {
            showPermissionError();
        } else {
            handleSetupError(error);
        }
    }
}

/**
 * Checks for camera permissions.
 * If permission is not granted, it will show the permission overlay.
 * If permission is already granted, it populates the camera list.
 */
async function checkCameraPermissions() {
    try {
        // Try to get a stream to check permissions without showing the overlay
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        
        // Success! Populate cameras and stop the dummy stream
        await populateCameraList();
        stream.getTracks().forEach(track => track.stop());
        
        // Hide permission overlay if it was somehow visible
        permissionOverlay.classList.add("hidden");
        
    } catch (error) {
        // Permission was denied or not yet granted
        console.error("Camera permission error:", error);
        loadingContainer.classList.add("hidden");
        liveView.classList.remove("hidden");
        permissionOverlay.classList.remove("hidden");
        // Re-throw the error to stop the setupApp process
        throw error;
    }
}

/**
 * Handles the "Grant Permission" button click.
 */
permissionButton.addEventListener("click", async () => {
    // Hide the permission overlay and show the loader
    permissionOverlay.classList.add("hidden");
    loadingContainer.classList.remove("hidden");
    loadingMessage.textContent = "Waiting for permission...";
    
    // Re-run the setup process
    // This will re-trigger the getUserMedia prompt
    await setupApp(); 
});

/**
 * Populates the camera dropdown list.
 */
async function populateCameraList() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        videoDevices = devices.filter(device => device.kind === 'videoinput');
        
        if (videoDevices.length > 1) {
            cameraSelectContainer.style.display = 'flex';
            cameraSelect.innerHTML = ''; // Clear existing options
            
            videoDevices.forEach(device => {
                const option = document.createElement('option');
                option.value = device.deviceId;
                // Try to create a user-friendly label
                let label = device.label || `Camera ${cameraSelect.options.length + 1}`;
                if (device.label.toLowerCase().includes('facing back')) {
                    label = 'Back Camera';
                } else if (device.label.toLowerCase().includes('facing front')) {
                    label = 'Front Camera';
                }
                option.textContent = label;
                cameraSelect.appendChild(option);
            });
        }
    } catch (error) {
        console.error("Error enumerating devices:", error);
    }
}

/**
 * Downloads a model file, tracks progress, and returns an ArrayBuffer.
 * This function is now more robust against Content-Length discrepancies.
 * @param {string} modelPath - The path to the .tflite model file.
 * @param {(percentage: number, downloadedMB: string, totalMB: string) => void} progressCallback - Function to update UI.
 * @returns {Promise<Uint8Array>} - The model data as a Uint8Array.
 */
async function downloadModelWithProgress(modelPath, progressCallback) {
    const response = await fetch(modelPath);
    if (!response.ok) {
        throw new Error(`Failed to fetch model: ${modelPath}. Server responded with ${response.status}`);
    }

    const reader = response.body.getReader();
    
    // Get total size from header. This is *only* for the progress bar.
    const totalSizeHeader = response.headers.get('Content-Length');
    const totalSize = totalSizeHeader ? parseInt(totalSizeHeader, 10) : 0;

    let downloadedSize = 0;
    let chunks = []; // Store downloaded chunks in a regular array

    while (true) {
        const { done, value } = await reader.read(); // value is a Uint8Array
        if (done) break;

        chunks.push(value); // Store the chunk
        downloadedSize += value.length;

        if (totalSize > 0) {
            const percentage = Math.round((downloadedSize / totalSize) * 100);
            const downloadedMB = (downloadedSize / 1024 / 1024).toFixed(1);
            const totalMB = (totalSize / 1024 / 1024).toFixed(1);
            progressCallback(percentage, downloadedMB, totalMB);
        } else {
            // Show progress in MB if total size is unknown
            const downloadedMB = (downloadedSize / 1024 / 1024).toFixed(1);
            progressCallback(0, downloadedMB, "??");
        }
    }

    // --- THIS IS THE FIX ---
    // 1. Create the final buffer with the *correct*, measured size
    const modelBuffer = new Uint8Array(downloadedSize);
    
    // 2. Copy all the chunks into it
    let offset = 0;
    for (const chunk of chunks) {
        modelBuffer.set(chunk, offset);
        offset += chunk.length;
    }
    // --- END OF FIX ---

    // Ensure progress bar hits 100%
    const finalMB = (downloadedSize / 1024 / 1024).toFixed(1);
    progressCallback(100, finalMB, finalMB);
    
    return modelBuffer;
}


/**
 * Callback function to update the correct progress bar (initial or overlay).
 * @param {boolean} isInitialLoad - True if it's the first page load.
 */
function createProgressCallback(isInitialLoad) {
    const pBar = isInitialLoad ? progressBar : overlayProgressBar;
    const pText = isInitialLoad ? progressText : overlayProgressText;
    const pContainer = isInitialLoad ? progressContainer : overlayProgressContainer;

    pContainer.classList.remove('hidden');

    return (percentage, downloadedMB, totalMB) => {
        pBar.style.width = `${percentage}%`;
        pText.textContent = `Downloading... ${downloadedMB} MB / ${totalMB} MB`;
    };
}


/**
 * Creates a new ObjectDetector or updates the existing one with new settings.
 */
async function createOrUpdateDetector() {
    const isInitialLoad = !objectDetector; // Is this the first time we're loading?
    
    // Show the correct loader
    if (isInitialLoad) {
        loadingContainer.classList.remove("hidden");
        loadingMessage.textContent = "Loading MediaPipe libraries...";
    } else {
        videoOverlay.classList.remove("hidden");
        overlaySpinner.classList.add("hidden"); // Hide spinner, show progress
        overlayMessage.textContent = "Switching AI model...";
    }

    try {
        const { ObjectDetector, vision } = await loadMediaPipe();

        const modelPath = modelSelect.value;
        const maxResults = parseInt(maxResultsSlider.value, 10);
        const scoreThreshold = parseFloat(thresholdSlider.value);

        // --- Download the model (from cache or network) ---
        let modelBuffer;
        if (modelCache.has(modelPath)) {
            modelBuffer = modelCache.get(modelPath);
            overlayMessage.textContent = "Loading model from cache...";
        } else {
            const progressCallback = createProgressCallback(isInitialLoad);
            modelBuffer = await downloadModelWithProgress(modelPath, progressCallback);
            modelCache.set(modelPath, modelBuffer); // Cache the downloaded model
        }

        // --- Create the detector ---
        const loadingMsgElement = isInitialLoad ? loadingMessage : overlayMessage;
        loadingMsgElement.textContent = "Initializing AI model...";

        // Close old detector if it exists
        if (objectDetector) {
            objectDetector.close();
        }

        objectDetector = await ObjectDetector.createFromOptions(vision, {
            baseOptions: {
                modelAssetBuffer: modelBuffer,
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            maxResults: maxResults,
            scoreThreshold: scoreThreshold
        });

        // Hide loaders
        if (isInitialLoad) {
            loadingContainer.classList.add("hidden");
            liveView.classList.remove("hidden");
        } else {
            videoOverlay.classList.add("hidden");
            overlaySpinner.classList.remove("hidden"); // Reset spinner
        }

    } catch (error) {
        handleSetupError(error);
    }
}

/**
 * Loads the MediaPipe libraries from the local folder.
 */
async function loadMediaPipe() {
    try {
        const { ObjectDetector, FilesetResolver } = await import(
            './mediapipe_wasm/vision_bundle.mjs'
        );
        const vision = await FilesetResolver.forVisionTasks(
            './mediapipe_wasm'
        );
        return { ObjectDetector, vision };
    } catch (error) {
        console.error("Critical error loading MediaPipe libraries:", error);
        error.message = "Failed to load critical AI libraries from 'mediapipe_wasm' folder.";
        throw error;
    }
}

/**
 * Starts or restarts the webcam stream with the selected device.
 */
async function startWebcam() {
    // Stop any existing stream
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
    }

    const deviceId = cameraSelect.value;
    const constraints = {
        video: {
            deviceId: deviceId ? { exact: deviceId } : undefined,
            width: { ideal: 1280 },
            height: { ideal: 720 }
        }
    };

    try {
        currentStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = currentStream;

        // Auto-flip based on camera facing mode
        autoFlipCamera(deviceId);
        
        // Start the detection loop
        video.addEventListener("loadeddata", predictWebcam);

    } catch (error) {
        console.error("Error starting webcam:", error);
        handleSetupError(error);
    }
}

/**
 * Automatically flips the video if the camera is user-facing.
 */
function autoFlipCamera(selectedDeviceId) {
    const selectedDevice = videoDevices.find(device => device.deviceId === selectedDeviceId);
    
    // Default to 'user' (front) if we can't determine
    let facingMode = 'user'; 
    
    if (selectedDevice) {
        facingMode = selectedDevice.facingMode || 'user';
    } else if (currentStream) {
        // Fallback: check the track's settings
        const trackSettings = currentStream.getVideoTracks()[0].getSettings();
        facingMode = trackSettings.facingMode || 'user';
    }

    // Flip if 'user' (front camera), don't flip if 'environment' (back camera)
    isFlipped = (facingMode === 'user');
    video.classList.toggle('flipped', isFlipped);
}

/**
 * Binds all the event listeners for the control panel.
 */
function addControlListeners() {
    // --- Smooth Label Updates (on 'input') ---
    maxResultsSlider.addEventListener("input", () => {
        maxResultsValue.textContent = maxResultsSlider.value;
    });

    thresholdSlider.addEventListener("input", () => {
        thresholdValue.textContent = `${Math.round(parseFloat(thresholdSlider.value) * 100)}%`;
    });

    // --- Heavy AI Updates (on 'change', when user releases) ---
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
            currentThreshold = currentThreshold;
            createOrUpdateDetector();
        }
    });
    
    // --- Other Controls ---
    cameraSelect.addEventListener("change", startWebcam);
    
    flipButton.addEventListener("click", () => {
        isFlipped = !isFlipped;
        video.classList.toggle('flipped', isFlipped);
    });
}

/**
 * The main detection loop.
 */
async function predictWebcam() {
    if (video.readyState < 2) {
      window.requestAnimationFrame(predictWebcam);
      return;
    }
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    if (objectDetector && video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        
        // --- FIX ---
        // Changed Date.Now() to Date.now() (lowercase 'n')
        const results = objectDetector.detectForVideo(video, Date.now());

        canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
        
        for (const detection of results.detections) {
            drawDetection(detection);
        }
    }

    // Keep the loop going
    window.requestAnimationFrame(predictWebcam);
}

/**
 * Draws a single detection (box and label) onto the canvas.
 * This function is now "flip-aware".
 * @param {object} detection - A single detection object from MediaPipe.
 */
function drawDetection(detection) {
    const box = detection.boundingBox;
    
    // --- 1. Calculate Coordinates ---
    let x, textX, textBgX;
    
    if (isFlipped) {
        // Flipped calculation
        x = canvas.width - box.originX - box.width;
    } else {
        // Normal calculation
        x = box.originX;
    }

    const y = box.originY;
    const w = box.width;
    const h = box.height;

    // --- 2. Draw the Bounding Box ---
    canvasCtx.beginPath();
    // --- FIX ---
    // Replaced invalid CSS `var()` with a simple string
    canvasCtx.strokeStyle = "#00bcd4"; 
    canvasCtx.lineWidth = Math.max(2, canvas.width * 0.003); // Responsive line width
    canvasCtx.rect(x, y, w, h);
    canvasCtx.stroke();
    
    // --- 3. Draw the Label ---
    const label = `${detection.categories[0].categoryName} (${Math.round(detection.categories[0].score * 100)}%)`;
    
    const fontSize = Math.max(16, canvas.width * 0.012);
    canvasCtx.font = `bold ${fontSize}px Arial`;
    const textWidth = canvasCtx.measureText(label).width;
    const textHeight = fontSize * 1.4;

    // Set text alignment based on flip state
    if (isFlipped) {
        textX = x + w - textWidth - 5; // Align text to the right inside the box
        textBgX = x + w - textWidth - 10;
    } else {
        textX = x + 5; // Align text to the left inside the box
        textBgX = x;
    }
    
    // Handle label position (move inside if at the top edge)
    let textY = y + textHeight * 0.8;
    let textBgY = y;
    
    if (textBgY < textHeight) { // If label is near the top edge
        textY = y + h - (textHeight * 0.2);
        textBgY = y + h - textHeight;
    }

    // Draw the text background
    canvasCtx.fillStyle = "rgba(0, 0, 0, 0.7)";
    canvasCtx.fillRect(textBgX, textBgY, textWidth + 10, textHeight);
    
    // Draw the text
    // --- FIX ---
    // Replaced invalid CSS `var()` with a simple string
    canvasCtx.fillStyle = "#00bcd4";
    canvasCtx.fillText(label, textX, textY);
}

/**
 * A centralized error handler for setup failures.
 * @param {Error} error - The error object.
 */
function handleSetupError(error) {
    console.error("Full error:", error);
    const msg = error.message || "An unknown error occurred.";

    if (msg.includes("Failed to load a critical file")) {
        // This is the detailed error from our check
        loadingMessage.innerHTML = msg; // Use innerHTML to render line breaks
    } else {
        loadingMessage.textContent = `Error: ${msg}. Please check the console (F12) and refresh.`;
    }
    
    loadingMessage.style.color = "#FF5252"; // Red
    loadingContainer.classList.remove("hidden");
    liveView.classList.add("hidden");
}

/**
 * Shows the specific error for "Permission Denied".
 */
function showPermissionError() {
    loadingContainer.classList.add("hidden");
    liveView.classList.remove("hidden");
    permissionOverlay.classList.remove("hidden");
}