// --- FINAL VERSION: Load all files locally ---

// --- Global variable to hold the vision libraries
let vision;

async function loadMediaPipe() {
    // 1. Import the JS bundle from the *local* mediapipe_wasm folder
    const { ObjectDetector, FilesetResolver } = await import(
        './mediapipe_wasm/vision_bundle.mjs'
    );
    
    // 2. Point the FilesetResolver to the *local* mediapipe_wasm folder
    vision = await FilesetResolver.forVisionTasks(
        './mediapipe_wasm'
    );

    return { ObjectDetector }; // Only return ObjectDetector
}

// --- HTML Elements --- //
const video = document.getElementById("webcam");
const canvas = document.getElementById("outputCanvas");
const canvasCtx = canvas.getContext("2d");
const loadingContainer = document.getElementById("loadingContainer");
const loadingMessage = document.getElementById("loadingMessage");
const liveView = document.getElementById("liveView");

// --- NEW --- Progress bar elements
const progressContainer = document.getElementById("progressContainer");
const progressBar = document.getElementById("progressBar");
const progressText = document.getElementById("progressText");
const overlayProgressContainer = document.getElementById("overlayProgressContainer");
const overlayProgressBar = document.getElementById("overlayProgressBar");
const overlayProgressText = document.getElementById("overlayProgressText");

const modelSelect = document.getElementById("modelSelect");

const maxResultsSlider = document.getElementById("maxResultsSlider");
const maxResultsValue = document.getElementById("maxResultsValue");
const thresholdSlider = document.getElementById("thresholdSlider");
const thresholdValue = document.getElementById("thresholdValue");

const cameraSelectContainer = document.getElementById("cameraSelectContainer");
const cameraSelect = document.getElementById("cameraSelect");

const flipButton = document.getElementById("flipButton");

const videoOverlay = document.getElementById("videoOverlay");
const overlayMessage = document.getElementById("overlayMessage");
const overlaySpinner = document.querySelector(".overlaySpinner"); // Get spinner

// --- Permission overlay elements
const permissionOverlay = document.getElementById("permissionOverlay");
const permissionButton = document.getElementById("permissionButton");

let objectDetector;
let lastVideoTime = -1;
let currentStream = null;
let currentDeviceId = null;
let isVideoFlipped = false; 

// Variables to store the last *applied* settings
let lastModel = "";
let lastMaxResults = -1;
let lastThreshold = -1.0;


// --- NEW --- Helper function to download model with progress
/**
 * Downloads a model file with progress updates.
 * @param {string} modelPath - The path to the .tflite model file.
 * @param {Function} onProgress - Callback function for progress updates.
 */
async function downloadModelWithProgress(modelPath, onProgress) {
    const response = await fetch(modelPath);
    if (!response.ok) {
        throw new Error(`Failed to fetch model: ${response.statusText}`);
    }

    const contentLength = response.headers.get('Content-Length');
    if (!contentLength) {
        // Fallback for servers that don't send Content-Length
        onProgress({ percentage: null, loaded: 0, total: 0 });
        const buffer = await response.arrayBuffer();
        return new Uint8Array(buffer);
    }

    const total = parseInt(contentLength, 10);
    let loaded = 0;

    const reader = response.body.getReader();
    const chunks = []; // Store chunks here

    while (true) {
        const { done, value } = await reader.read();
        if (done) {
            break;
        }
        chunks.push(value);
        loaded += value.length;

        const percentage = Math.round((loaded / total) * 100);
        onProgress({ percentage, loaded, total });
    }

    // Combine chunks into a single Uint8Array
    const modelBuffer = new Uint8Array(total);
    let offset = 0;
    for (const chunk of chunks) {
        modelBuffer.set(chunk, offset);
        offset += chunk.length;
    }

    return modelBuffer;
}
// --- END NEW FUNCTION ---


// --- Main Function --- //
async function setupApp() {
    try {
        // 1. Load the Wasm files for MediaPipe
        loadingMessage.textContent = "Loading MediaPipe libraries...";
        const { ObjectDetector } = await loadMediaPipe();

        // 2. Function to create/re-create the detector
        async function createOrUpdateDetector(isInitialLoad = false) {
            // Get current values from ALL controls
            const modelPath = modelSelect.value;
            const maxResults = parseInt(maxResultsSlider.value, 10);
            const scoreThreshold = parseFloat(thresholdSlider.value);
            
            // Store these values as the "last applied" settings
            lastModel = modelPath;
            lastMaxResults = maxResults;
            lastThreshold = scoreThreshold;
            
            // Update UI labels (good for initial load)
            maxResultsValue.textContent = maxResults;
            thresholdValue.textContent = `${Math.round(scoreThreshold * 100)}%`;
            
            // --- MODIFIED --- Setup progress UI
            let currentProgressContainer, currentProgressBar, currentProgressText;
            
            if (isInitialLoad) {
                loadingMessage.textContent = "Downloading AI model...";
                currentProgressContainer = progressContainer;
                currentProgressBar = progressBar;
                currentProgressText = progressText;
            } else {
                overlayMessage.textContent = "Downloading AI model...";
                overlaySpinner.classList.add("hidden"); // Hide spinner
                currentProgressContainer = overlayProgressContainer;
                currentProgressBar = overlayProgressBar;
                currentProgressText = overlayProgressText;
                videoOverlay.classList.remove("hidden"); // Show overlay
            }
            
            currentProgressContainer.classList.remove("hidden");
            currentProgressBar.style.width = "0%";
            currentProgressText.textContent = "";
            
            // --- MODIFIED --- Download model with progress
            const modelBuffer = await downloadModelWithProgress(modelPath, ({ percentage, loaded, total }) => {
                if (percentage !== null) {
                    currentProgressBar.style.width = `${percentage}%`;
                    const loadedMB = (loaded / 1024 / 1024).toFixed(1);
                    const totalMB = (total / 1024 / 1024).toFixed(1);
                    currentProgressText.textContent = `Downloading... ${loadedMB}MB / ${totalMB}MB`;
                } else {
                    currentProgressText.textContent = `Downloading... (size unknown)`;
                }
            });

            // --- MODIFIED --- Update UI post-download
            if (isInitialLoad) {
                loadingMessage.textContent = "Initializing AI model...";
                currentProgressText.textContent = "Download complete. Initializing...";
            } else {
                overlayMessage.textContent = "Initializing AI model...";
                currentProgressText.textContent = "Download complete. Initializing...";
            }

            // --- MODIFIED --- Create ObjectDetector from buffer
            objectDetector = await ObjectDetector.createFromOptions(vision, {
                baseOptions: {
                    modelAssetBuffer: modelBuffer, // Pass the downloaded buffer
                    delegate: "GPU"
                },
                runningMode: "VIDEO",
                scoreThreshold: scoreThreshold, 
                maxResults: maxResults          
            });
            
            // Hide the correct loading message
            if (isInitialLoad) {
                // Don't hide loading container yet, enableWebcam will
            } else {
                videoOverlay.classList.add("hidden"); // Hide overlay
                // Reset overlay for next time
                overlaySpinner.classList.remove("hidden");
                currentProgressContainer.classList.add("hidden");
                currentProgressText.textContent = "";
            }
        }
        
        // 3. Add event listeners
        
        // Listen for 'input' (smooth dragging) to update *only* the text labels
        maxResultsSlider.addEventListener("input", () => {
            maxResultsValue.textContent = maxResultsSlider.value;
        });
        
        thresholdSlider.addEventListener("input", () => {
            const threshold = parseFloat(thresholdSlider.value);
            thresholdValue.textContent = `${Math.round(threshold * 100)}%`;
        });

        // Listen for 'change' (on release) to update the *actual* model
        const handleSliderChange = async () => {
            const newMaxResults = parseInt(maxResultsSlider.value, 10);
            const newThreshold = parseFloat(thresholdSlider.value);
            
            if (newMaxResults !== lastMaxResults || newThreshold !== lastThreshold) {
                await createOrUpdateDetector(false);
            }
        };
        
        maxResultsSlider.addEventListener("change", handleSliderChange);
        thresholdSlider.addEventListener("change", handleSliderChange);

        const handleModelChange = async () => {
            if (modelSelect.value !== lastModel) {
                await createOrUpdateDetector(false);
            }
        };
        modelSelect.addEventListener("change", handleModelChange);

        cameraSelect.addEventListener('change', switchCamera);
        
        flipButton.addEventListener("click", () => {
            isVideoFlipped = !isVideoFlipped; // Toggle the state
            video.classList.toggle("flipped", isVideoFlipped); // Toggle the CSS class
        });
        
        // --- Add event listener for permission button
        permissionButton.addEventListener("click", async () => {
            // Hide the permission overlay and try enabling the webcam again
            permissionOverlay.classList.add("hidden");
            // Show a temporary loading message in the main loader
            loadingMessage.textContent = "Requesting camera permission...";
            progressContainer.classList.add("hidden"); // Hide progress bar
            progressText.textContent = "";
            loadingContainer.classList.remove("hidden");
            await enableWebcam();
            // The loading container will be hidden by enableWebcam if successful
        });
        
        // 4. Create the detector for the first time
        await createOrUpdateDetector(true);

        // 5. Start the webcam (which also populates the camera list)
        await enableWebcam();

    } catch (error) {
        console.error("Error during setup:", error);
        
        let detailedMessage = "An unknown error occurred.";

        if (error instanceof Error) {
            detailedMessage = `JavaScript Error: ${error.message}`;
        } else if (error && error.type === 'error') {
            detailedMessage = `Failed to load a critical file. This is usually due to one of two problems:
                <br><br>
                1. <strong>File Not Found (404):</strong> Check that <strong>vision_bundle.mjs</strong>, <strong>vision_wasm_internal.wasm</strong>, and <strong>vision_wasm_internal.js</strong> are inside a folder named <strong>mediapipe_wasm</strong>.
                <br><br>
                2. <strong>Server Issue:</strong> Your local server (Live Server) might not be serving '.mjs' files correctly. Check the browser's console (F12) for a 'MIME type' error.`;
        } else if (typeof error === 'string') {
            detailedMessage = error;
        } else {
            try {
                detailedMessage = `An unexpected error object was caught: ${JSON.stringify(error)}`;
            } catch {
                detailedMessage = `An unexpected error object was caught: ${error.toString()}`;
            }
        }

        loadingMessage.innerHTML = `<strong>Error loading dependencies:</strong><br>${detailedMessage}<br><br>Please check the browser console (F12) for more details and then refresh.`;
        loadingMessage.style.color = "#FF5252";
        loadingMessage.style.textAlign = "left";
        loadingMessage.style.padding = "10px";
        loadingMessage.style.border = "1px solid #FF5252";
        loadingMessage.style.borderRadius = "8px";
        loadingMessage.style.backgroundColor = "rgba(255, 82, 82, 0.1)";
        
        // Hide progress bar on error
        progressContainer.classList.add("hidden");
        progressText.textContent = "";
        
        loadingContainer.classList.remove("hidden");
    }
}

// --- Function to handle camera switching ---
async function switchCamera() {
    currentDeviceId = cameraSelect.value;
    overlayMessage.textContent = "Switching camera...";
    overlaySpinner.classList.remove("hidden"); // Show spinner
    overlayProgressContainer.classList.add("hidden"); // Hide progress
    overlayProgressText.textContent = "";
    videoOverlay.classList.remove("hidden"); // Show overlay
    
    // Auto-set flip state based on the selected camera
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const selectedDevice = devices.find(d => d.deviceId === currentDeviceId && d.kind === 'videoinput');
        
        if (selectedDevice && selectedDevice.facingMode === 'user') {
            isVideoFlipped = true;
        } else {
            isVideoFlipped = false; // Default for 'environment' (back) or unknown
        }
        video.classList.toggle("flipped", isVideoFlipped); // Apply CSS class
    } catch (e) {
        console.error("Could not enumerate devices to set flip state:", e);
        isVideoFlipped = false;
        video.classList.toggle("flipped", isVideoFlipped);
    }

    await enableWebcam(); // Re-run the webcam setup with the new device ID
    
    videoOverlay.classList.add("hidden");
}

async function enableWebcam() {
    // 1. Stop any existing stream
    if (currentStream) {
        currentStream.getTracks().forEach(track => {
            track.stop();
        });
    }

    // 2. Set new constraints
    const constraints = {
        video: {
            width: 640,
            height: 480,
            ...(currentDeviceId && { deviceId: { exact: currentDeviceId } })
        }
    };

    try {
        // 3. Get the new video stream
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        currentStream = stream; // Save the new stream
        video.srcObject = stream;

        // 4. Clear previous event listener and add a new one
        video.removeEventListener("loadeddata", predictWebcam);
        video.addEventListener("loadeddata", predictWebcam);
        
        // 5. Populate camera list and set initial flip state *only if it's the first time*
        if (cameraSelect.options.length === 0) {
            await populateCameraList();
            
            const devices = await navigator.mediaDevices.enumerateDevices();
            const activeDevice = devices.find(d => d.deviceId === currentDeviceId && d.kind === 'videoinput');

            if (activeDevice && activeDevice.facingMode === 'user') {
                isVideoFlipped = true;
            } else {
                isVideoFlipped = false;
            }
            video.classList.toggle("flipped", isVideoFlipped);
        }

        // --- SUCCESS! ---
        // Hide any overlays that might be visible
        loadingContainer.classList.add("hidden"); // Hide main loader
        permissionOverlay.classList.add("hidden"); // Hide permission overlay
        liveView.classList.remove("hidden"); // Ensure live view is visible

    } catch (error) {
        console.error("Error accessing webcam:", error);

        // --- PERMISSION HANDLING ---
        if (error.name === "NotAllowedError" || error.name === "PermissionDeniedError") {
            // User denied permission
            loadingContainer.classList.add("hidden"); // Hide the main loader
            liveView.classList.remove("hidden"); // Show the video container
            permissionOverlay.classList.remove("hidden"); // Show the permission overlay
        } else {
            // Other errors (e.g., camera not found, etc.)
            loadingMessage.textContent = `Could not access webcam: ${error.message}. Please check device and permissions.`;
            loadingMessage.style.color = "#FF5252";
            progressContainer.classList.add("hidden"); // Hide progress bar
            progressText.textContent = "";
            loadingContainer.classList.remove("hidden"); // Make sure error is visible
            liveView.classList.add("hidden"); // Hide the video view
        }
        // --- END HANDLING ---
    }
}

// --- Helper function to get camera list ---
async function populateCameraList() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
        console.warn("enumerateDevices() is not supported.");
        return;
    }

    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(device => device.kind === 'videoinput');
    
    const activeStreamDeviceId = currentStream.getVideoTracks()[0].getSettings().deviceId;

    cameraSelect.innerHTML = ''; 
    
    videoDevices.forEach(device => {
        const option = document.createElement('option');
        option.value = device.deviceId;
        
        let label = device.label || `Camera ${cameraSelect.options.length + 1}`;
        if (device.facingMode) {
            label = `${label.split('(')[0].trim()} (${device.facingMode})`;
        }
        option.textContent = label;
        
        if (device.deviceId === activeStreamDeviceId) {
            option.selected = true;
            currentDeviceId = device.deviceId;
        }
        cameraSelect.appendChild(option);
    });
    
    if (videoDevices.length > 1) {
        cameraSelectContainer.style.display = 'flex';
    }
}

/**
 * The main detection loop: grabs a frame, detects, draws, and repeats.
 */
async function predictWebcam() {
    // Set canvas size on every frame
    if (canvas.width !== video.videoWidth) {
        canvas.width = video.videoWidth;
    }
    if (canvas.height !== video.videoHeight) {
        canvas.height = video.videoHeight;
    }

    if (video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        
        if (objectDetector) { 
            const results = objectDetector.detectForVideo(video, Date.now());

            canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

            for (const detection of results.detections) {
                
                canvasCtx.beginPath();
                canvasCtx.strokeStyle = "#00bcd4";
                canvasCtx.lineWidth = 2;
                
                let true_x = detection.boundingBox.originX;
                // Flip the x-coordinate if the video is flipped
                if (isVideoFlipped) {
                    true_x = canvas.width - detection.boundingBox.originX - detection.boundingBox.width;
                }
                
                canvasCtx.rect(
                    true_x,
                    detection.boundingBox.originY,
                    detection.boundingBox.width,
                    detection.boundingBox.height
                );
                canvasCtx.stroke();
                
                const label = `${detection.categories[0].categoryName} (${Math.round(detection.categories[0].score * 100)}%)`;
                
                canvasCtx.font = "16px Arial";
                const textWidth = canvasCtx.measureText(label).width;
                
                canvasCtx.fillStyle = "rgba(0, 0, 0, 0.7)";
                // Draw the label background, adjusting for flipped video
                canvasCtx.fillRect(true_x - 1, detection.boundingBox.originY - 20, textWidth + 10, 20);
                
                canvasCtx.fillStyle = "#00bcd4";
                // Draw the label text, adjusting for flipped video
                canvasCtx.fillText(label, true_x + 4, detection.boundingBox.originY - 5);
            }
        }
    }

    window.requestAnimationFrame(predictWebcam);
}

// Wait for the DOM to be fully loaded before running setupApp()
document.addEventListener("DOMContentLoaded", setupApp);