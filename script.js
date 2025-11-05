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

// --- Slider elements ---
const maxResultsSlider = document.getElementById("maxResultsSlider");
const maxResultsValue = document.getElementById("maxResultsValue");
const thresholdSlider = document.getElementById("thresholdSlider");
const thresholdValue = document.getElementById("thresholdValue");

// --- Camera selector elements ---
const cameraSelectContainer = document.getElementById("cameraSelectContainer");
const cameraSelect = document.getElementById("cameraSelect");

// --- Video overlay elements ---
const videoOverlay = document.getElementById("videoOverlay");
const overlayMessage = document.getElementById("overlayMessage");

let objectDetector;
let lastVideoTime = -1;
let currentStream = null;
let currentDeviceId = null;

// --- NEW --- Variables to store the last *applied* slider values
let lastMaxResults = -1;
let lastThreshold = -1.0;


// --- Main Function --- //
async function setupApp() {
    try {
        // 1. Load the Wasm files for MediaPipe
        loadingMessage.textContent = "Loading MediaPipe libraries...";
        const { ObjectDetector } = await loadMediaPipe();

        // 2. Function to create/re-create the detector
        async function createOrUpdateDetector(isInitialLoad = false) {
            // Get current values from sliders
            const maxResults = parseInt(maxResultsSlider.value, 10);
            const scoreThreshold = parseFloat(thresholdSlider.value);
            
            // --- NEW --- Store these values as the "last applied" settings
            lastMaxResults = maxResults;
            lastThreshold = scoreThreshold;
            
            // Update UI labels (good for initial load)
            maxResultsValue.textContent = maxResults;
            thresholdValue.textContent = `${Math.round(scoreThreshold * 100)}%`;
            
            // Show the correct loading message
            if (isInitialLoad) {
                loadingMessage.textContent = "Initializing AI model...";
                loadingContainer.classList.remove("hidden");
            } else {
                overlayMessage.textContent = "Updating AI model...";
                videoOverlay.classList.remove("hidden");
            }

            // Create the ObjectDetector with the new settings
            objectDetector = await ObjectDetector.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: "efficientdet_lite2.tflite", // Load local model
                    delegate: "GPU"
                },
                runningMode: "VIDEO",
                scoreThreshold: scoreThreshold, // Use slider value
                maxResults: maxResults          // Use slider value
            });
            
            // Hide the correct loading message
            if (isInitialLoad) {
                loadingContainer.classList.add("hidden");
                liveView.classList.remove("hidden"); // Show video for the first time
            } else {
                videoOverlay.classList.add("hidden");
            }
        }
        
        // 3. Add event listeners to sliders
        
        // --- NEW --- Listen for 'input' (smooth dragging) to update *only* the text labels
        maxResultsSlider.addEventListener("input", () => {
            maxResultsValue.textContent = maxResultsSlider.value;
        });
        
        thresholdSlider.addEventListener("input", () => {
            const threshold = parseFloat(thresholdSlider.value);
            thresholdValue.textContent = `${Math.round(threshold * 100)}%`;
        });

        // --- NEW --- Listen for 'change' (on release) to update the *actual* model
        const handleSliderChange = async () => {
            const newMaxResults = parseInt(maxResultsSlider.value, 10);
            const newThreshold = parseFloat(thresholdSlider.value);
            
            // Only update if the values are different from the last *applied* settings
            if (newMaxResults !== lastMaxResults || newThreshold !== lastThreshold) {
                await createOrUpdateDetector(false);
            }
        };
        
        maxResultsSlider.addEventListener("change", handleSliderChange);
        thresholdSlider.addEventListener("change", handleSliderChange);

        // Add event listener for camera selector
        cameraSelect.addEventListener('change', switchCamera);
        
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
        
        // Ensure the loading container is visible to show the error
        loadingContainer.classList.remove("hidden");
    }
}

// Start the whole process
setupApp();

// --- Function to handle camera switching ---
async function switchCamera() {
    currentDeviceId = cameraSelect.value;
    // Show the overlay while the camera switches
    overlayMessage.textContent = "Switching camera...";
    videoOverlay.classList.remove("hidden");
    
    await enableWebcam(); // Re-run the webcam setup with the new device ID
    
    // Hide the overlay once the new stream is loaded
    videoOverlay.classList.add("hidden");
}

// --- MODIFIED --- This function now handles stream switching and list population
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
        
        // 5. Populate the camera list *only if it's the first time*
        if (cameraSelect.options.length === 0) {
            await populateCameraList();
        }

    } catch (error) {
        console.error("Error accessing webcam:", error);
        loadingMessage.textContent = "Could not access webcam. Please grant permission and refresh.";
        loadingMessage.style.color = "#FF5252";
        loadingContainer.classList.remove("hidden"); // Make sure error is visible
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
    
    // Get the device ID of the stream that is *actually* active
    const activeStreamDeviceId = currentStream.getVideoTracks()[0].getSettings().deviceId;

    // Clear any existing options
    cameraSelect.innerHTML = ''; 
    
    // Add an option for each camera
    videoDevices.forEach(device => {
        const option = document.createElement('option');
        option.value = device.deviceId;
        option.textContent = device.label || `Camera ${cameraSelect.options.length + 1}`;
        
        // Pre-select the one that is currently active
        if (device.deviceId === activeStreamDeviceId) {
            option.selected = true;
            currentDeviceId = device.deviceId; // Sync the global variable
        }
        cameraSelect.appendChild(option);
    });
    
    // Only show the dropdown if there's more than one camera
    if (videoDevices.length > 1) {
        cameraSelectContainer.style.display = 'flex';
    }
}

/**
 * The main detection loop: grabs a frame, detects, draws, and repeats.
 */
async function predictWebcam() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    if (video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        
        if (objectDetector) { 
            const results = objectDetector.detectForVideo(video, Date.now());

            canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

            for (const detection of results.detections) {
                
                // --- Draw the Bounding Box --- //
                canvasCtx.beginPath();
                canvasCtx.strokeStyle = "#00bcd4";
                canvasCtx.lineWidth = 2;
                
                // Flip the X-coordinate for the mirrored video
                const true_x = canvas.width - detection.boundingBox.originX - detection.boundingBox.width;
                
                canvasCtx.rect(
                    true_x,
                    detection.boundingBox.originY,
                    detection.boundingBox.width,
                    detection.boundingBox.height
                );
                canvasCtx.stroke();
                
                // --- Draw the Label --- //
                const label = `${detection.categories[0].categoryName} (${Math.round(detection.categories[0].score * 100)}%)`;
                
                canvasCtx.font = "16px Arial";
                const textWidth = canvasCtx.measureText(label).width;
                
                canvasCtx.fillStyle = "rgba(0, 0, 0, 0.7)";
                canvasCtx.fillRect(true_x - 1, detection.boundingBox.originY - 20, textWidth + 10, 20);
                
                canvasCtx.fillStyle = "#00bcd4";
                canvasCtx.fillText(label, true_x + 4, detection.boundingBox.originY - 5);
            }
        }
    }

    // Always request the next frame
    window.requestAnimationFrame(predictWebcam);
}