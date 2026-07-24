# Live AI Object Detector 🎯

A high-performance, real-time object detection web application built with **MediaPipe Tasks Vision**, **TensorFlow Lite (TFLite)**, and vanilla JavaScript. The application runs entirely client-side in the browser using WebAssembly (WASM) and hardware-accelerated GPU delegates, eliminating the need for server-side processing or external API calls.

[![Live Demo](https://img.shields.io/badge/Live_Demo-Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://live-object-detector-five.vercel.app/)
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![WebAssembly](https://img.shields.io/badge/WebAssembly-Enabled-purple.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Vision-0097a7.svg)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6%2B-yellow.svg)

---

## 🌐 Live Working Application

Try the application live directly in your web browser:  
👉 **[https://live-object-detector-five.vercel.app/](https://live-object-detector-five.vercel.app/)**

---

## ✨ Features

- ⚡ **Real-Time Browser Detection**: Runs live inferences on webcam streams with minimal latency using GPU-accelerated WebGL/GPU delegates (with automatic CPU fallback).
- 🧠 **Dynamic AI Model Switching**: Toggle on-the-fly between **EfficientDet-Lite2** (High Quality) and **EfficientDet-Lite0** (High Performance/Low Latency). Features model memory caching and download progress indicators.
- 📊 **Telemetry HUD**: Displays real-time FPS (Frames Per Second), model inference latency (in milliseconds), and the current count of detected objects.
- 🎛️ **Configurable Detector Settings**:
  - **Confidence Threshold**: Adjust minimum detection score threshold (10% to 95%).
  - **Max Results**: Limit maximum detected objects rendered simultaneously (1 to 20).
- 📷 **Multi-Camera & Mirroring Support**:
  - Automatically enumerates available cameras (Front, Rear, External webcams).
  - Auto-mirrors front-facing video feeds with standard flip controls.
- 📸 **Snapshot Generator**: Capture and download high-resolution PNG snapshots combining the video frame and bounding box overlay.
- 🎨 **Modern Glassmorphism UI**: Beautiful dark-mode user interface built with responsive HTML5 & Vanilla CSS. Includes category-based dynamic neon bounding box color matching.
- 🔒 **Privacy-First**: All video frame processing is conducted strictly client-side within the browser—no video data is transmitted to external servers.

---

## 🛠️ Tech Stack & Dependencies

- **Core**: HTML5, Vanilla JavaScript (ES Modules), CSS3
- **AI & Machine Learning Framework**: [MediaPipe Tasks Vision WebAssembly Bundle](https://developers.google.com/mediapipe/solutions/vision/object_detector)
- **Computer Vision Models**: TensorFlow Lite (`.tflite`) EfficientDet Models
- **Web APIs**: `navigator.mediaDevices` (WebRTC), HTML5 Canvas 2D API, WebAssembly (WASM), WebGL

---

## 🤖 Pre-loaded Models

| Model | Size | Best For | Key Characteristic |
|---|---|---|---|
| **EfficientDet-Lite2** | ~12.1 MB | High Quality | Higher detection accuracy and object coverage |
| **EfficientDet-Lite0** | ~7.2 MB | Speed / Mobile | Lightweight, optimized for lower-end devices |

*Note: Models are downloaded on demand when selected and cached in-memory for immediate subsequent switches.*

---

## 🚀 Getting Started

Because the application uses WebAssembly and ES Modules, browser security policies require serving the project via an HTTP/HTTPS local server (opening `index.html` directly via `file://` will cause WebAssembly and CORS errors).

### Prerequisites

You only need a modern web browser and a local HTTP server utility.

### Running Locally

1. **Clone or download this repository**:
   ```bash
   git clone https://github.com/Kapilbhadu0017/live_object_detector.git
   cd live_object_detector
   ```

2. **Start a local HTTP server** using any of the following methods:

   - **Using Python 3**:
     ```bash
     python3 -m http.server 8000
     ```

   - **Using Node.js (`http-server`)**:
     ```bash
     npx http-server -p 8000
     ```

   - **Using Node.js (`serve`)**:
     ```bash
     npx serve .
     ```

   - **Using VS Code**:
     Install the **Live Server** extension, right-click `index.html`, and select **"Open with Live Server"**.

3. **Open the Application**:
   Navigate to `http://localhost:8000` (or the port specified by your server) in your web browser.

4. **Grant Camera Permissions**:
   Allow camera access when prompted by the browser to begin live object detection.

---

## 🕹️ User Controls & Usage

| Control | Description |
|---|---|
| **Model Quality** | Select between High Quality (`EfficientDet-Lite2`) or Low Quality (`EfficientDet-Lite0`). |
| **Camera Source** | Switch between available camera inputs (if multiple devices are present). |
| **Max Objects** | Slider to control maximum objects detected simultaneously per frame. |
| **Confidence Threshold** | Slider to filter out low-confidence object predictions. |
| **Pause / Resume** | Freeze object detection processing while maintaining webcam stream state. |
| **Snapshot** | Download a merged `.png` image of the current video frame with bounding boxes. |
| **Boxes: On / Off** | Toggle visual rendering of bounding boxes on the canvas overlay. |
| **Flip Video** | Mirror/unmirror the webcam display horizontally. |

---

## 📁 Project Structure

```text
live_object_detector/
├── index.html                  # Main Application UI structure & controls layout
├── script.js                   # Main application logic (MediaPipe setup, webcam & rendering loops)
├── style.css                   # Custom CSS (Glassmorphism design system, typography & HUD styles)
├── efficientdet_lite0.tflite   # Lightweight TensorFlow Lite object detection model
├── efficientdet_lite2.tflite   # High-precision TensorFlow Lite object detection model
└── mediapipe_wasm/             # Offline MediaPipe Vision WASM binaries & JS bundle
    └── vision_bundle.mjs
```

---

## 🌐 Browser Compatibility

- **Google Chrome / Chromium / Edge / Brave**: Full Support (Recommended for best WebGL/GPU acceleration)
- **Mozilla Firefox**: Full Support
- **Apple Safari** (macOS / iOS 15+): Full Support

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
