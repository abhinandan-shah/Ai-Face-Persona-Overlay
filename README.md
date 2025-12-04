# AI Face Persona Overlay

A real-time facial emotion detection and HUD overlay system using **MediaPipe FaceMesh** with optional **ONNX emotion model** support. Includes calibration, smoothing, and a neon-style UI for clean visual feedback.

---

## 🚀 Features
- Real-time **MediaPipe Face Mesh** landmark tracking  
- Emotion detection via:
  - **ONNX model** (if available)
  - **Heuristic classifier with calibration** (fallback)  
- Neon **HUD overlay** (emotion + confidence + FPS)  
- **Screenshot** saving  
- **Label smoothing** to reduce flicker  
- **2-second neutral calibration** for better accuracy  

---

## 📦 Requirements
- Python **3.10**
- Install dependencies:

pip install -r requirements.txt

requirements.txt includes:
opencv-python  
mediapipe  
numpy  
onnxruntime  # optional
