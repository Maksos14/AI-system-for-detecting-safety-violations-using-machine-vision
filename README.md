# AI System for Detecting Safety Violations 

## Description

A local C++ application using OpenCV for automatic detection of industrial safety violations in real time. The system processes video streams from cameras and USB devices, recognizes the absence of helmets and protective clothing, and records incidents with screenshots saved to Yandex.Disk.

---

## Development Plan

### 1. Drawing Up a Plan and Creating a GitHub Repository
- Define the project structure (folders, modules)
- Create a repository on GitHub
- Configure `.gitignore` for C++ projects
- Draw up and approve a work plan

---

### 2. Connecting Libraries and Installing Utilities
- Set up the development environment (CMake, compiler)
- Connect OpenCV for working with video and images
- Connect a library for neural network inference (OpenVINO / ONNX Runtime)
- Connect a library for logging (spdlog)
- Configure the project build

---

### 3. Connecting the AI and Configuring It for Correct Operation
- Implement a video capture module 
- Choose a neural network architecture (YOLO / SSD)
- Integrate a pre-trained object detection model
- Configure inference on the target platform (CPU/GPU)

---

### 4. Training a Neural Network to Recognize Helmets and Implementing Basic Logic
- Prepare and annotate a dataset (people, helmets)
- Perform model training
- Export the model to a C++ inference format
- Check recognition quality
- Implement basic violation detection logic

---

### 5. Adding a Phone Camera, Recognizing Other Protective Equipment, and Saving
- Implement support for RTSP stream from a phone 
- Expand the dataset: add classes for protective vests, gloves, goggles
- Additionally train the model on new classes
- Integrate recognition of additional protective equipment into the main logic
- Set up separate rules for each violation type
- Implement saving violation screenshots to Yandex.Disk

---

### 6. Documentation and Finalization
- Write technical documentation (architecture, installation, configuration)
- Write user documentation (launching, working with the application)
- Prepare instructions for connecting cameras
- Prepare README for GitHub
- Prepare the final report
