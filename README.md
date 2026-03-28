# Artificial intelligence system for detecting safety violations 

## Description

A local application using Yolo to automatically detect industrial safety violations in real time. The system processes video streams from cameras and USB devices, recognizes the absence of helmets and protective clothing, and records incidents with screenshots saved locally.

---

## Development plan

###1. Making a plan and creating a repository on GitHub
- Defining the project structure (folders, modules)
- Create a repository on GitHub
- Set it up. "gitignore"
- Draw up and approve a work plan

---

###2. Connecting libraries and installing utilities
- Setting up the development environment (CMake, compiler)
- PPE connection for working with videos and images
- Connecting a library for neural network output
- Connecting the library for logging 
- Setting up the project build

---

### 3. We connect the AI and configure it to work correctly
- We are implementing a video capture module 
- Choosing the neural network architecture (YOLO / SSD)
- Integrate a pre-trained object detection model
- Configuring logical output on the target platform (CPU/GPU)

---

### 4. Neural network training for helmet recognition and implementation of basic logic
- Preparation and annotation of a dataset (people, helmets, vests)
- Performing model training
- Display of detected violations on the screen
- Checking the quality of speech recognition
- Implementation of the basic logic for detecting violations

---

###5. Add the phone's camera, recognize other security features, and save them
- We implement support for broadcasting from the phone 
- Expanding the data set: adding classes for protective vests, gloves, glasses
- Additionally, we are teaching the model new classes
- Integrate the recognition of additional security features into the main logic
- Set up separate rules for each type of violation
- Implement saving screenshots of violations

---

### 6. Documentation and revision
- Write technical documentation (architecture, installation, configuration)
- Write user documentation (launching, working with the application)
- Prepare instructions for connecting cameras
- Prepare a README for GitHub
- Prepare the final report
