# CYN-X Vision 👁️

**Vision and perception system for CYN-X**

CYN-X Vision is the computer-vision subsystem of the CYN-X platform.

Its purpose is to give CYN-X the ability to **see, interpret, and understand its environment** through cameras and other visual sensors.

The project is designed to start as a standalone software system and eventually connect directly into the larger CYN-X architecture and physical robot platform.

---

## 🧠 What Is CYN-X Vision?

CYN-X Vision transforms raw camera data into structured information that the rest of CYN-X can understand.

```text
                CAMERA
                   │
                   ▼
          ┌─────────────────┐
          │  Vision System  │
          │                 │
          │ Capture         │
          │ Detection       │
          │ Tracking        │
          │ Perception      │
          └────────┬────────┘
                   │
                   ▼
          Structured Perception
                   │
                   ▼
          ┌─────────────────┐
          │     CYN-X       │
          │                 │
          │ World Model     │
          │ Reasoning       │
          │ Decision Making │
          └─────────────────┘
```

Instead of CYN-X directly dealing with raw camera frames, Vision provides meaningful information such as:

* People
* Objects
* Object positions
* Movement
* Confidence scores
* Scene changes
* Tracking information
* Future spatial/depth information

---

## 🎯 Goals

The initial goals of CYN-X Vision are:

1. Capture video from a camera.
2. Process camera frames.
3. Detect objects.
4. Track objects over time.
5. Convert detections into structured perception data.
6. Send perception events to CYN-X.
7. Allow CYN-X to maintain an environmental **World Model**.
8. Provide a foundation for future robotic vision.

---

## 🏗️ Architecture

CYN-X Vision is intentionally separated from the physical robot.

```text
┌──────────────────────────────────────────┐
│                CYN-X                     │
│                                          │
│  Reasoning → Planning → World Model     │
└─────────────────────▲────────────────────┘
                      │
                Perception API
                      │
┌─────────────────────┴────────────────────┐
│             CYN-X Vision                 │
│                                          │
│  Camera → Detection → Tracking           │
│              → Perception                │
└─────────────────────▲────────────────────┘
                      │
                    Camera
```

This allows the vision system to work on:

* A development PC
* A laptop
* A Raspberry Pi
* An embedded Linux computer
* The future CYN-X robot computer

---

## 📁 Project Structure

```text
cynx-vision/
│
├── app.py
├── config.py
│
├── vision/
│   ├── __init__.py
│   ├── camera.py
│   ├── detector.py
│   └── perception.py
│
├── cynx/
│   ├── __init__.py
│   ├── client.py
│   └── events.py
│
├── models/
│   └── README.md
│
├── tests/
│   └── __init__.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 👁️ Vision Pipeline

The vision pipeline is divided into several stages.

### 1. Camera

The camera subsystem is responsible for acquiring frames.

```text
Camera
  ↓
Frame
```

The camera interface should eventually support multiple camera types without requiring changes to the rest of the vision system.

---

### 2. Detection

The detector analyzes frames and identifies objects.

Example:

```json
{
  "type": "person",
  "confidence": 0.94
}
```

The detector should remain replaceable so different computer-vision models can be used later.

---

### 3. Tracking

Tracking allows CYN-X to understand that an object detected in multiple frames is the **same object**.

Example:

```text
Frame 1 → Person #1
Frame 2 → Person #1
Frame 3 → Person #1
Frame 4 → Person #1 moved
```

This creates persistent object identities.

---

### 4. Perception

The perception layer converts raw detections into information meaningful to CYN-X.

Example:

```json
{
  "timestamp": "2026-09-08T12:00:00Z",
  "objects": [
    {
      "id": 1,
      "type": "person",
      "confidence": 0.94,
      "position": {
        "x": 0.21,
        "y": -0.08
      }
    }
  ]
}
```

---

## 🔌 CYN-X Integration

CYN-X Vision communicates with the larger CYN-X system through an API/event interface.

Possible events include:

```text
VISION.PERSON_DETECTED
VISION.OBJECT_DETECTED
VISION.PERSON_MOVED
VISION.OBJECT_MOVED
VISION.SCENE_CHANGED
VISION.NO_OBJECTS
```

The vision system should not make high-level decisions.

For example:

```text
Vision:
"Person detected at x=0.21, y=-0.08."

CYN-X:
"There is a person in front of me."

CYN-X reasoning:
"The person appears to be approaching."

Robot system:
"Adjust head/camera orientation."
```

This separation keeps perception, reasoning, and physical control independent.

---

# 🌎 World Model

CYN-X Vision is intended to become one of the primary sources of information for the CYN-X World Model.

Eventually, CYN-X could maintain something similar to:

```text
WORLD
│
├── Person #1
│   ├── Position
│   ├── Movement
│   └── Confidence
│
├── Laptop #1
│   ├── Position
│   └── State
│
├── Chair #1
│   └── Position
│
└── Environment
    ├── Lighting
    ├── Room
    └── Scene State
```

The World Model can then combine vision with other sensors.

```text
              ┌───────────┐
              │  Vision   │
              └─────┬─────┘
                    │
┌───────────┐       │       ┌───────────┐
│    IMU    │───────┼──────▶│           │
└───────────┘       │       │ WORLD     │
                    ├──────▶│ MODEL     │
┌───────────┐       │       │           │
│   Audio   │───────┤       └─────┬─────┘
└───────────┘       │             │
                    │             ▼
              ┌─────┴─────┐  CYN-X Reasoning
              │  Sensors  │
              └───────────┘
```

---

# 🚀 Development Roadmap

## Phase 1 — Camera

* [ ] Open webcam
* [ ] Capture frames
* [ ] Display live video
* [ ] Create camera abstraction

## Phase 2 — Object Detection

* [ ] Add object detector
* [ ] Detect people
* [ ] Detect common objects
* [ ] Add confidence scores
* [ ] Draw detection boxes

## Phase 3 — Tracking

* [ ] Assign object IDs
* [ ] Track objects between frames
* [ ] Detect movement
* [ ] Handle objects entering/leaving the scene

## Phase 4 — Perception API

* [ ] Create perception data model
* [ ] Create JSON representation
* [ ] Create CYN-X client
* [ ] Send perception events
* [ ] Handle connection failures

## Phase 5 — CYN-X World Model

* [ ] Create world-state interface
* [ ] Store detected entities
* [ ] Update entity positions
* [ ] Track entity state
* [ ] Merge information from multiple sensors

## Phase 6 — Advanced Vision

Future capabilities may include:

* [ ] Face detection
* [ ] Pose estimation
* [ ] Hand tracking
* [ ] Depth perception
* [ ] Spatial mapping
* [ ] Optical flow
* [ ] Scene recognition
* [ ] Object relationships
* [ ] 3D object positioning

## Phase 7 — Robot Integration

Eventually CYN-X Vision can run on the CYN-X robot computer.

```text
Camera
   ↓
CYN-X Vision
   ↓
CYN-X Perception
   ↓
World Model
   ↓
CYN-X Intelligence
   ↓
Robot Control
```

---

# 🧩 Design Principles

### Hardware Independence

Vision should not depend on one specific camera.

```text
Camera Interface
       │
 ┌─────┼─────┐
 ▼     ▼     ▼
USB   CSI   IP Camera
```

---

### Model Independence

The perception system should not depend permanently on one AI model.

```text
Detector Interface
       │
 ┌─────┼──────────┐
 ▼     ▼          ▼
YOLO  OpenCV   Future Model
```

---

### CYN-X Independence

The vision subsystem should be able to operate independently from the rest of CYN-X during development.

```text
CYN-X Vision
     │
     ├── Standalone Mode
     │
     └── CYN-X Connected Mode
```

---

### Structured Data

Vision should communicate **information**, not raw implementation details.

Bad:

```text
"YOLO detected class 0."
```

Better:

```json
{
  "type": "person",
  "confidence": 0.94
}
```

This allows CYN-X to remain independent from the underlying vision technology.

---

# 🛠️ Technology

The initial implementation is planned around:

* **Python**
* **OpenCV**
* Computer-vision / object-detection models
* REST or event-based communication
* JSON perception data
* Linux

Additional technologies can be introduced as the system evolves.

---

# 🔮 Long-Term Vision

CYN-X Vision is intended to become more than a camera program.

The long-term goal is to create a **general perception layer for CYN-X**.

```text
                    CYN-X
                      │
              ┌───────┴───────┐
              │  World Model  │
              └───────┬───────┘
                      │
                 Perception
                      │
        ┌─────────────┼─────────────┐
        │             │             │
      Vision        Audio         Sensors
        │             │             │
      Camera       Microphone     Hardware
```

The same perception architecture can eventually be used whether CYN-X is running on a desktop computer, embedded system, or physical robot.

---

# 📜 Status

**Early Development**

CYN-X Vision is currently a foundational project for the larger CYN-X ecosystem.

The initial priority is building a reliable software perception pipeline before connecting it to the physical robot platform.

---

# 🤖 CYN-X

CYN-X is a modular AI and robotics platform focused on connecting **intelligence, perception, software, hardware, and the physical world**.

CYN-X Vision provides the eyes.

**CYN-X provides the mind.**
#   c y n - x - v i s i o n  
 