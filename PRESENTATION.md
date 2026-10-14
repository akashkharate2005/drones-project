# AI-Based Smart Crowd Monitoring Payload for Surveillance Drones
## Autonomous Aerial Robotics & Computer Vision Engineering Presentation Deck

---

### Slide 1: Title & Executive Overview
* **Project**: AI-Based Smart Crowd Monitoring Payload for Surveillance Drones
* **Domain**: Autonomous Aerial Robotics | Edge AI | Computer Vision | Mechatronics
* **Architecture**: Simulation-First Hardware Abstraction (*Webots 3D + Video Feed + Webcam + SITL MAVLink*)
* **Key Capabilities**: Real-time person detection, ByteTrack tracking, 2D Gaussian crowd density heatmaps, close-proximity face recognition, 4-axis PID flight autonomy, and acoustic/visual alert deterrence.

---

### Slide 2: Problem Statement & Motivation
* **The Challenge**: Traditional ground surveillance has blind spots and fixed coverage during large gatherings, festivals, and public rallies.
* **The Danger**: Crowd congestion can rapidly escalate into stampedes and safety hazards without rapid aerial situational awareness.
* **The Solution**: An autonomous drone payload capable of:
  1. Continual high-altitude wide-area patrol (28m survey).
  2. Automated hotspot density detection and autonomous orbital tracking.
  3. Close-range descending facial inspection (4.5m) to verify authorized vs flagged individuals.
  4. Instant physical ground alerting via microcontroller-driven optical & acoustic deterrence.
  5. Multi-tiered failsafe logic (e.g. 20% critical battery automated Return-to-Home).

---

### Slide 3: End-to-End System Architecture

```
+-------------------------------------------------------------------------+
|                      SENSORS & SIMULATION INTERFACE                     |
|    WebotsBackend (Mavic 2 Pro)  |  VideoFileBackend  |  WebcamBackend   |
+------------------------------------+------------------------------------+
                                     | Camera Frame (960x720 BGR)
                                     v
+-------------------------------------------------------------------------+
|                             VISION PIPELINE                             |
|  Ultralytics YOLOv8n (Person)  ->  ByteTrack Multi-Object Tracker       |
|  2D Gaussian Density Heatmap   ->  InsightFace (ArcFace 512-D Embeddings)|
+------------------------------------+------------------------------------+
                                     | Count, Hotspot Centroid, Face Matches
                                     v
+-------------------------------------------------------------------------+
|                      DECISION & ALERTING SUBSYSTEM                      |
|  Debounced State Machine (NORMAL / WARNING / CRITICAL)                  |
|  Virtual Arduino Serial Protocol: <ALERT:...,LED:...,BUZ:...>           |
+-------------------+--------------------------------+--------------------+
                    | Mission Trigger                | Visual/Acoustic Packets
                    v                                v
+------------------------------------+  +---------------------------------+
|       FLIGHT CONTROL & AUTONOMY    |  |     OPERATOR HUD & TELEMETRY    |
|  Mission Manager (PATROL/ORBIT/RTH)|  |  OpenCV 1280x720 Tactical HUD   |
|  4-Axis PID Controller & Mixer     |  |  Structured CSV & JSON Telemetry|
+------------------------------------+  +---------------------------------+
```

---

### Slide 4: Computer Vision Pipeline

#### 1. Person Detection & Tracking
* **Detector**: Ultralytics YOLOv8 Nano (`yolov8n.pt`), inference size 960px.
* **Class Filtering**: Strictly COCO Class 0 (`person`), confidence cutoff $conf = 0.35$.
* **Tracker**: ByteTrack association matching high- and low-confidence detection boxes using Kalman motion prediction, preventing ID switches.

#### 2. Spatial Density Heatmap
* **2D Grid**: Computes spatial occupancy over a 10x10 normalized plaza grid.
* **Kernel Smoothing**: 2D isotropic Gaussian kernel:
  $$K(d) = \exp\left(-\frac{d^2}{2\sigma^2}\right)$$
* **Hotspot Centroid**: Identifies the mathematical center of mass of the densest crowd cluster, commanding the flight controller to orbit the centroid.

---

### Slide 5: Deep Facial Recognition Subsystem
* **Architecture**: Multi-backend fallback hierarchy:
  1. **InsightFace (ArcFace ResNet)** with ONNX Runtime: Extracts 512-dimensional normalized face embeddings:
     $$D(u, v) = 1.0 - (u \cdot v)$$
  2. **dlib / `face_recognition`**: 128-dimensional Euclidean distance fallback.
  3. **OpenCV Cascade**: Detection heuristic (disallowed for identity claims to prevent false positives).
* **Reference Galleries**:
  * **Authorized**: Pre-enrolled personnel (Alice, Bob, Akash, Laksh, Ritu).
  * **Flagged / Suspect**: Flagged persons of interest (e.g. Charlie).
  * **Inspection Logic**: Activates dynamically when descending below $z \le 8.0\,\text{m}$.

---

### Slide 6: Finite State Machine & Decision Engine
* **Hysteresis Filtering**: 8-frame sliding temporal debounce prevents rapid alert flickering caused by transient detection fluctuations.
* **Alert States**:
  * `NORMAL`: Crowd count $< 12$, battery nominal, no suspect detected.
  * `WARNING`: Crowd count $12 \le count < 25$, or manual inspection trigger.
  * `CRITICAL`: Crowd count $\ge 25$, suspect person confirmed, or battery $\le 20\%$.
* **Failsafe Priority Overrides**:
  * Critical Battery ($\le 20\%$) immediately preempts all missions and enforces `RETURN_HOME` landing.

---

### Slide 7: Flight Dynamics & Autonomy
* **4-Axis PID Flight Controller**:
  * Altitude ($Z$), Roll ($\phi$), Pitch ($\theta$), and Yaw ($\psi$) control loops.
  * Rotor Mixer: Converts 4-axis control efforts to individual angular velocities for 4 Mavic 2 Pro brushless motors.
  * Geofence Guard: Constrains operations within 60x60m bounds $[-35, +35]\,\text{m}$.
* **Mission Modes**:
  1. `PATROL`: 28m cruise altitude lawnmower survey.
  2. `CROWD_HOTSPOT`: Autonomous circular orbit at radius $r = 12\,\text{m}$ around cluster centroid.
  3. `INSPECT_FACES`: Nadir pitch descent to $4.5\,\text{m}$ for high-resolution facial capture.
  4. `RETURN_HOME`: Climb to obstacle clearance altitude, transit to origin $(0, 0)$, and vertical touchdown landing.

---

### Slide 8: Virtual Arduino & Hardware Payload Interface
* **UART Serial Protocol**: 115200 baud streaming ASCII packets:
  ```
  <ALERT:LEVEL,LED:COLOR,BUZ:STATE>
  ```
* **Payload Actuation**:
  * `NORMAL`: Green LED solid, Buzzer Muted (`BUZ:0`).
  * `WARNING`: Amber LED blinking (2Hz), Pulsed Tone (`BUZ:1`).
  * `CRITICAL`: Red LED solid, Continuous 2000Hz Alarm (`BUZ:2`).
* **Hardware Porting**: Toggle `alerts.use_hardware_serial: true` in `config.yaml` to interface directly with physical Arduino Nano pins via PySerial.

---

### Slide 9: Experimental Verification & Quantitative Results

#### Comprehensive Benchmark Over 1,200 Evaluation Frames:
| Evaluation Metric | Measured Value | Standard Benchmark Target |
|---|---|---|
| **Global Count MAE** | **6.73 persons** | $\le 8.0$ persons |
| **Global Count RMSE** | **7.70 persons** | $\le 10.0$ persons |
| **Global Detection Precision** | **81.7 %** | $\ge 75.0 \%$ |
| **Global Detection Recall** | **97.9 %** | $\ge 90.0 \%$ |
| **Global F1-Score** | **89.1 %** | $\ge 80.0 \%$ |
| **Mean Alert Dispatch Latency**| **0.60 seconds** | $\le 1.0$ second |

#### Altitude Stratification:
* **High Altitude ($\ge 25\,\text{m}$)**: F1-Score **90.4%** (Optimal broad surveillance).
* **Mid Altitude ($15 - 25\,\text{m}$)**: F1-Score **85.6%** (Transit phase).
* **Low Altitude ($< 15\,\text{m}$)**: F1-Score **68.0%** (Tight inspection FOV, 100% recall).

---

### Slide 10: Deliverables & Presentation Artifacts
1. **Interactive Terminal Launcher (`launch.py` / `launch.bat`)**: 1-click execution for all 9 project capabilities.
2. **Web Mission Control (`showcase.html`)**: Self-contained browser application featuring live Chart.js telemetry, phase HUD explorer, teleprompter timer, and benchmark zooms.
3. **Automated Unit Test Suite (`pytest tests/`)**: 19 offline unit tests covering perception, state machine, flight control, and serial protocols.
4. **Standalone Face Demo (`face_demo.py`)**: Instant multi-person test gallery detection and gallery verification grid.
5. **Evaluation Engine (`evaluate.py`)**: Automated generation of 4 publication-ready figures in `results/`.

---

### Slide 11: Hardware Deployment Roadmap
* **Step 1**: Flash Arduino Nano with the provided 15-line UART sketch driving Digital Pins 2, 3, 4 (LEDs) and 5 (Piezo).
* **Step 2**: Connect Raspberry Pi 4 USB 3.0 to optical camera, and USB serial to Arduino.
* **Step 3**: Connect Pi UART (`/dev/ttyAMA0`) to Pixhawk `TELEM2` port running ArduPilot.
* **Step 4**: Enable PyMAVLink dispatch in `src/sim/mavlink_stub.py` to route waypoint commands (`SET_POSITION_TARGET_LOCAL_NED`) and gimbal pitch commands directly to physical ESCs and servos.

---

### Slide 12: Conclusion & Q&A
* Fully integrated end-to-end aerial surveillance pipeline.
* Seamless abstraction between 3D simulation and physical avionics.
* High accuracy ($F_1 = 89.1\%$) and sub-second alert response ($0.6\,\text{s}$).
* Production-grade repository hygiene with comprehensive testing, interactive showcase, and packaging.

*Thank you! Questions and live demonstrations welcomed.*

