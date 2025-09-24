# Walkway Safety Monitoring System

## Features

### Walkway Violation Detection
- Detects humans in live video streams using **YOLO**.  
- Tracks movements with **SORT (Simple Online Realtime Tracking)**.  
- Determines if a person is inside or outside the walkway polygon.  
- Implemented in:  
  - `human_tracking.py`  
  - `human_tracking_alarm.py`  
  - `human_walkway_SAM.py`  
  - `human_walkway_SAM_alarm.py`  

---

### Alarms & Alerts
- **Audible alarms** triggered via Tapo cameras (speaker API).  
- Alarm automatically starts when a violation is detected and stops when cleared.  
- Implemented in:  
  - `human_tracking_alarm.py`  
  - `human_walkway_SAM_alarm.py`  

---

### Incident Logging
- Logs each violation with:  
  - Person ID  
  - Start & end time  
  - Duration of violation  
  - Captured image evidence  
  - Image/video uploaded to **Firebase Storage**  
  - Records stored in **Firebase Realtime Database**  
- Implemented in:  
  - `human_tracking.py`  
  - `human_tracking_alarm.py`  

---

### Mobile & Cloud Integration
- Real-time logs can be accessed via **Firebase**.  
- Designed to integrate with a mobile app for notifications and log review.  

---

### Independent Multi-Camera Support 
- Supports both **indoor corridors** and **outdoor areas**.  

---

### System Adaptability
- Walkway zones can be set manually (**polygon drawing**) in:  
  - `human_tracking.py`  
  - `human_tracking_alarm.py`  

- Or automatically via **MATLAB-assisted segmentation** in:  
  - `human_walkway_SAM.py`  
  - `human_walkway_SAM_alarm.py`  

- Adjustable to varying walkway widths, lighting, and environments.  

---

### Scalable Logging
- Local CSV logs:  
  - `total_positive_detections.csv` → people detected per frame  
  - `tracking_durations.csv` → per-person violation durations  
- Cloud logs stored in **Firebase** for centralized monitoring.  

---

## System Components
- `human_tracking.py` → Core detection & tracking with Firebase integration.  
- `human_tracking_alarm.py` → Adds alarm system (Tapo camera speaker).  
- `human_walkway_SAM.py` → Uses MATLAB engine for walkway segmentation.  
- `human_walkway_SAM_alarm.py` → Combines MATLAB segmentation + alarm system.  
