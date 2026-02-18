# Real-Time Emotion Detection System

### RetinaFace + DeepFace + Multi-Face Tracking + Heatmap + CSV + PNG Export

## Overview

This project implements a **real-time emotion detection system** using:

* **RetinaFace** for face detection
* **DeepFace** for emotion classification
* **Multi-face tracking with persistent IDs**
* **Heatmap visualization**
* **Statistical logging**
* **CSV export**
* **Final emotion trend graph exported as PNG**

The system is designed for:

* Computer vision experimentation
* Behavioral analytics
* Emotion distribution analysis
* Research and prototyping
* Dataset generation

---

## Key Features

### Face Detection

* RetinaFace backend (robust under varied lighting and angles)
* Multi-face support
* Stable tracking with unique face IDs

### Emotion Analysis

* DeepFace emotion model
* Configurable analysis frequency (performance optimization)
* Cached inference to reduce GPU/CPU usage

### Visualization

* Real-time bounding boxes
* Emotion labels with confidence percentage
* Optional emotional heatmap overlay
* FPS monitor
* HUD toggle

### Data & Analytics

* Per-frame emotion logging
* Global emotion statistics
* Per-face emotion statistics
* CSV export (full session data)
* Final rolling emotion trend graph exported as PNG

---

## Architecture

```
EmotionDetector/
├── app.py
├── requirements.txt
├── emotion_log.csv         # generated after run
├── emotion_plot.png        # generated after run
```

---

## Installation

### Requirements

* Python 3.8+
* Webcam
* CPU (GPU optional but recommended)

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

### Default run

```bash
python app.py
```

This will:

* Start camera
* Detect faces using RetinaFace
* Classify emotions
* Generate:

  * `emotion_log.csv`
  * `emotion_plot.png`

---

### Enable heatmap

```bash
python app.py --heatmap
```

---

### Improve performance (less frequent inference)

```bash
python app.py --analyze-every 10
```

---

### Custom CSV and PNG output

```bash
python app.py --csv session1.csv --plot-png session1.png
```

---

## Command Line Arguments

| Argument          | Description                            |
| ----------------- | -------------------------------------- |
| `--camera`        | Camera index (default 0)               |
| `--width`         | Frame width                            |
| `--height`        | Frame height                           |
| `--analyze-every` | Run emotion analysis every N frames    |
| `--heatmap`       | Enable emotional heatmap overlay       |
| `--csv`           | Output CSV file path                   |
| `--plot-png`      | Final PNG graph output path            |
| `--series-window` | Number of ticks stored for final graph |

---

## Controls (Keyboard)

| Key | Action         |
| --- | -------------- |
| `Q` | Quit           |
| `H` | Toggle HUD     |
| `M` | Toggle heatmap |

---

## How It Works

### 1. Detection

Faces are detected using:

```
DeepFace.extract_faces(..., detector_backend="retinaface")
```

This provides high-quality face localization.

---

### 2. Tracking

A lightweight centroid-based tracker assigns stable IDs to faces across frames.

This prevents:

* ID switching
* Emotion mixing across subjects

---

### 3. Emotion Classification

DeepFace analyzes cropped face regions.

To improve performance:

* Analysis runs every N frames
* Results are cached between updates

---

### 4. Heatmap

A decaying heatmap accumulates face presence weighted by dominant emotion confidence.

Visualization:

* Gaussian blobs
* Inferno colormap
* Alpha blending over video

---

### 5. CSV Logging

Each detection writes:

```
timestamp
face_id
emotion
confidence
bounding_box
full emotion scores (JSON)
```

Example row:

```
2025-02-18T21:03:44,1,happy,87.2,120,90,80,80,{"happy":87.2,"sad":2.1,...}
```

---

### 6. Final PNG Graph Export

At program exit:

* Rolling emotion counts are plotted
* Graph is saved as PNG

This allows:

* Post-session analysis
* Reports
* Presentation-ready visuals

---

## Output Files

### CSV

Contains full session logs for analysis.

### PNG

Displays rolling emotion frequency per tick.

Example graph includes:

* angry
* disgust
* fear
* happy
* sad
* surprise
* neutral

---

## Performance Notes

* RetinaFace is computationally heavier than Haar.
* Increase `--analyze-every` for better FPS.
* Reduce resolution for better performance.
* GPU significantly improves inference speed.

---

## Limitations

* Emotion detection is probabilistic
* Accuracy varies by lighting conditions
* Not identity recognition
* Not optimized for low-power hardware
* Emotion inference may fluctuate frame-to-frame
