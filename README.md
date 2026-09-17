# Real-Time Traffic Analytics

**A computer vision system that detects and tracks vehicles in dashcam video and estimates their real-world speed using perspective transformation (homography).**

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![YOLOv11](https://img.shields.io/badge/model-YOLOv11-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How Speed Estimation Works](#how-speed-estimation-works)
- [Installation](#installation)
- [Configuration](#configuration)
- [Output](#output)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

**Real-Time Traffic Analytics** processes dashcam footage to detect vehicles, track them consistently across frames, and estimate their speed in real-world units. It uses a homography transform to correct for perspective distortion, so speed estimates reflect actual distance traveled rather than raw pixel movement, and flags any vehicle exceeding a configurable speed threshold.

## Features

- **Vehicle Detection** — Detects cars, trucks, motorcycles, and buses using a YOLOv11 model
- **Object Tracking** — Maintains consistent vehicle IDs across frames with ByteTrack
- **Speed Estimation** — Maps pixel coordinates to real-world meters via perspective transform
- **Anomaly Detection** — Highlights vehicles exceeding the speed limit in red
- **Live Dashboard** — On-video overlay showing real-time statistics

## How Speed Estimation Works

Objects farther from the camera appear smaller due to perspective distortion, so the same pixel distance represents different real-world distances depending on where it falls in the frame. A fixed pixel-to-meter ratio can't account for this on its own.

To correct for it, the system uses a **homography**, a 3×3 matrix that maps points from the image plane onto a flat ground plane:

```
[x']   [h11 h12 h13]   [x]
[y'] = [h21 h22 h23] * [y]
[w']   [h31 h32 h33]   [1]

x' = (h11*x + h12*y + h13) / w'
y' = (h21*x + h22*y + h23) / w'
```

By defining a known rectangular region of road in the source video and mapping it to its actual real-world dimensions, pixel movement can be converted into meters, and meters per frame into a reliable speed estimate.

## Installation

### Requirements

- Python 3.8+
- Google Colab (recommended) or a local Python environment
- `ultralytics`, `supervision`, `opencv-python`, `numpy`

### Quick Start (Google Colab)

1. Open Google Colab.
2. Copy the cells from `traffic_analytics_colab.py` in order.
3. Run each cell top to bottom.
4. Adjust `SOURCE_POLYGON` to match the road region in your own video.

## Configuration

Update the following in `config.py` (or directly in the notebook) for your video:

```python
# Detection zone: 4 points marking a rectangular road section, in pixels
SOURCE_POLYGON = np.array([
    [400, 300],   # Top-left
    [880, 300],   # Top-right
    [1100, 600],  # Bottom-right
    [180, 600],   # Bottom-left
], dtype=np.float32)

# Real-world dimensions of that same road section, in meters
TARGET_RECT = np.array([
    [0, 0], [3.7, 0], [3.7, 20], [0, 20]  # 3.7m wide, 20m long
], dtype=np.float32)

SPEED_LIMIT_KMH = 100.0  # Speeding threshold
```

`SOURCE_POLYGON` must trace a real rectangular section of road (e.g. a lane between two lane markings) for the speed calculation to be accurate. `TARGET_RECT` should reflect that section's true width and length.

## Output

The system produces an annotated video containing:

- A cyan overlay marking the detection zone
- Green bounding boxes for vehicles within the speed limit
- Red bounding boxes for vehicles exceeding it
- A speed label on each tracked vehicle
- A live statistics dashboard overlay

## Project Structure

```
real-time-traffic-analytics/
├── traffic_analytics_colab.py   # Main Colab notebook (copy cells)
├── config.py                     # Configuration parameters
├── perspective_transform.py      # Homography math module
├── tracker.py                    # Vehicle detection & tracking
├── visualizer.py                 # Annotation & drawing
└── README.md                      # Project documentation
```

## Tech Stack

| Component | Tool |
|---|---|
| Detection model | YOLOv11-Medium (`ultralytics`) |
| Tracking | ByteTrack (`supervision`) |
| Perspective transform | `cv2.getPerspectiveTransform` |
| Visualization | OpenCV + `supervision` |

## Limitations

- Speed accuracy depends on how precisely `SOURCE_POLYGON` and `TARGET_RECT` match the real road geometry.
- Assumes a flat road surface; significant inclines or curves will distort the homography.
- Detection quality depends on video resolution, lighting, and camera stability.
- Designed and tuned around dashcam footage; other camera angles may need different detection zone tuning.

## Roadmap

- [ ] Multi-lane speed tracking with per-lane statistics
- [ ] Exportable session logs (CSV/JSON) of tracked vehicles and speeds
- [ ] Automatic detection-zone calibration from lane markings
- [ ] Support for additional vehicle classes (bicycles, pedestrians)
- [ ] Local (non-Colab) run script for standalone use

## License

MIT License
