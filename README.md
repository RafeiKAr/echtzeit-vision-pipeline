# Echtzeit-Vision-Pipeline

> Real-time face detection from a webcam with built-in performance measurement (FPS, latency). Runs on a laptop today; the same code is ready for a Raspberry Pi 5.

<!-- AUTO:START (wird von `projekt.py readme` aktualisiert) -->
| | |
|---|---|
| **Status** | 🚧 in progress (0 %) |
| **Goal** | ≥ 25 Bilder/s am Laptop bei 640×480, Tabelle mit 3 Bildgrößen in der README |
| **Tech** | Python, OpenCV, MediaPipe, Laptop-Kamera oder USB-Webcam · Bonus: Raspberry Pi 5 |
| **Focus** | Embedded Vision |
| **Bonus (hardware)** | ≥ 15 Bilder/s auf dem Raspberry Pi 5 bei 640×480, Vergleich Laptop vs. Pi in der README – planned |

<!-- AUTO:END -->

## Overview
A small computer-vision pipeline that reads a camera stream, detects faces and measures how fast it runs.
It compares three image sizes to show the trade-off between resolution and speed.
Image size and camera are settings, so the same code can run on embedded hardware such as a Raspberry Pi 5.

<!-- 🔴 1–2 Sätze ergänzen: Was war die größte Herausforderung? -->

## How it works
1. Capture frames from the laptop camera / USB webcam with OpenCV
2. Detect faces (OpenCV / MediaPipe)
3. Measure FPS and per-frame latency and draw them on the image
4. Compare three image sizes (1280×720, 640×480, 320×240)
5. Hardware-ready: resolution and camera are command-line options

## Results (laptop)
Each value: mean of 3 runs à 30 s. Details: `results/messung.md`

| Resolution | FPS | Latency per frame |
|---|---|---|
| 1280×720 | 🔴 | 🔴 |
| 640×480 (target ≥ 25 FPS) | 🔴 | 🔴 |
| 320×240 | 🔴 | 🔴 |

<!-- Bilder/GIFs aus results/ hier einbinden: ![demo](results/demo.gif) -->

## Quick start
```bash
pip install -r requirements.txt
python src/main.py --breite 640 --hoehe 480 --kamera 0
pytest
```

## Raspberry Pi (planned)
The code is prepared for a Raspberry Pi 5 (target: ≥ 15 FPS at 640×480).
Measurements on the Pi will be added here as soon as the hardware is available.

```bash
# on the Raspberry Pi 5 (Raspberry Pi OS 64-bit)
pip install -r requirements.txt
python src/main.py --breite 640 --hoehe 480 --kamera 0
```

| Device | Resolution | FPS | Latency |
|---|---|---|---|
| Laptop | 640×480 | see above | see above |
| Raspberry Pi 5 | 640×480 | planned | planned |

## What I learned
- 🔴

---
Author: [Khaled Rafei](https://github.com/RafeiKAr)
