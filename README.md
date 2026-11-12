# Echtzeit-Vision-Pipeline

> Real-time face detection from a webcam with built-in performance measurement (FPS, latency) – first on a laptop, then on a Raspberry Pi 5.

<!-- AUTO:START (wird von `projekt.py readme` aktualisiert) -->
| | |
|---|---|
| **Status** | 🚧 in progress |
| **Goal** | ≥ 15 Bilder/s auf dem Raspberry Pi bei 640×480, Messtabelle in der README |
| **Tech** | Python, OpenCV, MediaPipe, USB-Webcam, Raspberry Pi 5 (Labor oder gekauft) |
| **Focus** | Embedded Vision |
<!-- AUTO:END -->

## Overview
A small computer-vision pipeline that reads a camera stream, detects faces and measures how fast it runs.
The same code runs on a laptop and on a Raspberry Pi 5, so the results show what is realistic on embedded hardware.

<!-- 🔴 1–2 Sätze ergänzen: Was war die größte Herausforderung? -->

## How it works
1. Capture frames from a USB webcam with OpenCV
2. Detect faces (OpenCV / MediaPipe)
3. Measure FPS and per-frame latency and draw them on the image
4. Deploy and run the same code on a Raspberry Pi 5
5. Compare laptop vs. Raspberry Pi in a results table

## Results
| Metric | Target | Achieved |
|---|---|---|
| FPS on Raspberry Pi 5 (640×480) | ≥ 15 | 🔴 |
| FPS on laptop (640×480) | – | 🔴 |
| Latency per frame | – | 🔴 |

<!-- Bilder/GIFs aus results/ hier einbinden: ![demo](results/demo.gif) -->

## Quick start
```bash
pip install -r requirements.txt
python src/main.py
pytest
```

## What I learned
- 🔴

---
Author: [Khaled Rafei](https://github.com/RafeiKAr)
