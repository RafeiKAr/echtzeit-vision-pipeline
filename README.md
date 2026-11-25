# Echtzeit-Vision-Pipeline

> Real-time face detection from a webcam with built-in performance measurement (FPS, end-to-end latency). Runs on a laptop today; resolution and camera are command-line options, so the same code is ready for a Raspberry Pi 5.

<!-- AUTO:START (wird von `projekt.py readme` aktualisiert) -->
| | |
|---|---|
| **Status** | ✅ finished |
| **Goal** | ≥ 25 Bilder/s am Laptop bei 640×480, Tabelle mit 3 Bildgrößen in der README |
| **Tech** | Python, OpenCV, MediaPipe, Laptop-Kamera oder USB-Webcam · Bonus: Raspberry Pi 5 |
| **Focus** | Embedded Vision |
| **Bonus (hardware)** | ≥ 15 Bilder/s auf dem Raspberry Pi 5 bei 640×480, Vergleich Laptop vs. Pi in der README – planned |

<!-- AUTO:END -->

## Overview
A small computer-vision pipeline in Python: it reads a camera stream with OpenCV, detects faces with MediaPipe, draws the results and measures how fast the whole loop runs.
A built-in benchmark compares three requested resolutions (3 runs × 30 s each, 5 s warm-up) and writes the results to `results/messung.md` automatically.

**Main challenge:** separating what the *pipeline* costs from what the *camera* allows. The measurements showed the same 14.25 FPS at every resolution – the webcam, not the face detection, sets the limit (see [Verification](results/verify.md)).

## How it works
1. Capture frames from the laptop camera / USB webcam with OpenCV
2. Convert BGR → RGB and detect faces with MediaPipe Face Detection
3. Draw bounding boxes, confidence, FPS, latency and face count on the image
4. Benchmark mode: 3 resolutions × 3 runs in rotating order, requested **and** actual resolution logged per run
5. Hardware-ready: `--breite/--width`, `--hoehe/--height`, `--kamera/--camera`, `--confidence`, `--duration`, `--warmup`

## Results (laptop)
Laptop camera: HP True Vision 720p HD · mean of 3 runs à 30 s · details: [`results/messung.md`](results/messung.md)

| Requested resolution | Delivered by camera | FPS | Latency per frame |
|---|---|---:|---:|
| 1280×720 | 640×480 | 14.25 | 65.0 ms |
| 640×480 (target ≥ 25 FPS) | 640×480 | 14.25 | 64.3 ms |
| 320×240 | 320×240 | 14.25 | 65.7 ms |

**Result:** target of ≥ 25 FPS at 640×480 **not reached** with this webcam. FPS is identical at all resolutions and the latency (~65 ms) matches the camera's frame interval (1 / 14.25 FPS ≈ 70 ms), so the camera's frame rate is the bottleneck, not the processing. Reasoning and next steps: [`results/verify.md`](results/verify.md).

## Quick start
```bash
pip install -r requirements.txt
python src/main.py --breite 640 --hoehe 480 --kamera 0   # live view, quit with q or Esc
python src/main.py --benchmark                          # 3 resolutions × 3 runs → results/messung.md
pytest                                                  # tests
```

## Tests
`pytest` runs three tests for the command-line validation (normal case, boundary case, error case).
They replace OpenCV and MediaPipe with small stubs when those libraries are missing, so they run without a camera – also in GitHub Actions on every push.

## Raspberry Pi (planned)
The code is prepared for a Raspberry Pi 5 (target: ≥ 15 FPS at 640×480).
Measurements on the Pi will be added here as soon as the hardware is available.

```bash
# on the Raspberry Pi 5 (Raspberry Pi OS 64-bit)
pip install -r requirements.txt
python src/main.py --breite 640 --hoehe 480 --kamera 0
python src/main.py --benchmark
```

| Device | Resolution | FPS | Latency |
|---|---|---|---|
| Laptop | 640×480 | 14.25 | 64.3 ms |
| Raspberry Pi 5 | 640×480 | planned | planned |

## What I learned
- A camera does not always deliver the resolution you ask for – always log the *actual* frame size.
- Measure before you optimise: identical FPS at every resolution pointed to the camera, not to the code.
- Define latency precisely (here: camera read → detection → drawing → window update) and use warm-up time and repeated runs for reproducible numbers.
- Write tests that run without hardware, so CI can check the code on every push.

---
Author: [Khaled Rafei](https://github.com/RafeiKAr)
