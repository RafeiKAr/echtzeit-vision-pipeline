# Verification – P01 Echtzeit-Vision-Pipeline

**Goal:** ≥ 25 FPS at 640×480 on the laptop, with a table for 3 resolutions.
**Status:** table delivered ✅ · FPS target **not reached** ❌ (reason below)

## Measured (from `messung.md`)
Setup: HP True Vision 720p HD laptop camera, MediaPipe Face Detection (model 0, confidence 0.5),
3 runs × 30 s per resolution after 5 s warm-up, resolutions measured in rotating order.

| Requested | Delivered | FPS (mean) | Latency (mean) |
|---|---|---:|---:|
| 1280×720 | 640×480 | 14.25 | 65.01 ms |
| 640×480 | 640×480 | 14.25 | 64.28 ms |
| 320×240 | 320×240 | 14.25 | 65.67 ms |

## Why the target was not reached
1. **FPS is the same at every resolution.** If face detection were the bottleneck, 320×240 would be
   clearly faster than 640×480. It is not, so the limit comes from the camera.
2. **Latency ≈ camera frame interval.** 1 / 14.25 FPS ≈ 70 ms; the measured latency is 64–66 ms.
   Most of each loop is spent waiting in `camera.read()` for the next frame.
3. **1280×720 was not delivered.** The camera returned 640×480 for this request, so the "720p" row
   is in fact a second 640×480 measurement.

Conclusion: with this webcam and its current settings, the pipeline is camera-bound at about
14–15 FPS. The ≥ 25 FPS target cannot be shown with this setup.

## Next steps
- Request MJPG and 30 FPS from the camera (`CAP_PROP_FOURCC = MJPG`, `CAP_PROP_FPS = 30`),
  measure in bright light, and check supported modes (`v4l2-ctl --list-formats-ext` on Linux).
- Log `read`, `convert` and `detect` times as averages per run, to show the processing cost
  separately from the camera wait.
- Bonus: repeat the benchmark on a Raspberry Pi 5 (target ≥ 15 FPS at 640×480).
