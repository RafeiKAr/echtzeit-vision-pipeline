"""P01 tests – run with: pytest"""

import sys
import types
from pathlib import Path

import pytest


# Add the src directory to sys.path so the module can be imported as main.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# Replace cv2 so that the three argument tests run without a real camera library.
try:
    import cv2
except ModuleNotFoundError:
    cv2 = types.ModuleType("cv2")
    cv2.VideoCapture = object
    cv2.CAP_PROP_FRAME_WIDTH = 3
    cv2.CAP_PROP_FRAME_HEIGHT = 4
    cv2.destroyAllWindows = lambda: None
    sys.modules["cv2"] = cv2


# Replace mediapipe so that only the command-line argument validation is tested.
try:
    import mediapipe
except ModuleNotFoundError:
    mediapipe = types.ModuleType("mediapipe")
    mediapipe.solutions = types.SimpleNamespace(
        face_detection=types.SimpleNamespace(FaceDetection=object)
    )
    sys.modules["mediapipe"] = mediapipe


# Import main after adding the src path and defining the optional library stubs.
import main


# Normal case: valid German camera options are read correctly.
def test_valid_camera_options(monkeypatch):
    # Simulate starting python src/main.py with three valid options.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--breite",
            "640",
            "--hoehe",
            "480",
            "--kamera",
            "0",
        ],
    )

    # parse_arguments reads and validates the simulated command-line values.
    args = main.parse_arguments()

    # The German width option must be stored as the width value 640.
    assert args.width == 640
    # The German height option must be stored as the height value 480.
    assert args.height == 480
    # The German camera index must be stored as the camera value 0.
    assert args.camera == 0


# Boundary case: a positive minimum duration and zero warmup are valid.
def test_positive_measurement_duration_and_zero_warmup(monkeypatch):
    # Simulate a very short measurement run without a warmup phase.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--duration",
            "0.01",
            "--warmup",
            "0",
        ],
    )

    # parse_arguments checks both boundary values.
    args = main.parse_arguments()

    # 0.01 seconds is the positive minimum duration used in this test.
    assert args.duration == 0.01
    # A warmup time of zero seconds is still valid.
    assert args.warmup == 0


# Error case: width and height must be specified together.
def test_incomplete_resolution_is_rejected(monkeypatch):
    # sys.argv contains only the width and not the corresponding height.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--breite",
            "640",
        ],
    )

    # argparse raises SystemExit when invalid command-line input is detected.
    with pytest.raises(SystemExit):
        main.parse_arguments()
