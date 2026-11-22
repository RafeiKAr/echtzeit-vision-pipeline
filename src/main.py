"""Echtzeit-Vision-Pipeline – Einstiegspunkt."""

import argparse

import cv2
import mediapipe as mp



def main() -> None:
    print("Echtzeit-Vision-Pipeline läuft.")

    parser = argparse.ArgumentParser(
        description="Gesichtserkennung mit MediaPipe Face Detection"
    )
    parser.add_argument("--camera", type=int, default=0, help="Kamera-Index")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Mindest-Confidence von 0 bis 1 (Standard: 0.5)",
    )
    args = parser.parse_args()

    if not 0 <= args.confidence <= 1:
        parser.error("--confidence muss zwischen 0 und 1 liegen.")

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Kamera {args.camera} konnte nicht geöffnet werden.")

    face_detection = mp.solutions.face_detection

    # model_selection=0 erkennt Gesichter aus kurzer Distanz (z. B. Webcam).
    # model_selection=1 ist für größere Entfernungen gedacht.
    with face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=args.confidence,
    ) as detector:
        try:
            print("MediaPipe läuft. Beenden mit q oder Esc.")
            while True:
                success, frame = camera.read()
                if not success:
                    raise RuntimeError("Es konnte kein Kamerabild gelesen werden.")

                # MediaPipe erwartet RGB statt des von OpenCV verwendeten BGR.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = detector.process(rgb_frame)

                frame_height, frame_width = frame.shape[:2]
                detections = results.detections or []

                for detection in detections:
                    bounding_box = detection.location_data.relative_bounding_box
                    x = max(0, int(bounding_box.xmin * frame_width))
                    y = max(0, int(bounding_box.ymin * frame_height))
                    width = min(frame_width - x, int(bounding_box.width * frame_width))
                    height = min(frame_height - y, int(bounding_box.height * frame_height))

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + width, y + height),
                        (255, 0, 0),
                        2,
                    )

                    confidence = detection.score[0]
                    cv2.putText(
                        frame,
                        f"Gesicht {confidence:.2f}",
                        (x, max(25, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2,
                    )

                cv2.putText(
                    frame,
                    f"Gesichter: {len(detections)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Gesichtserkennung - MediaPipe", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
