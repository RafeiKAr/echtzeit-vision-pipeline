"""Echtzeit-Vision-Pipeline – Einstiegspunkt."""


import argparse                     # argparse liest optionale Startparameter aus der Kommandozeile.
from collections import deque       # deque speichert die Zeitpunkte der letzten Bilder effizient als Warteschlange.
import time                         # time stellt eine hochaufloesende Uhr fuer die Messungen bereit.
import cv2                          # OpenCV liest die Kamera und zeichnet Texte sowie Rechtecke ins Bild.
import mediapipe as mp              # MediaPipe stellt die Gesichtserkennung bereit.


# main startet die Kamera, die Gesichtserkennung und das Anzeigefenster.
def main() -> None:


    # parser verwaltet die optionalen Einstellungen beim Programmstart.
    parser = argparse.ArgumentParser(
        description="Gesichtserkennung mit MediaPipe Face Detection"
    )
    

    parser.add_argument("--camera", type=int, default=0, help="Kamera-Index")                   # --camera waehlt die Kamera aus; standardmaessig wird Kamera 0 verwendet.


    # --confidence legt die minimale Erkennungssicherheit als Zahl von 0 bis 1 fest.
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Mindest-Confidence von 0 bis 1 (Standard: 0.5)",
    )
    
    args = parser.parse_args()                                                                  # parse_args liest die angegebenen Kommandozeilenwerte ein.


    # Eine Confidence ausserhalb des erlaubten Bereichs wird als Eingabefehler abgelehnt.
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence muss zwischen 0 und 1 liegen.")

    
    camera = cv2.VideoCapture(args.camera)                                                      # VideoCapture verbindet OpenCV mit der ausgewaehlten Kamera.
    

    # Ohne geoeffnete Kamera kann keine Bildverarbeitung gestartet werden.
    if not camera.isOpened():
        camera.release()                                                                        # Gibt die Kameraressource auch im Fehlerfall wieder frei.
        raise RuntimeError(f"Kamera {args.camera} konnte nicht geöffnet werden.")               # Meldet dem Benutzer explizit, dass die Kamera nicht verfuegbar ist.


    
    face_detection = mp.solutions.face_detection                                                # face_detection enthaelt die Gesichtserkennungsfunktionen von MediaPipe.

    # model_selection=0 erkennt Gesichter aus kurzer Distanz (z. B. Webcam).
    # model_selection=1 ist für größere Entfernungen gedacht.


    # Der with-Block schliesst den MediaPipe-Detektor nach der Verwendung korrekt.
    with face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=args.confidence,
    ) as detector:
        
        frame_times: deque[float] = deque()                                                     # frame_times enthaelt die Startzeitpunkte der kuerzlich verarbeiteten Bilder.
        fps = 0.0                                                                               # fps wird erst berechnet, wenn mindestens zwei Bilder gemessen wurden.
        last_fps_update = 0.0                                                                   # last_fps_update merkt sich, wann die FPS zuletzt neu berechnet wurden.


        # try/finally stellt sicher, dass Kamera und Fenster auch bei Fehlern aufraeumen.
        try:
            
            print("MediaPipe läuft. Beenden mit q oder Esc.")                                   # Gibt Bedienhinweis und Tastenkombinationen im Terminal aus.


            # Die Schleife liest und verarbeitet Bilder, bis der Benutzer beendet.
            while True:
                
                frame_start = time.perf_counter()                                               # frame_start startet die Zeitmessung fuer dieses Bild.
                success, frame = camera.read()                                                  # read liest das naechste Bild von der Kamera.


                # Ein fehlgeschlagener Lesevorgang wird nicht stillschweigend ignoriert.
                if not success:
                    raise RuntimeError("Es konnte kein Kamerabild gelesen werden.")

                # MediaPipe erwartet RGB statt des von OpenCV verwendeten BGR.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                              # cvtColor wandelt deshalb die Farbkanalreihenfolge des Bildes um.
                rgb_frame.flags.writeable = False                                               # MediaPipe darf das Eingabebild nicht veraendern; das kann die Verarbeitung beschleunigen.
                results = detector.process(rgb_frame)                                           # process erkennt Gesichter im RGB-Bild.
                latency_ms = (time.perf_counter() - frame_start) * 1000                         # latency_ms misst Kamera-Lesezeit plus Gesichtserkennung in Millisekunden.

                frame_height, frame_width = frame.shape[:2]                                     # frame_height und frame_width enthalten die Bildabmessungen in Pixeln.
                
                detections = results.detections or []                                           # Wenn keine Gesichter gefunden wurden, wird eine leere Liste verwendet.


                # Diese Schleife zeichnet fuer jedes erkannte Gesicht einen Rahmen und Confidence.
                for detection in detections:
                    
                    bounding_box = detection.location_data.relative_bounding_box                # bounding_box enthaelt die normierte Lage und Groesse des Gesichts.
                    
                    x = max(0, int(bounding_box.xmin * frame_width))                            # xmin wird in Pixel umgerechnet und am linken Bildrand begrenzt.
                    y = max(0, int(bounding_box.ymin * frame_height))                           # ymin wird in Pixel umgerechnet und am oberen Bildrand begrenzt.
                    
                    width = min(frame_width - x, int(bounding_box.width * frame_width))         # Die Breite wird in Pixel umgerechnet und auf das Bild begrenzt.
                    height = min(frame_height - y, int(bounding_box.height * frame_height))     # Die Hoehe wird in Pixel umgerechnet und auf das Bild begrenzt.


                    # rectangle zeichnet den Rahmen um das erkannte Gesicht.
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + width, y + height),
                        (255, 0, 0),
                        2,
                    )

                    
                    confidence = detection.score[0]                                             # confidence enthaelt den Sicherheitswert fuer diese Erkennung.


                    # putText schreibt den Confidence-Wert ueber den Gesichtsrahmen.
                    cv2.putText(
                        frame,
                        f"Gesicht {confidence:.2f}",
                        (x, max(25, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2,
                    )

                
                frame_times.append(frame_start)                                                 # Jeder Bildzeitpunkt wird fuer die gleitende FPS-Messung gespeichert.


                # Zeitpunkte ausserhalb des letzten Sekundenfensters werden entfernt.
                while frame_times and frame_start - frame_times[0] > 1.0:
                    frame_times.popleft()
                

                # FPS wird regelmaessig aus den Zeitpunkten des letzten Sekundenfensters berechnet.
                if frame_start - last_fps_update >= 0.5:
                    
                    # Bei mindestens zwei Bildern ist die Bildrate aus Zeitspanne und Bildanzahl berechenbar.
                    if len(frame_times) >= 2:
                        
                        # Die gemessene Rate zaehlt Bildabstaende pro Sekunde.
                        fps = (len(frame_times) - 1) / (
                            frame_times[-1] - frame_times[0]
                        )
                    
                    # Ohne zwei Messpunkte gibt es noch keine gueltige FPS-Berechnung.
                    else:
                        fps = 0.0
                    
                    last_fps_update = frame_start                                               # Speichert den Zeitpunkt der Berechnung, damit sie nicht jedes Bild ausgefuehrt wird.


                # Der Status wird oben links in das Kamerabild geschrieben.
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f} (F/s)  |   Latenz: {latency_ms:.1f} (ms)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),                                                                # Rot
                    2,                                                                          # Dickere Schrift
                    cv2.LINE_AA                                                                 # Kantenglättung
                )



                # Diese zweite Zeile zeigt die Anzahl der erkannten Gesichter.
                cv2.putText(
                    frame,
                    f"Gesichter: {len(detections)}",
                    (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 255),                                                                # Rot
                    1,                                                                          # Dickere Schrift
                    cv2.LINE_AA                                                                 # Kantenglättung
                )


                
                cv2.imshow("Gesichtserkennung - MediaPipe", frame)                              # imshow zeigt das mit Erkennungen und Messwerten beschriftete Bild.

                
                key = cv2.waitKey(1) & 0xFF                                                     # waitKey aktualisiert das Fenster und liest eine Tastatureingabe.
                
                # q oder Esc beendet die Verarbeitungsschleife.
                if key == ord("q") or key == 27:
                    break
        
        finally:
            
            camera.release()                                                                    # release schliesst die Kamera und gibt ihre Systemressourcen frei.
            cv2.destroyAllWindows()                                                             # destroyAllWindows schliesst alle von OpenCV geoeffneten Fenster.



# Dieser Einstiegspunkt ruft main nur auf, wenn die Datei direkt gestartet wird.
if __name__ == "__main__":
    main()

