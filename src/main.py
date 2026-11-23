"""Echtzeit-Vision-Pipeline – Einstiegspunkt."""


import argparse                             # argparse liest Optionen ein, die beim Start des Programms angegeben werden.

from collections import deque               # deque speichert die Bildzeitpunkte fuer die laufende FPS-Anzeige.
from dataclasses import dataclass           # dataclass erstellt einen einfachen Datentyp fuer einen Messlauf.
from datetime import datetime               # datetime ermittelt das Datum fuer den Messbericht.
from pathlib import Path                    # Path verwaltet Dateipfade unabhaengig vom Betriebssystem.
import statistics                           # statistics berechnet Mittelwerte aus den drei Messlaeufen.
import time                                 # time liefert eine hochaufloesende Uhr fuer Dauer- und Latenzmessungen.

import cv2                                  # OpenCV liest die Kamera und zeichnet die Gesichtserkennung ins Bild.
import mediapipe as mp                      # MediaPipe stellt den Gesichtserkennungs-Detektor bereit.



RESOLUTIONS = ((1280, 720), (640, 480), (320, 240))                                                  # RESOLUTIONS enthaelt die drei fuer den Vergleich geforderten Bildgroessen.

MEASUREMENTS_PER_RESOLUTION = 3                                                                     # MEASUREMENTS_PER_RESOLUTION legt drei Messlaeufe pro Bildgroesse fest.

REPORT_PATH = Path(__file__).resolve().parent.parent / "results" / "messung.md"                     # REPORT_PATH ist der Speicherort des automatisch erstellten Messberichts.

# Globale Latenzwerte
LATENZ = {
    "read": 0.0,
    "convert": 0.0,
    "detect": 0.0
}

# Measurement speichert die Messwerte eines einzelnen 30-Sekunden-Laufs.
@dataclass
class Measurement:
    
    requested_width: int                    # requested_width ist die bei OpenCV angeforderte Bildbreite.
    requested_height: int                   # requested_height ist die bei OpenCV angeforderte Bildhoehe.
    actual_width: int                       # actual_width ist die vom Kamerabild tatsaechlich gelieferte Breite.
    actual_height: int                      # actual_height ist die vom Kamerabild tatsaechlich gelieferte Hoehe.
    fps: float                              # fps ist die Zahl der verarbeiteten Bilder pro Sekunde im Messlauf.
    latency_ms: float                       # latency_ms ist die mittlere Latenz pro Bild in Millisekunden.
    frames: int                             # frames ist die Zahl der waehrend der Messung verarbeiteten Bilder.
    duration_seconds: float                 # duration_seconds ist die gemessene Wandzeit des Messlaufs.


# parse_arguments definiert und prueft alle Kommandozeilenoptionen.
def parse_arguments() -> argparse.Namespace:
    
    # parser verwaltet die Beschreibung und Optionen des Programms.
    parser = argparse.ArgumentParser(
        description="MediaPipe-Gesichtserkennung mit optionalem Aufloesungsvergleich."
    )
    
    # --camera waehlt die Kamera aus; standardmaessig wird Kamera 0 verwendet.
    parser.add_argument("--camera", "--kamera", type=int, dest="camera", default=0, help="Kamera-Index (Standard: 0)")
    
    # --confidence legt die minimale Erkennungssicherheit von 0 bis 1 fest.
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Mindest-Confidence von 0 bis 1 (Standard: 0.5)",
    )
    
    # --width fordert im normalen Kameramodus eine bestimmte Bildbreite an.
    parser.add_argument(
        "--width", "--breite",
        type=int,
        dest="width",
        help="Angeforderte Kamera-Bildbreite fuer den normalen Modus",
    )
    
    # --height fordert im normalen Kameramodus eine bestimmte Bildhoehe an.
    parser.add_argument(
        "--height", "--hoehe",
        type=int,
        dest="height",
        help="Angeforderte Kamera-Bildhoehe fuer den normalen Modus",
    )
    
    # --benchmark startet alle drei Aufloesungen mit je drei Messlaeufen.
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Misst 3 Aufloesungen je 3 Mal und erstellt results/messung.md",
    )
    
    # --duration bestimmt die Dauer jeder Messung in Sekunden.
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Dauer jedes Messlaufs in Sekunden (Standard: 30)",
    )
    
    # --warmup bestimmt die Aufwaermzeit vor jedem Messlauf in Sekunden.
    parser.add_argument(
        "--warmup",
        type=float,
        default=5.0,
        help="Aufwaermzeit vor jeder Messung in Sekunden (Standard: 5)",
    )
    

    args = parser.parse_args()                                                      # args enthaelt die von argparse gelesenen Kommandozeilenwerte.

    # Eine Confidence ausserhalb des erlaubten Bereichs ist ein Eingabefehler.
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence muss zwischen 0 und 1 liegen.")
    

    # Breite und Hoehe muessen im normalen Modus gemeinsam angegeben werden.
    if (args.width is None) != (args.height is None):
        parser.error("--width und --height muessen gemeinsam angegeben werden.")
    

    # Eine angeforderte Bildgroesse muss aus positiven Pixelwerten bestehen.
    if args.width is not None and (args.width <= 0 or args.height <= 0):
        parser.error("--width und --height muessen groesser als 0 sein.")
    

    # Der Benchmark nutzt feste Aufloesungen und kann keine einzelne Groesse annehmen.
    if args.benchmark and (args.width is not None or args.height is not None):
        parser.error("--width und --height koennen nicht mit --benchmark kombiniert werden.")
    

    # Messdauer muss positiv sein, damit FPS und Mittelwerte bestimmbar sind.
    if args.duration <= 0:
        parser.error("--duration muss groesser als 0 sein.")
    

    # Eine negative Aufwaermzeit ist nicht sinnvoll.
    if args.warmup < 0:
        parser.error("--warmup darf nicht negativ sein.")

    return args                                                                         # Gibt die geprueften Programmoptionen zurueck.


# open_camera oeffnet die Kamera und fordert optional die gewuenschte Bildgroesse an.
def open_camera(
    camera_index: int,
    width: int | None = None,
    height: int | None = None,
) -> cv2.VideoCapture:
    
    camera = cv2.VideoCapture(camera_index)                                             # VideoCapture verbindet OpenCV mit der ausgewaehlten Kamera.
    
    # Ohne geoeffnete Kamera kann keine Verarbeitung gestartet werden.
    if not camera.isOpened():
        
        camera.release()                                                                # Gibt die Kameraressource auch im Fehlerfall wieder frei.
        raise RuntimeError(f"Kamera {camera_index} konnte nicht geoeffnet werden.")     # Meldet explizit, dass die Kamera nicht verfuegbar ist.


    # Fordert nur dann eine Bildbreite an, wenn eine angegeben wurde.
    if width is not None and height is not None:
        set_camera_resolution(camera, width, height)
    
    return camera                                                                       # Gibt die geoeffnete Kamera zurueck.


# set_camera_resolution fordert eine Bildgroesse an und meldet abgelehnte Einstellungen.
def set_camera_resolution(camera: cv2.VideoCapture, width: int, height: int) -> None:
    
    width_accepted = camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)                        # OpenCV meldet mit dem Rueckgabewert, ob es die angeforderte Breite uebernommen hat.
    
    height_accepted = camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)                     # OpenCV meldet mit dem Rueckgabewert, ob es die angeforderte Hoehe uebernommen hat.


    # Eine abgelehnte Breite wird als Warnung gemeldet; das tatsaechliche Frame wird spaeter geprueft.
    if not width_accepted:
        print(f"Warnung: Kamera hat die angeforderte Breite {width} nicht bestaetigt.")
    

    # Eine abgelehnte Hoehe wird als Warnung gemeldet; das tatsaechliche Frame wird spaeter geprueft.
    if not height_accepted:
        print(f"Warnung: Kamera hat die angeforderte Hoehe {height} nicht bestaetigt.")


# read_and_detect liest ein Kamerabild, erkennt Gesichter und zeichnet die Rahmen.
def read_and_detect(
    camera: cv2.VideoCapture,
    detector: mp.solutions.face_detection.FaceDetection,
) -> tuple[object, float, int]:
    
    # Zeit für das Einlesen des Kamerabildes
    read_start = time.perf_counter()                                                       # read_start erfasst den Anfang der Latenzmessung vor dem Kamera-Read.
    success, frame = camera.read()                                                      # camera.read liest das naechste Bild und meldet, ob das Lesen erfolgreich war.
    read_ms = (time.perf_counter() - read_start) * 1000
    

    # Ein fehlgeschlagener Lesevorgang wird explizit als Fehler gemeldet.
    if not success:
        raise RuntimeError("Es konnte kein Kamerabild gelesen werden.")


    # Zeit für die Farbkonvertierung
    convert_start = time.perf_counter()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                  # MediaPipe erwartet RGB statt der von OpenCV verwendeten BGR-Farbkanaele.
    convert_ms = (time.perf_counter() - convert_start) * 1000


    # Zeit für die Gesichtserkennung
    detect_start = time.perf_counter()
    rgb_frame.flags.writeable = False                                                   # Das Eingabebild wird schreibgeschuetzt, was die Verarbeitung beschleunigen kann.
    results = detector.process(rgb_frame)                                               # detector.process fuehrt die Gesichtserkennung auf dem RGB-Bild aus.
    detect_ms = (time.perf_counter() - detect_start) * 1000    


    # Gesamtlatenz
    latency_ms = read_ms + convert_ms + detect_ms                                # latency_ms umfasst Kamera-Read, Farbkonvertierung und Gesichtserkennung.
    
    LATENZ["read"] = read_ms
    LATENZ["convert"] = convert_ms
    LATENZ["detect"] = detect_ms


    frame_height, frame_width = frame.shape[:2]                                         # frame_height und frame_width enthalten die Bildabmessungen in Pixeln.
    detections = results.detections or []                                               # detections wird auch dann eine Liste, wenn MediaPipe nichts erkennt.

    # Diese Schleife zeichnet fuer jedes erkannte Gesicht Rahmen und Confidence.
    for detection in detections:

        bounding_box = detection.location_data.relative_bounding_box                    # bounding_box enthaelt die normierte Lage und Groesse des Gesichts.
        
        # x und y wandeln die linke obere Ecke in Bildpixel um und begrenzen sie.
        x = max(0, int(bounding_box.xmin * frame_width))
        y = max(0, int(bounding_box.ymin * frame_height))
        
        # width und height wandeln die Boxgroesse um und begrenzen sie auf das Bild.
        width = min(frame_width - x, int(bounding_box.width * frame_width))
        height = min(frame_height - y, int(bounding_box.height * frame_height))
        
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)           # rectangle zeichnet einen blauen Rahmen um das erkannte Gesicht.

        confidence = detection.score[0]                                                 # confidence enthaelt den Sicherheitswert fuer diese Gesichtserkennung.


        # putText schreibt den Confidence-Wert oberhalb des Gesichtsrahmens.
        cv2.putText(
            frame,
            f"Gesicht {confidence:.2f}",
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

    # Gibt das beschriftete Bild, die Latenz und die Anzahl der Gesichter zurueck.
    return frame, latency_ms, len(detections)


# show_frame blendet Statuswerte ein und zeigt das Kamerabild.
def show_frame(
    frame: object,
    fps: float,
    latency_ms: float,
    face_count: int,
    status: str = "",
) -> int:
    
    frame_width = frame.shape[1]                                                        # frame_width bestimmt die Schriftgroesse, damit der Text auch bei 320 Pixeln passt.
    
    font_scale = 0.45 if frame_width < 640 else 0.7                                     # Kleine Kamerabilder verwenden kleinere Schrift als breite Bilder.
    
    metrics_y = 48 if status else 30                                                    # metrics_y verschiebt die Messwerte nach unten, falls ein Status angezeigt wird.


    # putText zeigt den aktuellen Benchmark-Status in einer eigenen Zeile.
    if status:
        cv2.putText(
            frame,
            status,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )


    # putText zeigt FPS und Latenz oben links im Bild.
    cv2.putText(
        frame,
        f"FPS: {fps:.1f} (f/s)  |  Latenz: {latency_ms:.1f} (ms)",
        (10, metrics_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 0, 0),
        1 if frame_width < 640 else 2,
        cv2.LINE_AA,
    )


    face_count_y = metrics_y + (22 if frame_width < 640 else 28)                        # face_count_y platziert die Gesichterzeile unterhalb der Messwerte.


    # putText zeigt die Zahl der erkannten Gesichter unter den Messwerten.
    cv2.putText(
        frame,
        f"Gesichter: {face_count}",
        (10, face_count_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 255),
        1 if frame_width < 640 else 2,
        cv2.LINE_AA,
    )


    cv2.imshow("Gesichtserkennung - MediaPipe", frame)                                  # imshow zeigt das mit Erkennungen und Messwerten beschriftete Bild.
    
    return cv2.waitKey(1) & 0xFF                                                        # waitKey aktualisiert das Fenster und gibt den Tastencode zurueck.


# run_live zeigt die Gesichtserkennung fortlaufend mit optionaler Kameraaufloesung.
def run_live(args: argparse.Namespace, detector: mp.solutions.face_detection.FaceDetection) -> None:

    camera = open_camera(args.camera, args.width, args.height)                          # open_camera oeffnet die Kamera mit der optional angeforderten Bildgroesse.
    frame_times: deque[float] = deque()                                                 # frame_times enthaelt die Zeitpunkte der Bilder im gleitenden Ein-Sekunden-Fenster.
    fps = 0.0                                                                           # fps wird bis zur ersten gueltigen Messung mit null angezeigt.    
    last_fps_update = 0.0                                                               # last_fps_update verhindert eine Neuberechnung bei jedem einzelnen Bild.
    display_latency_ms = 0.0                                                            # display_latency_ms zeigt die End-to-End-Latenz des vorherigen Bildes an.

    # try/finally gibt Kamera und Fenster auch bei Fehlern frei.
    try:
        
        print("MediaPipe laeuft. Beenden mit q oder Esc.")                              # Gibt den Bedienhinweis und die Beenden-Tasten im Terminal aus.


        # Die Schleife liest und verarbeitet Bilder bis zum Beenden.
        while True:
            frame_start = time.perf_counter()                                           # frame_start startet die FPS- und End-to-End-Latenzmessung fuer dieses Bild.
            frame, latency_ms, face_count = read_and_detect(camera, detector)           # read_and_detect liest das Bild, erkennt Gesichter und misst die Latenz.
            frame_height, frame_width = frame.shape[:2]                                 # frame_height und frame_width geben die tatsaechliche Kameraaufloesung an.


            # Die tatsaechliche Bildgroesse wird beim ersten Bild im Terminal ausgegeben.
            if not frame_times:
                print(f"Tatsaechliche Aufloesung: {frame_width}x{frame_height}")
            
            frame_times.append(frame_start)                                             # Der aktuelle Bildzeitpunkt kommt in das gleitende FPS-Zeitfenster.

            
            # Zeitpunkte ausserhalb der letzten Sekunde werden verworfen.
            while frame_times and frame_start - frame_times[0] > 1.0:
                frame_times.popleft()
            

            # FPS wird zweimal pro Sekunde aus den Bildabstaenden berechnet.
            if frame_start - last_fps_update >= 0.5:
                

                # Mindestens zwei Zeitpunkte sind fuer die Bildrate erforderlich.
                if len(frame_times) >= 2:
                    
                    fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])   # fps ist die Zahl der Bildabstaende pro Sekunde.
                

                # Ohne zwei Messpunkte bleibt die angezeigte Bildrate null.
                else:
                    fps = 0.0
                
                last_fps_update = frame_start                                           # Merkt sich den Zeitpunkt der letzten FPS-Berechnung.
            
            key = show_frame(frame, fps, display_latency_ms, face_count)                # show_frame zeigt die End-to-End-Latenz des vorherigen Bildes.
            display_latency_ms = (time.perf_counter() - frame_start) * 1000             # display_latency_ms umfasst Kamera, Erkennung, Zeichnen und Fensteraktualisierung.
            

            # q oder Esc beendet den normalen Kameramodus.
            if key == ord("q") or key == 27:
                break
    

    # finally gibt die Kamera und alle OpenCV-Fenster frei.
    finally:
        
        camera.release()                                                                # release schliesst die Kamera und gibt ihre Systemressourcen frei.
        cv2.destroyAllWindows()                                                         # destroyAllWindows schliesst alle von OpenCV geoeffneten Fenstern.

    # Ergebnisse ausgeben
    print('\n\n',15*'==')
    print_latenz(args.width, args.height, LATENZ)
    print('',15*'==', '\n\n')

# run_measurement fuehrt Aufwaermzeit und einen zeitlich begrenzten Messlauf aus.
def run_measurement(
    camera: cv2.VideoCapture,
    detector: mp.solutions.face_detection.FaceDetection,
    requested_width: int,
    requested_height: int,
    duration: float,
    warmup: float,
    run_number: int,
) -> Measurement | None:

    # Fordert vor jedem Lauf die aktuelle gewuenschte Kameraaufloesung an.
    set_camera_resolution(camera, requested_width, requested_height)
    
    # Meldet, welche Aufloesung und welcher Wiederholungslauf gerade starten.
    print(
        f"\nAufloesung {requested_width}x{requested_height}, "
        f"Messlauf {run_number}/{MEASUREMENTS_PER_RESOLUTION}"
    )
    
    warmup_start = time.perf_counter()                                                  # warmup_start startet die separat nicht ausgewertete Aufwaermphase.
    warmup_display_latency_ms = 0.0                                                     # warmup_display_latency_ms zeigt die End-to-End-Latenz des vorherigen Aufwaermbildes an.


    # Die Aufwaermphase verarbeitet Bilder, damit Kamera und Detektor anlaufen.
    while True:
        frame_start = time.perf_counter()                                               # frame_start startet die End-to-End-Messung des Aufwaermbildes.
        frame, latency_ms, face_count = read_and_detect(camera, detector)               # read_and_detect verarbeitet das Aufwaermbild wie ein normales Kamerabild.
        # Zeigt die Aufwaermphase an, die nicht in die Messwerte einfliesst.
        key = show_frame(
            frame,
            0.0,
            warmup_display_latency_ms,
            face_count,
            f"Aufwaermen {requested_width}x{requested_height}",
        )
        warmup_display_latency_ms = (time.perf_counter() - frame_start) * 1000          # Erfasst Kamera, Erkennung, Zeichnen und Fensteraktualisierung.


        # q oder Esc bricht den gesamten Benchmark vor dem Bericht ab.
        if key == ord("q") or key == 27:
            return None


        # Mindestens ein Bild wird auch bei einer Aufwaermzeit von null eingelesen.
        if time.perf_counter() - warmup_start >= warmup:
            break


    latencies: list[float] = []                                                         # latencies sammelt die Latenz jedes Bildes innerhalb der Messdauer.
    
    frames = 0                                                                          # frames zaehlt die Bilder, die innerhalb des Messlaufs verarbeitet wurden.
    actual_width: int | None = None                                                     # actual_width wird aus dem ersten gewerteten Messbild ermittelt.
    actual_height: int | None = None                                                    # actual_height wird aus dem ersten gewerteten Messbild ermittelt.
    display_latency_ms = 0.0                                                            # display_latency_ms zeigt die End-to-End-Latenz des vorherigen Messbildes an.
    measurement_start = time.perf_counter()                                             # measurement_start startet die Wandzeitmessung nach der Aufwaermphase.
    measurement_end = measurement_start                                                 # measurement_end dient zur Pruefung, ob die eingestellte Dauer erreicht ist.


    # Die Messschleife laeuft bis zur Dauer oder bis zum Abbruch durch q/Esc.
    while measurement_end - measurement_start < duration:
        
        frame_start = time.perf_counter()                                               # frame_start startet die End-to-End-Zeitmessung vor dem Kamera-Read.
        frame, latency_ms, face_count = read_and_detect(camera, detector)               # read_and_detect liefert das Messbild samt Erkennungslatenz und Gesichtszahl.
        frame_height, frame_width = frame.shape[:2]                                     # Die Bildgroesse wird direkt aus dem aktuellen Messbild gelesen.
        # Die Ist-Aufloesung des ersten gewerteten Bildes wird im Ergebnis gespeichert.
        if actual_width is None or actual_height is None:
            actual_width = frame_width
            actual_height = frame_height
            print(f"Tatsaechliche Aufloesung im Messlauf: {actual_width}x{actual_height}")
            # Eine abweichende Bildgroesse wird sichtbar gemeldet, aber weiter gemessen.
            if (actual_width, actual_height) != (requested_width, requested_height):
                print(
                    f"Warnung: Angefordert waren {requested_width}x{requested_height}, "
                    f"die Kamera liefert {actual_width}x{actual_height}."
                )
        frames += 1                                                                     # Zaehlt das fertig verarbeitete Bild fuer die FPS-Berechnung.
        measured_fps = frames / max(time.perf_counter() - measurement_start, 1e-9)      # measured_fps zeigt die laufende Durchschnittsrate waehrend des Tests.
        

        # Zeigt Messwerte und Fortschrittsstatus im Kamerafenster an.
        key = show_frame(
            frame,
            measured_fps,
            display_latency_ms,
            face_count,
            f"Messung {requested_width}x{requested_height}",
        )
        measurement_end = time.perf_counter()                                           # measurement_end enthaelt auch Zeichnen und Fensteraktualisierung.
        latency_ms = (measurement_end - frame_start) * 1000                             # Die Benchmark-Latenz umfasst die gesamte Verarbeitung bis zur Anzeige.
        display_latency_ms = latency_ms                                                 # Der naechste Frame zeigt diese vollstaendige End-to-End-Latenz an.
        latencies.append(latency_ms)                                                    # Speichert die End-to-End-Latenz fuer den Laufmittelwert.

        # q oder Esc bricht den gesamten Benchmark ab.
        if key == ord("q") or key == 27:
            return None
    

    measured_duration = measurement_end - measurement_start                             # measured_duration ist die tatsaechlich verstrichene Zeit dieses Messlaufs.
    average_latency = statistics.mean(latencies)                                        # average_latency ist der arithmetische Mittelwert aller Bildlatenzen.
    fps = frames / measured_duration                                                    # fps nutzt die Bildanzahl und die tatsaechliche Wandzeit.

    # Ohne gewertetes Bild kann kein gueltiger Messwert zurueckgegeben werden.
    if actual_width is None or actual_height is None:
        raise RuntimeError("Der Messlauf hat kein Bild mit gueltiger Aufloesung geliefert.")

    # Gibt alle angeforderten und tatsaechlich gemessenen Laufwerte zurueck.
    return Measurement(
        requested_width=requested_width,
        requested_height=requested_height,
        actual_width=actual_width,
        actual_height=actual_height,
        fps=fps,
        latency_ms=average_latency,
        frames=frames,
        duration_seconds=measured_duration,
    )


# write_report erstellt die Ergebnistabelle mit Einzelwerten und Mittelwerten.
def write_report(
    measurements: list[Measurement],
    camera_index: int,
    duration: float,
    warmup: float,
) -> None:

    # lines sammelt die Berichtzeilen, bevor sie in die Markdown-Datei geschrieben werden.
    lines = [
        "# Vergleich der Kameraaufloesungen",
        "",
        f"- Datum: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
        f"- Kamera-Index: {camera_index}",
        "- Laptop-/Kameramodell: bitte nach dem Test ergaenzen.",
        f"- Messdauer: {duration:g} Sekunden je Lauf, nach {warmup:g} Sekunden Aufwaermzeit.",
        "- FPS: verarbeitete Bilder geteilt durch die tatsaechliche Messdauer einschliesslich Kamera-Read und Anzeige.",
        "- Latenz: mittlere End-to-End-Zeit vom Start des Kamera-Reads bis zur OpenCV-Fensteraktualisierung, einschliesslich Erkennung, Zeichnen und Text-Overlay.",
        "- Die tatsaechliche Bildschirmdarstellung kann durch das Betriebssystem oder Display-Backend zusaetzlich verzoegert sein.",
        "- Die angeforderte und die tatsaechlich gelieferte Kameraaufloesung werden je Lauf notiert.",
        "",
        "| Aufloesung (Soll) | Lauf 1: FPS / Latenz (Ist) | Lauf 2: FPS / Latenz (Ist) | Lauf 3: FPS / Latenz (Ist) | Mittel FPS | Mittel Latenz (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    
    # Fuegt je Soll-Aufloesung eine Zeile mit drei Messungen und ihren Mittelwerten hinzu.
    for requested_width, requested_height in RESOLUTIONS:
        
        # runs filtert die drei Einzelmessungen fuer die aktuelle Soll-Aufloesung.
        runs = [
            item
            for item in measurements
            if item.requested_width == requested_width
            and item.requested_height == requested_height
        ]
        
        # run_cells formatiert FPS, Latenz und Ist-Aufloesung jedes Messlaufs.
        run_cells = [
            (
                f"{item.fps:.2f} / {item.latency_ms:.2f} ms "
                f"({item.actual_width}x{item.actual_height})"
            )
            for item in runs
        ]
        
        avg_fps = statistics.mean(item.fps for item in runs)                            # avg_fps berechnet den Mittelwert der drei FPS-Laufwerte.
        
        avg_latency = statistics.mean(item.latency_ms for item in runs)                 # avg_latency berechnet den Mittelwert der drei mittleren Latenzen.

        # Fuegt die formatierte Ergebniszeile der Tabelle hinzu.
        lines.append(
            f"| {requested_width}×{requested_height} | "
            f"{run_cells[0]} | {run_cells[1]} | {run_cells[2]} | "
            f"{avg_fps:.2f} | {avg_latency:.2f} |"
        )


    # Fuegt eine Leerzeile und den Hinweis zum verwendeten Messmodus an.
    lines.extend(
        [
            "",
            "Die Kamera kann eine andere Bildgroesse als angefordert liefern. "
            "Fuer den Vergleich ist deshalb die Ist-Aufloesung in jeder Laufzelle angegeben.",
            "",
        ]
    )
    
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)                               # Legt den Ergebnisordner an, falls er noch nicht existiert.
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")                          # Schreibt den vollstaendigen Bericht in UTF-8 an den festgelegten Speicherort.
    print(f"\nMessbericht gespeichert: {REPORT_PATH}")                                  # Meldet dem Benutzer den Pfad der erfolgreich geschriebenen Datei.


# run_benchmark misst alle geforderten Aufloesungen dreimal und erstellt den Bericht.
def run_benchmark(
    args: argparse.Namespace,
    detector: mp.solutions.face_detection.FaceDetection,
) -> None:
    
    camera = open_camera(args.camera)                                                   # open_camera oeffnet die Kamera zunaechst ohne feste Aufloesung.
    measurements: list[Measurement] = []                                                # measurements sammelt alle neun vollstaendigen Messlaeufe.


    # try/finally gibt Kamera und Fenster auch bei Abbruch oder Fehler frei.
    try:
        
        # Informiert vor dem Start ueber Anzahl und Laufzeit des Benchmark-Programms.
        total_seconds = len(RESOLUTIONS) * MEASUREMENTS_PER_RESOLUTION * (
            args.duration + args.warmup
        )
        print(
            f"Benchmark: {len(RESOLUTIONS)} Aufloesungen x "
            f"{MEASUREMENTS_PER_RESOLUTION} Laeufe, etwa "
            f"{total_seconds / 60:.1f} Minuten."
        )
        
        # Jede Aufloesung wird pro Runde in einer rotierten Reihenfolge gemessen.
        for run_number in range(1, MEASUREMENTS_PER_RESOLUTION + 1):
            # offset rotiert die Startposition, damit jede Aufloesung frueh, mittig und spaet drankommt.
            for offset in range(len(RESOLUTIONS)):
                # resolution_index waehlt die naechste Aufloesung der rotierten Reihenfolge.
                resolution_index = (run_number - 1 + offset) % len(RESOLUTIONS)
                # requested_width und requested_height sind die Einstellungen dieses Laufs.
                requested_width, requested_height = RESOLUTIONS[resolution_index]
                
                # Fuehrt Aufwaermphase und den zeitlich begrenzten Messlauf aus.
                measurement = run_measurement(
                    camera,
                    detector,
                    requested_width,
                    requested_height,
                    args.duration,
                    args.warmup,
                    run_number,
                )
                
                # Ein None-Ergebnis bedeutet, dass der Benutzer den Lauf abgebrochen hat.
                if measurement is None:
                    print("Benchmark abgebrochen; es wurde kein Messbericht geschrieben.")
                    return
                
                # Speichert den vollstaendigen Lauf zur spaeteren Tabellenauswertung.
                measurements.append(measurement)
                

                # Gibt Einzelwerte zur Kontrolle direkt im Terminal aus.
                print(
                    f"Ergebnis: {measurement.fps:.2f} FPS, "
                    f"{measurement.latency_ms:.2f} ms mittlere Latenz, "
                    f"{measurement.frames} Bilder in "
                    f"{measurement.duration_seconds:.2f} s."
                )

                # Latenzwerte dieser Auflösung ausgeben
                print('\n',15*'==')
                print_latenz(requested_width, requested_height, LATENZ)
                print('', 15*'==','\n')
        
        # Erstellt den Bericht, wenn alle neun Messlaeufe abgeschlossen sind.
        write_report(measurements, args.camera, args.duration, args.warmup)
    


    # finally schliesst die Kamera auch bei einem unerwarteten Fehler.
    finally:
        
        camera.release()                                                                # release gibt die Kameraverbindung frei.
        cv2.destroyAllWindows()                                                         # destroyAllWindows schliesst alle OpenCV-Fenster.




# Gibt die gemessenen Latenzwerte einer Auflösung aus.
def print_latenz(width: int, height: int, latenz: dict) -> None:

    print(f"\n--- Latenz-Auswertung ({width}x{height}) ---")

    for name, value in latenz.items():
        print(f"{name:<10}: {value:.2f} ms")



# main waehlt den normalen Kameramodus oder den automatischen Benchmark aus.
def main() -> None:
    
    args = parse_arguments()                                                            # args enthaelt die geprueften Einstellungen der Kommandozeile.
    
    face_detection = mp.solutions.face_detection                                        # face_detection enthaelt die MediaPipe-Funktionen fuer Gesichtserkennung.
    
    # Der with-Block schliesst den MediaPipe-Detektor nach seiner Verwendung korrekt.
    with face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=args.confidence,
    ) as detector:
        
        # Der Benchmark misst alle Aufloesungen und schreibt den Ergebnisbericht.
        if args.benchmark:
            run_benchmark(args, detector)
        
        # Ohne --benchmark bleibt die normale Echtzeit-Gesichtserkennung aktiv.
        else:
            run_live(args, detector)


# Startet main nur, wenn diese Datei direkt ausgefuehrt wird. Ruft die Hauptfunktion des Programms auf.
if __name__ == "__main__":
    main()

