# Vergleich der Kameraaufloesungen

- Datum: 2026-11-23 01:21 CET
- Kamera-Index: 0
- Laptop-/Kameramodell: HP True Vision 720p HD
- Messdauer: 30 Sekunden je Lauf, nach 5 Sekunden Aufwaermzeit.
- FPS: verarbeitete Bilder geteilt durch die tatsaechliche Messdauer.
- Latenz: mittlere Zeit pro Bild vom Start des Kamera-Reads bis zum Abschluss der MediaPipe-Erkennung.
- Die angeforderte und die tatsaechlich gelieferte Kameraaufloesung werden je Lauf notiert.

| Aufloesung (Soll) | Lauf 1: FPS / Latenz (Ist) | Lauf 2: FPS / Latenz (Ist) | Lauf 3: FPS / Latenz (Ist) | Mittel FPS | Mittel Latenz (ms) |
|---|---:|---:|---:|---:|---:|
| 1280×720 | 14.25 / 65.01 ms (640x480) | 14.25 / 64.99 ms (640x480) | 14.25 / 65.02 ms (640x480) | 14.25 | 65.01 |
| 640×480 | 14.25 / 65.05 ms (640x480) | 14.25 / 63.92 ms (640x480) | 14.25 / 63.88 ms (640x480) | 14.25 | 64.28 |
| 320×240 | 14.25 / 65.64 ms (320x240) | 14.25 / 65.76 ms (320x240) | 14.25 / 65.61 ms (320x240) | 14.25 | 65.67 |

Die Kamera kann eine andere Bildgroesse als angefordert liefern. Fuer den Vergleich ist deshalb die Ist-Aufloesung in jeder Laufzelle angegeben.
