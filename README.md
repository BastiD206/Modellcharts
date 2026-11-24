# Modellcharts für die Wetter-App

Erzeugt Karten im WXCharts-Stil (feste Ausschnitte **Europa** und **Nordhalbkugel**) aus dem globalen
Wettermodell **GFS** (NOAA, 0,5°) und stellt sie über GitHub Pages bereit. Die Wetter-App liest nur
`manifest.json` und die Bilder.

Parameter: Temperatur 850 hPa, Temperatur-850-hPa-Anomalie, Geopotential 500 hPa, Geopotential-Anomalie,
Bodendruck + Schichtdicke, Temperatur 2 m, Wind 850 hPa, Jetstream 250 hPa, Niederschlagsrate, Bewölkung.
Die Anomalien sind die Abweichung vom Tagesmittel 1991–2020 (NCEP/NCAR-Reanalyse, NOAA PSL).

## Einrichten (einmalig, ca. 5 Minuten)
1. Auf GitHub ein neues Repository **Modellcharts** anlegen (öffentlich).
2. Alle Dateien dieses Ordners hineinladen (inkl. des Ordners `.github/workflows`).
3. Im Repository: **Settings → Pages → Source: GitHub Actions** wählen.
4. Im Reiter **Actions** den Workflow **Modellcharts** öffnen und **Run workflow** drücken.
   Der erste Lauf dauert etwas länger (lädt einmalig die Klimatologie), danach läuft er viermal am Tag von selbst.
5. Die Daten liegen dann unter `https://<dein-github-name>.github.io/Modellcharts/`.
   Die Wetter-App erwartet `https://bastid206.github.io/Modellcharts` (Konstante `GLOBAL_CHARTS_BASE` in der HTML-Datei).

## Testen ohne Internet
`python build_charts.py --synthetic --out site --max-hours 12` erzeugt Karten mit Fantasiedaten.

## Hinweise
- Ohne erreichbare Klimatologie entfallen nur die beiden Anomalie-Karten, alles andere läuft weiter.
- Niederschlag und Bewölkung gibt es im Analyse-Schritt (+0 h) nicht, dort zeigt die App „nicht verfügbar“.
