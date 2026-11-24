#!/usr/bin/env python3
"""
Modell-Charts im WXCharts-Stil: feste Kartenausschnitte (Europa, Nordhalbkugel)
aus dem globalen Wettermodell GFS (NOAA, 0,5 Grad, frei verfügbar).

Ablauf:  neuesten GFS-Lauf finden -> nötige Felder je Vorhersageschritt laden
         (nur die gebrauchten GRIB-Nachrichten per Byte-Range) -> Karten rendern (WebP)
         -> manifest.json schreiben.  Die Wetter-App liest nur manifest.json und die Bilder.

Aufruf:  python build_charts.py --out site            (echter Lauf)
         python build_charts.py --out site --synthetic (Test mit Fantasiedaten, ohne Internet)
"""
import argparse, datetime as dt, json, os, sys, tempfile, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import maximum_filter, minimum_filter

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except Exception:   # ohne Cartopy gibt es nur einfache Test-Karten (Länge/Breite), nie im echten Lauf
    HAVE_CARTOPY = False

S3 = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
GRID = "0p50"
CLIMO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "climo", "ncep_ltm.npz")
PSL = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.derived/pressure/"
WD = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]

# Welche GRIB-Nachrichten gebraucht werden: (Schlüssel, Variable, Ebene) – Ebene mit * = "beginnt mit"
WANTED = [
    ("t850", "TMP", "850 mb"), ("z500", "HGT", "500 mb"), ("z1000", "HGT", "1000 mb"),
    ("mslp", "PRMSL", "mean sea level"), ("t2m", "TMP", "2 m above ground"),
    ("u850", "UGRD", "850 mb"), ("v850", "VGRD", "850 mb"),
    ("u250", "UGRD", "250 mb"), ("v250", "VGRD", "250 mb"),
    ("prate", "PRATE", "surface"), ("tcc", "TCDC", "entire atmosphere*"),
]

REGIONS = {
    "europe": dict(title="Europa", extent=[-26, 46, 30, 71], size=(10.0, 8.6), coast="50m", sub=(20, 80, -45, 65),
                   proj=lambda: ccrs.LambertConformal(central_longitude=10, central_latitude=50, standard_parallels=(35, 65))),
    "nhem": dict(title="Nordhalbkugel", extent=[-180, 180, 24, 90], size=(10.0, 10.0), coast="110m", sub=(10, 90, -180, 180),
                 proj=lambda: ccrs.NorthPolarStereo(central_longitude=-20)),
}

# ---------- Farbskalen ----------
def cmap_from(name, n):
    return plt.get_cmap(name, n)
PRECIP_COLORS = ["#b9e3ff", "#7cc4f5", "#3f9be0", "#1f6fc4", "#2ca64a", "#a8d62f", "#f2d41f", "#f29a1f", "#e0431f", "#b3126e"]

def lv(a, b, s):
    return list(np.arange(a, b + s / 2, s))

# key, kurz, titel, Einheit, Pflichtfelder, Anomalie-Schlüssel
PARAMS = [
    dict(key="t850", short="T 850 hPa", title="Temperatur 850 hPa", unit="°C", needs=["t850"]),
    dict(key="t850a", short="T 850 Anomalie", title="Temperatur 850 hPa · Abweichung vom Klima", unit="K", needs=["t850"], climo="t850"),
    dict(key="z500", short="Z 500 hPa", title="Geopotential 500 hPa", unit="gpdam", needs=["z500"]),
    dict(key="z500a", short="Z 500 Anomalie", title="Geopotential 500 hPa · Abweichung vom Klima", unit="gpdam", needs=["z500"], climo="z500"),
    dict(key="pmsl", short="Druck & Dicke", title="Bodendruck und Schichtdicke 500/1000 hPa", unit="gpdam", needs=["mslp", "z500", "z1000"]),
    dict(key="t2m", short="Temp. 2 m", title="Temperatur 2 m", unit="°C", needs=["t2m"]),
    dict(key="wind850", short="Wind 850 hPa", title="Wind 850 hPa", unit="km/h", needs=["u850", "v850"]),
    dict(key="jet250", short="Jetstream", title="Jetstream 250 hPa", unit="km/h", needs=["u250", "v250"]),
    dict(key="prate", short="Niederschlag", title="Niederschlagsrate", unit="mm/h", needs=["prate"]),
    dict(key="cloud", short="Bewölkung", title="Bewölkung gesamt", unit="%", needs=["tcc"]),
]

def spec_for(key, F, A):
    """Liefert fill-Daten, Stufen, Farbskala, Zusatzlinien … für einen Parameter (F = Felder, A = Anomalie-Felder)."""
    if key == "t850":
        d = F["t850"] - 273.15
        return dict(data=d, levels=lv(-32, 32, 2), cmap=cmap_from("RdYlBu_r", 33), extend="both",
                    lines=dict(data=d, levels=lv(-40, 40, 4), bold=[0]), ticks=lv(-30, 30, 10))
    if key == "t850a":
        return dict(data=A["t850"], levels=lv(-16, 16, 2), cmap=cmap_from("RdBu_r", 17), extend="both",
                    lines=dict(data=F["t850"] - 273.15, levels=[0], bold=[0]), ticks=lv(-16, 16, 4))
    if key == "z500":
        d = F["z500"] / 10.0
        return dict(data=d, levels=lv(480, 600, 4), cmap=cmap_from("Spectral_r", 31), extend="both",
                    lines=dict(data=d, levels=lv(480, 600, 4), bold=[552]), ticks=lv(480, 600, 20))
    if key == "z500a":
        return dict(data=A["z500"], levels=lv(-45, 45, 5), cmap=cmap_from("RdBu_r", 19), extend="both",
                    lines=dict(data=F["z500"] / 10.0, levels=lv(480, 600, 8), bold=[552]), ticks=lv(-40, 40, 10))
    if key == "pmsl":
        thick = (F["z500"] - F["z1000"]) / 10.0
        return dict(data=thick, levels=lv(480, 600, 4), cmap=cmap_from("RdYlBu_r", 31), extend="both",
                    lines=dict(data=F["mslp"] / 100.0, levels=lv(940, 1060, 4), bold=[1012], label=True), ticks=lv(480, 600, 20),
                    extrema=F["mslp"] / 100.0)
    if key == "t2m":
        d = F["t2m"] - 273.15
        return dict(data=d, levels=lv(-32, 44, 2), cmap=cmap_from("turbo", 38), extend="both",
                    lines=dict(data=d, levels=[0], bold=[0]), ticks=lv(-30, 40, 10))
    if key == "wind850":
        sp = np.hypot(F["u850"], F["v850"]) * 3.6
        return dict(data=sp, levels=lv(0, 140, 10), cmap=cmap_from("YlGnBu", 14), extend="max", arrows=(F["u850"], F["v850"]), ticks=lv(0, 140, 20))
    if key == "jet250":
        sp = np.hypot(F["u250"], F["v250"]) * 3.6
        return dict(data=sp, levels=lv(80, 400, 20), cmap=cmap_from("plasma", 16), extend="max", arrows=(F["u250"], F["v250"]), ticks=lv(100, 400, 100))
    if key == "prate":
        d = F["prate"] * 3600.0
        return dict(data=d, levels=[0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, 40], cmap=ListedColormap(PRECIP_COLORS), extend="max",
                    ticks=[0.1, 0.5, 1, 2, 5, 12, 40])
    if key == "cloud":
        return dict(data=F["tcc"], levels=lv(0, 100, 10), cmap=cmap_from("Greys", 10), extend="neither", ticks=lv(0, 100, 20))
    raise KeyError(key)

# ---------- Daten holen ----------
def find_run(session, last_step):
    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    cand = now.replace(hour=now.hour - now.hour % 6)
    for k in range(10):
        run = cand - dt.timedelta(hours=6 * k)
        url = f"{S3}/gfs.{run:%Y%m%d}/{run:%H}/atmos/gfs.t{run:%H}z.pgrb2.{GRID}.f{last_step:03d}.idx"
        try:
            if session.head(url, timeout=30).status_code == 200:
                return run
        except Exception:
            pass
    raise SystemExit("Kein vollständiger GFS-Lauf gefunden.")

def fetch_step(session, run, fhr):
    import pygrib
    base = f"{S3}/gfs.{run:%Y%m%d}/{run:%H}/atmos/gfs.t{run:%H}z.pgrb2.{GRID}.f{fhr:03d}"
    r = session.get(base + ".idx", timeout=60); r.raise_for_status()
    rows = [l.split(":") for l in r.text.strip().splitlines()]
    offs = [int(x[1]) for x in rows]
    parts, keys = [], []
    for key, var, level in WANTED:
        for n, x in enumerate(rows):
            ok = x[4].startswith(level[:-1]) if level.endswith("*") else x[4] == level
            if x[3] == var and ok:
                end = offs[n + 1] - 1 if n + 1 < len(rows) else ""
                rr = session.get(base, headers={"Range": f"bytes={offs[n]}-{end}"}, timeout=180); rr.raise_for_status()
                parts.append(rr.content); keys.append(key); break
    if not parts:
        return None
    with tempfile.NamedTemporaryFile(suffix=".grib2") as tf:   # Nachrichten stehen in der Reihenfolge der Anfragen
        tf.write(b"".join(parts)); tf.flush()
        msgs = list(pygrib.open(tf.name))
        lat, lon = msgs[0].latlons()
        F = {k: np.ma.filled(g.values.astype("float32"), np.nan) for k, g in zip(keys, msgs)}
    return F, lat[:, 0], lon[0, :]

# ---------- Klimatologie (für Anomalien) ----------
def build_climo():
    """Tages-Klimamittel 1991-2020 aus der NCEP/NCAR-Reanalyse (NOAA PSL). Wird einmal geladen und gespeichert."""
    import requests, netCDF4
    out = {}
    for key, var, names, level in [("t850", "air", ["air.day.1991-2020.ltm.nc", "air.day.1981-2010.ltm.nc"], 850),
                                   ("z500", "hgt", ["hgt.day.1991-2020.ltm.nc", "hgt.day.1981-2010.ltm.nc"], 500)]:
        ok = False
        for nm in names:
            try:
                print("Lade Klimatologie", nm, flush=True)
                with requests.get(PSL + nm, stream=True, timeout=600) as r, tempfile.NamedTemporaryFile(suffix=".nc") as tf:
                    r.raise_for_status()
                    for chunk in r.iter_content(1 << 20): tf.write(chunk)
                    tf.flush()
                    ds = netCDF4.Dataset(tf.name)
                    li = int(np.argmin(np.abs(np.array(ds["level"][:]) - level)))
                    a = np.array(ds[var][:, li, :, :], dtype="float32")
                    units = str(getattr(ds[var], "units", "")).lower()
                    if key == "t850" and ("k" in units and "c" not in units.replace("kelvin", "k")): a = a - 273.15
                    out[key] = a; out["lat"] = np.array(ds["lat"][:]); out["lon"] = np.array(ds["lon"][:])
                    ds.close()
                ok = True; break
            except Exception as e:
                print("  nicht möglich:", e, flush=True)
        if not ok:
            return None
    os.makedirs(os.path.dirname(CLIMO_PATH), exist_ok=True)
    np.savez_compressed(CLIMO_PATH, **out)
    return out

def load_climo():
    if os.path.exists(CLIMO_PATH):
        z = np.load(CLIMO_PATH); return {k: z[k] for k in z.files}
    try:
        return build_climo()
    except Exception as e:
        print("Klimatologie nicht verfügbar:", e, flush=True); return None

def climo_on_grid(C, key, valid, lat1d, lon1d):
    from scipy.interpolate import RegularGridInterpolator
    doy = min(valid.timetuple().tm_yday, 365) - 1
    a = C[key][doy]; la = C["lat"]; lo = C["lon"]
    if la[0] > la[-1]: a = a[::-1]; la = la[::-1]
    a = np.concatenate([a, a[:, :1]], axis=1); lo = np.concatenate([lo, [lo[0] + 360]])
    f = RegularGridInterpolator((la, lo), a, bounds_error=False, fill_value=None)
    LON, LAT = np.meshgrid(lon1d % 360, lat1d)
    return f(np.stack([LAT.ravel(), LON.ravel()], axis=-1)).reshape(LAT.shape).astype("float32")

# ---------- Zeichnen ----------
def region_arrays(R, lat1d, lon1d):
    """Gitter auf -180..180 drehen, auf den Ausschnitt verkleinern und zyklisch schließen."""
    lon = np.where(lon1d > 180, lon1d - 360, lon1d)
    order = np.argsort(lon); lon = lon[order]
    la0, la1, lo0, lo1 = R["sub"]
    li = np.where((lat1d >= la0) & (lat1d <= la1))[0]
    ci = order[(lon >= lo0) & (lon <= lo1)]
    lons = lon[(lon >= lo0) & (lon <= lo1)]
    return li, ci, lat1d[li], lons

def sub(arr, li, ci):
    return arr[np.ix_(li, ci)]

def find_extrema(a, lat, lon, ext, size=14):
    res = []
    for mode, fn in (("H", maximum_filter), ("T", minimum_filter)):
        m = (fn(a, size=size, mode="nearest") == a)
        for i, j in zip(*np.where(m)):
            la, lo = lat[i], lon[j]
            if ext[0] + 2 < lo < ext[1] - 2 and ext[2] + 2 < la < ext[3] - 2 and not np.isnan(a[i, j]):
                res.append(("H" if mode == "H" else "T", la, lo, a[i, j]))
    return res

def render_one(rk, pk, F, A, grid, run, fhr, out_path):
    R = REGIONS[rk]; P = next(p for p in PARAMS if p["key"] == pk)
    lat1d, lon1d = grid
    li, ci, lat, lon = region_arrays(R, lat1d, lon1d)
    S = spec_for(pk, F, A)
    LON, LAT = np.meshgrid(lon, lat)
    valid = run + dt.timedelta(hours=fhr)
    fig = plt.figure(figsize=R["size"], dpi=100, facecolor="white")
    if HAVE_CARTOPY:
        ax = fig.add_axes([0.01, 0.075, 0.98, 0.835], projection=R["proj"]())
        ax.set_extent(R["extent"], crs=ccrs.PlateCarree()); tr = dict(transform=ccrs.PlateCarree())
    else:
        ax = fig.add_axes([0.06, 0.12, 0.92, 0.79]); ax.set_xlim(R["extent"][0], R["extent"][1]); ax.set_ylim(R["extent"][2], R["extent"][3]); tr = {}
    ax.set_facecolor("#f3f4f6")
    data = sub(S["data"], li, ci)
    cf = ax.contourf(LON, LAT, data, levels=S["levels"], cmap=S["cmap"], extend=S["extend"], **tr)
    if "lines" in S:
        L = S["lines"]; ld = sub(L["data"], li, ci)
        cs = ax.contour(LON, LAT, ld, levels=L["levels"], colors="#1a1a1a", linewidths=0.5, **tr)
        if L.get("bold"):
            try: ax.contour(LON, LAT, ld, levels=L["bold"], colors="#000000", linewidths=1.6, **tr)
            except Exception: pass
        if L.get("label"):
            try: ax.clabel(cs, fmt="%d", fontsize=7)
            except Exception: pass
    if "arrows" in S:
        u, v = sub(S["arrows"][0], li, ci), sub(S["arrows"][1], li, ci); st = 6 if rk == "europe" else 12
        kw = dict(color="#1b1b1b", scale=900, width=0.0016, headwidth=4, alpha=0.75)
        ax.quiver(LON[::st, ::st], LAT[::st, ::st], u[::st, ::st], v[::st, ::st], **kw, **tr)
    if "extrema" in S:
        for t, la, lo, val in find_extrema(sub(S["extrema"], li, ci), lat, lon, R["extent"]):
            col = "#c1121f" if t == "H" else "#1d4ed8"
            ax.text(lo, la, "H" if t == "H" else "T", color=col, fontsize=17, fontweight="bold", ha="center", va="center", clip_on=True, **tr)
            ax.text(lo, la - 1.6, f"{val:.0f}", color=col, fontsize=8, ha="center", va="top", clip_on=True, **tr)
    if HAVE_CARTOPY:
        ax.coastlines(resolution=R["coast"], linewidth=0.7, color="#222222")
        ax.add_feature(cfeature.BORDERS.with_scale(R["coast"]), linewidth=0.35, edgecolor="#444444")
        ax.gridlines(draw_labels=False, linewidth=0.3, color="#555555", alpha=0.45, linestyle=":")
    else:
        ax.grid(True, linewidth=0.3, alpha=0.5)
    cax = fig.add_axes([0.14, 0.04, 0.72, 0.016])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=S.get("ticks"))
    cb.ax.tick_params(labelsize=8); cb.set_label(P["unit"], fontsize=8, labelpad=1)
    fig.text(0.012, 0.968, P["title"], fontsize=15, fontweight="bold", ha="left", va="center", color="#111111")
    fig.text(0.012, 0.935, f"GFS 0,5° · {R['title']} · Lauf {run:%d.%m.%Y %H} UTC", fontsize=9.5, ha="left", va="center", color="#555555")
    fig.text(0.988, 0.968, f"{WD[valid.weekday()]} {valid:%d.%m.%Y %H} UTC", fontsize=13, fontweight="bold", ha="right", va="center", color="#111111")
    fig.text(0.988, 0.935, f"+{fhr} h", fontsize=10, ha="right", va="center", color="#555555")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, format="webp", dpi=100, pil_kwargs={"quality": 80, "method": 4})
    plt.close(fig)

# ---------- Synthetische Testdaten ----------
def synthetic_step(fhr):
    lat = np.arange(90, -90.01, -0.5); lon = np.arange(0, 360, 0.5)
    LON, LAT = np.meshgrid(lon, lat); r = np.deg2rad; ph = fhr / 24.0
    wave = np.sin(r(LON * 3 + 40 * ph)) * np.cos(r(LAT))
    F = dict(
        t850=273.15 + 22 * np.cos(r(LAT)) - 14 + 9 * wave, z500=5500 + 380 * np.cos(r(LAT * 1.0)) - 300 + 130 * wave,
        z1000=60 + 80 * np.sin(r(LON * 2 + 30 * ph)) * np.cos(r(LAT)), mslp=101300 + 2200 * np.sin(r(LON * 2 + 30 * ph)) * np.cos(r(LAT)) + 600 * np.cos(r(LAT * 3)),
        t2m=273.15 + 28 * np.cos(r(LAT)) - 12 + 7 * wave, u850=18 * np.cos(r(LAT * 2)) + 8 * wave, v850=9 * np.sin(r(LON * 3 + 40 * ph)),
        u250=45 * np.exp(-((LAT - 40) / 18) ** 2) + 20 * wave, v250=18 * np.sin(r(LON * 4 + 50 * ph)),
        prate=np.maximum(0, np.sin(r(LON * 5 + 30 * ph)) * np.cos(r(LAT * 2))) * 4e-3, tcc=np.clip(50 + 55 * np.sin(r(LON * 4 + 20 * ph)) * np.cos(r(LAT)), 0, 100))
    return {k: v.astype("float32") for k, v in F.items()}, lat, lon

def synthetic_climo():
    la = np.arange(90, -90.1, -2.5); lo = np.arange(0, 360, 2.5); LON, LAT = np.meshgrid(lo, la); r = np.deg2rad
    t = (22 * np.cos(r(LAT)) - 14).astype("float32"); z = (5500 + 380 * np.cos(r(LAT)) - 300).astype("float32")
    return dict(lat=la, lon=lo, t850=np.repeat(t[None], 365, 0), z500=np.repeat(z[None], 365, 0))

# ---------- Hauptprogramm ----------
_STATE = {}
def _task(args):
    rk, pk, fhr, out = args
    try:
        render_one(rk, pk, _STATE["F"], _STATE["A"], _STATE["grid"], _STATE["run"], fhr, out)
        return True
    except Exception as e:
        print(f"  Fehler {rk}/{pk}/f{fhr:03d}: {e}", flush=True); return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site"); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--max-hours", type=int, default=168); ap.add_argument("--step", type=int, default=6)
    ap.add_argument("--regions", default="europe,nhem"); ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    a = ap.parse_args()
    steps = list(range(0, a.max_hours + 1, a.step)); regions = [r for r in a.regions.split(",") if r in REGIONS]
    session = None
    if a.synthetic:
        run = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0, hour=0); climo = synthetic_climo()
    else:
        import requests
        session = requests.Session(); session.headers["User-Agent"] = "Modellcharts/1.0"
        run = find_run(session, steps[-1]); climo = load_climo()
    print("Lauf:", run, "Schritte:", len(steps), "Klimatologie:", "ja" if climo else "NEIN (Anomalie-Karten entfallen)", flush=True)
    done = {}   # (param) -> Anzahl erzeugter Bilder
    steps_ok = []
    import multiprocessing as mp
    t0 = time.time()
    for fhr in steps:
        if a.synthetic: F, lat, lon = synthetic_step(fhr)
        else:
            got = None
            for attempt in range(3):
                try:
                    got = fetch_step(session, run, fhr); break
                except Exception as e:
                    print(f"  Schritt {fhr}: Versuch {attempt + 1} fehlgeschlagen ({e})", flush=True); time.sleep(5 * (attempt + 1))
            if not got: print("Schritt", fhr, "ohne Daten"); continue
            F, lat, lon = got
        valid = run + dt.timedelta(hours=fhr)
        A = {}
        if climo:
            for k in ("t850", "z500"):
                if k in F: A[k] = (F[k] - 273.15 if k == "t850" else F[k]) - climo_on_grid(climo, k, valid, lat, lon)
                if k == "z500" and k in A: A[k] = A[k] / 10.0
        tasks = []
        for rk in regions:
            for P in PARAMS:
                if any(n not in F for n in P["needs"]): continue
                if P.get("climo") and P["climo"] not in A: continue
                tasks.append((rk, P["key"], fhr, os.path.join(a.out, rk, P["key"], f"f{fhr:03d}.webp")))
        _STATE.update(F=F, A=A, grid=(lat, lon), run=run)
        if a.workers > 1 and sys.platform != "win32":
            with mp.get_context("fork").Pool(a.workers) as pool: res = pool.map(_task, tasks)
        else:
            res = [_task(t) for t in tasks]
        for t, ok in zip(tasks, res):
            if ok: done[t[1]] = done.get(t[1], 0) + 1
        if any(res): steps_ok.append(fhr)
        print(f"f{fhr:03d}: {sum(res)}/{len(tasks)} Karten ({time.time() - t0:.0f} s)", flush=True)
    params = [dict(key=p["key"], short=p["short"], title=p["title"], unit=p["unit"]) for p in PARAMS if done.get(p["key"])]
    man = dict(run=run.strftime("%Y-%m-%dT%H:%M:%SZ"), generated=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), model="GFS 0,5°",
               steps=steps_ok, regions=[dict(key=k, title=REGIONS[k]["title"]) for k in regions], params=params)
    os.makedirs(a.out, exist_ok=True)
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), ensure_ascii=False)
    open(os.path.join(a.out, ".nojekyll"), "w").close()
    open(os.path.join(a.out, "index.html"), "w").write("<!doctype html><meta charset=utf-8><title>Modellcharts</title><p>Modellcharts-Daten für die Wetter-App. Lauf: " + man["run"] + "</p>")
    print("Fertig:", len(params), "Parameter,", sum(done.values()), "Bilder")

if __name__ == "__main__":
    main()
