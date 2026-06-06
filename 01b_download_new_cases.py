"""
Batch 2: 8 nuevos casos con evidencia en literatura de lluvia → sismicidad.
Tectónicamente diversos: rift, colisión, arco, karst, zona de falla.
"""
import os, time
import pandas as pd
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

OUT = "data"
os.makedirs(OUT, exist_ok=True)

NEW_CASES = [
    # ── WEST BOHEMIA / VOGTLAND, Czech Republic / Germany ──────────────────
    # Swarm clásico Oct-Nov 2018; precedido por lluvias otoñales intensas.
    # Referencia: Fischer et al. (2014, 2023), Bräuer et al.
    dict(name="west_bohemia_cz",  lat=50.25, lon=12.45, r=60,
         rain="2018-09-01", pre=90,  post=200, mmin=1.5),

    # ── ZAGROS FOLD BELT, Iran ──────────────────────────────────────────────
    # Inundaciones catastróficas abril 2019 (Lorestan/Khuzestan, 70+ muertos).
    # Zona de seismicidad por fluidos en cinturón plegado.
    dict(name="zagros_iran",      lat=32.5,  lon=48.0,  r=150,
         rain="2019-04-01", pre=120, post=180, mmin=2.5),

    # ── CASCADES / MT HOOD, Oregon, USA ────────────────────────────────────
    # Boettcher & McNamara (2014) documentaron sismicidad inducida por lluvias.
    # Atmospheric river catastrófico nov 2021 (inundaciones históricas en PNW).
    dict(name="cascades_oregon",  lat=45.4,  lon=-121.7, r=100,
         rain="2021-11-15", pre=120, post=180, mmin=1.5),

    # ── PAPUA NEW GUINEA (Highlands / Markham Valley) ──────────────────────
    # Enero 2018: lluvia extrema + enjambre sísmico documentado.
    # Alta sismicidad natural, zona de subducción activa.
    dict(name="papua_new_guinea", lat=-5.5,  lon=146.5, r=150,
         rain="2018-01-25", pre=90,  post=180, mmin=3.0),

    # ── ETHIOPIA / AFAR RIFT ────────────────────────────────────────────────
    # Temporada de lluvias jul-ago 2020 (excepcional).
    # Sismicidad inducida en rift documentada por Wilks et al. (2017).
    dict(name="ethiopia_rift",    lat=9.0,   lon=40.0,  r=150,
         rain="2020-07-15", pre=120, post=180, mmin=2.5),

    # ── PYRENEES (Francia/España) ───────────────────────────────────────────
    # Rigo et al. (2008, 2015): sismicidad kárstica relacionada con agua.
    # Oct 2013: episodio de lluvia intensa + actividad sismica en Ariège.
    dict(name="pyrenees_fr",      lat=42.7,  lon=1.2,   r=80,
         rain="2013-10-15", pre=120, post=180, mmin=1.5),

    # ── ASSAM / NE INDIA (Plataforma Shillong) ─────────────────────────────
    # Jul 2017: inundaciones catastróficas en Assam (5M afectados).
    # Zona de alta sismicidad histórica + monzón intenso.
    dict(name="assam_india",      lat=25.5,  lon=91.5,  r=150,
         rain="2017-07-25", pre=120, post=180, mmin=2.5),

    # ── COSTA RICA (Falla de Nicoya / interior) ─────────────────────────────
    # Huracán Otto, nov 2016 (primero en cruzar Costa Rica en noviembre).
    # Villegas-Lanza et al. documentaron sismicidad post-huracán en la región.
    dict(name="costa_rica",       lat=10.2,  lon=-84.0, r=120,
         rain="2016-11-24", pre=90,  post=180, mmin=2.0),
]

client = Client("ISC")
records = []

for c in NEW_CASES:
    rain_dt = UTCDateTime(c["rain"])
    t0 = rain_dt - c["pre"]  * 86400
    t1 = rain_dt + c["post"] * 86400

    print(f"\n→ {c['name']}  [{t0.date} → {t1.date}]  M≥{c['mmin']}", flush=True)
    try:
        cat = client.get_events(
            starttime=t0, endtime=t1,
            latitude=c["lat"], longitude=c["lon"],
            maxradius=c["r"] / 111.0,
            minmagnitude=c["mmin"],
        )
        rows = []
        for ev in cat:
            orig = ev.preferred_origin() or ev.origins[0]
            mag  = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
            rows.append({
                "time":      orig.time.datetime,
                "lat":       orig.latitude,
                "lon":       orig.longitude,
                "depth":     (orig.depth or 0) / 1000.0,
                "mag":       mag.mag if mag else float("nan"),
                "region":    c["name"],
                "rain_date": c["rain"],
            })
        df = pd.DataFrame(rows)
        path = f"{OUT}/{c['name']}.csv"
        df.to_csv(path, index=False)
        print(f"   {len(df)} eventos → {path}")
        records.append({"region": c["name"], "n_events": len(df),
                        "rain_date": c["rain"], "status": "OK"})
    except Exception as e:
        print(f"   ERROR: {e}")
        records.append({"region": c["name"], "n_events": 0,
                        "rain_date": c["rain"], "status": str(e)[:80]})
    time.sleep(3)

summary = pd.DataFrame(records)
print("\n\n=== RESUMEN DESCARGA BATCH 2 ===")
print(summary.to_string(index=False))
