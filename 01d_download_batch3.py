"""
Batch 3: 9 nuevas regiones — diversidad tectónica máxima.
Subducción, rift volcánico, zona de falla, colisión continental.
"""
import os, time
import pandas as pd
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

OUT = "data"
os.makedirs(OUT, exist_ok=True)

BATCH3 = [
    # Java, Indonesia — inundaciones extremas ene-feb 2017
    dict(name="java_indonesia",    client="ISC",  lat=-7.5,  lon=110.5, r=150,
         rain="2017-01-31", pre=90,  post=180, mmin=3.0),

    # Himachal Pradesh, India — monzón catastrófico jul 2023 (200+ muertos)
    dict(name="himachal_india",    client="ISC",  lat=31.5,  lon=77.0,  r=100,
         rain="2023-07-09", pre=120, post=180, mmin=2.5),

    # SE España (Murcia) — DANA septiembre 2019, falla de Alhama
    dict(name="murcia_spain",      client="ISC",  lat=37.8,  lon=-1.5,  r=100,
         rain="2019-09-13", pre=120, post=180, mmin=1.5),

    # Reykjanes, Islandia — enjambre 2020-2021, precipitación invernal intensa
    dict(name="reykjanes_iceland", client="ISC",  lat=63.9,  lon=-22.4, r=80,
         rain="2021-02-01", pre=120, post=240, mmin=1.0),

    # Colombia Andina — lluvias intensas may 2022, falla Cauca-Romeral
    dict(name="colombia_andes",    client="ISC",  lat=4.5,   lon=-75.5, r=150,
         rain="2022-05-01", pre=120, post=180, mmin=2.5),

    # Calabria, Italia S — inundaciones oct 2022, sismicidad apenínca
    dict(name="calabria_italy",    client="ISC",  lat=38.5,  lon=16.0,  r=100,
         rain="2022-10-15", pre=120, post=180, mmin=1.5),

    # Chile central — tormentas invernales extremas jun 2023, subducción
    dict(name="chile_central",     client="ISC",  lat=-35.0, lon=-71.0, r=150,
         rain="2023-06-25", pre=120, post=180, mmin=2.5),

    # Guerrero, México — Huracán Pamela oct 2021, zona de deslizamiento lento
    dict(name="guerrero_mexico",   client="USGS", lat=17.5,  lon=-99.5, r=150,
         rain="2021-10-05", pre=120, post=180, mmin=2.5),

    # Mindanao, Filipinas — temporada de tifones dic 2023, zona activa
    dict(name="mindanao_ph",       client="ISC",  lat=7.5,   lon=126.0, r=150,
         rain="2023-12-15", pre=90,  post=180, mmin=3.0),
]

records = []
for c in BATCH3:
    rain_dt = UTCDateTime(c["rain"])
    t0 = rain_dt - c["pre"]  * 86400
    t1 = rain_dt + c["post"] * 86400
    print(f"\n→ {c['name']} [{c['client']}]  [{t0.date} → {t1.date}]  M≥{c['mmin']}", flush=True)
    try:
        cli = Client(c["client"])
        cat = cli.get_events(
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
                "time": orig.time.datetime, "lat": orig.latitude,
                "lon":  orig.longitude, "depth": (orig.depth or 0)/1000.0,
                "mag":  mag.mag if mag else float("nan"),
                "region": c["name"], "rain_date": c["rain"],
            })
        df = pd.DataFrame(rows)
        df.to_csv(f"{OUT}/{c['name']}.csv", index=False)
        print(f"   {len(df)} eventos")
        records.append({"region": c["name"], "n_events": len(df), "status": "OK"})
    except Exception as e:
        print(f"   ERROR: {e}")
        records.append({"region": c["name"], "n_events": 0, "status": str(e)[:80]})
    time.sleep(3)

print("\n=== BATCH 3 ===")
print(pd.DataFrame(records).to_string(index=False))
