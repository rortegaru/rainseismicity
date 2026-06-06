"""
Fixes para casos con pocos eventos en ISC.
West Bohemia → ISC con mmin más bajo y radio mayor.
Cascades    → USGS (mejor cobertura local).
Ethiopia    → ISC radio mayor, mmin más bajo.
Assam       → ISC radio mayor, mmin más bajo.
Zagros      → ISC radio mayor.
Pyrenees    → ISC radio mayor, mmin más bajo.
"""
import os, time
import pandas as pd
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

OUT = "data"

FIXES = [
    # West Bohemia: swarms con M<2, bajamos umbral y expandimos área
    dict(name="west_bohemia_cz",  client="ISC",  lat=50.25, lon=12.45, r=90,
         rain="2018-09-01", pre=90, post=200, mmin=1.0),

    # Cascades: USGS tiene PNSN y regional networks
    dict(name="cascades_oregon",  client="USGS", lat=45.4,  lon=-121.7, r=150,
         rain="2021-11-15", pre=120, post=180, mmin=1.0),

    # Ethiopia: expandir a 300km, mmin=2.0
    dict(name="ethiopia_rift",    client="ISC",  lat=9.0,   lon=40.0,  r=300,
         rain="2020-07-15", pre=120, post=180, mmin=2.0),

    # Assam: expandir a 200km, mmin=2.0
    dict(name="assam_india",      client="ISC",  lat=25.5,  lon=91.5,  r=200,
         rain="2017-07-25", pre=120, post=180, mmin=2.0),

    # Zagros: expandir radio
    dict(name="zagros_iran",      client="ISC",  lat=32.5,  lon=48.0,  r=250,
         rain="2019-04-01", pre=120, post=180, mmin=2.5),

    # Pyrenees: bajar mmin
    dict(name="pyrenees_fr",      client="ISC",  lat=42.7,  lon=1.2,   r=100,
         rain="2013-10-15", pre=120, post=180, mmin=1.0),
]

records = []
for c in FIXES:
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
        records.append({"region": c["name"], "n_events": len(df), "status": "OK"})
    except Exception as e:
        print(f"   ERROR: {e}")
        records.append({"region": c["name"], "n_events": 0, "status": str(e)[:80]})
    time.sleep(3)

print("\n=== RESUMEN FIXES ===")
print(pd.DataFrame(records).to_string(index=False))
