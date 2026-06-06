"""
Download ISC catalog for known heavy-rainfall / seismic-swarm case studies.
Saves one CSV per region in data/.
"""
import os
import time
import pandas as pd
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

OUT = "data"
os.makedirs(OUT, exist_ok=True)

# Each entry: name, lat, lon, radius_km, rain_date, window_days_before, window_days_after, minmag
CASES = [
    # --- YOUR OWN PAPER ---
    dict(name="los_cabos_mx",     lat=23.05,  lon=-109.72, r=80,
         rain="2024-09-14", pre=180, post=250, mmin=1.0),

    # --- NOTO PENINSULA JAPAN (Science Advances 2024) ---
    # Sustained swarm from late 2020, heavy rains documented 2021
    dict(name="noto_japan",       lat=37.5,   lon=137.2,   r=80,
         rain="2021-09-16", pre=180, post=365, mmin=1.5),

    # --- NEPAL HIMALAYA (Bollinger et al.) ---
    # Monsoon peak August, winter seismicity higher: use Aug 2015 monsoon
    dict(name="nepal_himalaya",   lat=28.0,   lon=84.5,    r=150,
         rain="2015-08-01", pre=120, post=180, mmin=2.5),

    # --- TAIWAN (Science Advances 2021) ---
    # May-Sep monsoon → shallow earthquakes late winter/spring
    dict(name="taiwan",           lat=23.8,   lon=121.0,   r=150,
         rain="2021-06-01", pre=120, post=180, mmin=2.0),

    # --- CENTRAL APENNINES ITALY ---
    # Storm Ciaran Nov 2023 (record rainfall), seismic activity Apennines
    dict(name="apennines_italy",  lat=42.5,   lon=13.2,    r=120,
         rain="2023-11-02", pre=120, post=180, mmin=1.5),

    # --- HIMALAYA INDIA/PAKISTAN (2022 Pakistan monsoon floods) ---
    dict(name="pakistan_2022",    lat=30.5,   lon=68.5,    r=200,
         rain="2022-08-25", pre=120, post=180, mmin=2.5),

    # --- CORINTH GULF GREECE (known fluid-driven swarms, Mediterranean rains) ---
    dict(name="corinth_greece",   lat=38.1,   lon=22.5,    r=80,
         rain="2020-10-01", pre=120, post=180, mmin=1.5),

    # --- NEW ZEALAND (Marlborough, intense rainfall + known seismicity) ---
    dict(name="marlborough_nz",   lat=-41.7,  lon=174.0,   r=120,
         rain="2022-01-17", pre=120, post=180, mmin=1.5),
]

client = Client("ISC")

records = []
for c in CASES:
    rain_dt = UTCDateTime(c["rain"])
    t0 = rain_dt - c["pre"] * 86400
    t1 = rain_dt + c["post"] * 86400

    print(f"\n→ {c['name']}  [{t0.date} → {t1.date}]  M≥{c['mmin']}", flush=True)
    try:
        cat = client.get_events(
            starttime=t0, endtime=t1,
            latitude=c["lat"], longitude=c["lon"],
            maxradius=c["r"] / 111.0,   # km → degrees
            minmagnitude=c["mmin"],
        )
        rows = []
        for ev in cat:
            orig = ev.preferred_origin() or ev.origins[0]
            mag  = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
            rows.append({
                "time":  orig.time.datetime,
                "lat":   orig.latitude,
                "lon":   orig.longitude,
                "depth": (orig.depth or 0) / 1000.0,
                "mag":   mag.mag if mag else float("nan"),
                "region": c["name"],
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
    time.sleep(3)   # cortesía al servidor ISC

summary = pd.DataFrame(records)
summary.to_csv(f"{OUT}/download_summary.csv", index=False)
print("\n\n=== RESUMEN DESCARGA ===")
print(summary.to_string(index=False))
