"""
Configuración maestra de todas las regiones del estudio.
Coordenadas del epicentro de lluvia (no del centroide sísmico).
"""

REGIONS = {
    # ── BATCH 1 ─────────────────────────────────────────────────────────────
    "los_cabos_mx":     dict(lat=23.05,  lon=-109.72, r=80,  rain="2024-09-14", mc=1.5, f_inf=0.15),
    "noto_japan":       dict(lat=37.5,   lon=137.2,   r=80,  rain="2021-09-16", mc=1.5, f_inf=0.35),
    "nepal_himalaya":   dict(lat=28.0,   lon=84.5,    r=150, rain="2015-08-01", mc=3.6, f_inf=0.25),
    "taiwan":           dict(lat=23.8,   lon=121.0,   r=150, rain="2021-06-01", mc=2.2, f_inf=0.40),
    "apennines_italy":  dict(lat=42.5,   lon=13.2,    r=120, rain="2023-11-02", mc=1.5, f_inf=0.30),
    "pakistan_2022":    dict(lat=30.5,   lon=68.5,    r=200, rain="2022-08-25", mc=2.5, f_inf=0.10),
    "corinth_greece":   dict(lat=38.1,   lon=22.5,    r=80,  rain="2020-09-19", mc=1.6, f_inf=0.25),  # Medicane Ianos peak
    "marlborough_nz":   dict(lat=-41.7,  lon=174.0,   r=120, rain="2022-01-17", mc=1.8, f_inf=0.35),
    # ── BATCH 2 ─────────────────────────────────────────────────────────────
    "west_bohemia_cz":  dict(lat=50.25,  lon=12.45,   r=90,  rain="2018-09-01", mc=1.0, f_inf=0.30),
    "zagros_iran":      dict(lat=32.5,   lon=48.0,    r=250, rain="2019-04-01", mc=2.5, f_inf=0.10),
    "cascades_oregon":  dict(lat=45.4,   lon=-121.7,  r=150, rain="2021-11-15", mc=1.1, f_inf=0.45),
    "papua_new_guinea": dict(lat=-5.5,   lon=146.5,   r=150, rain="2018-01-25", mc=3.3, f_inf=0.50),
    "ethiopia_rift":    dict(lat=9.0,    lon=40.0,    r=300, rain="2020-07-15", mc=2.0, f_inf=0.20),
    "pyrenees_fr":      dict(lat=42.7,   lon=1.2,     r=100, rain="2013-10-15", mc=1.0, f_inf=0.40),
    "assam_india":      dict(lat=25.5,   lon=91.5,    r=200, rain="2017-07-25", mc=2.0, f_inf=0.35),
    "costa_rica":       dict(lat=10.2,   lon=-84.0,   r=120, rain="2016-11-24", mc=2.0, f_inf=0.45),
    # ── BATCH 3 ─────────────────────────────────────────────────────────────
    "java_indonesia":   dict(lat=-7.5,   lon=110.5,   r=150, rain="2017-01-31", mc=3.0, f_inf=0.50),
    "himachal_india":   dict(lat=31.5,   lon=77.0,    r=100, rain="2023-07-09", mc=2.5, f_inf=0.30),
    "murcia_spain":     dict(lat=37.8,   lon=-1.5,    r=100, rain="2019-09-13", mc=1.5, f_inf=0.20),
    "reykjanes_iceland":dict(lat=63.9,   lon=-22.4,   r=80,  rain="2021-02-01", mc=1.0, f_inf=0.20),
    "colombia_andes":   dict(lat=4.5,    lon=-75.5,   r=150, rain="2022-05-01", mc=2.5, f_inf=0.40),
    "calabria_italy":   dict(lat=38.5,   lon=16.0,    r=100, rain="2022-10-15", mc=1.5, f_inf=0.30),
    "chile_central":    dict(lat=-35.0,  lon=-71.0,   r=150, rain="2023-06-25", mc=2.5, f_inf=0.25),
    "guerrero_mexico":  dict(lat=17.5,   lon=-99.5,   r=150, rain="2021-10-05", mc=2.5, f_inf=0.20),
    "mindanao_ph":      dict(lat=7.5,    lon=126.0,   r=150, rain="2023-12-15", mc=3.0, f_inf=0.50),
    # ── SUPPLEMENTAL — national catalogs (lower Mc) ─────────────────────────
    "murcia_ign":       dict(lat=37.8,   lon=-1.5,    r=100, rain="2019-09-13", mc=0.9, f_inf=0.20),
    "calabria_ingv":    dict(lat=38.5,   lon=16.0,    r=100, rain="2022-10-15", mc=1.0, f_inf=0.30),
}
