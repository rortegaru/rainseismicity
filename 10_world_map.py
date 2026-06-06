"""
World map figure: all 25 study regions, colored by hydrological regime
and sized by ETAS excess ratio. Publication-ready.
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

FIG_DIR = "figures/english"
os.makedirs(FIG_DIR, exist_ok=True)

# All regions: lat, lon, regime, ETAS ratio, Bayesian P(+), label
REGIONS_PLOT = [
    # name,                    lat,    lon,    regime, etas_R, bayes_P, label_offset
    ("Los Cabos, Mexico",       23.05, -109.72, "I",    2.31,  0.00,  (1,0.5)),
    ("Murcia, Spain",           37.8,  -1.5,   "I",    None,  0.785, (1, 0.5)),
    ("Costa Rica",              10.2,  -84.0,  "I",    None,  0.226, (-2,-1)),
    ("Noto, Japan",             37.5,  137.2,  "II",   2.86,  1.00,  (1, 0.5)),
    ("Corinth, Greece",         38.1,  22.5,   "0*",   1.24,  1.00,  (1, 0.5)),
    ("Apennines, Italy",        42.5,  13.2,   "II",   1.16,  0.999, (-4, 0.5)),
    ("Calabria, Italy",         38.5,  16.0,   "0",    None,  0.056, (1,-1)),
    ("Pyrenees, France",        42.7,  1.2,    "0*",   1.01,  0.999, (-5, 1)),
    ("Reykjanes, Iceland",      63.9,  -22.4,  "0*",   None,  1.00,  (-1, 1.5)),
    ("W. Bohemia, Czech R.",    50.25, 12.45,  "0",    None,  0.00,  (0.5, 1)),
    ("Taiwan",                  23.8,  121.0,  "II",   0.90,  1.00,  (1, 0.5)),
    ("Nepal Himalaya",          28.0,  84.5,   "flag", None,  0.00,  (1,-1)),
    ("Assam, India",            25.5,  91.5,   "II",   0.72,  0.69,  (1, 0.5)),
    ("Himachal Pradesh",        31.5,  77.0,   "I",    None,  None,  (-1,-1.5)),
    ("Zagros, Iran",            32.5,  48.0,   "II",   0.51,  0.00,  (1, 0.5)),
    ("Pakistan",                30.5,  68.5,   "II",   0.72,  0.49,  (1,-1)),
    ("Ethiopia Rift",           9.0,   40.0,   "I",    None,  0.00,  (1, 0.5)),
    ("Marlborough, NZ",        -41.7,  174.0,  "0",    0.93,  0.20,  (1, 0.5)),
    ("Cascades, Oregon",        45.4, -121.7,  "III",  0.94,  0.35,  (-7, 0.5)),
    ("Papua New Guinea",        -5.5,  146.5,  "II",   1.00,  0.51,  (1, 0.5)),
    ("Java, Indonesia",         -7.5,  110.5,  "II",   None,  0.53,  (1,-1.5)),
    ("Colombia Andes",          4.5,  -75.5,   "0",    None,  0.30,  (-3,-1)),
    ("Chile Central",          -35.0, -71.0,   "II",   None,  0.40,  (-5, 0.5)),
    ("Guerrero, Mexico",        17.5,  -99.5,  "0",    None,  0.07,  (1,-1)),
    ("Mindanao, Philippines",   7.5,   126.0,  "0",    None,  0.00,  (1, 0.5)),
]

REGIME_COLORS = {
    "I":    "#E53935",   # red — isolated impulse, Darcy applicable
    "II":   "#FB8C00",   # orange — moderate contrast
    "III":  "#8BC34A",   # green — embedded in wet season
    "0*":   "#9C27B0",   # purple — ERA5 misses but likely isolated
    "0":    "#78909C",   # gray — no confirmed rain signal
    "flag": "#37474F",   # dark gray — flagged/contaminated
}
REGIME_LABELS = {
    "I":    "Regime I — Isolated impulse (Darcy applicable)",
    "II":   "Regime II — Moderate contrast",
    "III":  "Regime III — Embedded in wet season",
    "0*":   "Regime 0* — ERA5 blind, likely isolated",
    "0":    "Regime 0 — No confirmed rainfall signal",
    "flag": "Flagged (non-hydro trigger dominant)",
}

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 11),
                        subplot_kw={"projection": None})

# Simple Robinson-like world outline using matplotlib
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    fig.clear()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.LAND,  facecolor="#F5F5F0", edgecolor="#CCCCCC", lw=0.4)
    ax.add_feature(cfeature.OCEAN, facecolor="#E3F2FD")
    ax.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="#AAAAAA")
    ax.add_feature(cfeature.BORDERS, lw=0.3, edgecolor="#CCCCCC", linestyle=":")
    ax.gridlines(lw=0.3, color="#DDDDDD", linestyle="--")
    use_cartopy = True
    transform = ccrs.PlateCarree()
except ImportError:
    # Fallback: simple lat/lon axes
    ax.set_xlim(-180, 180)
    ax.set_ylim(-80, 85)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.grid(alpha=0.3)
    use_cartopy = False
    transform = None

for (name, lat, lon, regime, etas_R, bayes_P, lbl_off) in REGIONS_PLOT:
    color = REGIME_COLORS.get(regime, "#78909C")

    # Size: based on evidence strength
    evidence = 0
    if etas_R and etas_R > 1.2: evidence += 2
    if bayes_P and bayes_P > 0.90: evidence += 2
    if regime == "I": evidence += 2
    size = 40 + 60 * min(evidence, 4) / 4.0

    kwargs = dict(s=size, color=color, alpha=0.85, edgecolors="white",
                  linewidths=0.8, zorder=5)
    if use_cartopy:
        ax.scatter(lon, lat, transform=transform, **kwargs)
    else:
        ax.scatter(lon, lat, **kwargs)

    # Label for key sites
    if regime in ["I", "0*"] or (etas_R and etas_R > 1.5):
        txt_kwargs = dict(fontsize=7.5, fontweight="bold", color="#1A237E",
                          ha="left", va="center",
                          path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        if use_cartopy:
            ax.text(lon + lbl_off[0], lat + lbl_off[1], name,
                    transform=transform, **txt_kwargs)
        else:
            ax.text(lon + lbl_off[0], lat + lbl_off[1], name, **txt_kwargs)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_elements = [
    Patch(facecolor=REGIME_COLORS[r], edgecolor="white", label=REGIME_LABELS[r])
    for r in ["I","0*","II","III","0","flag"]
]
legend_elements += [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#888",
           markersize=s/6, label=lbl)
    for s, lbl in [(40,"Weak/no signal"),(80,"Moderate signal"),(100,"Strong signal")]
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=8,
          title="Hydrological regime & signal strength", title_fontsize=9,
          framealpha=0.9, ncol=2)

ax.set_title(
    "Global Survey of Rainfall-Triggered Seismicity\n"
    "25 study regions — color by hydrological regime, size by evidence strength",
    fontsize=14, fontweight="bold", pad=12)

plt.tight_layout()
fig_path = f"{FIG_DIR}/world_map_all_regions.png"
plt.savefig(fig_path, dpi=180, bbox_inches="tight")
plt.close()
print(f"World map saved → {fig_path}")
