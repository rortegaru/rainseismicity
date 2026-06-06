"""
Publication-quality figures for GRL submission.
All text in English. Clean, minimal, journal-ready.

Figure 1: World map
Figure 2: Regime conceptual diagram + IC vs SPI scatter
Figure 3: 4-panel results (Bayesian + ETAS for top 4 candidates)
Figure 4: Diffusion r²-t + D continuum
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import glob

warnings.filterwarnings("ignore")

FIG_DIR     = "figures/publication"
SUBCATS_DIR = "results/subcatalogs"
ORIGINS_CSV = "results/darcy_origins.csv"
os.makedirs(FIG_DIR, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "DejaVu Serif",
    "font.size":       9,
    "axes.labelsize":  9,
    "axes.titlesize":  10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi":      150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

REGIME_COLOR = {
    "I":    "#C62828",   # IC ≥ 10, isolated impulse, Darcy confirmed
    "I_nc": "#43A047",   # IC ≥ 10 but insufficient local catalog to confirm
    "V":    "#6A1B9A",   # IC/SPI NOT ESTIMABLE — ERA5 data too poor (formerly 0*)
    "II":   "#E65100",   # 3 ≤ IC < 10, moderate antecedent moisture
    "III":  "#2E7D32",   # IC < 3, wet season / elastic loading
    "IV":   "#546E7A",   # no confirmed post-rain anomaly
    "flag": "#263238",   # flagged — dominated by other large event
    "0*":   "#6A1B9A",   # alias kept for Figure 1 backward compat
    "0":    "#546E7A",   # alias kept for Figure 1 backward compat
}

# ── Pipeline helpers ──────────────────────────────────────────────────────────
def load_origins():
    if not os.path.exists(ORIGINS_CSV):
        return pd.DataFrame()
    return pd.read_csv(ORIGINS_CSV)

def load_subcatalog(region, sub_n):
    path = os.path.join(SUBCATS_DIR, f"{region}_sub{sub_n}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["time"])
    return df

def get_origin(origins, region):
    """Return the best (last-saved) origin row for a region."""
    rows = origins[origins["region"] == region]
    if rows.empty:
        return None
    return rows.iloc[-1]

def diffusion_envelope(D_ms2, t_days_arr):
    """r envelope in km for given D (m²/s) and t in days."""
    return np.sqrt(4 * D_ms2 * t_days_arr * 86400) / 1000


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — WORLD MAP
# ════════════════════════════════════════════════════════════════════════════

REGIONS_ALL = [
    ("Los Cabos, Mexico",    23.1, -109.7, "I",   2.31,  True),
    ("Murcia, Spain",        37.8,  -1.5,  "I",   1.49,  True),
    ("Costa Rica",           10.2, -84.0,  "I",   0.75,  False),
    ("Himachal Pradesh",     31.5,  77.0,  "I",   None,  False),
    ("Noto, Japan",          37.5, 137.2,  "I",   2.86,  True),
    ("Corinth, Greece",      38.1,  22.5,  "I",   5.75,  True),
    ("Pyrenees, France",     42.7,   1.2,  "0*",  2.38,  True),
    ("Reykjanes, Iceland",   63.9, -22.4,  "0*",  47.0,  True),
    ("Apennines, Italy",     42.5,  13.2,  "II",  1.32,  True),
    ("Taiwan",               23.8, 121.0,  "II",  1.16,  False),
    ("Assam, India",         25.5,  91.5,  "II",  1.59,  False),
    ("Zagros, Iran",         32.5,  48.0,  "II",  0.16,  False),
    ("Papua NG",             -5.5, 146.5,  "II",  1.07,  False),
    ("Pakistan",             30.5,  68.5,  "II",  1.15,  False),
    ("Java, Indonesia",      -7.5, 110.5,  "II",  1.34,  False),
    ("Chile",               -35.0, -71.0,  "II",  0.99,  False),
    ("Calabria, Italy",      38.5,  16.0,  "IV",  0.75,  True),
    ("Marlborough, NZ",     -41.7, 174.0,  "IV",  1.02,  True),
    ("Cascades, Oregon",     45.4,-121.7,  "III", 1.07,  False),
    ("Nepal",                28.0,  84.5,  "flag",0.02,  False),
    ("W. Bohemia, CZ",       50.2,  12.5,  "IV",  0.14,  False),
    ("Colombia",              4.5, -75.5,  "IV",  0.86,  False),
    ("Guerrero, MX",         17.5, -99.5,  "IV",  0.45,  False),
    ("Mindanao, PH",          7.5, 126.0,  "IV",  0.19,  False),
    ("Ethiopia",              9.0,  40.0,  "I",   0.20,  False),
]

fig, ax = plt.subplots(figsize=(9, 4.5))
try:
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    import matplotlib.patheffects as pe
    fig.clear()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.LAND,      facecolor="#F5F5F0", edgecolor="#BBBBBB", lw=0.3)
    ax.add_feature(cfeature.OCEAN,     facecolor="#EAF4FB")
    ax.add_feature(cfeature.COASTLINE, lw=0.4, color="#999999")
    ax.add_feature(cfeature.BORDERS,   lw=0.2, color="#CCCCCC", linestyle=":")
    ax.gridlines(lw=0.2, color="#DDDDDD", linestyle="--", alpha=0.5)
    transform   = ccrs.PlateCarree()
    use_cartopy = True
except ImportError:
    import matplotlib.patheffects as pe
    ax.set_xlim(-180, 180); ax.set_ylim(-75, 85)
    ax.set_facecolor("#EAF4FB")
    ax.grid(alpha=0.3, lw=0.5)
    transform   = None
    use_cartopy = False

label_offsets = {
    "Los Cabos, Mexico": (2, -2.5),
    "Noto, Japan": (1.5, 1.5),
    "Pyrenees, France": (-12, 2),
    "Reykjanes, Iceland": (2, 1.5),
    "Apennines, Italy": (-2, 2.5),   # above dot, avoids Calabria below
    "Marlborough, NZ":  (-7, 1.5),   # left of dot (near east map edge)
}
# Murcia, Corinth and Calabria use leader lines — handled separately below
MURCIA_DOT   = (37.8,  -1.5)   # lat, lon of the dot
MURCIA_LBL   = (24.5,  -9.5)   # lat, lon of the label anchor
CORINTH_DOT  = (38.1,  22.5)   # lat, lon of the dot
CORINTH_LBL  = (41.0,  58.0)   # lat, lon of the label anchor (far right, Caspian level)
CALABRIA_DOT = (38.5,  16.0)   # lat, lon of the dot
CALABRIA_LBL = (24.0,  20.0)   # lat, lon of the label anchor (far below, Libya level)

for (name, lat, lon, regime, signal, labeled) in REGIONS_ALL:
    color = REGIME_COLOR.get(regime, "#546E7A")
    sig   = signal if signal else 0.5
    size  = 20 + 60 * min(sig, 5) / 5.0
    kw    = dict(s=size, color=color, alpha=0.85, edgecolors="white",
                 linewidths=0.8, zorder=5)
    if use_cartopy:
        ax.scatter(lon, lat, transform=transform, **kw)
    else:
        ax.scatter(lon, lat, **kw)
    if labeled and name not in ("Murcia, Spain", "Corinth, Greece", "Calabria, Italy"):
        dx, dy = label_offsets.get(name, (1.5, 1.5))
        txt_kw = dict(fontsize=7.5, fontweight="bold", color="#1A237E",
                      path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        short = name.split(",")[0]
        if use_cartopy:
            ax.text(lon+dx, lat+dy, short, transform=transform, **txt_kw)
        else:
            ax.text(lon+dx, lat+dy, short, **txt_kw)

# ── Murcia leader line (down-left) ───────────────────────────────────────
_mlat, _mlon = MURCIA_DOT
_llat, _llon = MURCIA_LBL
_lc = REGIME_COLOR["I"]
_txt_kw = dict(fontsize=7.5, fontweight="bold", color="#1A237E",
               ha="center", va="top",
               path_effects=[pe.withStroke(linewidth=2, foreground="white")])
if use_cartopy:
    ax.plot([_mlon, _llon], [_mlat, _llat],
            color=_lc, lw=0.9, ls="-", transform=transform, zorder=4)
    ax.text(_llon, _llat - 1.2, "Murcia", transform=transform, **_txt_kw)
else:
    ax.plot([_mlon, _llon], [_mlat, _llat], color=_lc, lw=0.9, zorder=4)
    ax.text(_llon, _llat - 1.2, "Murcia", **_txt_kw)

# ── Corinth leader line (far right) ──────────────────────────────────────
_mlat, _mlon = CORINTH_DOT
_llat, _llon = CORINTH_LBL
_lc = REGIME_COLOR["I"]
_txt_kw2 = dict(fontsize=7.5, fontweight="bold", color="#1A237E",
                ha="left", va="center",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])
if use_cartopy:
    ax.plot([_mlon, _llon], [_mlat, _llat],
            color=_lc, lw=0.9, ls="-", transform=transform, zorder=4)
    ax.text(_llon + 1.0, _llat, "Corinth", transform=transform, **_txt_kw2)
else:
    ax.plot([_mlon, _llon], [_mlat, _llat], color=_lc, lw=0.9, zorder=4)
    ax.text(_llon + 1.0, _llat, "Corinth", **_txt_kw2)

# ── Calabria leader line (far below) ─────────────────────────────────────
_mlat, _mlon = CALABRIA_DOT
_llat, _llon = CALABRIA_LBL
_lc_cal = REGIME_COLOR["IV"]
_txt_kw_cal = dict(fontsize=7.5, fontweight="bold", color="#1A237E",
                   ha="center", va="top",
                   path_effects=[pe.withStroke(linewidth=2, foreground="white")])
if use_cartopy:
    ax.plot([_mlon, _llon], [_mlat, _llat],
            color=_lc_cal, lw=0.9, ls="-", transform=transform, zorder=4)
    ax.text(_llon, _llat - 1.2, "Calabria", transform=transform, **_txt_kw_cal)
else:
    ax.plot([_mlon, _llon], [_mlat, _llat], color=_lc_cal, lw=0.9, zorder=4)
    ax.text(_llon, _llat - 1.2, "Calabria", **_txt_kw_cal)

legend_elements = [
    mpatches.Patch(facecolor=REGIME_COLOR["I"],   label="Regime I — Isolated impulse"),
    mpatches.Patch(facecolor=REGIME_COLOR["V"],   label="Regime V — IC/SPI not estimable (ERA5-blind)"),
    mpatches.Patch(facecolor=REGIME_COLOR["II"],  label="Regime II — Moderate contrast"),
    mpatches.Patch(facecolor=REGIME_COLOR["III"], label="Regime III — Wet season"),
    mpatches.Patch(facecolor=REGIME_COLOR["IV"],  label="Regime IV — No confirmed signal"),
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=7,
          framealpha=0.9, edgecolor="#CCCCCC")
ax.set_title("Global survey of rainfall-triggered seismicity (25 regions)",
             fontsize=10, fontweight="bold", pad=8)

plt.tight_layout(pad=0.3)
plt.savefig(f"{FIG_DIR}/Figure1_world_map.pdf", dpi=300, bbox_inches="tight")
plt.savefig(f"{FIG_DIR}/Figure1_world_map.png", dpi=200, bbox_inches="tight")
plt.close()
print("Figure 1 saved")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — REGIME CONCEPTUAL + IC vs SPI SCATTER
# ════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(9, 4))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
ax_regI    = fig.add_subplot(gs[0])
ax_regIII  = fig.add_subplot(gs[1])
ax_scatter = fig.add_subplot(gs[2])

t = np.linspace(-40, 120, 400)

# Panel a: Regimes I & II overlaid
# ── Regime II (behind): moderate antecedent moisture, weaker impulse, smaller belly
baseline_II  = 2.8                                    # elevated pre-rain rate
seismicity_II = np.where(t >= 20,
                         baseline_II + 3.2 * np.exp(-(t-38)**2/420),
                         baseline_II)
rain_II = np.where((t >= 0) & (t < 3), 18, np.nan)   # smaller impulse (IC≈5)

# ── Regime I (front): dry baseline, sharp impulse, strong belly
baseline_I   = 0.5
seismicity_I = np.where(t >= 15,
                        baseline_I + 8.0 * np.exp(-(t-30)**2/300),
                        baseline_I)

ax = ax_regI

# Regime II — draw first (behind)
ax.fill_between(t, 0, np.where(t < 0, baseline_II, np.nan),
                color="#E65100", alpha=0.18, zorder=1)
ax.fill_between(t, 0, rain_II,
                color="#1565C0", alpha=0.35, zorder=2)
ax.fill_between(t, baseline_II,
                np.where(t >= 20, seismicity_II, np.nan),
                color="#E65100", alpha=0.30, zorder=2)
ax.plot(t[t >= 20], seismicity_II[t >= 20],
        color="#E65100", lw=1.2, ls="--", zorder=3)
ax.axhline(baseline_II, color="#E65100", lw=0.8, ls=":", alpha=0.7)
ax.text(60, seismicity_II[t >= 20].max() + 0.8, "Regime II",
        fontsize=8, color="#E65100", fontweight="bold")

# Regime I — draw on top
ax.fill_between(t, 0, np.where(t < 0, baseline_I, np.nan),
                color="#BBDEFB", alpha=0.7, zorder=3, label="Dry baseline\n(Reg. I)")
ax.fill_between(t, 0, np.where((t >= 0) & (t < 3), 45, np.nan),
                color="#1565C0", alpha=0.9, zorder=4, label="Rain impulse\n(47 mm, IC≈58)")
ax.fill_between(t, baseline_I,
                np.where(t >= 15, seismicity_I, np.nan),
                color="#C62828", alpha=0.70, zorder=4, label="Seismicity\nresponse")
ax.axvline(0,  color="#1565C0", lw=1.5, ls="--", zorder=5)
ax.axvline(15, color="#C62828", lw=1.5, ls="--", zorder=5, label="τ = 15 d\n(Reg. I)")
ax.text(60, seismicity_I[t >= 15].max() + 0.8, "Regime I",
        fontsize=8, color="#C62828", fontweight="bold")

# IC labels — white background box so they don't bleed into fill data
ax.text(-38, 41, "$\\mathrm{API}_\\ell \\approx 1$~mm/d\n(IC $\\approx$ 58)",
        fontsize=7, color="#C62828", zorder=10,
        bbox=dict(facecolor="white", edgecolor="#C62828",
                  boxstyle="round,pad=0.3", alpha=0.95, linewidth=0.8))
ax.text(-38, 32, "$\\mathrm{API}_\\ell \\approx 5$~mm/d\n(IC $\\approx$ 5)",
        fontsize=7, color="#E65100", zorder=10,
        bbox=dict(facecolor="white", edgecolor="#E65100",
                  boxstyle="round,pad=0.3", alpha=0.95, linewidth=0.8))

ax.set_xlim(-40, 120); ax.set_ylim(-1, 55)
ax.set_xlabel("Days relative to rain", fontsize=9)
ax.set_ylabel("Events/day  |  Rain (mm)", fontsize=9)
ax.set_title("(a) Regimes I and II — Impulse response\n(same mechanism, different contrast)",
             fontweight="bold", fontsize=9)
ax.legend(fontsize=6.5, loc="upper right",
          bbox_to_anchor=(1.0, 1.0), bbox_transform=ax.transAxes,
          framealpha=0.85, handlelength=1.4, handletextpad=0.4,
          borderpad=0.4, labelspacing=0.35)

# Panel b: Regime III schematic
np.random.seed(99)
rain_season = np.where(t < 0, 8 + 3*np.sin(t/7), 12 + 3*np.sin(t/5))
seism_III   = 3 + 0.5*np.sin(t/20) + 0.3*np.random.randn(400)
ax = ax_regIII
ax.fill_between(t, 0, rain_season, color="#1565C0", alpha=0.5, label="Sustained rainfall")
ax.fill_between(t, 0, seism_III,   color="#37474F", alpha=0.5, label="Seismicity (no response)")
ax.axvline(0, color="#1565C0", lw=1.5, ls="--")
ax.text(-35, 14, "$\\mathrm{API}_\\ell \\approx 8$~mm/d\n(IC $\\approx$ 1.5)",
        fontsize=7.5, color="#555", zorder=10,
        bbox=dict(facecolor="white", edgecolor="#777777",
                  boxstyle="round,pad=0.3", alpha=0.95, linewidth=0.8))
ax.set_xlim(-40, 120); ax.set_ylim(-0.5, 18)
ax.set_xlabel("Days relative to rain", fontsize=9)
ax.set_title("(b) Regime III — Wet season\n(loading mechanism)", fontweight="bold", fontsize=9)
ax.legend(fontsize=7, loc="upper right")

# Panel c: IC vs SPI scatter
DATA_POINTS = [
    # ── Regime I: IC ≥ 10, reliably estimated ───────────────────────────────
    # Labeled only if D was fitted from r²–t pipeline (8 cases total).
    # Murcia, Himachal, C.Rica, Ethiopia: valid IC but no D → unlabeled points.
    ("Los Cabos MX",  58,    3.5, "I",   True),   # D confirmed, label in loop
    ("Murcia ES",     15.8,  5.5, "I",   False),  # Regime I IC but catalog insufficient — no label
    ("Himachal IN",   16.1,  6.2, "I",   False),  # Regime I IC but catalog insufficient — no label
    ("C.Rica CR",     10.2,  2.9, "I",   False),  # Regime I IC but catalog insufficient — no label
    ("Ethiopia ET",   13.3,  3.6, "I",   False),  # Regime I IC but catalog insufficient — no label
    # ── Regime V: IC/SPI NOT ESTIMABLE — ERA5 too poor ──────────────────────
    # Cannot compute IC or SPI reliably; Darcy still analysed from seismicity.
    ("Noto JP",       28.0, -0.4, "I",   False),  # reclassified ERA5h IC≈28
    ("Corinth GR",    23.3, -0.2, "I",   False),  # reclassified Ianos IC=23.3
    ("Pyrenees FR",    4.5, -0.2, "V",   False),  # D confirmed — manual label
    ("Reykjanes IS",   0.0, -0.5, "V",   False),  # D confirmed — manual label
    # ── Regime II: 3 ≤ IC < 10 ──────────────────────────────────────────────
    ("Apennines IT",   3.9,  0.6, "II",  False),  # D confirmed — manual label
    ("Taiwan TW",      5.9,  1.9, "II",  False),
    ("Assam IN",       9.2,  1.3, "II",  False),
    ("Zagros IR",      9.2,  2.2, "II",  False),
    ("Papua NG",       6.7, -0.1, "II",  False),
    ("Pakistan PK",    6.8,  2.1, "II",  False),
    ("Java ID",        6.6,  0.5, "II",  False),
    ("Chile CL",       3.0,  0.5, "II",  False),
    # ── Regime III: IC < 3 ──────────────────────────────────────────────────
    ("Cascades US",    2.9,  0.2, "III", False),
    # ── Regime IV: no confirmed anomaly ─────────────────────────────────────
    ("Calabria IT",    0.1, -0.4, "IV",  False),  # D unconfirmed — manual label
    ("Marlborough NZ", 0.0, -0.4, "IV",  False),  # D confirmed — manual label
    ("W.Bohemia CZ",   0.6, -0.4, "IV",  False),
    ("Colombia CO",    0.3, -0.8, "IV",  False),
    ("Guerrero MX",    2.9, -0.4, "IV",  False),
    ("Mindanao PH",    0.7, -0.8, "IV",  False),
    # ── Flagged ─────────────────────────────────────────────────────────────
    ("Nepal NP",       1.1, -0.6, "flag",False),  # dominated by M7.8 aftershocks
]
ax = ax_scatter
for (name, ic, spi, regime, labeled) in DATA_POINTS:
    color = REGIME_COLOR.get(regime, "#546E7A")
    ax.scatter(ic, spi, s=50, color=color, alpha=0.85,
               edgecolors="white", linewidths=0.5, zorder=4)
    if labeled:
        # Los Cabos MX — only labeled point in main loop; has D → gets box
        ax.annotate(name, (ic, spi), textcoords="offset points",
                    xytext=(4, 3), fontsize=7, color=color,
                    fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor=color,
                              boxstyle="square,pad=0.25", linewidth=0.9),
                    zorder=10)
# ── Manual leader-line annotations for the 7 D-confirmed cases outside the
#    main loop (Los Cabos MX is handled in the main loop at IC=58).
#    All cluster at low IC — spread labels to avoid overlap.
_cV  = REGIME_COLOR["V"]
_cII = REGIME_COLOR["II"]
_cIV = REGIME_COLOR["IV"]

# All 7 D-confirmed cases outside the main loop — spread vertically to avoid overlap.
# Strategy: cluster at IC≈0 split into UP group and RIGHT group.

# ── UP group: labels above the low-IC cluster ───────────────────────────────
def _ann(ax, txt, xy, xytext, col):
    """Annotate with leader line and colored bounding box (= D confirmed).
    zorder=10 keeps these above all other elements."""
    ax.annotate(txt, xy=xy, xytext=xytext,
                fontsize=7, color=col, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=col, lw=0.9),
                bbox=dict(facecolor="white", edgecolor=col, linewidth=0.9,
                          boxstyle="square,pad=0.25"), zorder=10)

# ── Regime I — no local network: plain label, no box, no leader line ─────────
_cI = REGIME_COLOR["I"]
for nm, ic, spi, dx, dy in [
    ("Himachal IN", 16.1, 6.2,  0.4,  0.35),
    ("Murcia ES",   15.8, 5.5,  0.4, -0.45),
    ("Ethiopia ET", 13.3, 3.6,  0.4,  0.30),
]:
    ax.text(ic + dx, spi + dy, nm, fontsize=7, color=_cI, fontweight="bold",
            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")],
            zorder=5)

# C.Rica CR: lowest SPI among no-network reds — needs leader line to avoid
# overlapping with the Noto JP label that sits at IC≈10, SPI≈2.5
ax.annotate("C.Rica CR", xy=(10.2, 2.9), xytext=(28, 3.5),
            fontsize=7, color=_cI, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=_cI, lw=0.8),
            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")],
            zorder=5)

# ── UP group ─────────────────────────────────────────────────────────────────
_ann(ax, "Noto JP",    xy=(28.0, -0.4), xytext=(38,   1.5), col=_cI)
_ann(ax, "Corinth GR", xy=(23.3, -0.2), xytext=(52,  -0.5), col=_cI)
_ann(ax, "Reykjanes IS", xy=(0.0,  -0.5), xytext=(40,   0.1), col=_cV)

# ── RIGHT group ───────────────────────────────────────────────────────────────
_ann(ax, "Pyrenees FR",   xy=(4.5,  -0.2), xytext=(40,  0.9), col=_cV)
_ann(ax, "Apennines IT",  xy=(3.9,   0.6), xytext=(36,  2.5), col=_cII)
_ann(ax, "Marlborough NZ",xy=(0.0,  -0.4), xytext=(14, -1.2), col=_cIV)
_ann(ax, "Calabria IT†",  xy=(0.1,  -0.4), xytext=(30,  2.0), col=_cIV)

ax.axvline(10, color="#C62828", lw=1, ls="--", alpha=0.7, label="IC = 10 (Regime I threshold)")
ax.axhline(1.5, color="#E65100", lw=1, ls=":",  alpha=0.7, label="SPI = 1.5")
ax.set_xlabel("Impulse Contrast  (IC = $P_{\\rm event}/\\bar{P}_{30}$)", fontsize=9)
ax.set_ylabel("Standardized Precipitation Index (SPI)", fontsize=9)
ax.set_title("(c) Regime classification", fontweight="bold", fontsize=9)
ax.set_xlim(-1, 65); ax.set_ylim(-1.5, 7.5)
legend_elements2 = [
    mpatches.Patch(facecolor=REGIME_COLOR["I"],   label="Regime I"),
    mpatches.Patch(facecolor=REGIME_COLOR["II"],  label="Regime II"),
    mpatches.Patch(facecolor=REGIME_COLOR["III"], label="Regime III"),
    mpatches.Patch(facecolor=REGIME_COLOR["IV"],  label="Regime IV"),
    mpatches.Patch(facecolor=REGIME_COLOR["V"],   label="Regime V (ERA5-blind)"),
    # Marker style legend entry: box around label = D confirmed
    Line2D([0],[0], marker='s', color='w', markerfacecolor='white',
           markeredgecolor='#555', markersize=7, markeredgewidth=1.2,
           label="[box] = D fitted from pipeline"),
]
ax.legend(handles=legend_elements2, fontsize=7,
          loc="upper left",
          bbox_to_anchor=(50, 7.4), bbox_transform=ax.transData,
          framealpha=0.95, edgecolor="#CCCCCC",
          borderpad=0.5, labelspacing=0.3)

plt.savefig(f"{FIG_DIR}/Figure2_regimes.pdf", dpi=300, bbox_inches="tight")
plt.savefig(f"{FIG_DIR}/Figure2_regimes.png", dpi=200, bbox_inches="tight")
plt.close()
print("Figure 2 saved")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — RESULTS PANEL: 4 KEY CANDIDATES
# Uses pipeline subcatalogs for bottom row (distance from trigger point).
# ════════════════════════════════════════════════════════════════════════════

origins = load_origins()

CANDIDATES_FIG3 = [
    # (region, title, Mc, tau_best_days, etas_R)
    # Only the 4 most instructive manually reviewed Darcy cases.
    # Mc and tau_best from pipeline results; etas_R from ETAS analysis.
    ("los_cabos_mx",   "Los Cabos, Mexico\n(Regime I  |  47 mm  |  IC≈58)",       1.5, 15,  2.31),
    ("noto_japan",     "Noto Peninsula, Japan\n(Regime I  |  ETAS R=2.86)",        1.5, 55,  2.86),
    ("corinth_greece", "Corinth Gulf, Greece\n(Regime I  |  IC=23.3  |  5.75×)",  1.6, 90,  5.75),
    ("pyrenees_fr",    "Pyrenees, France\n(Regime V  |  Bayes 2.38×  |  D=10.2 m²/s)", 1.0, 155, 1.01),
]

fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
fig.suptitle("Statistical evidence and seismicity migration for post-rain anomalies",
             fontsize=11, fontweight="bold", y=1.01)

for col, (region, title, mc, tau_best, etas_r) in enumerate(CANDIDATES_FIG3):
    csv_raw = f"data/{region}.csv"
    if not os.path.exists(csv_raw):
        for row in range(2):
            axes[row, col].set_visible(False)
        continue

    # ── Top row: daily rate from raw catalog ─────────────────────────────
    df_raw = pd.read_csv(csv_raw, parse_dates=["time", "rain_date"])
    df_raw = df_raw[df_raw["mag"] >= mc].dropna(subset=["lat", "lon"]).copy()
    df_raw["time"]     = pd.to_datetime(df_raw["time"],     utc=True)
    rain_dt            = pd.Timestamp(df_raw["rain_date"].iloc[0], tz="UTC")
    df_raw["day_off"]  = (df_raw["time"] - rain_dt).dt.total_seconds() / 86400

    day_min   = int(df_raw["day_off"].min())
    day_max   = int(df_raw["day_off"].max())
    day_range = np.arange(day_min, day_max + 1)
    counts    = (df_raw.groupby(df_raw["day_off"].apply(int)).size()
                       .reindex(day_range, fill_value=0).values)

    ax = axes[0, col]
    ax.bar(day_range, np.where(day_range < 0, counts, 0),  color="#78909C", width=1, alpha=0.7)
    ax.bar(day_range, np.where(day_range >= 0, counts, 0), color="#C62828", width=1, alpha=0.75)
    pre_rate = counts[day_range < 0].mean() if (day_range < 0).sum() > 0 else 0
    ax.axhline(pre_rate, color="#78909C", lw=1.5, ls="--", label=f"Pre-rain: {pre_rate:.1f}/d")
    ax.axvline(0,        color="#1565C0", lw=2,   ls="-",  label="Rain event")
    ax.axvline(tau_best, color="#C62828", lw=1.5, ls="--", label=f"τ={tau_best}d")
    ax.set_title(title, fontsize=8, fontweight="bold")
    ax.set_xlabel("Days relative to rain")
    ax.set_ylabel("Events / day" if col == 0 else "")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.2)
    if etas_r:
        ax.text(0.02, 0.97, f"ETAS R={etas_r:.2f}×",
                transform=ax.transAxes, fontsize=8, va="top",
                color="#C62828" if etas_r > 1.2 else "#555", fontweight="bold")

    # ── Bottom row: r–t from pipeline subcatalog ──────────────────────────
    ax = axes[1, col]
    orig = get_origin(origins, region)

    if orig is not None:
        sub_n = int(orig["sub_n"])
        D     = float(orig["D_chosen"])
        sub   = load_subcatalog(region, sub_n)

        if sub is not None:
            post = sub[sub["t_from_origin"] > 0].copy()
            pre  = sub[sub["t_from_origin"] <= 0].copy()

            # Split post-origin events by inside_envelope:
            # inside  → support the Darcy interpretation → prominent
            # outside → tectonic background in the same box → faded
            inside_mask = post["inside_envelope"].astype(str).str.lower() == "true"
            post_in  = post[inside_mask].copy()
            post_out = post[~inside_mask].copy()

            # Pre-origin background (small, gray)
            ax.scatter(pre["t_from_origin"], pre["r_3d"],
                       s=3, c="#B0BEC5", alpha=0.25, zorder=2)
            # Outside-envelope: small, very faded gray (background noise in box)
            ax.scatter(post_out["t_from_origin"], post_out["r_3d"],
                       s=3, c="#CFD8DC", alpha=0.25, zorder=2)
            # Inside-envelope: larger, colored by depth — the Darcy cluster
            if len(post_in) > 0:
                sc = ax.scatter(post_in["t_from_origin"], post_in["r_3d"],
                                s=18,
                                c=post_in["depth"].clip(0, 35) if "depth" in post_in
                                  else "#C62828",
                                cmap="plasma_r", vmin=0, vmax=35,
                                alpha=0.85, edgecolors="white",
                                linewidths=0.3, zorder=4)

            # Diffusion envelope curves
            t_max = post["t_from_origin"].max() if len(post) > 0 else 30
            t_env = np.linspace(0.5, t_max * 1.05, 300)
            for Di, col_d, ls, lw in [
                (D,             "#C62828", "-",  2.0),
                (max(0.1, D/4), "#BBDEFB", "--", 1.0),
                (min(D*4, 200), "#FFA726", "--", 1.0),
            ]:
                lbl = f"D={D:.2f} m²s⁻¹" if Di == D else f"D={Di:.1f}"
                ax.plot(t_env, diffusion_envelope(Di, t_env),
                        color=col_d, lw=lw, ls=ls, label=lbl)

            ax.text(0.02, 0.97, f"D = {D:.2f} m²/s\nN = {len(post_in)} inside envelope",
                    transform=ax.transAxes, fontsize=7.5, va="top",
                    color="#C62828", fontweight="bold", zorder=10,
                    bbox=dict(facecolor="white", edgecolor="#C62828",
                              boxstyle="round,pad=0.35", linewidth=0.9,
                              alpha=0.95))
            ax.legend(fontsize=6.5, loc="upper right")

            # Y-axis: tight around inside-envelope events + envelope at t_max
            r_env_max = diffusion_envelope(D, t_max)
            r_in_max  = post_in["r_3d"].max() if len(post_in) > 0 else r_env_max
            r_max = max(r_env_max, r_in_max) * 1.20
            ax.set_ylim(0, r_max)
        else:
            ax.text(0.5, 0.5, "subcatalog not found", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="gray")
    else:
        ax.text(0.5, 0.5, "no pipeline origin", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="gray")

    ax.set_xlabel("Days from trigger point (t₀)")
    ax.set_ylabel("Distance r₃D (km)" if col == 0 else "")
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/Figure3_results_panel.pdf", dpi=300, bbox_inches="tight")
plt.savefig(f"{FIG_DIR}/Figure3_results_panel.png", dpi=200, bbox_inches="tight")
plt.close()
print("Figure 3 saved")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — (a) IC Sensitivity · (b) Multi-method convergence · (c) D continuum
# ════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(13, 4.8),
                         gridspec_kw={"width_ratios": [1, 1.1, 1]})
fig.suptitle("Regime framework robustness and hydraulic diffusivity",
             fontsize=11, fontweight="bold")

# ── Panel a: IC threshold sensitivity (F1 score) ─────────────────────────
ax = axes[0]
ic_sens = pd.read_csv("results/ic_sensitivity.csv")
ic_thresh = ic_sens["IC_threshold"].values
f1   = ic_sens["F1"].values
prec = ic_sens["PPV"].values   # precision
rec  = ic_sens["TPR"].values   # recall (sensitivity)

ax.plot(ic_thresh, f1,   color="#C62828", lw=2.5, label="F1 score", zorder=4)
ax.plot(ic_thresh, prec, color="#1565C0", lw=1.5, ls="--", label="Precision", zorder=3)
ax.plot(ic_thresh, rec,  color="#2E7D32", lw=1.5, ls=":",  label="Recall",    zorder=3)

# Shade stable F1 band (IC 8–15)
stable = (ic_thresh >= 8) & (ic_thresh <= 15)
ax.fill_between(ic_thresh[stable], 0, f1[stable],
                alpha=0.15, color="#C62828", label="Stable zone (IC 8–15)")
# Mark IC=10
ax.axvline(10, color="#C62828", lw=1.5, ls="--", alpha=0.8)
ax.text(10.3, 0.05, "IC = 10\n(adopted)", fontsize=7, color="#C62828")

ax.set_xlabel("Impulse Contrast threshold  $\\mathrm{IC}^*$", fontsize=9)
ax.set_ylabel("Score", fontsize=9)
ax.set_title("(a) IC threshold sensitivity\n(F1, Precision, Recall — 25 regions)",
             fontweight="bold", fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_xlim(ic_thresh.min(), ic_thresh.max())
ax.legend(fontsize=7, loc="upper left")
ax.grid(alpha=0.25)

# ── Panel b: Multi-method evidence convergence for the 4 confirmed cases ──
ax = axes[1]

# Data: (case, Bayes_P, ETAS_R, F_in, rho_tilde)
# Los Cabos Bayes P masked by pre-existing swarm → show ETAS only (marked †)
CASES_CONV = [
    ("Los Cabos MX",  np.nan, 2.31, 0.65, 0.89, "#C62828"),
    ("Noto JP",       1.00,   2.86, 0.62, 0.73, "#C62828"),
    ("Corinth GR",    1.00,   1.24, 0.87, 0.55, "#C62828"),
    ("Pyrenees FR",   0.999,  1.01, 0.82, 0.62, "#6A1B9A"),
]
METHODS   = ["Bayes $P$", "ETAS $\\mathcal{R}$", "$F_{\\rm in}$", "$1 - \\tilde{\\rho}$"]
N_CASES   = len(CASES_CONV)
N_METHODS = len(METHODS)

# Normalize each metric to [0, 1] for color scale
def norm_bayes(p):   return p if not np.isnan(p) else np.nan  # already 0-1
def norm_etas(r):    return min(r / 3.0, 1.0)                  # R=3 → 1.0
def norm_fin(f):     return f                                   # already 0-1
def norm_rho(rho):   return max(0, 1 - (rho - 0.4) / 0.8)     # rho=0.4→1, rho=1.2→0

vals = np.zeros((N_CASES, N_METHODS))
for i, (name, p, r, fin, rho, col) in enumerate(CASES_CONV):
    vals[i, 0] = norm_bayes(p)
    vals[i, 1] = norm_etas(r)
    vals[i, 2] = norm_fin(fin)
    vals[i, 3] = norm_rho(rho)

# Draw heatmap cells
cmap_ev = plt.cm.RdYlGn
for i in range(N_CASES):
    for j in range(N_METHODS):
        v = vals[i, j]
        color = "lightgray" if np.isnan(v) else cmap_ev(v)
        rect = mpatches.FancyBboxPatch((j - 0.45, i - 0.42), 0.90, 0.84,
                                       boxstyle="round,pad=0.05",
                                       facecolor=color, edgecolor="white",
                                       linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        # Value label
        name_c, p_c, r_c, fin_c, rho_c, _ = CASES_CONV[i]
        raw_vals = [p_c, r_c, fin_c, rho_c]
        raw = raw_vals[j]
        if np.isnan(raw):
            txt, col_t = "masked†", "gray"
        elif j == 1:   # ETAS R: show raw
            txt, col_t = f"{raw:.2f}×", ("white" if v > 0.5 else "#333")
        elif j == 0:   # Bayes P
            txt, col_t = f"{raw:.3f}", ("white" if v > 0.5 else "#333")
        else:
            txt, col_t = f"{raw:.2f}", ("white" if v > 0.5 else "#333")
        ax.text(j, i, txt, ha="center", va="center",
                fontsize=8, color=col_t, fontweight="bold", zorder=3)

ax.set_xticks(range(N_METHODS))
ax.set_xticklabels(METHODS, fontsize=8.5)
ax.set_yticks(range(N_CASES))
ax.set_yticklabels([c[0] for c in CASES_CONV], fontsize=8.5)
ax.set_xlim(-0.5, N_METHODS - 0.5)
ax.set_ylim(-0.5, N_CASES - 0.5)
ax.set_title("(b) Multi-method evidence convergence\n(4 confirmed cases)",
             fontweight="bold", fontsize=9)
ax.tick_params(bottom=False, left=False)
for spine in ax.spines.values():
    spine.set_visible(False)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap_ev, norm=plt.Normalize(0, 1))
sm.set_array([])
cb = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, aspect=20)
cb.set_label("Normalized evidence strength", fontsize=7)
cb.ax.tick_params(labelsize=7)

ax.text(0.0, -0.15, "† Bayesian $P$ masked by pre-existing swarm; ETAS $\\mathcal{R}$ correctly identifies anomaly",
        fontsize=6.5, color="gray", ha="left", va="top", transform=ax.transAxes, clip_on=False)

# ── Panel c: D continuum — read from darcy_origins.csv ───────────────────
ax = axes[2]

# Solid pipeline cases to display (region, label, tectonic context, color)
SOLID_CASES = [
    # (region, label, color, override_D, use_override)
    # Ordered by D value (ascending). Los Cabos uses FEFLOW/SRL value.
    # Calabria marked † = unconfirmed candidate (POOR tier).
    ("los_cabos_mx",      "Los Cabos MX\n(crystalline rift)",      "#C62828", 0.94,  True),
    ("noto_japan",        "Noto JP\n(fractured basement)",          "#C62828", None,  False),
    ("corinth_greece",    "Corinth GR\n(extensional/karst)",        "#C62828", None,  False),
    ("pyrenees_fr",       "Pyrenees FR\n(karst/fractured)",         "#6A1B9A", None,  False),
    ("apennines_italy",   "Apennines IT\n(thrust/karst)",           "#E65100", None,  False),
    ("marlborough_nz",    "Marlborough NZ\n(strike-slip)",          "#546E7A", None,  False),
    ("calabria_italy",    "Calabria IT†\n(Apennine thrust)",        "#546E7A", None,  False),
    ("reykjanes_iceland", "Reykjanes IS\n(volcanic rift)",          "#1E88E5", None,  False),
]

bar_data = []
for (region, label, color, override_D, use_override) in SOLID_CASES:
    if use_override:
        bar_data.append((label, override_D, color, "(FEFLOW/SRL)"))
    else:
        orig = get_origin(origins, region)
        if orig is not None:
            bar_data.append((label, float(orig["D_chosen"]), color, "(pipeline)"))

names_d  = [b[0] for b in bar_data]
D_vals   = [b[1] for b in bar_data]
colors_d = [b[2] for b in bar_data]
sources  = [b[3] for b in bar_data]
y_pos    = np.arange(len(bar_data))

ax.barh(y_pos, D_vals, color=colors_d, alpha=0.85, edgecolor="white", height=0.6)
for i, (D_val, col, src) in enumerate(zip(D_vals, colors_d, sources)):
    ax.text(D_val * 1.15, i, f"{D_val:.2f}", va="center",
            fontsize=7.5, color=col, fontweight="bold")

ax.axvspan(0.3,   5,   alpha=0.15, color="#C62828", label="Compact crystalline")
ax.axvspan(5,    20,   alpha=0.15, color="#9C27B0", label="Fractured / karst")
ax.axvspan(20,  200,   alpha=0.15, color="#1565C0", label="Volcanic / hydrothermal")

ax.set_yticks(y_pos)
ax.set_yticklabels(names_d, fontsize=8)
ax.set_xlabel("Hydraulic diffusivity $D_h$ (m² s⁻¹)")
ax.set_xscale("log")
ax.set_xlim(0.3, 300)
ax.set_title("(c) $D_h$ across tectonic settings\n(three-decade continuum — all pipeline cases)",
             fontweight="bold", fontsize=9)
ax.text(0.02, -0.23, "† Calabria IT: unconfirmed candidate ($\\tilde{\\rho}=1.42$, POOR tier)",
        transform=ax.transAxes, fontsize=6.5, color="#546E7A", style="italic")
ax.legend(fontsize=7, loc="lower right")
ax.grid(alpha=0.3, which="both", axis="x")

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.savefig(f"{FIG_DIR}/Figure4_diffusion.pdf", dpi=300, bbox_inches="tight")
plt.savefig(f"{FIG_DIR}/Figure4_diffusion.png", dpi=200, bbox_inches="tight")
plt.close()
print("Figure 4 saved")

print(f"\nAll figures saved to {FIG_DIR}/")
print("PDFs ready for journal submission.")
