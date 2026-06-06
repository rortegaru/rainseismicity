"""
Space-Time-Depth diagnostic plots for Regime I candidates.
For each candidate region, plots:
  Panel A: Time vs. Depth (with rain day marked, diffusion envelope)
  Panel B: Time vs. Epicentral distance from centroid
  Panel C: Daily event rate (Poisson model + observed)
  Panel D: Map of seismicity pre vs. post rain

Candidates: noto_japan, corinth_greece, murcia_spain, los_cabos_mx,
            reykjanes_iceland, pyrenees_fr, apennines_italy
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

DATA_DIR = "data"
FIG_DIR  = "figures/english"
os.makedirs(FIG_DIR, exist_ok=True)

# Hydraulic diffusivity curves (m²/s) for diffusion envelopes
D_VALUES = [0.1, 1.0, 10.0]   # m²/s
COLORS_D = ["#2196F3", "#FF9800", "#E91E63"]

CANDIDATES = [
    dict(name="noto_japan",        mc=1.5,  depth_max=30,  r_max=80),
    dict(name="corinth_greece",    mc=1.6,  depth_max=25,  r_max=80),
    dict(name="murcia_spain",      mc=1.5,  depth_max=25,  r_max=100),
    dict(name="los_cabos_mx",      mc=1.5,  depth_max=20,  r_max=80),
    dict(name="reykjanes_iceland", mc=1.0,  depth_max=15,  r_max=80),
    dict(name="pyrenees_fr",       mc=1.0,  depth_max=25,  r_max=100),
    dict(name="apennines_italy",   mc=1.5,  depth_max=30,  r_max=120),
    dict(name="taiwan",            mc=2.2,  depth_max=35,  r_max=150),
]

# Lookup for rainfall labels (for better figure annotation)
RAIN_LABELS = {
    "noto_japan":        "Heavy rainfall\nSept 16, 2021",
    "corinth_greece":    "Mediterranean storm\nOct 1, 2020",
    "murcia_spain":      "DANA storm\nSept 13, 2019\n(SPI = 5.5σ)",
    "los_cabos_mx":      "Tropical cyclone\nSept 14, 2024\n(47 mm / 7 hr)",
    "reykjanes_iceland": "Winter storm\nFeb 1, 2021",
    "pyrenees_fr":       "Autumn rain\nOct 15, 2013",
    "apennines_italy":   "Storm Ciaran\nNov 2, 2023",
    "taiwan":            "Monsoon peak\nJun 1, 2021",
}

def diffusion_envelope(t_days_array, D_m2s, depth_m_ref=None):
    """
    r²(t) = 4 * D * t  [m²]  →  r(t) in km
    or equivalently: depth front z(t) = sqrt(4*D*t)
    """
    t_sec = t_days_array * 86400.0
    t_sec = np.clip(t_sec, 1e-3, None)
    r_km  = np.sqrt(4 * D_m2s * t_sec) / 1000.0
    return r_km


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def make_spacetime_plot(region_name, mc, depth_max, r_max):
    csv_path = f"{DATA_DIR}/{region_name}.csv"
    if not os.path.exists(csv_path):
        print(f"  SKIP: {csv_path} not found")
        return

    df = pd.read_csv(csv_path, parse_dates=["time", "rain_date"])
    df = df.dropna(subset=["lat","lon","depth","mag"]).copy()
    df = df[df["mag"] >= mc].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    rain_dt = pd.Timestamp(df["rain_date"].iloc[0], tz="UTC")

    # Days relative to rain event
    df["day_offset"] = (df["time"] - rain_dt).dt.total_seconds() / 86400.0

    # Epicentral distance from post-rain centroid
    post = df[df["day_offset"] > 0]
    if len(post) < 5:
        lat0 = df["lat"].mean(); lon0 = df["lon"].mean()
    else:
        lat0 = post["lat"].median(); lon0 = post["lon"].median()
    df["dist_km"] = df.apply(
        lambda r: haversine_km(lat0, lon0, r["lat"], r["lon"]), axis=1)

    # Color by period
    pre_mask  = df["day_offset"] < 0
    post_mask = df["day_offset"] >= 0

    # ── FIGURE ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f"{region_name.replace('_', ' ').title()}\n"
                 f"Spatio-temporal seismicity analysis  |  M≥{mc}  |  "
                 f"N={len(df)} events",
                 fontsize=13, fontweight="bold")

    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.3)
    ax_td  = fig.add_subplot(gs[0, 0])   # Time vs Depth
    ax_tr  = fig.add_subplot(gs[0, 1])   # Time vs Distance
    ax_rate= fig.add_subplot(gs[1, 0])   # Daily rate
    ax_map = fig.add_subplot(gs[1, 1])   # Map

    rain_label = RAIN_LABELS.get(region_name, f"Rain event\n{rain_dt.date()}")

    # ─ Panel A: Time vs Depth ─────────────────────────────────────────────────
    ax = ax_td
    ax.scatter(df.loc[pre_mask,  "day_offset"], df.loc[pre_mask,  "depth"],
               s=4*df.loc[pre_mask,"mag"]**2, c="#90A4AE", alpha=0.5,
               label="Pre-rain", zorder=2)
    ax.scatter(df.loc[post_mask, "day_offset"], df.loc[post_mask, "depth"],
               s=4*df.loc[post_mask,"mag"]**2, c="#E53935", alpha=0.6,
               label="Post-rain", zorder=3)

    # Diffusion envelopes in depth: z(t) = sqrt(4Dt)
    t_pos = np.linspace(0.1, df["day_offset"].max(), 400)
    for D, col in zip(D_VALUES, COLORS_D):
        z_env = np.sqrt(4 * D * t_pos * 86400.0) / 1000.0
        z_env = np.clip(z_env, 0, depth_max)
        ax.plot(t_pos, z_env, color=col, lw=1.5, ls="--",
                label=f"D={D} m²/s", alpha=0.8)

    ax.axvline(0, color="navy", lw=2, ls="-", label=rain_label)
    ax.set_xlabel("Days relative to rain event", fontsize=10)
    ax.set_ylabel("Depth (km)", fontsize=10)
    ax.set_ylim(depth_max, 0)   # inverted
    ax.set_title("(A) Time — Depth", fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    # ─ Panel B: Time vs Distance ──────────────────────────────────────────────
    ax = ax_tr
    ax.scatter(df.loc[pre_mask,  "day_offset"], df.loc[pre_mask,  "dist_km"],
               s=4*df.loc[pre_mask,"mag"]**2, c="#90A4AE", alpha=0.5, zorder=2)
    ax.scatter(df.loc[post_mask, "day_offset"], df.loc[post_mask, "dist_km"],
               s=4*df.loc[post_mask,"mag"]**2, c="#E53935", alpha=0.6, zorder=3)

    # Diffusion front r(t)
    for D, col in zip(D_VALUES, COLORS_D):
        r_env = diffusion_envelope(t_pos, D)
        r_env = np.clip(r_env, 0, r_max)
        ax.plot(t_pos, r_env, color=col, lw=1.5, ls="--", alpha=0.8)

    ax.axvline(0, color="navy", lw=2, ls="-")
    ax.set_xlabel("Days relative to rain event", fontsize=10)
    ax.set_ylabel("Distance from centroid (km)", fontsize=10)
    ax.set_ylim(0, r_max)
    ax.set_title("(B) Time — Epicentral Distance", fontweight="bold")
    ax.grid(alpha=0.3)

    # ─ Panel C: Daily event rate ──────────────────────────────────────────────
    ax = ax_rate
    day_min = int(df["day_offset"].min())
    day_max = int(df["day_offset"].max())
    bins    = np.arange(day_min, day_max + 1)
    counts, _ = np.histogram(df["day_offset"], bins=bins)
    bin_centers = bins[:-1] + 0.5

    pre_c  = np.where(bin_centers < 0, counts, 0)
    post_c = np.where(bin_centers >= 0, counts, 0)

    ax.bar(bin_centers, pre_c,  color="#90A4AE", alpha=0.7, width=1, label="Pre-rain")
    ax.bar(bin_centers, post_c, color="#E53935", alpha=0.7, width=1, label="Post-rain")

    # Pre-rain mean rate
    n_pre_days = max(1, len(df[pre_mask]))
    pre_span   = max(1, abs(day_min))
    pre_rate   = counts[:abs(day_min)].mean() if abs(day_min) > 0 else 0
    ax.axhline(pre_rate, color="#546E7A", lw=1.5, ls="--",
               label=f"Pre-rain mean ({pre_rate:.1f} ev/d)")
    ax.axvline(0, color="navy", lw=2, ls="-", label="Rain day")
    ax.set_xlabel("Days relative to rain event", fontsize=10)
    ax.set_ylabel("Events / day", fontsize=10)
    ax.set_title("(C) Daily Event Rate", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ─ Panel D: Map ───────────────────────────────────────────────────────────
    ax = ax_map
    sc_pre  = ax.scatter(df.loc[pre_mask,  "lon"], df.loc[pre_mask,  "lat"],
                         s=5*df.loc[pre_mask,"mag"]**2, c=df.loc[pre_mask,"depth"],
                         cmap="Blues_r", alpha=0.4, vmin=0, vmax=depth_max, zorder=2)
    sc_post = ax.scatter(df.loc[post_mask, "lon"], df.loc[post_mask, "lat"],
                         s=5*df.loc[post_mask,"mag"]**2, c=df.loc[post_mask,"depth"],
                         cmap="Reds_r", alpha=0.6, vmin=0, vmax=depth_max, zorder=3)
    ax.plot(lon0, lat0, "k*", markersize=12, zorder=5, label="Centroid")
    plt.colorbar(sc_post, ax=ax, label="Depth (km)", shrink=0.8)
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.set_title("(D) Spatial Distribution\n(Blue=pre-rain, Red=post-rain)",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Magnitude legend
    for mag_ex in [1.5, 2.5, 3.5]:
        ax.scatter([], [], s=5*mag_ex**2, c="gray", alpha=0.6,
                   label=f"M{mag_ex}")
    ax.legend(fontsize=7, loc="upper right")

    plt.savefig(f"{FIG_DIR}/{region_name}_spacetime.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {FIG_DIR}/{region_name}_spacetime.png")


# ── MAIN ──────────────────────────────────────────────────────────────────────
print("Generating space-time-depth diagnostic plots...\n")
for c in CANDIDATES:
    print(f"Processing: {c['name']}")
    make_spacetime_plot(c["name"], c["mc"], c["depth_max"], c["r_max"])

print("\nDone. All figures saved to", FIG_DIR)
