"""
Pore-pressure diffusion analysis: r² vs t plots for Regime I candidates.
Estimates hydraulic diffusivity D from the seismicity migration front.
Method: Shapiro et al. (1997), Rothert & Shapiro (2003), Ortega et al. (2026)
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit
import glob

warnings.filterwarnings("ignore")

DATA_DIR = "data"
FIG_DIR  = "figures/english"
RES_DIR  = "results"
os.makedirs(FIG_DIR, exist_ok=True)

# Physical constants
SECONDS_PER_DAY = 86400.0

CANDIDATES = [
    dict(name="noto_japan",        mc=1.5, lat_rain=37.5,  lon_rain=137.2,  post_days=200),
    dict(name="corinth_greece",    mc=1.6, lat_rain=38.1,  lon_rain=22.5,   post_days=200),
    dict(name="murcia_spain",      mc=1.5, lat_rain=37.8,  lon_rain=-1.5,   post_days=180),
    dict(name="los_cabos_mx",      mc=1.5, lat_rain=23.05, lon_rain=-109.72,post_days=250),
    dict(name="pyrenees_fr",       mc=1.0, lat_rain=42.7,  lon_rain=1.2,    post_days=180),
    dict(name="apennines_italy",   mc=1.5, lat_rain=42.5,  lon_rain=13.2,   post_days=180),
    dict(name="reykjanes_iceland", mc=1.0, lat_rain=63.9,  lon_rain=-22.4,  post_days=240),
    dict(name="taiwan",            mc=2.2, lat_rain=23.8,  lon_rain=121.0,  post_days=180),
]

D_TEST = np.logspace(-2, 2, 400)   # m²/s range for envelope scanning


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def diffusion_front_km(t_days, D_m2s):
    """r(t) = sqrt(4 * D * t)  in km"""
    return np.sqrt(4 * D_m2s * t_days * SECONDS_PER_DAY) / 1000.0


def fit_diffusion_D(r2_m2, t_sec):
    """Linear fit r² = 4D·t → D = slope/4"""
    if len(r2_m2) < 5:
        return np.nan, np.nan, np.nan
    slope, intercept, r_val, p_val, stderr = linregress(t_sec, r2_m2)
    D_fit = slope / 4.0
    return D_fit, r_val**2, stderr / 4.0


def percentile_front(df_post, percentile=90):
    """
    Compute the expanding distance front: for each day, take the
    {percentile}th percentile of distances of events up to that day.
    Returns (day_array, r_front_km).
    """
    days = np.arange(0, int(df_post["day_offset"].max()) + 1)
    r_front = []
    for d in days:
        mask = (df_post["day_offset"] >= 0) & (df_post["day_offset"] <= d)
        sub = df_post.loc[mask, "dist_km"]
        if len(sub) >= 3:
            r_front.append(np.percentile(sub, percentile))
        else:
            r_front.append(np.nan)
    return np.array(days, dtype=float), np.array(r_front)


results = []

for c in CANDIDATES:
    region = c["name"]
    csv_path = f"{DATA_DIR}/{region}.csv"
    if not os.path.exists(csv_path):
        print(f"SKIP: {region} (no data)")
        continue

    print(f"\n{'='*55}\nDiffusion analysis: {region}")

    df = pd.read_csv(csv_path, parse_dates=["time", "rain_date"])
    df = df.dropna(subset=["lat","lon","depth","mag"]).copy()
    df = df[df["mag"] >= c["mc"]].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    rain_dt = pd.Timestamp(df["rain_date"].iloc[0], tz="UTC")
    df["day_offset"] = (df["time"] - rain_dt).dt.total_seconds() / SECONDS_PER_DAY
    df["dist_km"] = df.apply(
        lambda r: haversine_km(c["lat_rain"], c["lon_rain"], r["lat"], r["lon"]), axis=1)
    df["dist_m2"] = (df["dist_km"] * 1000.0)**2   # r² in m²
    df["t_sec"]   = df["day_offset"] * SECONDS_PER_DAY

    df_post = df[(df["day_offset"] > 0) & (df["day_offset"] <= c["post_days"])].copy()
    df_pre  = df[df["day_offset"] <= 0].copy()
    n_post  = len(df_post)
    print(f"  Post-rain events: {n_post}")

    if n_post < 10:
        print("  SKIP: too few post-rain events")
        continue

    # ── Percentile front ─────────────────────────────────────────────────────
    days_arr, r_front = percentile_front(df_post, percentile=90)
    valid = ~np.isnan(r_front)
    r2_front = (r_front * 1000.0)**2   # m²

    # ── Fit D from percentile front ──────────────────────────────────────────
    t_sec_valid = days_arr[valid] * SECONDS_PER_DAY
    r2_valid    = r2_front[valid]
    D_fit, r2_score, D_err = fit_diffusion_D(r2_valid, t_sec_valid)
    print(f"  D_fit (90th pct front) = {D_fit:.3f} m²/s  R²={r2_score:.3f}")

    # ── Scan best-fit D by minimizing envelope residuals ─────────────────────
    def envelope_residual(D):
        r_env = diffusion_front_km(days_arr[valid], D)
        return np.mean((r_env - r_front[valid])**2)

    residuals = np.array([envelope_residual(d) for d in D_TEST])
    D_best    = D_TEST[np.argmin(residuals)]
    print(f"  D_best (envelope scan) = {D_best:.3f} m²/s")

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{region.replace('_',' ').title()}  —  Pore-Pressure Diffusion Analysis  |  M≥{c['mc']}",
                 fontsize=13, fontweight="bold")

    # Panel A: r²–t scatter + linear fit
    ax = axes[0]
    sc = ax.scatter(df_post["day_offset"], df_post["dist_km"],
                    c=df_post["depth"], cmap="viridis_r", s=10, alpha=0.4,
                    vmin=0, vmax=30, label="Post-rain events")
    plt.colorbar(sc, ax=ax, label="Depth (km)", shrink=0.8)

    t_plot = np.linspace(0.1, c["post_days"], 400)
    D_display = [0.1, 1.0, D_best, 10.0]
    D_labels  = [f"D=0.1", f"D=1.0", f"D={D_best:.2f} (best fit)", f"D=10.0"]
    D_colors  = ["#90CAF9","#1976D2","#E53935","#F57F17"]
    for D, lbl, col in zip(D_display, D_labels, D_colors):
        lw = 2.5 if "best" in lbl else 1.2
        ax.plot(t_plot, diffusion_front_km(t_plot, D), color=col, lw=lw,
                ls="-" if "best" in lbl else "--", label=lbl)

    ax.plot(days_arr[valid], r_front[valid], "k-", lw=2, label="90th pct front")
    ax.set_xlabel("Days after rain event", fontsize=11)
    ax.set_ylabel("Distance from rain source (km)", fontsize=11)
    ax.set_title("(A) r–t diagram with diffusion envelopes", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    # Panel B: r²–t  (linear: r² = 4Dt → D from slope)
    ax = axes[1]
    ax.scatter(df_post["t_sec"]/SECONDS_PER_DAY, df_post["dist_m2"]/1e6,
               c=df_post["depth"], cmap="viridis_r", s=8, alpha=0.3)
    if not np.isnan(D_fit) and D_fit > 0:
        t_line = np.linspace(0, c["post_days"], 300)
        r2_line = 4 * D_fit * t_line * SECONDS_PER_DAY / 1e6
        ax.plot(t_line, r2_line, "r-", lw=2,
                label=f"$r^2 = 4Dt$\nD = {D_fit:.3f} m²/s\n$R^2$={r2_score:.2f}")
    ax.set_xlabel("Days after rain event", fontsize=11)
    ax.set_ylabel("$r^2$ (km²)", fontsize=11)
    ax.set_title("(B) $r^2$–$t$ scaling (Darcy diffusion)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel C: Residual scan → best-fit D
    ax = axes[2]
    ax.semilogx(D_TEST, residuals, "b-", lw=2)
    ax.axvline(D_best, color="red", lw=2, ls="--",
               label=f"Best-fit D = {D_best:.3f} m²/s")
    ax.axvspan(0.1, 10, alpha=0.1, color="green", label="Typical crustal range")
    ax.set_xlabel("Hydraulic diffusivity D (m²/s)", fontsize=11)
    ax.set_ylabel("Mean squared residual (km²)", fontsize=11)
    ax.set_title("(C) D estimation: residual scan", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    fig_path = f"{FIG_DIR}/{region}_diffusion_r2t.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure → {fig_path}")

    results.append({
        "region":      region,
        "n_post":      n_post,
        "D_fit_m2s":   round(D_fit, 4) if not np.isnan(D_fit) else None,
        "D_best_m2s":  round(D_best, 4),
        "r2_score":    round(r2_score, 3) if not np.isnan(r2_score) else None,
        "t_diff_10km_days": round(10000**2 / (4 * D_best * SECONDS_PER_DAY), 1),
    })

df_res = pd.DataFrame(results)
df_res.to_csv(f"{RES_DIR}/diffusion_results.csv", index=False)

print("\n\n" + "="*70)
print("DIFFUSION RESULTS — Hydraulic diffusivity estimates")
print("="*70)
print(df_res.to_string(index=False))
