#!/usr/bin/env python3
"""
20_SI_spacetime_subcatalog.py
==============================
Regenerate Figures S1–S8 (space-time-depth diagnostics) using the
manually-picked subcatalogs from scripts 16–17. Each figure shows
exactly the subregion used for the Darcy r²-t analysis.

Does NOT modify 08_spacetime_depth_plots.py.

Output: figures/SI_subcatalog/{region}_spacetime.png
Copy to AGU_PAPER: cp figures/SI_subcatalog/*.png /Users/roberto/AGU_PAPER/figures_SI/

Panels per figure:
  (A) Depth — time  : all subcatalog events; rain day + origin marked
  (B) r — t         : post-origin only; D_chosen envelope; inside/outside color
  (C) r² — t        : same as B; linear slope = 4D
  (D) Map            : subcatalog events colored by depth; origin star
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from config_regions import REGIONS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SUBCATS_DIR = os.path.join(RESULTS_DIR, "subcatalogs")
ORIGINS_CSV = os.path.join(RESULTS_DIR, "darcy_origins.csv")
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR     = os.path.join(os.path.dirname(__file__), "figures", "SI_subcatalog")
os.makedirs(OUT_DIR, exist_ok=True)

DEPTH_MAX = 35   # km — color ceiling

# Active sub_n per region (from manual picking session Jun-2026).
# Only regions with an actual subcatalog from scripts 16-17.
ACTIVE_SUBS = {
    "los_cabos_mx":      3,
    "noto_japan":        3,
    "corinth_greece":    2,
    "murcia_spain":      2,   # D=100 (slider max) → no valid envelope
    "pyrenees_fr":       2,
    "reykjanes_iceland": 2,
    "apennines_italy":   2,
    "marlborough_nz":    1,
    "calabria_italy":    5,
}

# All regions with actual manual Darcy analysis — Taiwan excluded (no subcatalog)
S_FIGURES = [
    "los_cabos_mx",
    "noto_japan",
    "corinth_greece",
    "murcia_spain",
    "pyrenees_fr",
    "reykjanes_iceland",
    "apennines_italy",
    "marlborough_nz",
    "calabria_italy",
]

# Reference D lines shown faint (m²/s)
D_REF = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]


def darcy_r(D_ms2, t_days):
    """Diffusion front r(t) in km.  r = sqrt(4·D·t)"""
    return np.sqrt(np.maximum(4.0 * D_ms2 * t_days * 86400.0, 0.0)) / 1000.0


def _sz(mag_series, mc):
    return np.clip(18 * 10 ** (0.5 * (mag_series - mc)), 5, 300)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def make_figure(region):
    sub_n = ACTIVE_SUBS.get(region)
    cfg   = REGIONS[region]
    rain_date = pd.Timestamp(cfg["rain"])
    mc        = cfg.get("mc", 1.5)

    # ── Load subcatalog ───────────────────────────────────────────────────────
    sub_path = os.path.join(SUBCATS_DIR, f"{region}_sub{sub_n}.csv")
    if not os.path.exists(sub_path):
        print(f"  SKIP {region}: subcatalog not found → {sub_path}")
        return

    df = pd.read_csv(sub_path, parse_dates=["time"])

    origins = pd.read_csv(ORIGINS_CSV)
    orig = origins[(origins["region"] == region) & (origins["sub_n"] == sub_n)].iloc[-1]
    origin_t     = pd.Timestamp(orig["origin_time"])
    origin_lat   = float(orig["origin_lat"])
    origin_lon   = float(orig["origin_lon"])
    origin_depth = float(orig["origin_depth"])
    D_chosen     = float(orig["D_chosen"])
    dist_type    = str(orig["dist_type"])

    # Distance column
    dist_col = {"3d": "r_3d", "horizontal": "r_horizontal",
                "vertical": "r_vertical"}.get(dist_type, "r_3d")
    if dist_col not in df.columns:
        dist_col = "r_3d"

    # Rain day position on the t_from_origin axis (negative = before origin)
    rain_from_origin = (rain_date - origin_t).total_seconds() / 86400.0

    t_col   = "t_from_origin"
    x_label = f"Days from origin ({origin_t.strftime('%Y-%m-%d')})"
    valid_D = (D_chosen < 98.0)   # D=100 is the slider-max flag for murcia

    # ── Apply mc filter and clip depth ───────────────────────────────────────
    df = df[df["mag"] >= mc].copy()
    df["depth_clip"] = df["depth"].clip(0, DEPTH_MAX)

    pre_mask  = df[t_col] < 0
    post_mask = df[t_col] >= 0
    n_post    = int(post_mask.sum())

    inside_col = df["inside_envelope"].astype(bool) if "inside_envelope" in df.columns \
                 else pd.Series(False, index=df.index)

    # ── Diffusion time axis ──────────────────────────────────────────────────
    t_post_vals = df.loc[post_mask, t_col].values
    t_max  = float(t_post_vals.max()) * 1.10 if len(t_post_vals) > 0 else 90.0
    t_arr  = np.linspace(0.01, t_max, 500)

    r_post = df.loc[post_mask, dist_col].values if dist_col in df.columns \
             else np.zeros(n_post)
    r_max  = float(np.nanmax(r_post)) * 1.20 if len(r_post) > 0 and np.nanmax(r_post) > 0 \
             else 50.0

    # ── Colors for panels B & C (inside vs outside envelope) ─────────────────
    inside_post = inside_col[post_mask].values
    c_post = np.where(inside_post, "#e53935", "#aaaaaa")   # red inside, gray outside

    # ── FIGURE ────────────────────────────────────────────────────────────────
    norm_depth = Normalize(vmin=0, vmax=DEPTH_MAX)
    cmap_d     = cm.plasma_r

    fig = plt.figure(figsize=(16, 14), facecolor="white")

    d_str = (f"D$_h$ = {D_chosen:.2f} m²/s" if valid_D
             else ("D = 100 m²/s (slider max — discarded)" if D_chosen and D_chosen >= 98
                   else "Null result — Regime II"))
    fig.suptitle(
        f"{region.replace('_', ' ').title()}  [sub{sub_n}]   "
        f"M≥{mc}  |  N={len(df)} events  ({n_post} post-origin)\n"
        f"Rain: {rain_date.strftime('%Y-%m-%d')}     "
        f"Origin: {origin_t.strftime('%Y-%m-%d')}  "
        + (f"({origin_lat:.3f}°, {origin_lon:.3f}°, {origin_depth:.1f} km)   " if origin_depth else "")
        + d_str,
        fontsize=11, fontweight="bold"
    )

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.40, wspace=0.30,
                           top=0.88, bottom=0.06, left=0.08, right=0.95)
    ax_td  = fig.add_subplot(gs[0, 0])
    ax_rt  = fig.add_subplot(gs[0, 1])
    ax_r2t = fig.add_subplot(gs[1, 0])
    ax_map = fig.add_subplot(gs[1, 1])

    # ── Panel A: Depth — Time ─────────────────────────────────────────────────
    ax = ax_td
    t_all_min = float(df[t_col].min()) * 1.05 if pre_mask.any() else -5
    t_all_max = float(df[t_col].max()) * 1.05

    if pre_mask.any():
        ax.scatter(df.loc[pre_mask, t_col], df.loc[pre_mask, "depth_clip"],
                   s=_sz(df.loc[pre_mask, "mag"], mc) * 0.4,
                   c="lightgray", alpha=0.55, zorder=2, label="Pre-origin")

    sc_td = ax.scatter(df.loc[post_mask, t_col], df.loc[post_mask, "depth_clip"],
                       s=_sz(df.loc[post_mask, "mag"], mc),
                       c=df.loc[post_mask, "depth_clip"],
                       cmap=cmap_d, norm=norm_depth,
                       alpha=0.80, linewidths=0.2, edgecolors="k", zorder=3,
                       label="Post-origin")
    fig.colorbar(sc_td, ax=ax, label="Depth (km)", shrink=0.75, pad=0.02)

    ax.axvline(0, color="#c62828", lw=2.0, ls="-",  zorder=5,
               label=f"Origin  t=0")
    ax.axvline(rain_from_origin, color="steelblue", lw=2.0, ls="--", zorder=4,
               label=f"Rain  {rain_date.strftime('%b %d, %Y')}")

    ax.set_xlim(t_all_min, t_all_max)
    ax.set_ylim(DEPTH_MAX, 0)
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel("Depth (km)", fontsize=9)
    ax.set_title("(A)  Depth — Time", fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    # ── Panel B: r — t  (post-origin) ────────────────────────────────────────
    ax = ax_rt

    ax.scatter(df.loc[post_mask, t_col], r_post,
               s=_sz(df.loc[post_mask, "mag"], mc),
               c=c_post, alpha=0.75, linewidths=0.2, edgecolors="k", zorder=4)

    # Reference D curves (faint)
    for D_r in D_REF:
        r_ref = np.clip(darcy_r(D_r, t_arr), 0, r_max)
        ax.plot(t_arr, r_ref, lw=0.6, color="#b0cce8", ls="--", zorder=1)
        if r_ref[-1] < r_max * 0.98:
            ax.text(t_arr[-1], r_ref[-1], f"{D_r}", fontsize=5.5,
                    color="#4a80b0", ha="left", va="center", clip_on=True)

    # Fitted D_chosen envelope (bold red)
    if valid_D:
        r_fit = np.clip(darcy_r(D_chosen, t_arr), 0, r_max)
        ax.plot(t_arr, r_fit, lw=2.5, color="#c62828", zorder=6,
                label=f"D$_h$ = {D_chosen:.2f} m²/s")
        n_in = int(inside_col[post_mask].sum())
        ax.scatter([], [], c="#e53935", s=40, label=f"Inside ({n_in}/{n_post})")
        ax.scatter([], [], c="#aaaaaa", s=40, label=f"Outside ({n_post-n_in}/{n_post})")
        ax.legend(fontsize=7, loc="upper left")

    ax.set_xlim(0, t_max)
    ax.set_ylim(0, r_max)
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel(f"r  [{dist_type}]  (km)", fontsize=9)
    ax.set_title(f"(B)  r — t   [{dist_type}]", fontweight="bold")
    ax.grid(alpha=0.3)

    # ── Panel C: r² — t  (post-origin) ───────────────────────────────────────
    ax = ax_r2t
    r2_post = r_post ** 2
    r2_max  = float(np.nanmax(r2_post)) * 1.20 if len(r2_post) > 0 and np.nanmax(r2_post) > 0 \
              else 100.0

    ax.scatter(df.loc[post_mask, t_col], r2_post,
               s=_sz(df.loc[post_mask, "mag"], mc),
               c=c_post, alpha=0.75, linewidths=0.2, edgecolors="k", zorder=4)

    for D_r in D_REF:
        r_ref = darcy_r(D_r, t_arr)
        ax.plot(t_arr, np.clip(r_ref**2, 0, r2_max),
                lw=0.6, color="#b0cce8", ls="--", zorder=1)

    if valid_D:
        r_fit = darcy_r(D_chosen, t_arr)
        ax.plot(t_arr, np.clip(r_fit**2, 0, r2_max),
                lw=2.5, color="#c62828", zorder=6,
                label=f"D$_h$ = {D_chosen:.2f} m²/s  (slope = 4D)")
        ax.legend(fontsize=7, loc="upper left")

    ax.set_xlim(0, t_max)
    ax.set_ylim(0, r2_max)
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel(f"r²  [{dist_type}]  (km²)", fontsize=9)
    ax.set_title("(C)  r² — t   (linear = pure Darcy)", fontweight="bold")
    ax.grid(alpha=0.3)

    # ── Panel D: Map ──────────────────────────────────────────────────────────
    ax = ax_map

    if pre_mask.any():
        ax.scatter(df.loc[pre_mask, "lon"], df.loc[pre_mask, "lat"],
                   s=_sz(df.loc[pre_mask, "mag"], mc) * 0.4,
                   c="lightgray", alpha=0.4, zorder=2)

    sc_map = ax.scatter(df.loc[post_mask, "lon"], df.loc[post_mask, "lat"],
                        s=_sz(df.loc[post_mask, "mag"], mc),
                        c=df.loc[post_mask, "depth_clip"],
                        cmap=cmap_d, norm=norm_depth,
                        alpha=0.80, linewidths=0.2, edgecolors="k", zorder=3)
    fig.colorbar(sc_map, ax=ax, label="Depth (km)", shrink=0.75, pad=0.02)

    if origin_depth is not None:
        ax.plot(origin_lon, origin_lat, "r*", ms=16, zorder=6,
                markeredgecolor="#7f0000", markeredgewidth=1.2, label="Origin")

    # Magnitude legend
    for mag_ex in [mc, mc + 1, mc + 2]:
        s_ex = float(_sz(pd.Series([mag_ex]), mc).iloc[0])
        ax.scatter([], [], s=s_ex, c="gray", alpha=0.7, label=f"M {mag_ex:.1f}")

    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.set_title("(D)  Subcatalog Map\n(post-origin color = depth)", fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, f"{region}_spacetime.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating S1–S8 from subcatalogs (script 20)...\n")
    for region in S_FIGURES:
        print(f"Processing: {region}")
        make_figure(region)
    print(f"\nDone. All figures in:\n  {OUT_DIR}")
    print(f"\nTo deploy to Overleaf:")
    print(f"  cp {OUT_DIR}/*.png /Users/roberto/AGU_PAPER/figures_SI/")
