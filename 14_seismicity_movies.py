#!/usr/bin/env python3
"""
14_seismicity_movies.py
=======================
Animated seismicity maps for visual inspection of spatio-temporal migration.

Each frame = one day (cumulative events up to that day).
  - Color  = depth (shallow=yellow, deep=purple via plasma_r)
  - Size   = magnitude
  - Rain date marked with red vertical line on the cumulative-count strip

Output: figures/movies/{region}.mp4

Usage:
    python 14_seismicity_movies.py                  # all regions
    python 14_seismicity_movies.py --region noto_japan
    python 14_seismicity_movies.py --region noto_japan los_cabos_mx
"""

import argparse
import os
import sys

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.dirname(__file__))
from config_regions import REGIONS

# ── global parameters ─────────────────────────────────────────────────────────
PRE_DAYS  = 30    # days before rain_date to include
POST_DAYS = 90    # days after  rain_date to include
FPS       = 8     # frames per second in the output MP4
DPI       = 120
DEPTH_MAX = 35    # km — colorscale ceiling (events deeper are clipped visually)

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures", "movies")
os.makedirs(OUT_DIR, exist_ok=True)

CMAP = cm.plasma_r   # shallow → bright yellow, deep → dark purple


def _size(mag, mc):
    """Scatter point area proportional to magnitude above completeness."""
    s = 15 * 10 ** (0.5 * (mag - mc))
    return np.clip(s, 6, 250)


def _deg_span(r_km, clat):
    dlat = r_km / 111.32
    dlon = r_km / (111.32 * np.cos(np.radians(clat)))
    return dlat, dlon


def make_movie(region, cfg, verbose=True):
    csv_path = os.path.join(os.path.dirname(__file__), "data", f"{region}.csv")
    if not os.path.exists(csv_path):
        print(f"  [skip] CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path, parse_dates=["time"])
    rain_date = pd.Timestamp(cfg["rain"])
    t0 = rain_date - pd.Timedelta(days=PRE_DAYS)
    t1 = rain_date + pd.Timedelta(days=POST_DAYS)

    df = df[(df["time"] >= t0) & (df["time"] <= t1)].copy()
    if len(df) < 3:
        print(f"  [skip] only {len(df)} events in window for {region}")
        return

    df["depth_clip"] = df["depth"].clip(0, DEPTH_MAX)
    df["day"] = ((df["time"] - t0) / pd.Timedelta(days=1)).astype(int)

    clat, clon = cfg["lat"], cfg["lon"]
    dlat, dlon = _deg_span(cfg["r"], clat)
    mc = cfg.get("mc", 1.5)
    total_days = PRE_DAYS + POST_DAYS
    rain_day   = PRE_DAYS   # index of rain date in the animation

    norm = Normalize(vmin=0, vmax=DEPTH_MAX)

    # pre-compute cumulative count per day for the bottom strip
    cum_counts = np.array([int((df["day"] <= d).sum()) for d in range(total_days + 1)])
    max_count  = max(cum_counts.max(), 1)

    # ── figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8, 8.5), facecolor="white")
    gs  = fig.add_gridspec(2, 1, height_ratios=[6, 1.2], hspace=0.4)
    ax_map  = fig.add_subplot(gs[0])
    ax_time = fig.add_subplot(gs[1])

    sm   = cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_map, fraction=0.028, pad=0.02)
    cbar.set_label("Depth (km)", fontsize=9)

    days_axis = np.arange(total_days + 1)

    def draw_frame(day):
        ax_map.clear()
        ax_time.clear()

        subset = df[df["day"] <= day]
        current_date = (t0 + pd.Timedelta(days=int(day))).strftime("%Y-%m-%d")

        # ── MAP panel ─────────────────────────────────────────────────────────
        ax_map.set_facecolor("#ddeeff")
        ax_map.set_xlim(clon - dlon, clon + dlon)
        ax_map.set_ylim(clat - dlat, clat + dlat)
        ax_map.set_aspect("equal", adjustable="box")
        ax_map.grid(True, color="white", lw=0.6, alpha=0.7)
        ax_map.set_xlabel("Longitude", fontsize=9)
        ax_map.set_ylabel("Latitude", fontsize=9)

        if not subset.empty:
            ax_map.scatter(
                subset["lon"], subset["lat"],
                c=subset["depth_clip"], cmap=CMAP, norm=norm,
                s=_size(subset["mag"], mc),
                alpha=0.8, linewidths=0.3, edgecolors="k", zorder=4
            )

        # rain center marker
        ax_map.plot(clon, clat, "r*", ms=13, zorder=6,
                    label="Rain epicenter", markeredgecolor="darkred", markeredgewidth=0.5)
        ax_map.legend(fontsize=8, loc="upper right", framealpha=0.8)

        phase = "PRE-RAIN" if day < rain_day else f"+{day - rain_day}d post-rain"
        ax_map.set_title(
            f"{region.replace('_', ' ').title()}  |  {current_date}  ({phase})  |  N={len(subset)}",
            fontsize=10, fontweight="bold", pad=6
        )

        # ── TIME strip ────────────────────────────────────────────────────────
        ax_time.fill_between(days_axis, cum_counts, color="steelblue", alpha=0.5, step="mid")
        ax_time.axvline(rain_day, color="red", lw=2, label=f"Rain {rain_date.strftime('%Y-%m-%d')}", zorder=4)
        ax_time.axvline(day, color="black", lw=1.5, ls="--", zorder=5)
        ax_time.set_xlim(0, total_days)
        ax_time.set_ylim(0, max_count * 1.12)
        ax_time.set_xlabel(f"Days since {t0.strftime('%Y-%m-%d')}", fontsize=8)
        ax_time.set_ylabel("Cum. N", fontsize=8)
        ax_time.legend(fontsize=7, loc="upper left", framealpha=0.8)
        ax_time.tick_params(labelsize=7)

        # x-tick labels as calendar dates every 15 days
        tick_days = np.arange(0, total_days + 1, 15)
        ax_time.set_xticks(tick_days)
        ax_time.set_xticklabels(
            [(t0 + pd.Timedelta(days=int(d))).strftime("%m-%d") for d in tick_days],
            rotation=30, ha="right", fontsize=6
        )

    ani = animation.FuncAnimation(
        fig, draw_frame,
        frames=total_days + 1,
        interval=1000 // FPS,
        blit=False
    )

    out_path = os.path.join(OUT_DIR, f"{region}.mp4")
    writer = animation.FFMpegWriter(
        fps=FPS, bitrate=2000,
        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"]
    )
    ani.save(out_path, writer=writer, dpi=DPI,
             savefig_kwargs={"facecolor": "white"})
    plt.close(fig)
    if verbose:
        print(f"  [ok]  {out_path}  ({len(df)} events)")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate seismicity animation per region.")
    parser.add_argument("--region", nargs="*", default=None,
                        help="One or more region keys (default: all)")
    args = parser.parse_args()

    targets = args.region if args.region else list(REGIONS.keys())
    missing = [r for r in targets if r not in REGIONS]
    if missing:
        print(f"Unknown region(s): {missing}")
        print(f"Available: {list(REGIONS.keys())}")
        sys.exit(1)

    print(f"Generating {len(targets)} movie(s) → {OUT_DIR}\n")
    for i, reg in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] {reg} ...", end=" ", flush=True)
        make_movie(reg, REGIONS[reg])

    print("\nDone.")
