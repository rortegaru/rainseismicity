#!/usr/bin/env python3
"""
15_subregion_movie.py
=====================
Interactive subregion selector → restricted seismicity movie.

STEP 1 — Interactive window
  • Shows ALL events (full region) as scatter, color = depth.
  • Natural Earth background (coast, borders, land/ocean) via cartopy.
  • Drag a rectangle with the mouse to define the subregion.
  • Press ENTER (or close the window) to confirm and generate the movie.

STEP 2 — Movie (restricted to the selected box)
  • Top panel  : lat/lon map, color = depth, size = mag.
  • Bottom panel: depth-vs-time cross section (vertical migration diagnostic).

Output: figures/movies/{region}_sub{N}.mp4

Usage:
    python 15_subregion_movie.py --region noto_japan
    python 15_subregion_movie.py --region los_cabos_mx --pre 20 --post 60
"""

import argparse
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("MacOSX")          # interactive on macOS; change to TkAgg if needed

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import RectangleSelector
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, os.path.dirname(__file__))
from config_regions import REGIONS

# ── defaults ──────────────────────────────────────────────────────────────────
PRE_DAYS_DEFAULT  = 30
POST_DAYS_DEFAULT = 90
FPS       = 8
DPI       = 130
DEPTH_MAX = 35     # km — colorscale ceiling
CMAP      = cm.plasma_r   # shallow → bright yellow, deep → dark purple

OUT_DIR     = os.path.join(os.path.dirname(__file__), "figures", "movies")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
BOXES_FILE  = os.path.join(RESULTS_DIR, "subregion_boxes.json")
os.makedirs(OUT_DIR,     exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────
def _norm():
    return Normalize(vmin=0, vmax=DEPTH_MAX)

def _size(mag, mc):
    s = 18 * 10 ** (0.5 * (mag - mc))
    return np.clip(s, 6, 280)

def _next_sub_n(region):
    n = 1
    while os.path.exists(os.path.join(OUT_DIR, f"{region}_sub{n}.mp4")):
        n += 1
    return n

def _next_sub_path(region, force_n=None):
    n = force_n if force_n is not None else _next_sub_n(region)
    return os.path.join(OUT_DIR, f"{region}_sub{n}.mp4"), n

def save_box_to_json(region, sub_n, box, pre_days, post_days, max_depth=None):
    boxes = {}
    if os.path.exists(BOXES_FILE):
        with open(BOXES_FILE) as f:
            boxes = json.load(f)
    entry = {**box, "pre_days": pre_days, "post_days": post_days}
    if max_depth is not None:
        entry["max_depth"] = max_depth
    boxes[f"{region}_sub{sub_n}"] = entry
    with open(BOXES_FILE, "w") as f:
        json.dump(boxes, f, indent=2)
    return sub_n

def _cartopy_background(ax, extent):
    """Add Natural Earth features to a cartopy GeoAxes."""
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,       facecolor="#e8e0d0", zorder=0)
    ax.add_feature(cfeature.OCEAN,      facecolor="#c9dff0", zorder=0)
    ax.add_feature(cfeature.COASTLINE,  linewidth=0.7, edgecolor="#555", zorder=1)
    ax.add_feature(cfeature.BORDERS,    linewidth=0.5, edgecolor="#888", linestyle=":", zorder=1)
    ax.add_feature(cfeature.RIVERS,     linewidth=0.4, edgecolor="#88bbdd", zorder=1)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                      alpha=0.5, linestyle="--")
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — interactive selector
# ══════════════════════════════════════════════════════════════════════════════
def interactive_select(region, cfg, df, pre_days, post_days):
    """
    Show full seismicity map. User drags a rectangle.
    Returns (lon_min, lon_max, lat_min, lat_max) of selected box.
    """
    rain_date = pd.Timestamp(cfg["rain"])
    mc        = cfg.get("mc", 1.5)
    clat, clon = cfg["lat"], cfg["lon"]

    # full extent from config radius
    r_km  = cfg["r"]
    dlat  = r_km / 111.32
    dlon  = r_km / (111.32 * np.cos(np.radians(clat)))
    full_extent = [clon - dlon, clon + dlon, clat - dlat, clat + dlat]

    norm = _norm()

    fig = plt.figure(figsize=(10, 9))
    fig.suptitle(
        f"{region.replace('_',' ').title()}  —  ALL events (N={len(df)})  "
        f"|  rain: {rain_date.strftime('%Y-%m-%d')}\n"
        "Drag a rectangle to select subregion, then press ENTER",
        fontsize=11, fontweight="bold"
    )

    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    _cartopy_background(ax, full_extent)

    sc = ax.scatter(
        df["lon"], df["lat"],
        c=df["depth"].clip(0, DEPTH_MAX), cmap=CMAP, norm=norm,
        s=_size(df["mag"], mc),
        alpha=0.7, linewidths=0.3, edgecolors="k",
        transform=ccrs.PlateCarree(), zorder=5
    )

    # rain reference marker
    ax.plot(clon, clat, "r*", ms=14, transform=ccrs.PlateCarree(),
            zorder=8, markeredgecolor="darkred", label="Rain ref.")
    ax.legend(fontsize=9, loc="upper right")

    cbar = fig.colorbar(sc, ax=ax, fraction=0.028, pad=0.06)
    cbar.set_label("Depth (km)", fontsize=9)

    # info panel — depth histogram on the side
    ax_info = fig.add_axes([0.01, 0.12, 0.06, 0.3])
    ax_info.barh(
        np.arange(0, DEPTH_MAX, 2),
        np.histogram(df["depth"].clip(0, DEPTH_MAX), bins=np.arange(0, DEPTH_MAX+2, 2))[0],
        height=1.8, color=CMAP(norm(np.arange(1, DEPTH_MAX+1, 2))), edgecolor="none"
    )
    ax_info.invert_yaxis()
    ax_info.set_xlabel("N", fontsize=7)
    ax_info.set_ylabel("Depth (km)", fontsize=7)
    ax_info.tick_params(labelsize=6)
    ax_info.set_title("Depth\nhist.", fontsize=7)

    # storage for selected coords
    selected = {}
    rect_patch = [None]

    def on_select(eclick, erelease):
        x0, y0 = min(eclick.xdata, erelease.xdata), min(eclick.ydata, erelease.ydata)
        x1, y1 = max(eclick.xdata, erelease.xdata), max(eclick.ydata, erelease.ydata)
        selected["lon_min"] = x0
        selected["lon_max"] = x1
        selected["lat_min"] = y0
        selected["lat_max"] = y1
        n_in = len(df[
            (df["lon"] >= x0) & (df["lon"] <= x1) &
            (df["lat"] >= y0) & (df["lat"] <= y1)
        ])
        fig.suptitle(
            f"{region.replace('_',' ').title()}  —  "
            f"Box: [{x0:.3f}–{x1:.3f}°E, {y0:.3f}–{y1:.3f}°N]  "
            f"N_box={n_in}\n"
            "Press ENTER to generate movie  (drag again to re-select)",
            fontsize=10, fontweight="bold", color="navy"
        )
        fig.canvas.draw_idle()

    rs = RectangleSelector(
        ax, on_select,
        useblit=False,
        button=[1],
        minspanx=5e-4, minspany=5e-4,
        spancoords="data",
        interactive=True,
        props=dict(facecolor="cyan", edgecolor="blue", alpha=0.25, fill=True)
    )

    def on_key(event):
        if event.key == "enter" and selected:
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    if not selected:
        print("  No selection made — using full extent.")
        selected["lon_min"] = full_extent[0]
        selected["lon_max"] = full_extent[1]
        selected["lat_min"] = full_extent[2]
        selected["lat_max"] = full_extent[3]

    return selected


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — restricted movie
# ══════════════════════════════════════════════════════════════════════════════
def make_subregion_movie(region, cfg, df, box, pre_days, post_days, force_sub_n=None):
    """
    Generate MP4 for the spatial subregion defined by `box`.
    Two panels per frame:
      - Top : lat/lon map (color=depth, Natural Earth background)
      - Bottom : depth-vs-time cross section (vertical migration)
    """
    rain_date = pd.Timestamp(cfg["rain"])
    t0  = rain_date - pd.Timedelta(days=pre_days)
    mc  = cfg.get("mc", 1.5)
    total_days = pre_days + post_days
    rain_day   = pre_days

    # spatial filter
    sub = df[
        (df["lon"] >= box["lon_min"]) & (df["lon"] <= box["lon_max"]) &
        (df["lat"] >= box["lat_min"]) & (df["lat"] <= box["lat_max"])
    ].copy()

    if len(sub) < 3:
        print(f"  [skip] only {len(sub)} events in box — select a larger area.")
        return

    sub["depth_clip"] = sub["depth"].clip(0, DEPTH_MAX)
    sub["day"] = ((sub["time"] - t0) / pd.Timedelta(days=1)).astype(int)

    norm      = _norm()
    norm_time = Normalize(vmin=0, vmax=total_days)
    cmap_time = cm.viridis

    extent = [box["lon_min"], box["lon_max"], box["lat_min"], box["lat_max"]]
    cum_counts = np.array([int((sub["day"] <= d).sum()) for d in range(total_days + 1)])
    max_count  = max(cum_counts.max(), 1)
    days_axis  = np.arange(total_days + 1)

    out_path, sub_n = _next_sub_path(region, force_n=force_sub_n)
    print(f"  Subregion {sub_n}: {len(sub)} events  →  {out_path}")

    fig = plt.figure(figsize=(9, 10), facecolor="white")
    gs  = fig.add_gridspec(3, 1, height_ratios=[5, 2.5, 1], hspace=0.45)
    ax_map   = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
    ax_depth = fig.add_subplot(gs[1])
    ax_time  = fig.add_subplot(gs[2])

    # static colorbar for depth
    sm_d = cm.ScalarMappable(cmap=CMAP,     norm=norm)
    sm_d.set_array([])
    cbar_d = fig.colorbar(sm_d, ax=ax_map, fraction=0.028, pad=0.06)
    cbar_d.set_label("Depth (km)", fontsize=9)

    def draw_frame(day):
        ax_map.clear()
        ax_depth.clear()
        ax_time.clear()

        subset   = sub[sub["day"] <= day]
        cur_date = (t0 + pd.Timedelta(days=int(day))).strftime("%Y-%m-%d")
        phase    = "PRE" if day < rain_day else f"+{day-rain_day}d"

        # ── MAP ───────────────────────────────────────────────────────────────
        _cartopy_background(ax_map, extent)
        if not subset.empty:
            ax_map.scatter(
                subset["lon"], subset["lat"],
                c=subset["depth_clip"], cmap=CMAP, norm=norm,
                s=_size(subset["mag"], mc),
                alpha=0.8, linewidths=0.3, edgecolors="k",
                transform=ccrs.PlateCarree(), zorder=5
            )
        ax_map.set_title(
            f"{region.replace('_',' ').title()}  sub{sub_n}  |  "
            f"{cur_date}  ({phase})  |  N={len(subset)}",
            fontsize=10, fontweight="bold", pad=4
        )

        # ── DEPTH-TIME cross section ──────────────────────────────────────────
        # All events: faint gray background
        ax_depth.scatter(sub["day"], sub["depth_clip"],
                         c="lightgray", s=6, alpha=0.4, zorder=2)
        # Events up to current day: colored by depth
        if not subset.empty:
            ax_depth.scatter(
                subset["day"], subset["depth_clip"],
                c=subset["depth_clip"], cmap=CMAP, norm=norm,
                s=_size(subset["mag"], mc) * 0.4,
                alpha=0.85, linewidths=0.2, edgecolors="k", zorder=4
            )
        ax_depth.axvline(rain_day, color="red",   lw=2,   zorder=5,
                         label=f"Rain {rain_date.strftime('%Y-%m-%d')}")
        ax_depth.axvline(day,      color="black", lw=1.5, ls="--", zorder=6)
        ax_depth.invert_yaxis()                  # depth 0 at top
        ax_depth.set_xlim(0, total_days)
        ax_depth.set_ylim(DEPTH_MAX, 0)
        ax_depth.set_ylabel("Depth (km)", fontsize=9)
        ax_depth.set_xlabel(f"Days since {t0.strftime('%Y-%m-%d')}", fontsize=8)
        ax_depth.set_title("Depth–time cross section", fontsize=9)
        ax_depth.legend(fontsize=7, loc="upper left", framealpha=0.8)
        ax_depth.grid(True, alpha=0.3)
        ax_depth.tick_params(labelsize=7)

        # ── TIME strip (cumulative N) ─────────────────────────────────────────
        ax_time.fill_between(days_axis, cum_counts,
                             color="steelblue", alpha=0.5, step="mid")
        ax_time.axvline(rain_day, color="red",   lw=2,   zorder=4)
        ax_time.axvline(day,      color="black", lw=1.5, ls="--", zorder=5)
        ax_time.set_xlim(0, total_days)
        ax_time.set_ylim(0, max_count * 1.15)
        ax_time.set_ylabel("Cum. N", fontsize=8)
        ax_time.tick_params(labelsize=7)
        ax_time.set_xticks(np.arange(0, total_days + 1, 15))
        ax_time.set_xticklabels(
            [(t0 + pd.Timedelta(days=int(d))).strftime("%m-%d")
             for d in np.arange(0, total_days + 1, 15)],
            rotation=30, ha="right", fontsize=6
        )
        ax_time.grid(True, alpha=0.3)

    ani = animation.FuncAnimation(
        fig, draw_frame,
        frames=total_days + 1,
        interval=1000 // FPS,
        blit=False
    )

    writer = animation.FFMpegWriter(
        fps=FPS, bitrate=2200,
        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"]
    )
    ani.save(out_path, writer=writer, dpi=DPI,
             savefig_kwargs={"facecolor": "white"})
    plt.close(fig)
    print(f"  [ok] {out_path}")
    return sub_n


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region",        required=True)
    parser.add_argument("--pre",           type=int, default=PRE_DAYS_DEFAULT)
    parser.add_argument("--post",          type=int, default=POST_DAYS_DEFAULT)
    parser.add_argument("--generate-only", action="store_true",
                        help="Headless movie generation (reads box from JSON)")
    parser.add_argument("--sub",           type=int, default=None,
                        help="Subregion index (used with --generate-only)")
    parser.add_argument("--max_depth",     type=float, default=None,
                        help="Exclude events deeper than this (km)")
    args = parser.parse_args()

    if args.region not in REGIONS:
        print(f"Unknown region '{args.region}'. Available: {list(REGIONS.keys())}")
        sys.exit(1)

    cfg      = REGIONS[args.region]
    csv_path = os.path.join(os.path.dirname(__file__), "data", f"{args.region}.csv")
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path, parse_dates=["time"])
    rain_date = pd.Timestamp(cfg["rain"])
    t0 = rain_date - pd.Timedelta(days=args.pre)
    t1 = rain_date + pd.Timedelta(days=args.post)
    df = df[(df["time"] >= t0) & (df["time"] <= t1)].copy()
    if args.max_depth is not None:
        df = df[df["depth"] <= args.max_depth].copy()

    # ── HEADLESS MODE: just generate movie (spawned by interactive mode) ──────
    if args.generate_only:
        if not os.path.exists(BOXES_FILE):
            print("No boxes file found. Run interactively first.")
            sys.exit(1)
        with open(BOXES_FILE) as f:
            boxes = json.load(f)
        key = f"{args.region}_sub{args.sub}"
        if key not in boxes:
            print(f"Box '{key}' not found in {BOXES_FILE}")
            sys.exit(1)
        box       = boxes[key]
        pre_days  = box.get("pre_days",  args.pre)
        post_days = box.get("post_days", args.post)
        md        = box.get("max_depth", None)
        df2 = pd.read_csv(csv_path, parse_dates=["time"])
        rain_d = pd.Timestamp(cfg["rain"])
        df2 = df2[(df2["time"] >= rain_d - pd.Timedelta(days=pre_days)) &
                  (df2["time"] <= rain_d + pd.Timedelta(days=post_days))].copy()
        if md is not None:
            df2 = df2[df2["depth"] <= md].copy()
        make_subregion_movie(args.region, cfg, df2, box, pre_days, post_days,
                             force_sub_n=args.sub)
        sys.exit(0)

    # ── INTERACTIVE MODE ──────────────────────────────────────────────────────
    print(f"Region: {args.region}  |  {len(df)} events in window  "
          f"({t0.date()} → {t1.date()})"
          + (f"  depth≤{args.max_depth}km" if args.max_depth else ""))

    box = interactive_select(args.region, cfg, df, args.pre, args.post)
    print(f"\nSelected box: lon [{box['lon_min']:.4f}, {box['lon_max']:.4f}]  "
          f"lat [{box['lat_min']:.4f}, {box['lat_max']:.4f}]")

    sub_n = _next_sub_n(args.region)
    save_box_to_json(args.region, sub_n, box, args.pre, args.post, args.max_depth)
    print(f"  Box saved → sub{sub_n}")

    # Spawn headless movie generation in background
    env = {**os.environ, "MPLBACKEND": "Agg"}
    movie_proc = subprocess.Popen(
        [sys.executable, __file__,
         "--region", args.region,
         "--pre",  str(args.pre),
         "--post", str(args.post),
         "--generate-only",
         "--sub",  str(sub_n)],
        env=env
    )
    print(f"  Movie rendering in background (PID {movie_proc.pid}) ...")

    # Immediately launch origin picker in this process
    print(f"  Opening origin picker for sub{sub_n} ...\n")
    picker_path = os.path.join(os.path.dirname(__file__), "16_pick_origin.py")
    os.execv(sys.executable, [sys.executable, picker_path,
                               "--region", args.region,
                               "--sub", str(sub_n)])
