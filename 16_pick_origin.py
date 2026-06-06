#!/usr/bin/env python3
"""
16_pick_origin.py
=================
Interactive tool to select the diffusion origin (event zero, t₀, X₀, Y₀, Z₀).

Two panels:
  Left  : lat/lon map — post-rain events colored by depth, pre-rain in gray.
  Right : depth–time cross section — post-rain events colored by time progression.

Click any event in either panel → both panels highlight it.
Choose geometry (point / fault) with radio buttons.
Press ENTER to confirm and save.

Saved to: results/darcy_origins.csv
Then launches 17_pick_diffusivity.py automatically.

Usage:
    python 16_pick_origin.py --region noto_japan --sub 1
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("MacOSX")

import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.widgets import RadioButtons

import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, os.path.dirname(__file__))
from config_regions import REGIONS

RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
SUBCATS_DIR  = os.path.join(RESULTS_DIR, "subcatalogs")
BOXES_FILE   = os.path.join(RESULTS_DIR, "subregion_boxes.json")
ORIGINS_CSV  = os.path.join(RESULTS_DIR, "darcy_origins.csv")
DEPTH_MAX    = 35


def _load_box(region, sub_n):
    if not os.path.exists(BOXES_FILE):
        return None
    with open(BOXES_FILE) as f:
        return json.load(f).get(f"{region}_sub{sub_n}")


def _nearest_on_map(df, lon_c, lat_c):
    cos_lat = np.cos(np.radians(df["lat"].mean()))
    dx = (df["lon"] - lon_c) * cos_lat * 111.32
    dy = (df["lat"] - lat_c) * 111.32
    dist = np.sqrt(dx**2 + dy**2)
    idx = dist.values.argmin()
    return df.iloc[idx], dist.iloc[idx]


def _nearest_on_dt(df, day_c, depth_c, day_range, depth_range):
    nd = (df["day"] - day_c) / max(day_range, 1)
    nz = (df["depth_clip"] - depth_c) / max(depth_range, 1)
    dist = np.sqrt(nd**2 + nz**2)
    idx = dist.values.argmin()
    return df.iloc[idx], dist.iloc[idx]


def run(region, sub_n=1):
    cfg       = REGIONS[region]
    rain_date = pd.Timestamp(cfg["rain"])
    mc        = cfg.get("mc", 1.5)

    box = _load_box(region, sub_n)
    if box is None:
        r_km  = cfg["r"]
        clat, clon = cfg["lat"], cfg["lon"]
        dlat  = r_km / 111.32
        dlon  = r_km / (111.32 * np.cos(np.radians(clat)))
        box   = dict(lon_min=clon-dlon, lon_max=clon+dlon,
                     lat_min=clat-dlat, lat_max=clat+dlat,
                     pre_days=30, post_days=90)

    pre_days  = box.get("pre_days",  30)
    post_days = box.get("post_days", 90)
    t0 = rain_date - pd.Timedelta(days=pre_days)
    t1 = rain_date + pd.Timedelta(days=post_days)

    csv = os.path.join(os.path.dirname(__file__), "data", f"{region}.csv")
    df  = pd.read_csv(csv, parse_dates=["time"])
    df  = df[(df["time"] >= t0) & (df["time"] <= t1)].copy()
    df  = df[(df["lon"] >= box["lon_min"]) & (df["lon"] <= box["lon_max"]) &
             (df["lat"] >= box["lat_min"]) & (df["lat"] <= box["lat_max"])].copy()
    max_depth = box.get("max_depth", None)
    if max_depth is not None:
        df = df[df["depth"] <= max_depth].copy()
    df["depth_clip"] = df["depth"].clip(0, DEPTH_MAX)
    df["day"]  = ((df["time"] - t0) / pd.Timedelta(days=1)).astype(int)
    df = df.reset_index(drop=True)

    if len(df) < 2:
        print(f"  Too few events ({len(df)}) in box. Exiting.")
        return None

    rain_day   = pre_days
    total_days = pre_days + post_days
    extent     = [box["lon_min"], box["lon_max"], box["lat_min"], box["lat_max"]]

    post_df = df[df["day"] >= rain_day].copy()
    pre_df  = df[df["day"] <  rain_day].copy()

    norm_depth = Normalize(vmin=0, vmax=DEPTH_MAX)
    norm_time  = Normalize(vmin=rain_day, vmax=total_days)
    cmap_depth = cm.plasma_r
    cmap_time  = cm.viridis

    def _sz(mag):
        return np.clip(18 * 10 ** (0.5 * (mag - mc)), 6, 250)

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8), facecolor="white")
    fig.suptitle(
        f"{region.replace('_',' ').title()}  sub{sub_n}  "
        f"|  N={len(df)} events  (pre={len(pre_df)}, post={len(post_df)})\n"
        "Click any event to select origin (t₀, X₀, Y₀, Z₀)  —  ENTER to confirm",
        fontsize=11, fontweight="bold"
    )

    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.5, 1],
                           wspace=0.35, left=0.05, right=0.97,
                           top=0.88, bottom=0.12)
    ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
    ax_dt  = fig.add_subplot(gs[1])

    # ── MAP ───────────────────────────────────────────────────────────────────
    ax_map.set_extent(extent, crs=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.LAND,      facecolor="#e8e0d0", zorder=0)
    ax_map.add_feature(cfeature.OCEAN,     facecolor="#c9dff0", zorder=0)
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="#555", zorder=1)
    ax_map.add_feature(cfeature.BORDERS,   linewidth=0.5, edgecolor="#aaa",
                       linestyle=":", zorder=1)
    gl = ax_map.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                          alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    if not pre_df.empty:
        ax_map.scatter(pre_df["lon"], pre_df["lat"],
                       c="lightgray", s=_sz(pre_df["mag"]) * 0.4,
                       alpha=0.4, transform=ccrs.PlateCarree(), zorder=3)

    if not post_df.empty:
        ax_map.scatter(
            post_df["lon"], post_df["lat"],
            c=post_df["depth_clip"], cmap=cmap_depth, norm=norm_depth,
            s=_sz(post_df["mag"]),
            alpha=0.8, linewidths=0.3, edgecolors="k",
            transform=ccrs.PlateCarree(), zorder=4
        )

    sm = cm.ScalarMappable(cmap=cmap_depth, norm=norm_depth)
    sm.set_array([])
    fig.colorbar(sm, ax=ax_map, fraction=0.028, pad=0.08, label="Depth (km)")
    ax_map.set_title("Map  (post-rain: color = depth)", fontsize=9, pad=4)

    # ── DEPTH–TIME ────────────────────────────────────────────────────────────
    if not pre_df.empty:
        ax_dt.scatter(pre_df["day"], pre_df["depth_clip"],
                      c="lightgray", s=_sz(pre_df["mag"]) * 0.35,
                      alpha=0.4, zorder=2)

    if not post_df.empty:
        sc_dt = ax_dt.scatter(
            post_df["day"], post_df["depth_clip"],
            c=post_df["day"], cmap=cmap_time, norm=norm_time,
            s=_sz(post_df["mag"]) * 0.45,
            alpha=0.85, linewidths=0.2, edgecolors="k", zorder=4
        )
        fig.colorbar(sc_dt, ax=ax_dt, fraction=0.03, pad=0.04,
                     label=f"Day (0 = {t0.strftime('%Y-%m-%d')})")

    ax_dt.axvline(rain_day, color="red", lw=2,
                  label=f"Rain {rain_date.strftime('%Y-%m-%d')}", zorder=5)
    ax_dt.invert_yaxis()
    ax_dt.set_xlim(0, total_days)
    ax_dt.set_ylim(DEPTH_MAX, 0)
    ax_dt.set_xlabel(f"Days since {t0.strftime('%Y-%m-%d')}", fontsize=8)
    ax_dt.set_ylabel("Depth (km)", fontsize=8)
    ax_dt.set_title("Depth–time  (post-rain: color = time)", fontsize=9, pad=4)
    ax_dt.legend(fontsize=7, loc="upper right")
    ax_dt.grid(True, alpha=0.3)
    ax_dt.tick_params(labelsize=7)

    # ── selection state ───────────────────────────────────────────────────────
    state = {"event": None, "geometry": "point"}
    hl_map = [None]
    hl_dt  = [None]

    ax_radio = fig.add_axes([0.04, 0.01, 0.10, 0.08])
    radio = RadioButtons(ax_radio, ("point", "fault"), active=0,
                         label_props={"fontsize": [9, 9]})
    radio.on_clicked(lambda lbl: state.update({"geometry": lbl}))

    info_ax  = fig.add_axes([0.17, 0.01, 0.79, 0.07])
    info_ax.axis("off")
    info_txt = info_ax.text(
        0.0, 0.5,
        "Click an event in either panel to select the diffusion origin (event zero).",
        va="center", ha="left", fontsize=9, transform=info_ax.transAxes,
        color="#333"
    )

    def _highlight(ev):
        for h in [hl_map[0], hl_dt[0]]:
            if h is not None:
                try: h.remove()
                except: pass

        hl_map[0] = ax_map.plot(
            ev["lon"], ev["lat"], "r*", ms=20, zorder=10,
            transform=ccrs.PlateCarree(),
            markeredgecolor="darkred", markeredgewidth=1
        )[0]
        hl_dt[0] = ax_dt.plot(
            ev["day"], ev["depth_clip"], "r*", ms=20, zorder=10,
            markeredgecolor="darkred", markeredgewidth=1
        )[0]

        t_ev  = pd.Timestamp(ev["time"])
        dpost = int(ev["day"]) - rain_day
        info_txt.set_text(
            f"ORIGIN ▶  {t_ev.strftime('%Y-%m-%d %H:%M')}  "
            f"({'PRE' if dpost < 0 else f'+{dpost}d'} rain)  |  "
            f"lat {ev['lat']:.4f}°  lon {ev['lon']:.4f}°  "
            f"depth {ev['depth']:.1f} km  M{ev['mag']:.1f}  |  "
            f"Geometry: {state['geometry']}    ← ENTER to confirm"
        )
        state["event"] = ev
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is ax_map and event.xdata is not None:
            ev, d = _nearest_on_map(df, event.xdata, event.ydata)
            if d < 30:
                _highlight(ev)
        elif event.inaxes is ax_dt and event.xdata is not None:
            ev, _ = _nearest_on_dt(df, event.xdata, event.ydata,
                                   total_days, DEPTH_MAX)
            _highlight(ev)

    def on_key(event):
        if event.key == "enter" and state["event"] is not None:
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    if state["event"] is None:
        print("  No origin selected.")
        return None

    ev  = state["event"]
    row = {
        "region":       region,
        "sub_n":        sub_n,
        "origin_time":  pd.Timestamp(ev["time"]).isoformat(),
        "origin_lat":   round(float(ev["lat"]),   5),
        "origin_lon":   round(float(ev["lon"]),   5),
        "origin_depth": round(float(ev["depth"]), 2),
        "origin_mag":   round(float(ev["mag"]),   1),
        "geometry":     state["geometry"],
        "D_chosen":     float("nan"),
        "dist_type":    "",
        "notes":        ""
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.path.exists(ORIGINS_CSV):
        existing = pd.read_csv(ORIGINS_CSV)
        existing = existing[~((existing["region"] == region) &
                              (existing["sub_n"]  == sub_n))]
        new_df = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        new_df = pd.DataFrame([row])

    new_df.to_csv(ORIGINS_CSV, index=False)
    print(f"  [saved] {region} sub{sub_n}  origin: "
          f"({row['origin_lat']}, {row['origin_lon']}, {row['origin_depth']} km)  "
          f"{pd.Timestamp(row['origin_time']).strftime('%Y-%m-%d')}  "
          f"geom={row['geometry']}")

    # ── Save subcatalog with relative coordinates ─────────────────────────────
    origin_t   = pd.Timestamp(ev["time"])
    origin_lat = float(ev["lat"])
    origin_lon = float(ev["lon"])
    origin_dep = float(ev["depth"])

    cos_lat = np.cos(np.radians(origin_lat))
    dx = (df["lon"].values - origin_lon) * cos_lat * 111.32   # km E–W
    dy = (df["lat"].values - origin_lat) * 111.32              # km N–S
    dz = df["depth_clip"].values - origin_dep                  # km (+ = deeper)

    sub_cat = df[["time", "lat", "lon", "depth", "mag"]].copy()
    sub_cat["t_days_from_t0"]    = df["day"].values
    sub_cat["t_from_origin"]     = ((df["time"] - origin_t)
                                    .dt.total_seconds() / 86400.0).round(4)
    sub_cat["r_3d"]         = np.round(np.sqrt(dx**2 + dy**2 + dz**2), 4)
    sub_cat["r_horizontal"] = np.round(np.sqrt(dx**2 + dy**2), 4)
    sub_cat["r_vertical"]   = np.round(np.abs(dz), 4)
    sub_cat["dz_signed"]    = np.round(dz, 4)   # + = deeper than origin
    sub_cat["inside_envelope"] = np.nan          # filled by 17_pick_diffusivity

    os.makedirs(SUBCATS_DIR, exist_ok=True)
    sub_path = os.path.join(SUBCATS_DIR, f"{region}_sub{sub_n}.csv")
    sub_cat.to_csv(sub_path, index=False)
    n_post = int((sub_cat["t_from_origin"] >= 0).sum())
    print(f"  [saved] subcatalog: {sub_path}  "
          f"({len(sub_cat)} events total, {n_post} post-origin)")

    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True)
    parser.add_argument("--sub",    type=int, default=1)
    args = parser.parse_args()

    if args.region not in REGIONS:
        print(f"Unknown region: {args.region}")
        sys.exit(1)

    result = run(args.region, args.sub)

    if result is not None:
        print("\nLaunching D picker (17_pick_diffusivity.py) ...")
        os.execv(sys.executable,
                 [sys.executable,
                  os.path.join(os.path.dirname(__file__), "17_pick_diffusivity.py"),
                  "--region", args.region,
                  "--sub",    str(args.sub)])
