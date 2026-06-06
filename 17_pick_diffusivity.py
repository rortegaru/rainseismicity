#!/usr/bin/env python3
"""
17_pick_diffusivity.py
======================
Interactive D picker: r–t scatter + adjustable diffusion envelope.

Two scatter panels:
  Top    : r vs t  (raw distance from origin vs. days)
  Bottom : r² vs t (linear for pure Darcy — slope = 4D)

Red bold curve = currently selected D (adjust with slider).
Faint blue lines = reference D values for orientation.
Radio buttons switch distance metric (3D / horizontal / vertical).

Press ENTER to confirm and save D to results/darcy_origins.csv.

Usage:
    python 17_pick_diffusivity.py --region noto_japan --sub 1
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
from matplotlib.widgets import RadioButtons, Slider

sys.path.insert(0, os.path.dirname(__file__))
from config_regions import REGIONS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SUBCATS_DIR = os.path.join(RESULTS_DIR, "subcatalogs")
BOXES_FILE  = os.path.join(RESULTS_DIR, "subregion_boxes.json")
ORIGINS_CSV = os.path.join(RESULTS_DIR, "darcy_origins.csv")
DEPTH_MAX   = 35

# Reference D lines shown as faint background (m²/s)
D_REF = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]


def _r_curve(D_ms2, t_days):
    """Diffusion front r(t) in km.  r = sqrt(4 * D * t_seconds) / 1000."""
    return np.sqrt(np.maximum(4.0 * D_ms2 * t_days * 86400.0, 0)) / 1000.0


def _distances(df, origin):
    cos_lat = np.cos(np.radians(float(origin["origin_lat"])))
    dx = (df["lon"].values - float(origin["origin_lon"])) * cos_lat * 111.32
    dy = (df["lat"].values - float(origin["origin_lat"])) * 111.32
    dz = df["depth_clip"].values - float(origin["origin_depth"])
    r_h = np.sqrt(dx**2 + dy**2)
    r_v = np.abs(dz)
    r_3 = np.sqrt(dx**2 + dy**2 + dz**2)
    return {"3d": r_3, "horizontal": r_h, "vertical": r_v}


def run(region, sub_n=1):
    if not os.path.exists(ORIGINS_CSV):
        print(f"No origins file found. Run 16_pick_origin.py first.")
        return

    origins_df = pd.read_csv(ORIGINS_CSV)
    mask = (origins_df["region"] == region) & (origins_df["sub_n"] == sub_n)
    if not mask.any():
        print(f"No origin for {region} sub{sub_n}. Run 16_pick_origin.py first.")
        return
    origin = origins_df[mask].iloc[-1]

    cfg       = REGIONS[region]
    rain_date = pd.Timestamp(cfg["rain"])
    mc        = cfg.get("mc", 1.5)
    origin_t  = pd.Timestamp(origin["origin_time"])

    # Load box
    box = None
    if os.path.exists(BOXES_FILE):
        with open(BOXES_FILE) as f:
            box = json.load(f).get(f"{region}_sub{sub_n}")
    post_days = box.get("post_days", 90) if box else 90

    # Load & filter data
    csv = os.path.join(os.path.dirname(__file__), "data", f"{region}.csv")
    df  = pd.read_csv(csv, parse_dates=["time"])

    if box:
        df = df[(df["lon"] >= box["lon_min"]) & (df["lon"] <= box["lon_max"]) &
                (df["lat"] >= box["lat_min"]) & (df["lat"] <= box["lat_max"])].copy()

    t_end = origin_t + pd.Timedelta(days=post_days)
    df = df[(df["time"] >= origin_t) & (df["time"] <= t_end)].copy()
    df["depth_clip"] = df["depth"].clip(0, DEPTH_MAX)
    df["t_days"] = (df["time"] - origin_t).dt.total_seconds() / 86400.0
    df = df[df["t_days"] >= 0].reset_index(drop=True)

    if len(df) < 3:
        print(f"  Too few events ({len(df)}) after origin. Exiting.")
        return

    all_r = _distances(df, origin)
    state = {"dist_type": "3d", "D": 1.0}
    df["r"] = all_r["3d"]

    norm_depth = Normalize(vmin=0, vmax=DEPTH_MAX)
    cmap_depth = cm.plasma_r

    t_max  = df["t_days"].max() * 1.08
    t_arr  = np.linspace(0.01, t_max, 400)

    def _sz(mag):
        return np.clip(18 * 10 ** (0.5 * (mag - mc)), 5, 200)

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 10), facecolor="white")
    fig.suptitle(
        f"{region.replace('_',' ').title()}  sub{sub_n}  "
        f"|  Adjust D slider to fit data  |  ENTER to confirm\n"
        f"Origin: {origin_t.strftime('%Y-%m-%d')}  "
        f"({float(origin['origin_lat']):.4f}, {float(origin['origin_lon']):.4f}, "
        f"{float(origin['origin_depth']):.1f} km)  M{float(origin['origin_mag']):.1f}  "
        f"geom={origin['geometry']}",
        fontsize=10, fontweight="bold"
    )

    gs = gridspec.GridSpec(3, 1, figure=fig,
                           height_ratios=[3, 3, 0.7],
                           hspace=0.45,
                           top=0.88, bottom=0.04,
                           left=0.10, right=0.94)
    ax_rt  = fig.add_subplot(gs[0])
    ax_r2t = fig.add_subplot(gs[1])
    ax_bot = fig.add_subplot(gs[2])
    ax_bot.axis("off")

    # Colorbar (shared)
    sm = cm.ScalarMappable(cmap=cmap_depth, norm=norm_depth)
    sm.set_array([])
    fig.colorbar(sm, ax=[ax_rt, ax_r2t], fraction=0.015, pad=0.01,
                 label="Depth (km)")

    def _draw_scatter():
        r = df["r"].values
        t = df["t_days"].values
        for ax, y, ylabel, title in [
            (ax_rt,  r,    "r (km)",  "r  vs  t"),
            (ax_r2t, r**2, "r² (km²)", "r²  vs  t  —  linear = pure Darcy")
        ]:
            ax.scatter(t, y,
                       c=df["depth_clip"], cmap=cmap_depth, norm=norm_depth,
                       s=_sz(df["mag"]),
                       alpha=0.7, linewidths=0.2, edgecolors="k", zorder=4)
            ax.set_xlim(0, t_max)
            ymax = max(float(np.nanmax(y)) * 1.25, 0.1)
            ax.set_ylim(0, ymax)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(title, fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
        ax_r2t.set_xlabel(f"Days since origin  ({origin_t.strftime('%Y-%m-%d')})",
                          fontsize=9)

    _draw_scatter()

    # Reference D curves (faint)
    for D_r in D_REF:
        y_r = _r_curve(D_r, t_arr)
        ax_rt.plot(t_arr, y_r,    lw=0.7, color="#a0c4e8", ls="--", zorder=2)
        ax_r2t.plot(t_arr, y_r**2, lw=0.7, color="#a0c4e8", ls="--", zorder=2)
        # Label on right edge of r-vs-t
        ax_rt.text(t_arr[-1], y_r[-1], f"{D_r}", fontsize=6,
                   color="#4a90c4", ha="left", va="center",
                   clip_on=True)

    D_init = 1.0
    y_init = _r_curve(D_init, t_arr)
    line_rt,  = ax_rt.plot(t_arr, y_init,    lw=2.5, color="red", zorder=6,
                            label=f"D = {D_init:.3f} m²/s")
    line_r2t, = ax_r2t.plot(t_arr, y_init**2, lw=2.5, color="red", zorder=6,
                             label=f"D = {D_init:.3f} m²/s")
    ax_rt.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax_r2t.legend(fontsize=8, loc="upper left", framealpha=0.85)

    # ── Slider (log scale: val = log10(D), range -2..2 → D = 0.01..100) ──────
    ax_sl = fig.add_axes([0.12, 0.015, 0.55, 0.025])
    slider = Slider(ax_sl, "log₁₀(D)", -2.0, 2.0, valinit=0.0, color="tomato")

    d_label = fig.text(0.70, 0.02, f"D = {D_init:.3f} m²/s",
                       fontsize=12, fontweight="bold", color="red", va="center")

    def _update_D(val):
        D = 10 ** slider.val
        state["D"] = D
        y  = _r_curve(D, t_arr)
        line_rt.set_ydata(y)
        line_r2t.set_ydata(y**2)
        lbl = f"D = {D:.4g} m²/s"
        line_rt.set_label(lbl)
        line_r2t.set_label(lbl)
        ax_rt.legend(fontsize=8, loc="upper left", framealpha=0.85)
        ax_r2t.legend(fontsize=8, loc="upper left", framealpha=0.85)
        d_label.set_text(f"D = {D:.4g} m²/s")
        fig.canvas.draw_idle()

    slider.on_changed(_update_D)

    # ── Distance type radio ───────────────────────────────────────────────────
    ax_radio = fig.add_axes([0.78, 0.005, 0.13, 0.065])
    radio = RadioButtons(ax_radio, ("3D", "horizontal", "vertical"), active=0,
                         label_props={"fontsize": [8, 8, 8]})

    def _on_radio(label):
        dtype_map = {"3D": "3d", "horizontal": "horizontal", "vertical": "vertical"}
        dtype = dtype_map[label]
        state["dist_type"] = dtype
        df["r"] = all_r[dtype]
        # Redraw scatter (clear then redraw)
        for ax in [ax_rt, ax_r2t]:
            for coll in ax.collections:
                coll.remove()
        _draw_scatter()
        # Redraw reference lines
        for ax in [ax_rt, ax_r2t]:
            for line in list(ax.lines):
                if line not in [line_rt, line_r2t]:
                    line.remove()
        for D_r in D_REF:
            y_r = _r_curve(D_r, t_arr)
            ax_rt.plot(t_arr, y_r,    lw=0.7, color="#a0c4e8", ls="--", zorder=2)
            ax_r2t.plot(t_arr, y_r**2, lw=0.7, color="#a0c4e8", ls="--", zorder=2)
        # Update active curve
        D  = state["D"]
        y  = _r_curve(D, t_arr)
        line_rt.set_ydata(y)
        line_r2t.set_ydata(y**2)
        # Rescale
        rvals = df["r"].values
        ax_rt.set_ylim(0,  max(float(np.nanmax(rvals))*1.25, 0.1))
        ax_r2t.set_ylim(0, max(float(np.nanmax(rvals**2))*1.25, 0.1))
        fig.canvas.draw_idle()

    radio.on_clicked(_on_radio)

    # ── ENTER to confirm ──────────────────────────────────────────────────────
    def on_key(event):
        if event.key == "enter":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    # ── Save D to origins CSV ────────────────────────────────────────────────
    D_final    = state["D"]
    dist_final = state["dist_type"]
    df_orig = pd.read_csv(ORIGINS_CSV, dtype={"dist_type": str, "notes": str})
    mask2   = (df_orig["region"] == region) & (df_orig["sub_n"] == sub_n)
    df_orig.loc[mask2, "D_chosen"]  = round(D_final, 5)
    df_orig.loc[mask2, "dist_type"] = dist_final
    df_orig.to_csv(ORIGINS_CSV, index=False)
    print(f"  [saved] D = {D_final:.4g} m²/s  dist={dist_final}  →  {ORIGINS_CSV}")

    # ── Update inside_envelope in subcatalog ─────────────────────────────────
    sub_path = os.path.join(SUBCATS_DIR, f"{region}_sub{sub_n}.csv")
    if os.path.exists(sub_path):
        sc = pd.read_csv(sub_path, parse_dates=["time"])
        r_col = {"3d": "r_3d", "horizontal": "r_horizontal",
                 "vertical": "r_vertical"}.get(dist_final, "r_3d")
        t_pos  = sc["t_from_origin"].clip(lower=0)
        r_env  = np.sqrt(4.0 * D_final * t_pos * 86400.0) / 1000.0  # km
        sc["inside_envelope"] = sc[r_col] <= r_env
        sc["D_chosen"]   = D_final
        sc["dist_type"]  = dist_final
        sc["r_envelope"] = r_env.round(4)
        sc.to_csv(sub_path, index=False)
        n_in  = int(sc["inside_envelope"].sum())
        n_tot = len(sc)
        print(f"  [updated] subcatalog: {sub_path}  "
              f"({n_in}/{n_tot} events inside envelope)")
    else:
        print(f"  [warn] subcatalog not found: {sub_path}  (run 16 first)")

    return D_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True)
    parser.add_argument("--sub",    type=int, default=1)
    args = parser.parse_args()

    if args.region not in REGIONS:
        print(f"Unknown region: {args.region}")
        sys.exit(1)

    run(args.region, args.sub)
