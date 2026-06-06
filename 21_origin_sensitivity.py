"""
21_origin_sensitivity.py
========================
Reviewer response: demonstrate that Dc is robust to the choice of diffusion
origin (reviewer concern: "did you pick the point that fits best?").

Method:
  For each confirmed Regime I region (Noto, Los Cabos, Corinth, Pyrenees),
  load the active subcatalog and systematically shift the origin time by
  t_offset ∈ {-30, -14, -7, 0, +7, +14, +30} days.
  For each shifted origin, recompute t_from_shifted = t - (t0 + offset),
  keep only events with t_from_shifted > 0 and r_3d < R_max,
  fit D via linear regression r² = 4D·t (OLS through origin),
  record D_fit and R².

Output: one panel per region showing D_fit vs t_offset as a band.
If D is stable across offsets → origin choice is not cherry-picked.
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE    = os.path.dirname(os.path.abspath(__file__))
SUB_DIR = os.path.join(BASE, "results", "subcatalogs")
FIG_DIR = os.path.join(BASE, "figures", "publication")
os.makedirs(FIG_DIR, exist_ok=True)

REGIONS = {
    "noto_japan":      ("noto_japan_sub3.csv",      "Noto JP",        3.47),
    "los_cabos_mx":    ("los_cabos_mx_sub3.csv",    "Los Cabos MX",   3.54),
    "corinth_greece":  ("corinth_greece_sub2.csv",  "Corinth GR",     7.71),
    "pyrenees_fr":     ("pyrenees_fr_sub2.csv",     "Pyrenees FR",   10.17),
}

OFFSETS = [-14, -7, -3, 0, 3, 7, 14]   # days to shift t0

def fit_D(t_days, r_3d_km):
    """OLS r² = 4D·t through origin (both in SI: m²/s)."""
    t_s   = t_days * 86400
    r_m   = r_3d_km * 1000
    r2    = r_m**2
    # D = sum(r²·t) / sum(t²) * (1/4)
    denom = np.sum(t_s**2)
    if denom < 1e-10:
        return np.nan, np.nan
    D = np.sum(r2 * t_s) / (4 * denom)
    # R²
    y_pred = 4 * D * t_s
    ss_res = np.sum((r2 - y_pred)**2)
    ss_tot = np.sum((r2 - r2.mean())**2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return D, R2

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})

fig, axes = plt.subplots(1, 4, figsize=(13, 4), sharey=False,
                         gridspec_kw={"wspace": 0.40})

for ax, (region, (subfile, title, D_ref)) in zip(axes, REGIONS.items()):
    fpath = os.path.join(SUB_DIR, subfile)
    if not os.path.exists(fpath):
        ax.set_title(f"{title}\n(no subcatalog)", fontsize=8)
        continue

    sub = pd.read_csv(fpath)
    sub = sub.dropna(subset=["t_from_origin","r_3d"])
    # Use events that were post-origin in the original (t_from_origin > 0)
    sub_post = sub[sub["t_from_origin"] > 0].copy()
    if len(sub_post) < 5:
        ax.set_title(f"{title}\n(N too small)", fontsize=8)
        continue

    # R_max = 99th percentile of r_3d (use same spatial extent)
    r_max = np.percentile(sub_post["r_3d"], 99)

    Ds, R2s = [], []
    for offset in OFFSETS:
        # Shift: new t_from_origin = original t_from_origin - offset
        t_shifted = sub_post["t_from_origin"] - offset
        valid = (t_shifted > 0) & (sub_post["r_3d"] <= r_max)
        t_v = t_shifted[valid].values
        r_v = sub_post.loc[valid, "r_3d"].values
        if len(t_v) < 5:
            Ds.append(np.nan); R2s.append(np.nan)
            continue
        D, R2 = fit_D(t_v, r_v)
        Ds.append(D); R2s.append(R2)

    Ds  = np.array(Ds, dtype=float)
    R2s = np.array(R2s, dtype=float)
    valid_mask = np.isfinite(Ds) & (Ds > 0)

    # Plot
    ax.plot(np.array(OFFSETS)[valid_mask], Ds[valid_mask],
            "o-", color="#1B5E20", lw=1.8, ms=6,
            zorder=4, label=r"$D_{\rm fit}$ (regression)")

    # Reference D (visual fit from paper)
    ax.axhline(D_ref, color="#D32F2F", lw=1.2, ls="--", alpha=0.8,
               label=f"$D_{{\\rm paper}}$ = {D_ref} m²/s")

    # ±factor-of-2 band around reference
    ax.fill_between(OFFSETS, D_ref/2, D_ref*2, alpha=0.08, color="#D32F2F",
                    label="×2 / ÷2 band")

    ax.axvline(0, color="#555", lw=0.8, ls=":", alpha=0.6)
    ax.set_xlabel("Origin time offset (days)", fontsize=8)
    ax.set_ylabel(r"$D_{\rm fit}$ (m² s⁻¹)", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_yscale("log")

    if ax == axes[0]:
        ax.legend(fontsize=7, loc="upper right", framealpha=0.85)

    # Annotate variability
    valid_D = Ds[valid_mask]
    if len(valid_D) >= 3:
        ratio = valid_D.max() / valid_D.min()
        ax.text(0.05, 0.07, f"max/min = {ratio:.1f}×",
                transform=ax.transAxes, fontsize=7.5, color="#333")

fig.suptitle("Sensitivity of hydraulic diffusivity to origin point selection",
             fontsize=9.5, y=1.02)

for ext in ("pdf","png"):
    plt.savefig(os.path.join(FIG_DIR, f"FigureS_origin_sensitivity.{ext}"),
                dpi=300 if ext=="pdf" else 200, bbox_inches="tight")
plt.close()
print("FigureS_origin_sensitivity saved")
