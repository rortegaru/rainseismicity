"""
19_metrics_figure.py
====================
Generates a publication figure summarizing the two diffusion quality metrics
(F_in and ρ̃) computed in script 18 for every confirmed Darcy case.

WHAT WE ARE SHOWING AND WHY
----------------------------
The diffusivity D was obtained visually (script 17). Scripts 16–17 produced
a rigorous analysis — the spatial extent was chosen to capture only shallow,
migrating seismicity — but the fit is still visual.

This figure adds two N-independent quantitative companions:

  • F_in  = fraction of post-origin events inside the envelope r²≤4Dt
            → simple containment: what fraction of the migrating cluster
              does this D account for?

  • ρ̃    = median(rᵢ / r_envelope(tᵢ))  — normalized distance to the front
            → quality of the spatial-temporal pattern independent of N:
              ρ̃ < 1 means events cluster inside (good), ρ̃ ≈ 0.5–0.9 means
              they track the diffusion front well, ρ̃ > 1.2 is a warning.

Together these two metrics let us compare regions with very different N
(N=14 for Calabria vs N=988 for Corinth) on the same footing.

FIGURE LAYOUT
-------------
Two panels:
  Left  (a): scatter plot D_chosen vs ρ̃, marker size ∝ √N_post,
             color = F_in (viridis), region labels.
             Shaded bands show the ρ̃ quality zones.
  Right (b): horizontal lollipop chart — one row per region,
             left dot = F_in (0–1), right dot = ρ̃ (0–max),
             colored by tier.  Gives a clean ranked overview.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

BASE    = os.path.dirname(os.path.abspath(__file__))
MET_CSV = os.path.join(BASE, "results", "diffusion_quality_metrics.csv")
FIG_DIR = os.path.join(BASE, "figures", "publication")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})

# ── Load metrics, keep only active cases with valid numbers ──────────────────
df = pd.read_csv(MET_CSV)
active = df[(df["status"] == "active") & df["rho_median"].notna()].copy()

# ── Tier colors ──────────────────────────────────────────────────────────────
TIER_COLOR = {
    "GOOD":     "#1B5E20",   # dark green
    "OK":       "#F57F17",   # amber
    "MARGINAL": "#B71C1C",   # dark red
    "POOR":     "#880E4F",   # purple-red
}

# Short display labels
LABEL = {
    "noto_japan":        "Noto JP",
    "los_cabos_mx":      "Los Cabos MX",
    "corinth_greece":    "Corinth GR",
    "pyrenees_fr":       "Pyrenees FR",
    "reykjanes_iceland": "Reykjanes IS",
    "calabria_italy":    "Calabria IT",
    "apennines_italy":   "Apennines IT",
    "marlborough_nz":    "Marlborough NZ",
}
active["label"] = active["region"].map(LABEL).fillna(active["region"])
active["color"] = active["tier"].map(TIER_COLOR)

# Sort by D_chosen for consistent ordering
active = active.sort_values("D_chosen").reset_index(drop=True)

# ── Figure ───────────────────────────────────────────────────────────────────
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5.5),
                                  gridspec_kw={"wspace": 0.55})

# ════════════════════════════════════════════════════════════════════════════
# Panel (a): D_chosen vs ρ̃ — bubble chart
# ════════════════════════════════════════════════════════════════════════════
ax = ax_a

# Quality bands for ρ̃ interpretation
ax.axhspan(0,    0.4,  alpha=0.06, color="#BDBDBD")   # too inside (D over?)
ax.axhspan(0.4,  1.1,  alpha=0.10, color="#1B5E20")   # GOOD zone
ax.axhspan(1.1,  1.5,  alpha=0.08, color="#F9A825")   # OK zone
ax.axhspan(1.5,  6.0,  alpha=0.08, color="#B71C1C")   # POOR zone
ax.axhline(1.0, color="#555", lw=0.8, ls="--", alpha=0.6,
           label="ρ̃ = 1")

# Clip y-axis at 1.8 so the good cases fill the panel.
# Calabria (ρ̃=5.07) is annotated with a broken-axis symbol.
Y_MAX = 1.80

# Manual annotation positions (x_text, y_text) — all within Y_MAX
ANNOT = {
    "Noto JP":         (0.5,  0.92, "left"),
    "Los Cabos MX":    (7,    0.80, "left"),
    "Corinth GR":      (5.5,  0.38, "left"),
    "Pyrenees FR":     (14,   0.48, "left"),
    "Marlborough NZ":  (65,   0.90, "left"),
    "Apennines IT":    (65,   0.48, "left"),
    "Reykjanes IS":    (65,   0.70, "left"),
}

for _, row in active.iterrows():
    sz = max(40, 12 * np.sqrt(row["N_post"]))
    rho_plot = min(row["rho_median"], Y_MAX * 0.97)
    ax.scatter(row["D_chosen"], rho_plot,
               s=sz, color=row["color"], alpha=0.85,
               edgecolors="white", linewidths=0.8, zorder=4)

    if row["label"] == "Calabria IT":
        # Off-scale: show broken arrow and label at top
        ax.annotate("", xy=(row["D_chosen"], Y_MAX * 0.97),
                    xytext=(row["D_chosen"], Y_MAX * 0.84),
                    arrowprops=dict(arrowstyle="->", color=row["color"],
                                    lw=1.2, linestyle="dashed"))
        ax.text(row["D_chosen"] * 1.1, Y_MAX * 0.91,
                f"Calabria IT\n(ρ̃ = 5.07, POOR)",
                fontsize=7, color=row["color"], fontweight="bold",
                va="center", ha="left")
        continue

    if row["label"] not in ANNOT:
        continue
    xt, yt, ha = ANNOT[row["label"]]
    ax.annotate(row["label"],
                xy=(row["D_chosen"], row["rho_median"]),
                xytext=(xt, yt),
                fontsize=7.5, color=row["color"], fontweight="bold",
                va="center", ha=ha,
                arrowprops=dict(arrowstyle="-", color=row["color"],
                                lw=0.7, alpha=0.7),
                zorder=5)

ax.set_xscale("log")
ax.set_xlabel("Hydraulic diffusivity  D  (m² s⁻¹)", fontsize=9)
ax.set_ylabel("Median normalized distance to front  ρ̃", fontsize=9)
ax.set_title("(a) D vs diffusion-front proximity", fontsize=9, fontweight="bold")
ax.set_ylim(0, Y_MAX)
ax.set_xlim(0.3, 200)

# Band labels (fit within Y_MAX=1.8)
ax.text(1.2, 0.10, "events deep\ninside envelope", fontsize=6,
        color="#999", style="italic")
ax.text(1.2, 0.65, "GOOD\n(track front)", fontsize=6.5,
        color="#1B5E20", style="italic", fontweight="bold")
ax.text(1.2, 1.20, "OK", fontsize=6.5, color="#F57F17",
        style="italic", fontweight="bold")
ax.text(1.2, 1.55, "MARGINAL", fontsize=6, color="#B71C1C",
        style="italic")

# Size legend
for n_ex, lbl in [(10, "N=10"), (100, "N=100"), (500, "N=500")]:
    ax.scatter([], [], s=max(40, 12*np.sqrt(n_ex)), color="#555",
               alpha=0.7, label=lbl)
ax.legend(fontsize=7.5, bbox_to_anchor=(0.43, 0.98), loc="upper left",
          bbox_transform=ax.transAxes,
          framealpha=0.85, handlelength=0.8, borderpad=0.7,
          labelspacing=1.2)

# ════════════════════════════════════════════════════════════════════════════
# Panel (b): lollipop chart — F_in and ρ̃ per region
# ════════════════════════════════════════════════════════════════════════════
ax = ax_b

# Clip ρ̃ for POOR cases so they don't blow the scale
active["rho_plot"] = active["rho_median"].clip(upper=3.5)

y_pos = np.arange(len(active))

for i, row in active.iterrows():
    col = row["color"]
    y   = i

    # F_in bar (left axis, 0–1)
    ax.plot([0, row["F_in"]], [y + 0.15, y + 0.15],
            color=col, lw=2.5, alpha=0.7, solid_capstyle="round")
    ax.scatter(row["F_in"], y + 0.15, s=55, color=col,
               zorder=5, edgecolors="white", linewidths=0.6)
    ax.text(row["F_in"] + 0.02, y + 0.15,
            f"{row['F_in']:.2f}", fontsize=7, color=col, va="center")

    # ρ̃ bar (right axis, using transformed x: 1 + ρ_offset plotted separately)
    rho_x = 1.15 + (row["rho_plot"] - 0) * 0.25   # rescale ρ̃ to x > 1.1
    ax.plot([1.15, rho_x], [y - 0.15, y - 0.15],
            color=col, lw=2.5, alpha=0.7, ls="--", solid_capstyle="round")
    ax.scatter(rho_x, y - 0.15, s=55, color=col, marker="D",
               zorder=5, edgecolors="white", linewidths=0.6)
    rho_lbl = f">{row['rho_plot']:.2f}" if row["rho_median"] > 3.5 else f"{row['rho_median']:.2f}"
    ax.text(rho_x + 0.02, y - 0.15,
            f"ρ̃={rho_lbl}", fontsize=7, color=col, va="center")

ax.set_yticks(y_pos)
ax.set_yticklabels(active["label"], fontsize=8)
ax.set_xlim(-0.05, 2.2)
ax.set_ylim(-0.8, len(active) - 0.2)

# Divider line between F_in space and ρ̃ space
ax.axvline(1.1, color="#BDBDBD", lw=0.8, ls=":")
ax.axvline(1.0, color="#555",    lw=0.7, ls="--", alpha=0.5)  # F_in=1 reference
ax.axvline(1.15 + 1.0*0.25, color="#555", lw=0.7, ls="--",   # ρ̃=1 reference
           alpha=0.5)

# x-axis tick labels
fin_ticks  = [0, 0.25, 0.5, 0.75, 1.0]
rho_values = [0, 0.5,  1.0, 2.0,  3.5]
xticks     = fin_ticks + [1.15 + r*0.25 for r in rho_values]
xlabels    = [f"{v:.2f}" for v in fin_ticks] + \
             [f"{v:.1f}" for v in rho_values]
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, fontsize=7, rotation=45)

ax.text(0.5,  -0.65, "F_in  (● fraction inside envelope)",
        fontsize=7.5, ha="center", color="#333", transform=ax.transData)
ax.text(1.55, -0.65, "ρ̃  (◆ normalized distance to front)",
        fontsize=7.5, ha="center", color="#333", transform=ax.transData)

ax.set_title("(b) Quality metrics per region", fontsize=9, fontweight="bold")

# Tier legend
legend_els = [mpatches.Patch(color=c, label=f"{t}")
              for t, c in TIER_COLOR.items() if t != "MARGINAL"]
ax.legend(handles=legend_els, fontsize=7,
          bbox_to_anchor=(0.99, 0.93), loc="upper right",
          bbox_transform=ax.transAxes,
          framealpha=0.85, title="Tier", title_fontsize=7)

# ── Save ─────────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    path = os.path.join(FIG_DIR, f"FigureS_diffusion_metrics.{ext}")
    dpi = 300 if ext == "pdf" else 200
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
plt.close()
print("FigureS_diffusion_metrics saved (PDF + PNG)")

# ── Print clean table for paper / Overleaf ───────────────────────────────────
print("\n=== TABLA PARA EL PAPER ===")
print(f"{'Región':<20} {'D (m²/s)':>9}  {'N':>5}  {'F_in':>5}  {'ρ̃':>5}  "
      f"{'IQR ρ':>12}  {'Tier':<10}")
print("-" * 75)
for _, row in active.sort_values("D_chosen").iterrows():
    iqr = f"[{row['rho_q25']:.2f}–{row['rho_q75']:.2f}]"
    print(f"{row['label']:<20} {row['D_chosen']:>9.2f}  {int(row['N_post']):>5}  "
          f"{row['F_in']:>5.2f}  {row['rho_median']:>5.2f}  {iqr:>12}  {row['tier']:<10}")
