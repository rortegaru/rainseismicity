"""
20_ic_comparison.py
===================
Reviewer response: demonstrate that IC captures the physically relevant
dimension (dry-baseline isolation) that SPI and total precipitation miss.

NOT framed as "IC is best" — framed as "IC is the only index that reflects
the pore-pressure boundary condition; the others correlate with absolute
rainfall which is physically irrelevant for the Darcy mechanism."

Three panels:
  (a) P_event vs lambda-ratio — absolute rainfall does not separate classes
  (b) SPI       vs lambda-ratio — standardized anomaly does better but conflates
      large-event-in-wet-season with moderate-event-in-dry-season
  (c) IC        vs lambda-ratio — isolation index cleanly separates classes;
      IC=10 threshold shown

Color: green = positive signal (P>=0.78 AND ratio>1), gray = null/negative
Marker: circle = Regime I|II, triangle = Regime III|IV|V|F (non-Darcy context)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
def roc_auc_score(y_true, y_score):
    """Simple Mann-Whitney AUC."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    return np.mean([np.mean(p > neg) + 0.5 * np.mean(p == neg) for p in pos])

BASE    = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, "figures", "publication")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
ci  = pd.read_csv(os.path.join(BASE, "results", "climate_indices.csv"))
sw  = pd.read_csv(os.path.join(BASE, "results", "switchpoint_results.csv"))

df = ci.merge(sw[["region","ratio_after_before","p_rate_increase"]], on="region", how="inner")

# Drop rows with missing IC or ratio
df = df.dropna(subset=["IC","ratio_after_before","p_rate_increase"])

# Positive signal definition (consistent with paper)
df["positive"] = (df["p_rate_increase"] >= 0.78) & (df["ratio_after_before"] > 1.0)

# Clip IC for display (some are 0)
df["IC_plot"] = df["IC"].clip(lower=0.01)

# Ratio for display — clip extreme outliers (Reykjanes 47x)
df["ratio_plot"] = df["ratio_after_before"].clip(upper=10)

# Colors and markers
col = df["positive"].map({True: "#1B5E20", False: "#9E9E9E"})
regime_marker = df["hydro_regime"].apply(
    lambda r: "^" if any(x in str(r) for x in ["III","IV","V","F","SIN"]) else "o"
)

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True,
                         gridspec_kw={"wspace": 0.12})

PANELS = [
    ("P_event_mm",        r"Total rainfall  $P_{\rm event}$  (mm)",       "linear"),
    ("SPI_local",         r"SPI  (standard deviations)",                  "linear"),
    ("IC_plot",           r"Impulse Contrast  IC  (dimensionless)",        "log"),
]

labels = {r: r.replace("_"," ").replace(" mx","").title()
          for r in df["region"]}
label_short = {
    "los_cabos_mx": "Los Cabos",
    "noto_japan":   "Noto",
    "corinth_greece": "Corinth",
    "pyrenees_fr":  "Pyrenees",
    "reykjanes_iceland": "Reykjanes",
    "murcia_spain": "Murcia",
    "apennines_italy": "Apennines",
    "taiwan":       "Taiwan",
}

for ax, (xcol, xlabel, xscale) in zip(axes, PANELS):
    for _, row in df.iterrows():
        xv = row[xcol]
        yv = row["ratio_plot"]
        c  = "#1B5E20" if row["positive"] else "#9E9E9E"
        mk = "^" if any(x in str(row["hydro_regime"]) for x in ["III","IV","V","F","SIN"]) else "o"
        ax.scatter(xv, yv, color=c, marker=mk, s=55, alpha=0.85,
                   edgecolors="white", linewidths=0.6, zorder=4)

    # Annotate the 4 confirmed + key null cases
    for _, row in df.iterrows():
        rg = row["region"]
        if rg not in label_short:
            continue
        xv = row[xcol]
        yv = row["ratio_plot"]
        ax.annotate(label_short[rg], (xv, yv),
                    xytext=(5, 3), textcoords="offset points",
                    fontsize=6.5, color="#1B5E20" if row["positive"] else "#777")

    ax.axhline(1.0, color="#555", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=8.5)
    if xscale == "log":
        ax.set_xscale("log")
        ax.axvline(10, color="#D32F2F", lw=1.2, ls="--", alpha=0.8,
                   label="IC = 10 threshold")
        ax.legend(fontsize=7.5, loc="upper left")

    pass  # AUC not shown — metric framing not the purpose of this figure

axes[0].set_ylabel(r"Post/pre-rain rate ratio  $\lambda_{\rm after}/\lambda_{\rm before}$",
                   fontsize=8.5)
axes[0].set_ylim(0, 10.5)

legend_els = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#1B5E20",
           markersize=8, label="Positive signal (P ≥ 0.78)"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#9E9E9E",
           markersize=8, label="Null / negative"),
    Line2D([0],[0], marker="^", color="w", markerfacecolor="#555",
           markersize=8, label="Non-Darcy regime (III/IV/V)"),
]
axes[2].legend(handles=legend_els, fontsize=7, loc="upper right",
               framealpha=0.85, title="Signal class", title_fontsize=7)

panel_labels = ["(a)", "(b)", "(c)"]
for ax, pl in zip(axes, panel_labels):
    ax.text(0.02, 0.97, pl, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top")

fig.suptitle("IC isolates dry-baseline events; total rainfall and SPI do not",
             fontsize=9.5, y=1.01)

for ext in ("pdf","png"):
    plt.savefig(os.path.join(FIG_DIR, f"FigureS_IC_comparison.{ext}"),
                dpi=300 if ext=="pdf" else 200, bbox_inches="tight")
plt.close()
print("FigureS_IC_comparison saved")
