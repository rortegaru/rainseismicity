"""
Improved Figure 2: cleaner conceptual diagram of two triggering regimes.
More visual, publication-quality, suitable for GRL.
"""
import os, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.gridspec as gs
import matplotlib.patches as mpatches, matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch

os.makedirs("figures/publication", exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":9,
                     "axes.spines.top":False,"axes.spines.right":False})

fig = plt.figure(figsize=(10, 5.5))
G = gs.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.4,
                height_ratios=[1.2, 1])

# ── Top row: Regime I schematic (left 2/3) ───────────────────────────────────
ax_I_rain  = fig.add_subplot(G[0, 0])
ax_I_seis  = fig.add_subplot(G[1, 0])
ax_III_rain = fig.add_subplot(G[0, 1])
ax_III_seis = fig.add_subplot(G[1, 1])
ax_scatter  = fig.add_subplot(G[:, 2])

np.random.seed(42)
t = np.linspace(-60, 150, 1000)

# ─── Regime I ────────────────────────────────────────────────────────────────
# Precipitation: near-zero background then spike
precip_I = np.zeros_like(t)
precip_I[(t >= 0) & (t < 1)] = 47.0
precip_I_bg = np.where(t < 0, 0.3 + 0.2*np.abs(np.sin(t/7)), 0.0)

ax = ax_I_rain
ax.fill_between(t, 0, precip_I_bg, color="#90CAF9", alpha=0.5, step="mid")
ax.bar([0.5], [47], width=1.5, color="#1565C0", alpha=0.9, zorder=5)
ax.axvline(0, color="#1565C0", lw=1.5, ls="--", alpha=0.7)
ax.fill_between(t[t<0], 0, 2.0, alpha=0.12, color="#546E7A")
ax.set_xlim(-60, 150); ax.set_ylim(-1, 55)
ax.set_ylabel("Precip. (mm/d)", fontsize=8)
ax.set_xticklabels([])
ax.text(0.5, 47*0.7, "47 mm\n(7 hr)", fontsize=8, ha="center",
        color="white", fontweight="bold", va="top")
api_text = "API₃₀ ≈ 0.8 mm/d\nIC = 58  →  Regime I"
ax.text(-55, 42, api_text, fontsize=8, color="#C62828", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor="#C62828"))
ax.set_title("Regime I: Isolated impulse\n(Los Cabos, Murcia)", fontsize=9, fontweight="bold",
             color="#C62828")

# Seismicity response
def swarm_envelope(t, onset, peak, decay):
    return np.where(t>=onset, peak*np.exp(-((t-onset)**2)/(2*decay**2)), 0)

seis_I = 0.8 + swarm_envelope(t, 15, 12, 20) + 0.3*np.random.randn(len(t))*0.5
seis_I = np.clip(seis_I, 0, None)
t_diff_10km = 10000**2/(4*1.0*86400)  # ~289 days for D=1 m²/s

ax = ax_I_seis
ax.fill_between(t, 0, np.where(t<0, seis_I, np.nan), color="#78909C", alpha=0.5, step="mid")
ax.fill_between(t, 0, np.where(t>=0, seis_I, np.nan), color="#C62828", alpha=0.7, step="mid")
ax.axvline(0, color="#1565C0", lw=1.5, ls="--", alpha=0.7)
ax.axvline(15, color="#C62828", lw=1.5, ls="--", label="τ = 15 d")
# Diffusion arrow
ax.annotate("", xy=(15, 6), xytext=(0, 6),
            arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.5))
ax.text(7.5, 6.8, "τ", fontsize=9, ha="center", color="#C62828", fontweight="bold")
ax.text(100, 10, f"D ≈ 0.94 m²s⁻¹", fontsize=8, color="#C62828",
        bbox=dict(boxstyle="round", facecolor="#FFEBEE", alpha=0.9))
ax.set_xlabel("Days relative to rain event", fontsize=8)
ax.set_ylabel("Events / day", fontsize=8)
ax.set_xlim(-60, 150); ax.set_ylim(-0.5, 14)
ax.legend(fontsize=8)

# ─── Regime III ──────────────────────────────────────────────────────────────
# Precipitation: sustained wet season background + similar peak
precip_III_bg = 8 + 3*np.sin(t/10) + 1.5*np.random.randn(len(t))
precip_III_bg = np.clip(precip_III_bg, 0, None)
precip_III_event = np.zeros_like(t)
precip_III_event[(t>=0)&(t<2)] = 45

ax = ax_III_rain
ax.fill_between(t, 0, precip_III_bg, color="#90CAF9", alpha=0.5, step="mid", label="Season rain")
ax.bar([0.5], [45], width=1.5, color="#1565C0", alpha=0.9, zorder=5)
ax.axvline(0, color="#1565C0", lw=1.5, ls="--", alpha=0.7)
api_text3 = "API₃₀ ≈ 8 mm/d\nIC = 5.6  →  Regime II/III"
ax.text(-55, 42, api_text3, fontsize=8, color="#E65100", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0", edgecolor="#E65100"))
ax.set_xlim(-60, 150); ax.set_ylim(-1, 55)
ax.set_xticklabels([])
ax.set_title("Regime III: Wet season\n(Nepal, Taiwan)", fontsize=9, fontweight="bold",
             color="#E65100")

# Seismicity: no clear response
seis_III = 3 + 0.8*np.sin(t/15) + 0.5*np.random.randn(len(t))
seis_III = np.clip(seis_III, 0, None)
ax = ax_III_seis
ax.fill_between(t, 0, seis_III, color="#78909C", alpha=0.5, step="mid", label="Seismicity (no clear response)")
ax.axvline(0, color="#1565C0", lw=1.5, ls="--", alpha=0.7)
ax.text(60, 5.5, "No diagnostic\npost-rain anomaly", fontsize=8,
        color="#E65100", ha="center",
        bbox=dict(boxstyle="round", facecolor="#FFF3E0", alpha=0.9))
ax.set_xlabel("Days relative to rain event", fontsize=8)
ax.set_xlim(-60, 150); ax.set_ylim(-0.5, 7)
ax.legend(fontsize=8)

# ─── Right: IC vs SPI scatter ─────────────────────────────────────────────────
REGIME_COLOR = {"I":"#C62828","0*":"#6A1B9A","II":"#E65100","III":"#2E7D32","0":"#546E7A","flag":"#263238"}

DATA = [
    ("Los Cabos",  58,   3.5,  "I",   True),
    ("Murcia",     15.8, 5.5,  "I",   True),
    ("Himachal",   16.1, 6.2,  "I",   True),
    ("Costa Rica", 10.2, 2.9,  "I",   True),
    ("Ethiopia",   13.3, 3.6,  "I",   False),
    ("Noto",        2.1,-0.4,  "0*",  True),
    ("Corinth",     0.0,-0.2,  "0*",  True),
    ("Pyrenees",    4.5,-0.2,  "0*",  False),
    ("Reykjanes",   0.0,-0.5,  "0*",  False),
    ("Taiwan",      5.9, 1.9,  "II",  False),
    ("Assam",       9.2, 1.3,  "II",  False),
    ("Zagros",      9.2, 2.2,  "II",  False),
    ("Papua NG",    6.7,-0.1,  "II",  False),
    ("Pakistan",    6.8, 2.1,  "II",  False),
    ("Cascades",    2.9, 0.2,  "III", False),
    ("Nepal",       1.1,-0.6,  "flag",False),
    ("W.Bohemia",   0.6,-0.4,  "0",   False),
    ("Etna",       12.0, 1.2,  "I",   False),
]

ax = ax_scatter
# Background zones
ax.fill_betweenx([-2, 8], 10, 70, alpha=0.07, color="#C62828",
                 label="Regime I\n(IC ≥ 10)")
ax.fill_betweenx([-2, 8], 3, 10, alpha=0.07, color="#E65100",
                 label="Regime II\n(3 ≤ IC < 10)")
ax.fill_betweenx([-2, 8], 0, 3, alpha=0.07, color="#2E7D32",
                 label="Regime III\n(IC < 3)")

for (name, ic, spi, regime, labeled) in DATA:
    col = REGIME_COLOR.get(regime, "#546E7A")
    ax.scatter(ic, spi, s=55, color=col, alpha=0.9,
               edgecolors="white", linewidths=0.7, zorder=4)
    if labeled:
        ax.annotate(name, (ic, spi), textcoords="offset points",
                    xytext=(5,4), fontsize=7.5, color=col, fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])

ax.axvline(10, color="#C62828", lw=1.2, ls="--", alpha=0.8)
ax.axhline(1.5, color="#888", lw=0.8, ls=":", alpha=0.5)
ax.set_xlabel("Impulse Contrast  IC = $P_{\\rm event}$ / $\\bar{P}_{30}$", fontsize=9)
ax.set_ylabel("Standardized Precipitation Index (SPI, local)", fontsize=9)
ax.set_title("(c) Regime classification of 25 regions", fontsize=9, fontweight="bold")
ax.set_xlim(-2, 65); ax.set_ylim(-1.5, 7.5)
legend_els = [mpatches.Patch(facecolor=REGIME_COLOR[r], alpha=0.8, label=f"Regime {r}")
              for r in ["I","0*","II","III","0"]]
ax.legend(handles=legend_els, fontsize=7.5, loc="upper right", framealpha=0.9)
ax.text(-1.5, -1.2, "← wet season\nmasking", fontsize=7, color="#2E7D32", style="italic")
ax.text(50, -1.2, "isolated\npulse →", fontsize=7, color="#C62828", style="italic")
ax.grid(alpha=0.25)

# Panel labels
for ax_obj, lbl in [(ax_I_rain,"(a)"), (ax_III_rain,"(b)"), (ax_I_seis,""), (ax_III_seis,"")]:
    if lbl:
        ax_obj.text(0.01, 0.97, lbl, transform=ax_obj.transAxes,
                    fontsize=10, fontweight="bold", va="top")

plt.savefig("figures/publication/Figure2_regimes_v2.pdf", dpi=300, bbox_inches="tight")
plt.savefig("figures/publication/Figure2_regimes_v2.png", dpi=200, bbox_inches="tight")
plt.close()
print("Figure 2 (improved) saved")
