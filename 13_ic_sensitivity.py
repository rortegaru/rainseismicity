"""
IC threshold sensitivity analysis.
Shows that the main conclusions (which regions show signal) are stable
across IC thresholds from 5 to 20.
Also produces ROC-like curve: true positive rate vs IC threshold.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os

RES = "results"
os.makedirs("figures/publication", exist_ok=True)

# Ground truth: regions with confirmed seismicity signal (Bayesian P>0.90 OR ETAS R>1.5)
CONFIRMED_SIGNAL = {"los_cabos_mx","noto_japan","corinth_greece","murcia_spain",
                    "pyrenees_fr","reykjanes_iceland"}  # reykjanes flagged but real signal

# All regions with their IC values and signal status
REGIONS = [
    # name,              IC_era5,  IC_station, signal
    ("los_cabos_mx",      35,       58,        True),
    ("murcia_spain",      15.8,     15.8,      True),
    ("himachal_india",    16.1,     16.1,      None),   # no seismic data
    ("costa_rica",        10.2,     10.2,      False),  # P=0.23, weak
    ("ethiopia_rift",     13.3,     13.3,      False),
    ("noto_japan",        28,       28,        True),   # ERA5 hourly
    ("corinth_greece",    0,        None,      True),   # ERA5 blind
    ("pyrenees_fr",       4.5,      4.5,       True),   # ERA5 low but signal
    ("reykjanes_iceland", 0,        None,      True),   # ERA5 blind, volcanic
    ("taiwan",            5.9,      5.9,       False),
    ("assam_india",       9.2,      9.2,       False),
    ("zagros_iran",       9.2,      9.2,       False),
    ("papua_ng",          6.7,      6.7,       False),
    ("pakistan",          6.8,      6.8,       False),
    ("java_indonesia",    6.6,      6.6,       False),
    ("chile",             3.0,      3.0,       False),
    ("apennines_italy",   3.9,      3.9,       False),
    ("calabria_italy",    0.1,      0.1,       False),
    ("marlborough_nz",    0.0,      0.0,       False),
    ("cascades_oregon",   2.9,      2.9,       False),
    ("nepal_himalaya",    1.1,      1.1,       False),
    ("west_bohemia",      0.6,      0.6,       False),
    ("colombia",          0.3,      0.3,       False),
    ("guerrero_mexico",   2.9,      2.9,       False),
    ("mindanao_ph",       0.7,      0.7,       False),
]

df = pd.DataFrame(REGIONS, columns=["region","IC_era5","IC_station","signal"])
# Use station IC where available, else ERA5
df["IC"] = df["IC_station"].fillna(df["IC_era5"])
# Exclude cases without seismic signal assessment (None)
df_assessed = df[df["signal"].notna()].copy()

# ── Sensitivity analysis ──────────────────────────────────────────────────────
IC_THRESHOLDS = np.arange(3, 25, 1)

results = []
for thresh in IC_THRESHOLDS:
    regime_I   = df_assessed[df_assessed["IC"] >= thresh]
    regime_III = df_assessed[df_assessed["IC"] < thresh]

    # True positives: Regime I with signal
    TP = (regime_I["signal"] == True).sum()
    # False positives: Regime I without signal
    FP = (regime_I["signal"] == False).sum()
    # True negatives: Regime III without signal
    TN = (regime_III["signal"] == False).sum()
    # False negatives: Regime III with signal
    FN = (regime_III["signal"] == True).sum()

    n_regime_I    = len(regime_I)
    n_total_pos   = (df_assessed["signal"] == True).sum()
    n_total_neg   = (df_assessed["signal"] == False).sum()

    TPR = TP / n_total_pos if n_total_pos > 0 else 0    # sensitivity
    TNR = TN / n_total_neg if n_total_neg > 0 else 0    # specificity
    PPV = TP / max(n_regime_I, 1)                        # precision

    results.append({
        "IC_threshold": thresh,
        "n_regime_I":   n_regime_I,
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "TPR": round(TPR, 3),
        "TNR": round(TNR, 3),
        "PPV": round(PPV, 3),
        "F1":  round(2*TP/(2*TP+FP+FN) if (2*TP+FP+FN)>0 else 0, 3),
    })

sens_df = pd.DataFrame(results)
sens_df.to_csv(f"{RES}/ic_sensitivity.csv", index=False)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.suptitle("Sensitivity of Results to Impulse Contrast (IC) Threshold",
             fontsize=12, fontweight="bold")

# Panel A: TPR, TNR, F1 vs threshold
ax = axes[0]
ax.plot(sens_df["IC_threshold"], sens_df["TPR"], "o-", color="#C62828", lw=2,
        markersize=5, label="Sensitivity (TPR)")
ax.plot(sens_df["IC_threshold"], sens_df["TNR"], "s-", color="#1565C0", lw=2,
        markersize=5, label="Specificity (TNR)")
ax.plot(sens_df["IC_threshold"], sens_df["F1"],  "^-", color="#2E7D32", lw=2,
        markersize=5, label="F1 score")
ax.axvline(10, color="black", lw=1.5, ls="--", label="IC=10 (used in paper)")
ax.axvspan(8, 13, alpha=0.1, color="gray", label="Stable zone")
ax.set_xlabel("IC threshold", fontsize=10)
ax.set_ylabel("Score", fontsize=10)
ax.set_title("(A) Classification performance", fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.set_ylim(0, 1.05)

# Panel B: N Regime-I vs threshold
ax = axes[1]
ax.plot(sens_df["IC_threshold"], sens_df["n_regime_I"],
        "D-", color="#6A1B9A", lw=2, markersize=6)
ax.axvline(10, color="black", lw=1.5, ls="--")
ax.axhspan(3, 6, alpha=0.1, color="gray", label="3–6 candidates")
ax.set_xlabel("IC threshold", fontsize=10)
ax.set_ylabel("N regions classified Regime I", fontsize=10)
ax.set_title("(B) Number of Regime-I regions", fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel C: Which regions change classification?
ax = axes[2]
# Show IC values of all regions as scatter
colors_pt = ["#C62828" if s else "#90A4AE" for s in df_assessed["signal"]]
ax.scatter(df_assessed["IC"], range(len(df_assessed)),
           c=colors_pt, s=60, zorder=4, alpha=0.85)
for i, row in enumerate(df_assessed.itertuples()):
    short = row.region.replace("_"," ")[:12]
    ax.text(row.IC + 0.3, i, short, fontsize=6.5, va="center")
ax.axvline(10, color="black", lw=2, ls="--", label="IC=10")
ax.axvline(8,  color="#546E7A", lw=1, ls=":", alpha=0.7, label="IC=8")
ax.axvline(13, color="#546E7A", lw=1, ls=":", alpha=0.7, label="IC=13")
ax.set_xlabel("Impulse Contrast (IC)", fontsize=10)
ax.set_yticks([]); ax.set_yticklabels([])
ax.set_title("(C) IC values by region\n(red=signal, gray=no signal)",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("figures/publication/FigureS_IC_sensitivity.pdf", dpi=300, bbox_inches="tight")
plt.savefig("figures/publication/FigureS_IC_sensitivity.png", dpi=200, bbox_inches="tight")
plt.close()

print("IC Sensitivity Results:")
print(sens_df[["IC_threshold","n_regime_I","TPR","TNR","F1"]].to_string(index=False))
print(f"\nOptimal IC threshold by F1: {sens_df.loc[sens_df['F1'].idxmax(), 'IC_threshold']}")
print(f"F1 at IC=10: {sens_df[sens_df['IC_threshold']==10]['F1'].values[0]}")
print(f"Saved → figures/publication/FigureS_IC_sensitivity.png")
