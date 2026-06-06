"""
Regenerate Zaliapin SI figures in English from pre-computed _classified.csv.
Reads saved nnd_eta and is_cluster columns — no recomputation needed.
Output: figures/SI_english/{region}_zaliapin.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "data"
OUT_DIR  = "figures/SI_english"
os.makedirs(OUT_DIR, exist_ok=True)

# Only the 8 regions referenced in supporting_information.tex (Figs S17-S24)
REGIONS = [
    "los_cabos_mx",
    "noto_japan",
    "corinth_greece",
    "murcia_spain",
    "pyrenees_fr",
    "taiwan",
    "apennines_italy",
    "marlborough_nz",
]

for region in REGIONS:
    cl_path = f"{DATA_DIR}/{region}_classified.csv"
    if not os.path.exists(cl_path):
        print(f"SKIP (no _classified.csv): {region}")
        continue

    df = pd.read_csv(cl_path)
    print(f"Processing {region}: N={len(df)}")

    eta = df["nnd_eta"].values
    log_eta = np.log10(eta[eta > 0])
    log_eta = log_eta[np.isfinite(log_eta)]
    log_thresh = np.percentile(log_eta, 10)

    mc_est = round(df["mag"].quantile(0.25), 1)
    mags_above = df.loc[df["mag"] >= mc_est, "mag"].values
    if len(mags_above) > 1:
        b_val = round(1.0 / (np.log(10) * (np.mean(mags_above) - mc_est)), 3)
    else:
        b_val = float("nan")

    n_cluster  = int(df["is_cluster"].sum())
    n_bg       = int((~df["is_cluster"]).sum())
    frac_clust = n_cluster / len(df)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{region}  —  Zaliapin NND clustering", fontsize=12)

    ax = axes[0]
    ax.hist(log_eta, bins=50, color="steelblue", alpha=0.8)
    ax.axvline(log_thresh, color="red", lw=2, ls="--",
               label=f"threshold log η={log_thresh:.2f}")
    ax.set_xlabel("log₁₀(η) — nearest-neighbor distance")
    ax.set_ylabel("N events")
    ax.legend()
    ax.set_title(f"b={b_val}  Mc={mc_est}")

    ax = axes[1]
    bg = df[~df["is_cluster"]]
    cl = df[df["is_cluster"]]
    ax.scatter(bg["lon"], bg["lat"], s=4, c="steelblue", alpha=0.4, label="background")
    ax.scatter(cl["lon"], cl["lat"], s=8, c="firebrick", alpha=0.8, label="cluster")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(markerscale=2)
    ax.set_title(f"Clusters: {n_cluster} ({frac_clust:.0%})  Background: {n_bg}")

    plt.tight_layout()
    out_path = f"{OUT_DIR}/{region}_zaliapin.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  → {out_path}")

print("\nDone.")
