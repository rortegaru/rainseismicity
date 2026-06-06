"""
Zaliapin & Ben-Zion (2013) nearest-neighbor clustering.
For each region catalog, classifies events as background vs. cluster.
Adds classification column to each CSV.
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "data"
FIG_DIR  = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ── Nearest-neighbor distance (Zaliapin & Ben-Zion 2013) ─────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi  = np.radians(lat2 - lat1)
    dlam  = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def nearest_neighbor_distance(df, b=1.0, df_frac=1.6):
    """
    Compute rescaled nearest-neighbor distance eta_ij for each event.
    eta_ij = T_ij * R_ij^df_frac * 10^(-b*mi)
    Returns array of eta (distance to nearest preceding neighbor).
    """
    times = df["time_decimal"].values
    lats  = df["lat"].values
    lons  = df["lon"].values
    mags  = df["mag"].values

    n    = len(df)
    eta  = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=int)

    for i in range(1, n):
        dt  = times[i] - times[:i]        # years
        dt[dt <= 0] = 1e-10
        rij = np.array([haversine_km(lats[i], lons[i], lats[j], lons[j])
                         for j in range(i)])
        rij[rij <= 0] = 0.01

        eta_ij = dt * rij**df_frac * 10**(-b * mags[:i])
        idx_min = np.argmin(eta_ij)
        eta[i]  = eta_ij[idx_min]
        parent[i] = idx_min

    return eta, parent


def classify_clusters(eta, threshold_percentile=10):
    """
    Log-space threshold: events below threshold are 'clustered', rest 'background'.
    Default: bottom 10th percentile in log(eta) → clustered.
    """
    log_eta = np.log10(eta[1:])           # skip first event (no parent)
    thresh  = np.percentile(log_eta, threshold_percentile)
    is_cluster = np.concatenate([[False], log_eta <= thresh])
    return is_cluster, thresh


# ── b-value (simple MLE) ─────────────────────────────────────────────────────

def bvalue_mle(mags, mc):
    mags_above = mags[mags >= mc]
    if len(mags_above) < 20:
        return np.nan, np.nan
    b = np.log10(np.e) / (np.mean(mags_above) - mc + 0.05)
    sigma_b = b / np.sqrt(len(mags_above))
    return round(b, 3), round(sigma_b, 3)


# ── MAIN ──────────────────────────────────────────────────────────────────────

results = []
csv_files = sorted(glob.glob(f"{DATA_DIR}/*.csv"))
csv_files = [f for f in csv_files if "summary" not in f]

for csv_path in csv_files:
    region = os.path.basename(csv_path).replace(".csv", "")
    print(f"\n{'='*55}")
    print(f"  Zaliapin NND: {region}")

    df = pd.read_csv(csv_path, parse_dates=["time"])
    df = df.dropna(subset=["lat", "lon", "mag"]).copy()
    df = df.sort_values("time").reset_index(drop=True)

    if len(df) < 20:
        print(f"  SKIP: {len(df)} eventos")
        results.append({"region": region, "status": "muy pocos"})
        continue

    # decimal year for time metric
    t0_year = df["time"].iloc[0].year
    df["time_decimal"] = (
        df["time"].astype("int64") / 1e9 / (365.25 * 86400)
    )

    # magnitude completeness estimate (simple: mode of rounded mags)
    mc_est = round(df["mag"].quantile(0.25), 1)
    df_mc  = df[df["mag"] >= mc_est].reset_index(drop=True)
    print(f"  N total={len(df)}, Mc≈{mc_est}, N≥Mc={len(df_mc)}")

    if len(df_mc) < 20:
        print("  SKIP: muy pocos sobre Mc")
        results.append({"region": region, "status": "pocos sobre Mc"})
        continue

    try:
        b_val, b_err = bvalue_mle(df_mc["mag"].values, mc_est)
        print(f"  b-value = {b_val} ± {b_err}")

        eta, parent = nearest_neighbor_distance(df_mc, b=b_val if not np.isnan(b_val) else 1.0)
        is_cluster, log_thresh = classify_clusters(eta, threshold_percentile=10)

        df_mc["nnd_eta"]      = eta
        df_mc["is_cluster"]   = is_cluster
        df_mc["cluster_type"] = np.where(is_cluster, "cluster", "background")

        n_cluster  = is_cluster.sum()
        n_bg       = (~is_cluster).sum()
        frac_clust = n_cluster / len(df_mc)

        # ── figure: log(eta) distribution ──
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"{region}  —  Zaliapin NND clustering", fontsize=12)

        ax = axes[0]
        log_eta_vals = np.log10(eta[1:])
        ax.hist(log_eta_vals, bins=50, color="steelblue", alpha=0.8)
        ax.axvline(log_thresh, color="red", lw=2, ls="--",
                   label=f"threshold log η={log_thresh:.2f}")
        ax.set_xlabel("log₁₀(η) — nearest-neighbor distance")
        ax.set_ylabel("N events")
        ax.legend()
        ax.set_title(f"b={b_val}  Mc={mc_est}")

        ax = axes[1]
        bg  = df_mc[~df_mc["is_cluster"]]
        cl  = df_mc[df_mc["is_cluster"]]
        ax.scatter(bg["lon"], bg["lat"], s=4, c="steelblue", alpha=0.4, label="background")
        ax.scatter(cl["lon"], cl["lat"], s=8, c="firebrick", alpha=0.8, label="cluster")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.legend(markerscale=2)
        ax.set_title(f"Clusters: {n_cluster} ({frac_clust:.0%})  Background: {n_bg}")

        plt.tight_layout()
        fig_path = f"{FIG_DIR}/{region}_zaliapin.png"
        plt.savefig(fig_path, dpi=120)
        plt.close()
        print(f"  Clusters={n_cluster} ({frac_clust:.1%})  Background={n_bg} → {fig_path}")

        # save enriched catalog
        out_path = csv_path.replace(".csv", "_classified.csv")
        df_mc.to_csv(out_path, index=False)

        results.append({
            "region":         region,
            "n_total":        len(df),
            "mc":             mc_est,
            "n_above_mc":     len(df_mc),
            "b_value":        b_val,
            "b_error":        b_err,
            "n_cluster":      int(n_cluster),
            "n_background":   int(n_bg),
            "frac_clustered": round(frac_clust, 3),
            "log_eta_thresh": round(log_thresh, 3),
            "status":         "OK",
        })

    except Exception as e:
        print(f"  ERROR: {e}")
        results.append({"region": region, "status": f"ERROR: {str(e)[:100]}"})

df_res = pd.DataFrame(results)
df_res.to_csv("results/zaliapin_results.csv", index=False)
print("\n\n=== ZALIAPIN CLUSTERING RESULTS ===")
print(df_res.to_string(index=False))
