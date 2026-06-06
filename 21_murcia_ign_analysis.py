"""
Supplemental analysis for Murcia Spain using the IGN catalog (Mc~0.9).
Standalone script — reads data/murcia_ign.csv, writes only murcia_ign_* files.
Does NOT touch any existing murcia_spain_* or shared results CSVs.

Produces:
  figures/murcia_ign_spacetime.png
  figures/murcia_ign_zaliapin.png
  figures/murcia_ign_switchpoint.png
  data/murcia_ign_classified.csv
  results/murcia_ign_results.csv

Requires Roberto for scripts 15-17 (movie + interactive r²-t).
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pymc as pm
import arviz as az
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

CSV_IN   = "data/murcia_ign.csv"
FIG_DIR  = "figures"
RES_DIR  = "results"
REGION   = "murcia_ign"
MC       = 0.9
RAIN_STR = "2019-09-13"
RAIN_LAB = "DANA storm\nSept 13, 2019"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"  Murcia IGN supplemental analysis")
print(f"{'='*60}")

# ── load ──────────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(CSV_IN, parse_dates=["time"])
df_raw = df_raw.dropna(subset=["lat","lon","mag"]).sort_values("time").reset_index(drop=True)
rain_date = pd.Timestamp(RAIN_STR)
print(f"  Loaded {len(df_raw)} events  ({df_raw['time'].min().date()} → {df_raw['time'].max().date()})")

df_raw["day_offset"] = (df_raw["time"].dt.normalize() - rain_date).dt.days

# apply Mc filter
df = df_raw[df_raw["mag"] >= MC].reset_index(drop=True)
print(f"  After Mc={MC}: {len(df)} events")


# ══════════════════════════════════════════════════════════════════════════════
# 1. SPACETIME / DEPTH PLOT
# ══════════════════════════════════════════════════════════════════════════════
print("\n  [1/3] Spacetime-depth figure...")

# center for epicentral distance
lat_c = df["lat"].median()
lon_c = df["lon"].median()
R_EARTH = 6371.0

def epi_dist_km(lat, lon, lat0, lon0):
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat0))*np.cos(np.radians(lat))*np.sin(dlon/2)**2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))

df["r_km"] = epi_dist_km(df["lat"].values, df["lon"].values, lat_c, lon_c)

# diffusion envelopes (r²=4Dt)
D_VALUES  = [0.5, 5.0, 50.0]
D_COLORS  = ["#2196F3", "#FF9800", "#E91E63"]
D_LABELS  = ["D=0.5 m²/s", "D=5 m²/s", "D=50 m²/s"]
t_arr = np.linspace(1, 300, 500)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle(f"Murcia Spain (IGN, Mc={MC})  |  rain: {RAIN_STR}", fontsize=13, fontweight="bold")

# panel A: depth-time
ax = axes[0, 0]
pre  = df[df["day_offset"] < 0]
post = df[df["day_offset"] >= 0]
ax.scatter(pre["day_offset"],  pre["depth"],  s=4,  c="steelblue", alpha=0.5, label="pre-rain")
ax.scatter(post["day_offset"], post["depth"], s=6,  c="firebrick",  alpha=0.7, label="post-rain")
ax.axvline(0, color="red", lw=2, ls="--", label="rain")
ax.invert_yaxis()
ax.set_xlabel("Days relative to rain")
ax.set_ylabel("Depth (km)")
ax.set_title("Depth–time")
ax.legend(fontsize=8, markerscale=2)

# panel B: r-time with diffusion envelopes
ax = axes[0, 1]
ax.scatter(pre["day_offset"],  pre["r_km"],  s=4,  c="steelblue", alpha=0.5)
ax.scatter(post["day_offset"], post["r_km"], s=6,  c="firebrick",  alpha=0.7)
ax.axvline(0, color="red", lw=2, ls="--")
for D, col, lbl in zip(D_VALUES, D_COLORS, D_LABELS):
    r_env = np.sqrt(4 * D * t_arr * 86400) / 1000.0
    ax.plot(t_arr, r_env, color=col, lw=1.5, ls="--", label=lbl)
ax.set_xlabel("Days relative to rain")
ax.set_ylabel("Epicentral distance (km)")
ax.set_title("Distance–time (r²-t envelopes)")
ax.set_xlim(-120, 200)
ax.set_ylim(0, 100)
ax.legend(fontsize=7)

# panel C: daily rate
ax = axes[1, 0]
day_min = int(df["day_offset"].min())
day_max = int(df["day_offset"].max())
day_range = np.arange(day_min, day_max + 1)
counts = df.groupby("day_offset").size().reindex(day_range, fill_value=0).values
ax.bar(day_range, counts, color="steelblue", alpha=0.7, width=1)
ax.axvline(0, color="red", lw=2, ls="--", label="rain")
ax.set_xlabel("Days relative to rain")
ax.set_ylabel("Events/day")
ax.set_title("Daily seismicity rate")
ax.legend()
lambda_pre  = counts[day_range < 0].mean()
lambda_post = counts[day_range >= 0].mean()
ax.axhline(lambda_pre,  color="steelblue", lw=1.5, ls=":", alpha=0.8, label=f"λ_pre={lambda_pre:.2f}")
ax.axhline(lambda_post, color="firebrick",  lw=1.5, ls=":", alpha=0.8, label=f"λ_post={lambda_post:.2f}")
ax.legend(fontsize=8)

# panel D: map
ax = axes[1, 1]
sc = ax.scatter(pre["lon"],  pre["lat"],  s=4,  c="steelblue", alpha=0.4, label="pre-rain")
sc = ax.scatter(post["lon"], post["lat"], s=8,  c="firebrick",  alpha=0.7, label="post-rain")
ax.scatter([-1.5], [37.8], marker="*", s=200, c="red", zorder=5, label="rain epicenter")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title("Map")
ax.legend(fontsize=8, markerscale=1.5)

plt.tight_layout()
fig_path = f"{FIG_DIR}/murcia_ign_spacetime.png"
plt.savefig(fig_path, dpi=130)
plt.close()
print(f"    → {fig_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. ZALIAPIN CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n  [2/3] Zaliapin NND clustering...")

def haversine_km(lat1, lon1, lat2, lon2):
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))

def nearest_neighbor_distance(df_in, b=1.0, df_frac=1.6):
    times = df_in["time_decimal"].values
    lats  = df_in["lat"].values
    lons  = df_in["lon"].values
    mags  = df_in["mag"].values
    n     = len(df_in)
    eta   = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=int)
    for i in range(1, n):
        dt = times[i] - times[:i]
        dt[dt <= 0] = 1e-10
        rij = np.array([haversine_km(lats[i], lons[i], lats[j], lons[j]) for j in range(i)])
        rij[rij <= 0] = 0.01
        eta_ij = dt * rij**df_frac * 10**(-b * mags[:i])
        idx_min = np.argmin(eta_ij)
        eta[i] = eta_ij[idx_min]
        parent[i] = idx_min
    return eta, parent

def bvalue_mle(mags, mc):
    mags_a = mags[mags >= mc]
    if len(mags_a) < 20:
        return np.nan, np.nan
    b = np.log10(np.e) / (np.mean(mags_a) - mc + 0.05)
    return round(b, 3), round(b / np.sqrt(len(mags_a)), 3)

df["time_decimal"] = df["time"].astype("int64") / 1e9 / (365.25 * 86400)
b_val, b_err = bvalue_mle(df["mag"].values, MC)
print(f"    b-value = {b_val} ± {b_err}  (Mc={MC}, N={len(df)})")

eta, parent = nearest_neighbor_distance(df, b=b_val if not np.isnan(b_val) else 1.0)
log_eta = np.log10(eta[1:])
log_thresh = np.percentile(log_eta, 10)
is_cluster = np.concatenate([[False], log_eta <= log_thresh])

df["nnd_eta"]      = eta
df["is_cluster"]   = is_cluster
df["cluster_type"] = np.where(is_cluster, "cluster", "background")

n_cluster  = is_cluster.sum()
n_bg       = (~is_cluster).sum()
frac_clust = n_cluster / len(df)
print(f"    Clusters={n_cluster} ({frac_clust:.1%})  Background={n_bg}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"Murcia Spain (IGN)  —  Zaliapin NND  |  b={b_val}  Mc={MC}", fontsize=12)

ax = axes[0]
ax.hist(log_eta, bins=50, color="steelblue", alpha=0.8)
ax.axvline(log_thresh, color="red", lw=2, ls="--", label=f"threshold log η={log_thresh:.2f}")
ax.set_xlabel("log₁₀(η) — nearest-neighbor distance")
ax.set_ylabel("N events")
ax.legend()

ax = axes[1]
bg = df[~df["is_cluster"]]
cl = df[df["is_cluster"]]
ax.scatter(bg["lon"], bg["lat"], s=4,  c="steelblue", alpha=0.4, label="background")
ax.scatter(cl["lon"], cl["lat"], s=8,  c="firebrick",  alpha=0.8, label="cluster")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.legend(markerscale=2)
ax.set_title(f"Clusters: {n_cluster} ({frac_clust:.0%})  Background: {n_bg}")

plt.tight_layout()
fig_path = f"{FIG_DIR}/murcia_ign_zaliapin.png"
plt.savefig(fig_path, dpi=120)
plt.close()
print(f"    → {fig_path}")

df.to_csv("data/murcia_ign_classified.csv", index=False)
print(f"    → data/murcia_ign_classified.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 3. BAYESIAN SWITCHPOINT
# ══════════════════════════════════════════════════════════════════════════════
print("\n  [3/3] Bayesian switchpoint (PyMC)...")

SAMPLES = 2000
TUNE    = 1000

counts  = df.groupby("day_offset").size().reindex(day_range, fill_value=0).values
n_days  = len(counts)

with pm.Model() as model:
    lambda_1 = pm.Exponential("lambda_before", lam=1.0 / (counts.mean() + 1e-6))
    lambda_2 = pm.Exponential("lambda_after",  lam=1.0 / (counts.mean() + 1e-6))
    tau_idx  = pm.DiscreteUniform("tau_idx", lower=0, upper=n_days - 1)
    idx  = np.arange(n_days)
    rate = pm.math.switch(tau_idx >= idx, lambda_1, lambda_2)
    obs  = pm.Poisson("obs", mu=rate, observed=counts)
    trace = pm.sample(SAMPLES, tune=TUNE, cores=2, chains=2, progressbar=True)

tau_samples     = trace.posterior["tau_idx"].values.flatten()
tau_day_samples = day_range[tau_samples.astype(int)]
tau_median      = float(np.median(tau_day_samples))
tau_hdi         = np.percentile(tau_day_samples, [3, 97])

lb_mean = float(trace.posterior["lambda_before"].mean())
la_mean = float(trace.posterior["lambda_after"].mean())
ratio_s = (trace.posterior["lambda_after"].values /
           (trace.posterior["lambda_before"].values + 1e-9)).flatten()
ratio   = float(np.mean(ratio_s))
p_inc   = float(np.mean(trace.posterior["lambda_after"].values >
                         trace.posterior["lambda_before"].values))

print(f"    λ_before={lb_mean:.3f}  λ_after={la_mean:.3f}  ratio={ratio:.2f}  τ={tau_median:.0f}d  P={p_inc:.3f}")

fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle(f"Murcia Spain (IGN Mc={MC})  |  Bayesian switchpoint  |  rain: {RAIN_STR}",
             fontsize=13, fontweight="bold")

ax = axes[0]
ax.bar(day_range, counts, color="steelblue", alpha=0.7, width=1)
ax.axvline(0, color="red", lw=2, ls="--", label="rain")
ax.axvline(tau_median, color="darkorange", lw=2, ls="-", label=f"τ={tau_median:.0f}d")
ax.set_ylabel("Events/day"); ax.set_xlabel("Days relative to rain"); ax.legend()

ax = axes[1]
ax.hist(tau_day_samples, bins=60, color="darkorange", alpha=0.8, density=True)
ax.axvline(0, color="red", lw=2, ls="--", label="rain day")
ax.axvline(tau_median, color="black", lw=2, label=f"τ median={tau_median:.0f}d")
ax.set_xlabel("Days relative to rain (switchpoint τ)")
ax.set_ylabel("Posterior density"); ax.legend()

ax = axes[2]
lb_post = trace.posterior["lambda_before"].values.flatten()
la_post = trace.posterior["lambda_after"].values.flatten()
ax.hist(lb_post, bins=60, alpha=0.6, color="steelblue",
        density=True, label=f"λ_before  μ={lb_mean:.2f}")
ax.hist(la_post, bins=60, alpha=0.6, color="firebrick",
        density=True, label=f"λ_after   μ={la_mean:.2f}")
ax.set_xlabel("Rate (events/day)"); ax.set_ylabel("Posterior density"); ax.legend()

plt.tight_layout()
fig_path = f"{FIG_DIR}/murcia_ign_switchpoint.png"
plt.savefig(fig_path, dpi=120)
plt.close()
print(f"    → {fig_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. SAVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
summary = {
    "region":          REGION,
    "catalog":         "IGN",
    "mc":              MC,
    "n_total":         len(df_raw),
    "n_above_mc":      len(df),
    "b_value":         b_val,
    "b_error":         b_err,
    "n_cluster":       int(n_cluster),
    "frac_clustered":  round(frac_clust, 3),
    "lambda_before":   round(lb_mean, 3),
    "lambda_after":    round(la_mean, 3),
    "ratio":           round(ratio, 2),
    "tau_median_days": round(tau_median, 1),
    "tau_hdi_low":     round(tau_hdi[0], 1),
    "tau_hdi_high":    round(tau_hdi[1], 1),
    "p_rate_increase": round(p_inc, 3),
}

pd.DataFrame([summary]).to_csv(f"{RES_DIR}/murcia_ign_results.csv", index=False)
print(f"\n  Summary → {RES_DIR}/murcia_ign_results.csv")

print("\n" + "="*60)
print("  DONE. Next steps (require Roberto):")
print("  python 14_seismicity_movies.py  (add murcia_ign to CANDIDATES)")
print("  python 15_subregion_movie.py --region murcia_ign --post 180")
print("  python 16_pick_origin.py --region murcia_ign")
print("  python 17_pick_diffusivity.py")
print("="*60)
