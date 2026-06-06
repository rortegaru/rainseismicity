"""
Regenerate Bayesian switchpoint SI figures in English.
Runs MCMC for the 7 regions referenced in supporting_information.tex (Figs S9-S16).
Output: figures/SI_english/{region}_switchpoint.png
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

warnings.filterwarnings("ignore")

DATA_DIR = "data"
OUT_DIR  = "figures/SI_english"
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLES = 2000
TUNE    = 1000

# 9 regions with manual Darcy analysis — matches SI Figs S10-S18
# Taiwan removed; apennines, marlborough, calabria added
REGIONS = [
    "los_cabos_mx",
    "noto_japan",
    "corinth_greece",
    "murcia_spain",
    "pyrenees_fr",
    "reykjanes_iceland",
    "apennines_italy",
    "marlborough_nz",
    "calabria_italy",
]


def load_and_bin(csv_path, bin_days=1):
    df = pd.read_csv(csv_path, parse_dates=["time", "rain_date"])
    rain_date = pd.to_datetime(df["rain_date"].iloc[0])
    df["day_offset"] = (df["time"].dt.normalize() - rain_date).dt.days
    day_min   = int(df["day_offset"].min())
    day_max   = int(df["day_offset"].max())
    day_range = np.arange(day_min, day_max + 1)
    counts    = df.groupby("day_offset").size().reindex(day_range, fill_value=0).values
    return day_range, counts, rain_date


def run_switchpoint(counts, day_range, region_name):
    n = len(counts)
    with pm.Model() as model:
        lambda_1 = pm.Exponential("lambda_before", lam=1.0 / (counts.mean() + 1e-6))
        lambda_2 = pm.Exponential("lambda_after",  lam=1.0 / (counts.mean() + 1e-6))
        tau_idx  = pm.DiscreteUniform("tau_idx", lower=0, upper=n - 1)
        idx  = np.arange(n)
        rate = pm.math.switch(tau_idx >= idx, lambda_1, lambda_2)
        obs  = pm.Poisson("obs", mu=rate, observed=counts)
        trace = pm.sample(SAMPLES, tune=TUNE, cores=2, chains=2,
                          progressbar=False)
    return trace


def plot_result(counts, day_range, trace, region_name, rain_date, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f"{region_name}  |  rain: {rain_date.date()}",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.bar(day_range, counts, color="steelblue", alpha=0.7, width=1)
    ax.axvline(0, color="red", lw=2, ls="--", label="rain")
    ax.set_ylabel("Events/day")
    ax.set_xlabel("Days relative to rain")
    ax.legend()

    ax = axes[1]
    tau_samples     = trace.posterior["tau_idx"].values.flatten()
    tau_day_samples = day_range[tau_samples.astype(int)]
    ax.hist(tau_day_samples, bins=60, color="darkorange", alpha=0.8, density=True)
    ax.axvline(0, color="red", lw=2, ls="--", label="rain day")
    ax.axvline(float(np.median(tau_day_samples)), color="black", lw=2,
               label=f"τ median={np.median(tau_day_samples):.0f}d")
    ax.set_xlabel("Days relative to rain (switchpoint τ)")
    ax.set_ylabel("Posterior density")
    ax.legend()

    ax = axes[2]
    lb = trace.posterior["lambda_before"].values.flatten()
    la = trace.posterior["lambda_after"].values.flatten()
    ax.hist(lb, bins=60, alpha=0.6, color="steelblue",
            density=True, label=f"λ_before  μ={lb.mean():.2f}")
    ax.hist(la, bins=60, alpha=0.6, color="firebrick",
            density=True, label=f"λ_after   μ={la.mean():.2f}")
    ax.set_xlabel("Rate (events/day)")
    ax.set_ylabel("Posterior density")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


for region in REGIONS:
    csv_path = f"{DATA_DIR}/{region}.csv"
    if not os.path.exists(csv_path):
        print(f"SKIP (no CSV): {region}")
        continue

    print(f"\n{'='*60}")
    print(f"  {region}")
    try:
        day_range, counts, rain_date = load_and_bin(csv_path)
        print(f"  {counts.sum()} events, {len(counts)} days")
        if counts.sum() < 10:
            print("  SKIP: too few events")
            continue
        trace    = run_switchpoint(counts, day_range, region)
        out_path = f"{OUT_DIR}/{region}_switchpoint.png"
        plot_result(counts, day_range, trace, region, rain_date, out_path)
        print(f"  → {out_path}")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\nDone.")
