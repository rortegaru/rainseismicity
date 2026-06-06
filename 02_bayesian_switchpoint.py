"""
Bayesian switchpoint analysis for each region.
Model: Poisson process with rate that switches at tau days after rainfall.
Produces: posterior of lambda_before, lambda_after, tau (lag).
"""
import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

warnings.filterwarnings("ignore")

DATA_DIR   = "data"
FIG_DIR    = "figures"
RES_DIR    = "results"
for d in [FIG_DIR, RES_DIR]:
    os.makedirs(d, exist_ok=True)

SAMPLES = 2000
TUNE    = 1000


def load_and_bin(csv_path, bin_days=1):
    """Load catalog CSV, bin events per day, return arrays."""
    df = pd.read_csv(csv_path, parse_dates=["time", "rain_date"])
    rain_date = pd.to_datetime(df["rain_date"].iloc[0])

    df["day_offset"] = (df["time"].dt.normalize() - rain_date).dt.days
    total_span = int(df["day_offset"].max() - df["day_offset"].min()) + 1
    day_min    = int(df["day_offset"].min())
    day_max    = int(df["day_offset"].max())

    day_range  = np.arange(day_min, day_max + 1)
    counts     = df.groupby("day_offset").size().reindex(day_range, fill_value=0).values
    return day_range, counts, rain_date


def run_switchpoint(counts, day_range, region_name):
    """
    Bayesian switchpoint: tau is the day (offset from rain) where rate changes.
    tau is free — model finds it. We'll see if it clusters near day 0+ (post-rain).
    """
    n = len(counts)

    with pm.Model() as model:
        lambda_1 = pm.Exponential("lambda_before", lam=1.0 / (counts.mean() + 1e-6))
        lambda_2 = pm.Exponential("lambda_after",  lam=1.0 / (counts.mean() + 1e-6))

        # switchpoint index (0..n-1), mapped back to day_range later
        tau_idx = pm.DiscreteUniform("tau_idx", lower=0, upper=n - 1)

        idx   = np.arange(n)
        rate  = pm.math.switch(tau_idx >= idx, lambda_1, lambda_2)
        obs   = pm.Poisson("obs", mu=rate, observed=counts)

        trace = pm.sample(
            SAMPLES, tune=TUNE, cores=2, chains=2,
            progressbar=False,
            nuts_sampler="nutpie" if False else "pymc",
        )

    # posterior summaries
    lb_mean  = float(trace.posterior["lambda_before"].mean())
    lb_hdi   = az.hdi(trace, var_names=["lambda_before"], hdi_prob=0.94)["lambda_before"].values
    la_mean  = float(trace.posterior["lambda_after"].mean())
    la_hdi   = az.hdi(trace, var_names=["lambda_after"],  hdi_prob=0.94)["lambda_after"].values

    tau_samples = trace.posterior["tau_idx"].values.flatten()
    tau_day_samples = day_range[tau_samples.astype(int)]
    tau_median = float(np.median(tau_day_samples))
    tau_hdi_d  = np.percentile(tau_day_samples, [3, 97])

    # rate change ratio
    ratio_samples = (trace.posterior["lambda_after"].values /
                     (trace.posterior["lambda_before"].values + 1e-9)).flatten()
    ratio_mean = float(np.mean(ratio_samples))
    ratio_hdi  = np.percentile(ratio_samples, [3, 97])

    # probability that rate INCREASED after rain
    p_increase = float(np.mean(trace.posterior["lambda_after"].values >
                                trace.posterior["lambda_before"].values))

    return dict(
        region          = region_name,
        n_events        = int(counts.sum()),
        lambda_before   = round(lb_mean, 3),
        lb_hdi_low      = round(lb_hdi[0], 3),
        lb_hdi_high     = round(lb_hdi[1], 3),
        lambda_after    = round(la_mean, 3),
        la_hdi_low      = round(la_hdi[0], 3),
        la_hdi_high     = round(la_hdi[1], 3),
        ratio_after_before = round(ratio_mean, 2),
        ratio_hdi_low   = round(ratio_hdi[0], 2),
        ratio_hdi_high  = round(ratio_hdi[1], 2),
        tau_median_days = round(tau_median, 1),
        tau_hdi_low     = round(tau_hdi_d[0], 1),
        tau_hdi_high    = round(tau_hdi_d[1], 1),
        p_rate_increase = round(p_increase, 3),
    ), trace


def plot_result(counts, day_range, trace, region_name, rain_date):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f"{region_name}  |  rain: {rain_date.date()}", fontsize=13, fontweight="bold")

    # --- panel 1: daily counts + rain line ---
    ax = axes[0]
    ax.bar(day_range, counts, color="steelblue", alpha=0.7, width=1)
    ax.axvline(0, color="red", lw=2, ls="--", label="rain")
    ax.set_ylabel("Events/day")
    ax.set_xlabel("Days relative to rain")
    ax.legend()

    # --- panel 2: tau posterior ---
    ax = axes[1]
    tau_samples = trace.posterior["tau_idx"].values.flatten()
    tau_day_samples = day_range[tau_samples.astype(int)]
    ax.hist(tau_day_samples, bins=60, color="darkorange", alpha=0.8, density=True)
    ax.axvline(0, color="red", lw=2, ls="--", label="rain day")
    ax.axvline(float(np.median(tau_day_samples)), color="black", lw=2,
               label=f"τ median={np.median(tau_day_samples):.0f}d")
    ax.set_xlabel("Days relative to rain (switchpoint τ)")
    ax.set_ylabel("Posterior density")
    ax.legend()

    # --- panel 3: lambda before/after posteriors ---
    ax = axes[2]
    lb = trace.posterior["lambda_before"].values.flatten()
    la = trace.posterior["lambda_after"].values.flatten()
    ax.hist(lb, bins=60, alpha=0.6, color="steelblue",  density=True, label=f"λ_before  μ={lb.mean():.2f}")
    ax.hist(la, bins=60, alpha=0.6, color="firebrick",  density=True, label=f"λ_after   μ={la.mean():.2f}")
    ax.set_xlabel("Rate (events/day)")
    ax.set_ylabel("Posterior density")
    ax.legend()

    plt.tight_layout()
    path = f"{FIG_DIR}/{region_name}_switchpoint.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"   Figura → {path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
results = []

csv_files = sorted(glob.glob(f"{DATA_DIR}/*.csv"))
csv_files = [f for f in csv_files if "summary" not in f]

if not csv_files:
    print("No hay datos descargados todavía. Corre primero 01_download_isc.py")
else:
    for csv_path in csv_files:
        region_name = os.path.basename(csv_path).replace(".csv", "")
        print(f"\n{'='*60}")
        print(f"  Procesando: {region_name}")
        try:
            day_range, counts, rain_date = load_and_bin(csv_path)
            print(f"  {counts.sum()} eventos en {len(counts)} días")

            if counts.sum() < 10:
                print("  SKIP: muy pocos eventos")
                results.append({"region": region_name, "status": "pocos_eventos (<10)"})
                continue

            res, trace = run_switchpoint(counts, day_range, region_name)
            plot_result(counts, day_range, trace, region_name, rain_date)
            res["status"] = "OK"
            results.append(res)

            print(f"  λ_before={res['lambda_before']} [{res['lb_hdi_low']}-{res['lb_hdi_high']}]  "
                  f"λ_after={res['lambda_after']} [{res['la_hdi_low']}-{res['la_hdi_high']}]")
            print(f"  ratio={res['ratio_after_before']}  "
                  f"τ={res['tau_median_days']}d  P(aumento)={res['p_rate_increase']}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"region": region_name, "status": f"ERROR: {str(e)[:100]}"})

    df_res = pd.DataFrame(results)
    df_res.to_csv(f"{RES_DIR}/switchpoint_results.csv", index=False)
    print(f"\n\nResultados guardados → {RES_DIR}/switchpoint_results.csv")
