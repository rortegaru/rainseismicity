"""
22_negative_validation.py
=========================
Reviewer response: formal negative control.

For each of the 4 confirmed Regime I regions, apply the SAME rate-ratio
test (λ_after / λ_before) to 200 randomly drawn "pseudo-rain dates"
sampled from the pre-event catalog window.

If the pipeline detects structure and not noise:
  - The distribution of random λ-ratios should cluster near 1.0
  - The actual rain date λ-ratio should be an extreme outlier

Output: one panel per region (4 panels) showing:
  - Histogram of 200 random λ-ratios
  - Vertical red line = actual λ-ratio from the paper
  - p-value = fraction of random ratios >= actual ratio (empirical one-sided)
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
FIG_DIR  = os.path.join(BASE, "figures", "publication")
os.makedirs(FIG_DIR, exist_ok=True)

# Confirmed Regime I cases with their actual lambda ratios and rain dates
CASES = {
    "noto_japan":      {"file": "noto_japan.csv",
                        "rain_date": "2021-09-16",
                        "actual_ratio": 1.99,
                        "window_days": 90,
                        "title": "Noto JP"},
    "los_cabos_mx":    {"file": "los_cabos_mx.csv",
                        "rain_date": "2024-09-14",
                        "actual_ratio": 2.31,   # ETAS R (Bayesian confounded by pre-swarm)
                        "window_days": 90,
                        "title": "Los Cabos MX\n(ETAS R = 2.31)"},
    "corinth_greece":  {"file": "corinth_greece.csv",
                        "rain_date": "2020-09-19",
                        "actual_ratio": 5.75,
                        "window_days": 90,
                        "title": "Corinth GR"},
}

N_RANDOM = 200
WINDOW   = 90    # days before and after pseudo-date

def compute_ratio(times_days, pseudo_date_day, window=90):
    """λ_after / λ_before for a given pseudo-date in fractional days."""
    before = times_days[(times_days >= pseudo_date_day - window) &
                        (times_days < pseudo_date_day)]
    after  = times_days[(times_days >  pseudo_date_day) &
                        (times_days <= pseudo_date_day + window)]
    n_b = len(before)
    n_a = len(after)
    if n_b < 3 or n_a < 3:
        return np.nan
    return (n_a / window) / (n_b / window)

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})

fig, axes = plt.subplots(1, 3, figsize=(11, 4.2),
                         gridspec_kw={"wspace": 0.40})

for ax, (region, cfg) in zip(axes, CASES.items()):
    fpath = os.path.join(DATA_DIR, cfg["file"])
    if not os.path.exists(fpath):
        ax.set_title(f"{cfg['title']}\n(no catalog)", fontsize=8)
        continue

    cat = pd.read_csv(fpath, parse_dates=["time"])
    cat = cat.sort_values("time").reset_index(drop=True)

    rain_dt = pd.Timestamp(cfg["rain_date"])
    t0_num  = (cat["time"] - cat["time"].min()).dt.total_seconds() / 86400

    # Rain date in the same units
    rain_day = (rain_dt - cat["time"].min()).total_seconds() / 86400

    # Sample pseudo-dates from anywhere in the catalog EXCEPT the event window
    # Avoidance zone: rain_day ± 90 days
    AVOID = 90
    all_days = t0_num.values
    t_min, t_max = all_days.min() + WINDOW, all_days.max() - WINDOW
    if t_max - t_min < WINDOW:
        ax.set_title(f"{cfg['title']}\n(catalog too short)", fontsize=8)
        continue
    # Sample uniformly, reject if within AVOID days of rain_day
    candidates = []
    attempts   = 0
    while len(candidates) < N_RANDOM and attempts < 50000:
        pd_try = np.random.uniform(t_min, t_max)
        if abs(pd_try - rain_day) > AVOID:
            candidates.append(pd_try)
        attempts += 1
    pseudo_days = np.array(candidates)

    ratios = np.array([compute_ratio(t0_num.values, p) for p in pseudo_days])
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]

    # Empirical p-value
    actual = cfg["actual_ratio"]
    p_emp  = np.mean(ratios >= actual)

    # Plot
    ax.hist(ratios, bins=25, color="#90A4AE", edgecolor="white",
            linewidth=0.5, density=True, alpha=0.85, label=f"Random dates\n(N={len(ratios)})")
    ax.axvline(actual, color="#D32F2F", lw=2.0, ls="-",
               label=f"Actual rain date\n(λ = {actual:.2f}×)")
    ax.axvline(1.0, color="#555", lw=0.8, ls="--", alpha=0.6)

    ax.set_xlabel(r"$\lambda_{\rm after}/\lambda_{\rm before}$", fontsize=8.5)
    ax.set_ylabel("Density" if ax == axes[0] else "", fontsize=8.5)
    ax.set_title(cfg["title"], fontsize=9, fontweight="bold")

    # p-value annotation
    n_r = len(ratios)
    p_str = f"p = {p_emp:.3f}" if p_emp > 0 else (f"p < {1/n_r:.3f}" if n_r > 0 else "p < 0.005")
    ax.text(0.97, 0.95, p_str, transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5, color="#D32F2F",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", lw=0.8))

    ax.legend(fontsize=7, framealpha=0.85, loc="upper left")

fig.suptitle(
    f"Negative control: {N_RANDOM} random pseudo-rain dates vs. actual rain date signal",
    fontsize=9.5, y=1.02)

for ext in ("pdf","png"):
    plt.savefig(os.path.join(FIG_DIR, f"FigureS_negative_validation.{ext}"),
                dpi=300 if ext=="pdf" else 200, bbox_inches="tight")
plt.close()
print("FigureS_negative_validation saved")
