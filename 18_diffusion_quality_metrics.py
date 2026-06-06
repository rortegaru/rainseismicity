"""
18_diffusion_quality_metrics.py
================================
Computes quantitative quality metrics for each diffusion fit obtained
visually in scripts 16 (pick origin) and 17 (pick D with slider).

WHY THIS SCRIPT EXISTS
----------------------
Scripts 16-17 produced a diffusivity D for each region by visually fitting
the envelope r² = 4·D·t to the spatio-temporal seismicity pattern. That
process was rigorous — the spatial extent was chosen deliberately (shallow
events, visible migration) and the origin (X₀,Y₀,Z₀,t₀) was picked at the
initiating event. But the result is still a visual estimate.

This script adds two quantitative metrics that validate and characterize
each fit WITHOUT replacing the visual analysis:

  1. F_in  — Fraction of post-origin events inside the diffusion envelope.
             Tells you: how much of the migrating cluster is "explained" by
             this D.  Simple but depends on N (500/1000 = 0.5 is not the
             same confidence as 2/4 = 0.5).

  2. rho_median (ρ̃)  — Median normalized distance to the diffusion front:
                         ρᵢ = rᵢ / √(4·D·tᵢ)
             ρ < 1 → inside envelope, ρ > 1 → outside.
             This metric is NOT sensitive to N: it uses the actual position
             of every event relative to the front, so 4 well-placed events
             give a meaningful ρ̃ just as 400 do.
             Physical interpretation:
               ρ̃ ~ 0.0-0.3 → events near origin only, no clear migration
                              (D probably overestimated)
               ρ̃ ~ 0.5-0.85 → events track the diffusion front well
               ρ̃ ~ 0.85-1.0 → events tight to front, very clean diffusion
               ρ̃ > 1.0      → most events outside (D underestimated)

  3. D_regression  — Objective D estimate from linear regression r²= 4Dt
             through the origin, using all post-origin events inside the
             envelope. Gives a data-driven anchor to compare against the
             visual D_chosen.  NOT a replacement — just a cross-check.
             Note: a regression through ALL post-origin events (including
             those far from the front) systematically underestimates D
             because most events sit inside, not at the front. Use it as
             a lower bound and sanity check.

  4. R2_regression  — R² of that linear fit. Measures how well a straight
             line through the origin describes the r²–t scatter. R² close
             to 1 → clean diffusion pattern; low R² → noisy or mixed signal.

UNIT NOTES
----------
  r_3d        : km
  t_from_origin: days  (positive = after origin event)
  D_chosen    : m²/s  (stored in darcy_origins.csv)
  r_envelope  : km    (already computed in script 17 as √(4·D·t·86400)/1000)

  Conversion for regression:
    r² [km²] = 4 · D [m²/s] · t [days] · 86400 [s/day] / 10⁶ [m²/km²]
    r² [km²] = D [m²/s] · t [days] · 0.3456
    → D_fit [m²/s] = slope(r²[km²] / t[days]) / 0.3456

OUTPUT
------
  results/diffusion_quality_metrics.csv  — one row per subcatalog entry
  Console table with interpretation flags
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

# ── Paths ────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
SUB_DIR     = os.path.join(BASE, "results", "subcatalogs")
ORIGINS_CSV = os.path.join(BASE, "results", "darcy_origins.csv")
OUT_CSV     = os.path.join(BASE, "results", "diffusion_quality_metrics.csv")

# Unit conversion: r²[km²] = D[m²/s] * t[days] * CONV
CONV = 4 * 86400 / 1e6   # = 0.3456  km² / (m²/s · day)

# ── Cases flagged as invalid/discarded in the analysis ──────────────────────
# These are kept in darcy_origins.csv as a record but excluded from the
# paper's main results (D hit the slider ceiling or is physically impossible).
DISCARDED = {
    ("murcia_spain",    2): "ISC catalog insufficient; D=100 m²/s unphysical",
    ("calabria_italy",  2): "origin at 130 km depth, discarded (sub4 is current)",
    ("calabria_italy",  3): "superseded by sub4; D=6.22 underestimated vertical migration",
    ("calabria_italy",  4): "superseded by sub5 (refined box, D=42.23, rho=1.42)",
    ("apennines_italy", 1): "D hit slider ceiling (100 m²/s); two populations",
}

# ── Load darcy_origins ───────────────────────────────────────────────────────
origins = pd.read_csv(ORIGINS_CSV)

records = []

for _, row in origins.iterrows():
    region = row["region"]
    sub_n  = int(row["sub_n"])
    D      = float(row["D_chosen"])   # m²/s

    # Flag discarded cases — compute metrics anyway but mark them
    discarded_reason = DISCARDED.get((region, sub_n), None)

    # Load subcatalog
    fname = f"{region}_sub{sub_n}.csv"
    fpath = os.path.join(SUB_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  [MISSING] {fname}")
        continue

    df = pd.read_csv(fpath)

    # ── Filter to post-origin events only ───────────────────────────────────
    # t_from_origin > 0: event occurred AFTER the chosen origin event.
    # Events with t ≤ 0 are pre-rain background — they have r_envelope=0
    # and are meaningless for a diffusion metric that starts at t=0.
    post = df[df["t_from_origin"] > 0].copy()

    # Also require r_envelope > 0 (extra safety: avoids division by zero)
    post = post[post["r_envelope"] > 0]

    N_total_sub = len(df)          # all events in subcatalog (pre + post)
    N_post      = len(post)        # post-origin events used for metrics

    # N_early: events very close to origin in time (t < 1 day).
    # At t < 1 day the diffusion envelope r = √(4Dt) is still tiny (< 1–2 km
    # for typical D values) while the spatial extent can be 10–20 km across.
    # Events at large r and tiny t have ρ >> 1 not because D is wrong, but
    # because they were already in the spatial box: co-seismic response,
    # ongoing background, or events from a different cluster.
    # A high N_early / N_post ratio is a contamination warning for ρ̃.
    N_early = int(((post["t_from_origin"] > 0) & (post["t_from_origin"] < 1.0)).sum())

    if N_post < 3:
        print(f"  [SKIP] {fname}: only {N_post} post-origin events")
        records.append({
            "region": region, "sub_n": sub_n, "D_chosen": D,
            "N_total_sub": N_total_sub, "N_post": N_post,
            "F_in": np.nan, "rho_median": np.nan,
            "rho_q25": np.nan, "rho_q75": np.nan,
            "D_regression": np.nan, "R2_regression": np.nan,
            "status": "too few post-origin events",
        })
        continue

    # ── Metric 1: F_in ───────────────────────────────────────────────────────
    # Parse inside_envelope (stored as string "True"/"False" in CSV)
    inside = post["inside_envelope"].astype(str).str.strip().str.lower() == "true"
    F_in   = inside.sum() / N_post

    # ── Metric 2: ρ̃ — normalized distance to diffusion front ────────────────
    # ρᵢ = r_3d / r_envelope.  r_envelope = √(4·D·t·86400)/1000 [km]
    # Both columns already in the subcatalog CSV from script 17.
    rho    = post["r_3d"].values / post["r_envelope"].values
    rho_median = np.median(rho)
    rho_q25    = np.percentile(rho, 25)
    rho_q75    = np.percentile(rho, 75)

    # ── Metric 3 & 4: linear regression r² = 4·D·t ──────────────────────────
    # Use only inside-envelope events for the regression.
    # Outside events are likely unrelated background seismicity that leaked
    # into the spatial extent — including them would bias D downward.
    post_in = post[inside].copy()

    if len(post_in) >= 3:
        t_days = post_in["t_from_origin"].values
        r2_km2 = post_in["r_3d"].values ** 2

        # Regression through origin: slope = Σ(t·r²) / Σ(t²)
        slope      = np.dot(t_days, r2_km2) / np.dot(t_days, t_days)
        D_reg      = slope / CONV    # back to m²/s

        # R² relative to mean (standard interpretation)
        r2_pred    = slope * t_days
        ss_res     = np.sum((r2_km2 - r2_pred) ** 2)
        ss_tot     = np.sum((r2_km2 - r2_km2.mean()) ** 2)
        R2_reg     = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    else:
        D_reg  = np.nan
        R2_reg = np.nan

    # ── Quality tier ─────────────────────────────────────────────────────────
    # Automatic assessment combining F_in and ρ̃.
    # ρ̃ is the primary signal; F_in adds weight when N is large enough.
    # Also penalizes cases with heavy early contamination (N_early > 30% of N_post).
    early_frac = N_early / N_post if N_post > 0 else 0
    if discarded_reason:
        tier = "DISCARDED"
    elif early_frac > 0.30 or rho_median > 3.0:
        tier = "POOR"       # early contamination OR most events far outside
    elif rho_median > 1.5 or F_in < 0.40:
        tier = "MARGINAL"   # pattern present but weak
    elif rho_median <= 1.1 and F_in >= 0.60:
        tier = "GOOD"       # events track the diffusion front well
    else:
        tier = "OK"         # acceptable fit

    # ── Assemble record ──────────────────────────────────────────────────────
    status = discarded_reason if discarded_reason else "active"
    records.append({
        "region":       region,
        "sub_n":        sub_n,
        "D_chosen":     round(D, 3),
        "N_total_sub":  N_total_sub,
        "N_post":       N_post,
        "N_early":      N_early,
        "F_in":         round(F_in, 3),
        "rho_median":   round(rho_median, 3),
        "rho_q25":      round(rho_q25, 3),
        "rho_q75":      round(rho_q75, 3),
        "D_regression": round(D_reg, 3) if not np.isnan(D_reg) else np.nan,
        "R2_regression":round(R2_reg, 3) if not np.isnan(R2_reg) else np.nan,
        "tier":         tier,
        "status":       status,
    })

# ── Save ─────────────────────────────────────────────────────────────────────
results = pd.DataFrame(records)
results.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}\n")

# ── Print summary table ───────────────────────────────────────────────────────
print("=" * 95)
print(f"{'Region':<22} {'sub':>3}  {'D_chosen':>9}  {'N_post':>6}  "
      f"{'F_in':>5}  {'ρ̃':>5}  [Q25–Q75]        {'D_regr':>8}  {'R²':>5}  Status")
print("-" * 95)

TIER_ICON = {"GOOD": "✓", "OK": "~", "MARGINAL": "△", "POOR": "✗", "DISCARDED": "⊘"}

for _, r in results.iterrows():
    q_range = (f"[{r['rho_q25']:.2f}–{r['rho_q75']:.2f}]"
               if not np.isnan(r['rho_q25']) else "  —  ")
    D_r   = f"{r['D_regression']:.2f}" if not np.isnan(r['D_regression']) else "  —  "
    R2_r  = f"{r['R2_regression']:.2f}" if not np.isnan(r['R2_regression']) else "  —  "
    early = f"{r['N_early']}/{r['N_post']}" if not np.isnan(r['N_post']) else "—"
    icon  = TIER_ICON.get(r["tier"], "?")
    print(f"{icon} {r['region']:<22} {int(r['sub_n']):>3}  "
          f"{r['D_chosen']:>9.2f}  "
          f"{int(r['N_post']) if not np.isnan(r['N_post']) else '—':>6}  "
          f"early:{early:<7}  "
          f"F_in:{r['F_in']:.2f}  ρ̃:{r['rho_median']:.2f}  {q_range:<16}  "
          f"D_reg:{D_r:>7}  R²:{R2_r:>5}  [{r['tier']}]")

print("=" * 110)
print("""
TIER LEGEND
  ✓ GOOD      : ρ̃ ≤ 1.1 AND F_in ≥ 0.60 — events track the diffusion front well
  ~ OK        : acceptable fit, moderate containment
  △ MARGINAL  : weak pattern — ρ̃ > 1.5 or F_in < 0.40
  ✗ POOR      : early contamination (>30% events at t<1d) OR ρ̃ > 3 — unreliable
  ⊘ DISCARDED : known invalid case (see status column)

COLUMN GUIDE
  N_post      : post-origin events (t > 0) used for all metrics
  early N/N   : events at 0 < t < 1 day — if high fraction, ρ̃ is contaminated
                by co-seismic or pre-existing events (envelope too small at t~0)
  F_in        : fraction of post-origin events inside r²≤4Dt  [depends on N]
  ρ̃           : median(r / r_envelope) — N-independent distance-to-front metric
  [Q25–Q75]   : spread of the ρ distribution (tight range = coherent pattern)
  D_regression: objective D from forced-origin regression r²=4Dt,
                using only inside-envelope events [m²/s]
                → lower bound vs visual D_chosen (inside events sit before front)
  R²          : goodness of fit of r²–t linear relationship
                (R² → 1: clean diffusion; R² low: noisy/mixed signal)
""")
