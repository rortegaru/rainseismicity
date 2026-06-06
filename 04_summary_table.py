"""
Merge Bayesian switchpoint + Zaliapin results into one master summary table.
"""
import os
import pandas as pd
import numpy as np

RES = "results"

sw  = pd.read_csv(f"{RES}/switchpoint_results.csv")  if os.path.exists(f"{RES}/switchpoint_results.csv")  else pd.DataFrame()
zal = pd.read_csv(f"{RES}/zaliapin_results.csv")     if os.path.exists(f"{RES}/zaliapin_results.csv")     else pd.DataFrame()

# Gather rain_dates from all CSV data files directly
import glob
rain_lookup = {}
for f in glob.glob("data/*.csv"):
    if "summary" in f or "classified" in f:
        continue
    try:
        tmp = pd.read_csv(f, usecols=["region", "rain_date"], nrows=1)
        if not tmp.empty:
            rain_lookup[tmp["region"].iloc[0]] = tmp["rain_date"].iloc[0]
    except Exception:
        pass

# Base: all regions that have switchpoint results (exclude _classified duplicates)
if not sw.empty:
    sw = sw[~sw["region"].str.endswith("_classified")].copy()
    master = sw.copy()
    master["rain_date"] = master["region"].map(rain_lookup)
else:
    master = pd.DataFrame()

if not zal.empty and "region" in zal.columns:
    zal = zal[~zal["region"].str.endswith("_classified")].copy()
    master = master.merge(zal.drop(columns=["status", "n_total"], errors="ignore"),
                          on="region", how="left")

# Interpretive flag
def interpret(row):
    flags = []
    if pd.notna(row.get("p_rate_increase")) and row["p_rate_increase"] >= 0.90:
        flags.append("TASA↑")
    if pd.notna(row.get("ratio_after_before")) and row["ratio_after_before"] >= 1.5:
        flags.append(f"ratio={row['ratio_after_before']:.1f}x")
    if pd.notna(row.get("tau_median_days")) and 0 <= row["tau_median_days"] <= 60:
        flags.append(f"τ={row['tau_median_days']:.0f}d")
    if pd.notna(row.get("b_value")) and row["b_value"] >= 1.3:
        flags.append(f"b={row['b_value']}")
    if pd.notna(row.get("frac_clustered")) and row["frac_clustered"] >= 0.15:
        flags.append(f"clust={row['frac_clustered']:.0%}")
    return " | ".join(flags) if flags else "—"

master["interpretacion"] = master.apply(interpret, axis=1)

# Classify candidate
def candidate(row):
    score = 0
    if pd.notna(row.get("p_rate_increase")) and row["p_rate_increase"] >= 0.90: score += 2
    if pd.notna(row.get("ratio_after_before")) and row["ratio_after_before"] >= 1.5: score += 1
    if pd.notna(row.get("tau_median_days")) and 0 <= row["tau_median_days"] <= 60: score += 2
    if pd.notna(row.get("b_value")) and row["b_value"] >= 1.3: score += 2
    if pd.notna(row.get("frac_clustered")) and row["frac_clustered"] >= 0.15: score += 1
    if score >= 5: return "★★★ FUERTE"
    if score >= 3: return "★★  MODERADO"
    if score >= 1: return "★   DÉBIL"
    return "—"

master["candidato"] = master.apply(candidate, axis=1)

master.to_csv(f"{RES}/master_table.csv", index=False)

# Print clean display table
cols_display = [
    "region", "rain_date", "n_events",
    "lambda_before", "lambda_after", "ratio_after_before",
    "p_rate_increase", "tau_median_days",
    "b_value", "frac_clustered",
    "candidato"
]
cols_display = [c for c in cols_display if c in master.columns]
print("\n" + "="*110)
print("TABLA MAESTRA — Lluvia intensa → sismicidad: análisis Bayesiano + Zaliapin")
print("="*110)
print(master[cols_display].to_string(index=False))
print("="*110)
print(f"\nArchivo completo → {RES}/master_table.csv")
