"""
Tabla maestra integrada: Bayesian switchpoint + Zaliapin b-values + ETAS + Clima.
Genera tabla completa ordenada por señal y diagnóstico científico.
"""
import os, glob
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

RES = "results"

# ── Cargar resultados disponibles ─────────────────────────────────────────────
sw  = pd.read_csv(f"{RES}/switchpoint_results.csv")  if os.path.exists(f"{RES}/switchpoint_results.csv") else pd.DataFrame()
zal = pd.read_csv(f"{RES}/zaliapin_results.csv")     if os.path.exists(f"{RES}/zaliapin_results.csv")    else pd.DataFrame()
cli = pd.read_csv(f"{RES}/climate_indices.csv")      if os.path.exists(f"{RES}/climate_indices.csv")     else pd.DataFrame()
eta = pd.read_csv(f"{RES}/etas_results.csv")         if os.path.exists(f"{RES}/etas_results.csv")        else pd.DataFrame()

# Filtrar duplicados _classified
for df_name in ["sw","zal"]:
    df_obj = eval(df_name)
    if not df_obj.empty and "region" in df_obj.columns:
        exec(f"{df_name} = df_obj[~df_obj['region'].str.endswith('_classified')].copy()")

# ── Rain dates desde catálogos ────────────────────────────────────────────────
rain_lookup = {}
for f in glob.glob("data/*.csv"):
    if "summary" in f or "classified" in f: continue
    try:
        tmp = pd.read_csv(f, usecols=["region","rain_date"], nrows=1)
        if not tmp.empty:
            rain_lookup[tmp["region"].iloc[0]] = tmp["rain_date"].iloc[0]
    except Exception: pass

# ── Merge ─────────────────────────────────────────────────────────────────────
if sw.empty:
    print("No hay resultados de switchpoint todavía"); exit()

master = sw.copy()
master["rain_date"] = master["region"].map(rain_lookup)

if not zal.empty:
    master = master.merge(
        zal[["region","b_value","b_error","mc","n_above_mc"]].rename(columns={"mc":"mc_zal"}),
        on="region", how="left")

if not cli.empty:
    master = master.merge(
        cli[["region","P_event_mm","P_eff_mm","API_30d","SPI_local",
             "I_pulse_mm","dP_surf_kPa","dP_10km_kPa","HSCI"]],
        on="region", how="left")

if not eta.empty and "region" in eta.columns:
    eta_cols = ["region","n_pre","n_post","log10_mu","branching_ratio_proxy",
                "expected_bg","ratio_obs_exp"]
    eta_cols = [c for c in eta_cols if c in eta.columns]
    master = master.merge(eta[eta_cols], on="region", how="left")

# ── Score integrado ────────────────────────────────────────────────────────────
def score(row):
    s = 0
    # Señal sísmica
    if pd.notna(row.get("p_rate_increase")) and row["p_rate_increase"] >= 0.90: s += 2
    if pd.notna(row.get("ratio_after_before")) and row["ratio_after_before"] >= 1.5: s += 1
    if pd.notna(row.get("tau_median_days")) and 0 <= row["tau_median_days"] <= 120: s += 2
    if pd.notna(row.get("b_value")) and row["b_value"] >= 1.2: s += 2
    # Señal climática
    if pd.notna(row.get("P_eff_mm")) and row["P_eff_mm"] >= 10: s += 1
    if pd.notna(row.get("SPI_local")) and row["SPI_local"] >= 1.5: s += 1
    if pd.notna(row.get("HSCI")) and row["HSCI"] >= 0.3: s += 1
    return s

def candidato(s):
    if s >= 7: return "★★★ FUERTE"
    if s >= 5: return "★★  MODERADO"
    if s >= 3: return "★   DÉBIL"
    return "—"

master["score"]     = master.apply(score, axis=1)
master["candidato"] = master["score"].apply(candidato)
master = master.sort_values("score", ascending=False)

# ── Tabla display ─────────────────────────────────────────────────────────────
master.to_csv(f"{RES}/master_full.csv", index=False)

# Tabla 1: Señal sísmica
print("\n" + "="*120)
print("TABLA A — SEÑAL SÍSMICA (Bayesian switchpoint + b-value)")
print("="*120)
cols_a = ["region","rain_date","n_events","lambda_before","lambda_after",
          "ratio_after_before","p_rate_increase","tau_median_days","b_value","candidato"]
cols_a = [c for c in cols_a if c in master.columns]
print(master[cols_a].to_string(index=False))

# Tabla 2: Índices climáticos
print("\n" + "="*120)
print("TABLA B — ÍNDICES CLIMÁTICOS E INFILTRACIÓN")
print("="*120)
cols_b = ["region","rain_date","P_event_mm","P_eff_mm","API_30d","SPI_local",
          "I_pulse_mm","dP_surf_kPa","HSCI","candidato"]
cols_b = [c for c in cols_b if c in master.columns]
print(master[cols_b].to_string(index=False))

# Tabla 3: ETAS (si disponible)
if not eta.empty and "log10_mu" in master.columns:
    print("\n" + "="*120)
    print("TABLA C — PARÁMETROS ETAS PRE-LLUVIA")
    print("="*120)
    cols_c = ["region","rain_date","n_pre","n_post","log10_mu",
              "branching_ratio_proxy","expected_bg","ratio_obs_exp","candidato"]
    cols_c = [c for c in cols_c if c in master.columns]
    print(master[cols_c].dropna(subset=["log10_mu"]).to_string(index=False))

print(f"\n→ Tabla completa guardada en {RES}/master_full.csv")
print(f"→ Regiones analizadas: {len(master)}")
print(f"→ Candidatos fuertes (★★★): {(master['candidato']=='★★★ FUERTE').sum()}")
print(f"→ Candidatos moderados (★★): {(master['candidato']=='★★  MODERADO').sum()}")
