"""
Descarga datos climáticos históricos via Open-Meteo (sin API key)
y calcula índices de infiltración/triggering para cada región.

Índices calculados:
  - P_event     : precipitación total en el evento (mm)
  - P_7d        : precipitación acumulada 7 días previos (mm)
  - API_30       : Antecedent Precipitation Index 30 días (mm, k=0.85)
  - SPI_local   : cuántas sigmas sobre la media histórica mensual
  - P_eff       : precipitación efectiva = P - ET₀ (mm)
  - I_pulse     : pulso de infiltración = f_inf × P_eff (mm)  [f_inf por litología]
  - delta_P_surf: perturbación de presión en superficie (kPa)
  - delta_P_dep : perturbación a 10 km de profundidad via difusión (kPa)
  - t_diff_10km : tiempo de difusión a 10 km con D=1 m²/s (días)
  - HSCI        : Hydro-Seismic Coupling Index propuesto
"""
import os, json
import numpy as np
import pandas as pd
import requests
import glob
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "data"
RES_DIR  = "results"
os.makedirs(RES_DIR, exist_ok=True)

# Usar coordenadas del epicentro de lluvia, no del centroide sísmico
import sys
sys.path.insert(0, ".")
from config_regions import REGIONS as REGION_CFG

# ── Parámetros físicos ────────────────────────────────────────────────────────
RHO_W   = 1000.0    # kg/m³
G       = 9.81      # m/s²
D_HYDRO = 1.0       # m²/s  (difusividad hidráulica — valor Los Cabos)
SS      = 1e-6      # 1/m   (almacenamiento específico)
Z_SEIS  = 10_000.0  # m     profundidad sismogénica objetivo
K_API   = 0.85      # constante de recesión del suelo

# Factor de infiltración por tipo de ambiente (estimado de literatura)
F_INF = {
    "los_cabos_mx":     0.15,   # hipeárido costero
    "noto_japan":       0.35,   # subtropical húmedo
    "nepal_himalaya":   0.25,   # montañoso
    "taiwan":           0.40,   # subtropical
    "apennines_italy":  0.30,   # mediterráneo
    "pakistan_2022":    0.10,   # árido
    "corinth_greece":   0.25,   # mediterráneo kárstico
    "marlborough_nz":   0.35,   # oceánico templado
    "west_bohemia_cz":  0.30,   # continental húmedo
    "zagros_iran":      0.10,   # árido/semi-árido
    "cascades_oregon":  0.45,   # templado lluvioso
    "papua_new_guinea": 0.50,   # tropical húmedo
    "ethiopia_rift":    0.20,   # árido/semi-árido
    "pyrenees_fr":      0.40,   # kárstico-montañoso
    "assam_india":      0.35,   # subtropical húmedo
    "costa_rica":       0.45,   # tropical húmedo
    "java_indonesia":   0.50,   # tropical húmedo
    "himachal_india":   0.30,   # montañoso
    "murcia_spain":     0.20,   # mediterráneo semi-árido
    "reykjanes_iceland":0.20,   # volcánico/lávico
    "colombia_andes":   0.40,   # tropical montañoso
    "calabria_italy":   0.30,   # mediterráneo húmedo
    "chile_central":    0.25,   # mediterráneo
    "guerrero_mexico":  0.20,   # tropical semi-árido
    "mindanao_ph":      0.50,   # tropical húmedo
}

# Coordenadas del epicentro de lluvia desde config_regions
REGION_COORDS = {
    name: {"lat": cfg["lat"], "lon": cfg["lon"], "rain_date": cfg["rain"]}
    for name, cfg in REGION_CFG.items()
}


def fetch_openmeteo(lat, lon, start_date, end_date):
    """
    Descarga precipitación diaria + ET₀ de Open-Meteo Archive API (ERA5).
    Sin API key. Retorna DataFrame.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":    lat,
        "longitude":   lon,
        "start_date":  start_date,
        "end_date":    end_date,
        "daily":       "precipitation_sum,et0_fao_evapotranspiration,rain_sum",
        "timezone":    "UTC",
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame({
            "date":  pd.to_datetime(data["daily"]["time"]),
            "precip": data["daily"]["precipitation_sum"],
            "et0":    data["daily"]["et0_fao_evapotranspiration"],
            "rain":   data["daily"]["rain_sum"],
        })
        df = df.fillna(0)
        return df
    except Exception as e:
        print(f"   Open-Meteo error: {e}")
        return None


def compute_api(precip_series, k=K_API):
    """Antecedent Precipitation Index: API(t) = k*API(t-1) + P(t)."""
    api = np.zeros(len(precip_series))
    for i, p in enumerate(precip_series):
        api[i] = k * (api[i-1] if i > 0 else 0) + p
    return api


def compute_spi_local(precip_series, window_days=30):
    """
    SPI simplificado: cuántas sigmas está el evento sobre la media de ventanas
    del mismo período en años anteriores. Usa los datos disponibles.
    """
    if len(precip_series) < 60:
        return np.nan
    mu  = precip_series.mean()
    sig = precip_series.std()
    if sig < 0.01:
        return np.nan
    return float((precip_series.iloc[-1] - mu) / sig)


def pressure_perturbation_surface(infiltration_m):
    """ΔP en superficie = ρ_w × g × h_inf  [Pa → kPa]."""
    return RHO_W * G * infiltration_m / 1000.0


def pressure_at_depth_diffusion(delta_p0_kpa, lag_days, depth_m=Z_SEIS, D=D_HYDRO):
    """
    Solución de Green para difusión 1D de presión a profundidad z, tiempo t:
    ΔP(z,t) = ΔP₀ × erfc(z / sqrt(4Dt))
    (frente de difusión 1D desde superficie)
    """
    from scipy.special import erfc
    t_sec = lag_days * 86400.0
    if t_sec <= 0:
        return 0.0
    arg = depth_m / np.sqrt(4 * D * t_sec)
    return delta_p0_kpa * float(erfc(arg))


def diffusion_time_to_depth(depth_m=Z_SEIS, D=D_HYDRO):
    """Tiempo en días para que el frente difusivo alcance la profundidad z (r²/4D)."""
    return depth_m**2 / (4 * D) / 86400.0


def hsci(p_eff_mm, f_inf, lag_days, D=D_HYDRO):
    """HSCI = (P_eff × f_inf) / (lag_days^0.5 × D^0.5)"""
    if lag_days <= 0 or p_eff_mm <= 0:
        return 0.0
    return (p_eff_mm * f_inf) / (np.sqrt(lag_days) * np.sqrt(D))


def impulse_contrast(p_event_mm, api_30_mm):
    """
    Impulse Contrast (IC): razón entre el evento y la tasa diaria antecedente.
    IC = P_evento / (API_30 / 30)
    IC >> 1  → pulso aislado sobre fondo seco  (Régimen I — Darcy impulso)
    IC ~  1  → evento moderado sobre fondo húmedo (Régimen II — carga estacional)
    IC <  1  → evento embebido en temporada lluviosa (Régimen III — Bollinger)
    """
    daily_antecedent = api_30_mm / 30.0
    if daily_antecedent < 0.1:
        return p_event_mm * 10  # suelo muy seco — máximo contraste
    return p_event_mm / daily_antecedent


def hydrological_regime(ic, p_event_mm):
    """
    Clasifica el régimen hidrológico del evento.
    Solo Régimen I es apropiado para análisis de difusión Darcy tipo Los Cabos.
    """
    if p_event_mm < 5:
        return "0-SIN_LLUVIA"
    if ic >= 10:
        return "I-PULSO_AISLADO"
    if ic >= 3:
        return "II-CONTRASTE_MODERADO"
    return "III-TEMPORADA_HUMEDA"


# ── MAIN ──────────────────────────────────────────────────────────────────────
results = []

for region, info in sorted(REGION_COORDS.items()):
    lat       = info["lat"]
    lon       = info["lon"]
    rain_date = pd.to_datetime(info["rain_date"])
    f_inf_val = REGION_CFG.get(region, {}).get("f_inf", F_INF.get(region, 0.25))

    print(f"\n→ {region}  ({lat:.1f}, {lon:.1f})  lluvia: {rain_date.date()}")

    # Ventana de descarga: 60 días antes → 5 días después del evento
    t_start = (rain_date - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    t_end   = (rain_date + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    climate = fetch_openmeteo(lat, lon, t_start, t_end)
    if climate is None or len(climate) < 10:
        print("   SKIP: sin datos climáticos")
        results.append({"region": region, "status": "no_climate"})
        continue

    # Encontrar el día del evento de lluvia en la serie
    rain_idx = (climate["date"] - rain_date).abs().idxmin()

    # Precipitación en el evento (1 día) y ventana 3 días
    p_event  = float(climate.loc[rain_idx, "precip"])
    p_3d     = float(climate.loc[max(0, rain_idx-2):rain_idx, "precip"].sum())
    p_7d     = float(climate.loc[max(0, rain_idx-6):rain_idx-1, "precip"].sum())

    # ET₀ en el día del evento
    et0_event = float(climate.loc[rain_idx, "et0"])

    # API 30 días (todos los días antes)
    api_series = compute_api(climate["precip"])
    api_30     = float(api_series[min(rain_idx, len(api_series)-1)])

    # SPI local (usando serie disponible 60 días)
    spi = compute_spi_local(climate["precip"][:rain_idx+1])

    # Precipitación efectiva = P - ET₀ (mínimo 0)
    p_eff = max(0.0, p_event - et0_event)

    # Pulso de infiltración (mm → m)
    i_pulse_m  = f_inf_val * p_eff / 1000.0

    # Perturbación de presión en superficie
    dp_surf_kpa = pressure_perturbation_surface(i_pulse_m)

    # Tiempo de difusión a 10 km con D=1 m²/s
    t_diff_days = diffusion_time_to_depth(Z_SEIS, D_HYDRO)

    # Presión a 10 km con el lag observado del Bayesian switchpoint
    # (usamos t_diff_days como lag esperado para calcular ΔP)
    dp_depth_kpa = pressure_at_depth_diffusion(dp_surf_kpa, t_diff_days)

    # HSCI
    hsci_val = hsci(p_eff, f_inf_val, t_diff_days)

    ic_val   = impulse_contrast(p_event, api_30)
    regime   = hydrological_regime(ic_val, p_event)

    print(f"   P_evento={p_event:.1f}mm  P_eff={p_eff:.1f}mm  API30={api_30:.1f}mm")
    print(f"   SPI_local={spi:.2f}  IC={ic_val:.1f}  Régimen={regime}")
    print(f"   I_pulse={i_pulse_m*1000:.1f}mm  ΔP_sup={dp_surf_kpa:.3f}kPa  HSCI={hsci_val:.4f}")

    results.append({
        "region":         region,
        "rain_date":      info["rain_date"],
        "f_inf":          f_inf_val,
        "P_event_mm":     round(p_event, 1),
        "P_3d_mm":        round(p_3d, 1),
        "P_7d_prev_mm":   round(p_7d, 1),
        "ET0_mm":         round(et0_event, 1),
        "API_30d":        round(api_30, 1),
        "SPI_local":      round(spi, 2) if not np.isnan(spi) else None,
        "P_eff_mm":       round(p_eff, 1),
        "I_pulse_mm":     round(i_pulse_m * 1000, 2),
        "dP_surf_kPa":    round(dp_surf_kpa, 4),
        "dP_10km_kPa":    round(dp_depth_kpa, 6),
        "t_diff_10km_d":  round(t_diff_days, 1),
        "HSCI":            round(hsci_val, 5),
        "IC":              round(ic_val, 2),
        "hydro_regime":    regime,
        "status":          "OK",
    })

df = pd.DataFrame(results)
df.to_csv(f"{RES_DIR}/climate_indices.csv", index=False)

print("\n\n" + "="*100)
print("ÍNDICES CLIMÁTICOS E INFILTRACIÓN POR REGIÓN")
print("="*100)
cols = ["region","P_event_mm","P_eff_mm","API_30d","SPI_local","IC",
        "hydro_regime","I_pulse_mm","dP_surf_kPa","HSCI"]
cols = [c for c in cols if c in df.columns]
print(df[cols].to_string(index=False))
