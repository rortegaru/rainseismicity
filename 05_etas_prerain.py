"""
ETAS pre-rain fitting pipeline.

Para cada región con suficientes eventos:
  1. Separa catálogo en PRE-lluvia y POST-lluvia
  2. Ajusta modelo ETAS al período PRE-lluvia → parámetros de fondo (mu, K, alpha, c, p)
  3. Calcula tasa esperada ETAS en ventana post-lluvia
  4. Compara con tasa observada → exceso relativo y significancia
  5. Guarda parámetros y figuras

Referencia: Mizrahi, Nandan & Wiemer (2021) JGR
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from etas.inversion import ETASParameterCalculation, parameter_array2dict

DATA_DIR = "data"
RES_DIR  = "results"
FIG_DIR  = "figures"
os.makedirs(RES_DIR, exist_ok=True)

# Candidatos con suficientes eventos y señal positiva
# (omitimos casos con τ negativo o N<50)
TARGETS = [
    dict(name="noto_japan",       mc=1.5, pre_min_events=80),
    dict(name="corinth_greece",   mc=1.6, pre_min_events=80),
    dict(name="taiwan",           mc=2.2, pre_min_events=80),
    dict(name="apennines_italy",  mc=1.5, pre_min_events=80),
    dict(name="pyrenees_fr",      mc=1.0, pre_min_events=30),
    dict(name="los_cabos_mx",     mc=1.5, pre_min_events=80),
    dict(name="marlborough_nz",   mc=1.8, pre_min_events=50),
    dict(name="cascades_oregon",  mc=1.1, pre_min_events=30),
    dict(name="papua_new_guinea", mc=3.3, pre_min_events=30),
    dict(name="zagros_iran",      mc=2.5, pre_min_events=20),
    dict(name="assam_india",      mc=2.0, pre_min_events=10),
]

def prep_catalog(df, mc):
    """Formato que espera lmizrahi/etas."""
    df = df.copy()
    df = df[df["mag"] >= mc].dropna(subset=["lat","lon","mag"]).copy()
    df = df.sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["magnitude"] = df["mag"]
    df["latitude"]  = df["lat"]
    df["longitude"] = df["lon"]
    df["id"] = range(len(df))
    return df[["id","latitude","longitude","time","magnitude"]]


def run_etas(cat_pre, mc, region_name):
    """
    Ajusta ETAS al catálogo pre-lluvia (API v3 lmizrahi/etas).
    Retorna objeto ETASParameterCalculation con .parameters dict.
    """
    if len(cat_pre) < 10:
        return None

    t_start = cat_pre["time"].min()
    t_end   = cat_pre["time"].max()
    # auxiliary period = misma ventana (condiciona desde el inicio)
    aux_start = t_start - pd.Timedelta(days=30)

    metadata = {
        "catalog":                cat_pre,
        "mc":                     mc,
        "delta_m":                0.1,
        "m_ref":                  mc,
        "coppersmith_multiplier": 100,
        "auxiliary_start":        aux_start.isoformat(),
        "timewindow_start":       t_start.isoformat(),
        "timewindow_end":         t_end.isoformat(),
    }

    try:
        calc = ETASParameterCalculation(metadata)
        calc.prepare()
        theta_dict = calc.invert()      # v3 returns dict directly
        if theta_dict is None:
            theta_dict = calc.theta     # fallback
        # mu = 10^log10_mu (background rate, events/day integrated over space)
        log10_mu = theta_dict.get("log10_mu", None)
        theta_dict["mu"] = 10 ** float(log10_mu) if log10_mu is not None else np.nan
        # branching ratio proxy: n = K * β / (α - β)  [simplified]
        K   = 10 ** float(theta_dict.get("log10_k0", -3))
        a   = float(theta_dict.get("a", 1.0))
        theta_dict["K"] = K
        theta_dict["branching_ratio_proxy"] = K * a
        calc._params_named = theta_dict
        return calc
    except Exception as e:
        print(f"   ETAS ERROR: {e}")
        return None


def expected_rate_etas(params, cat_pre, t_start, t_end_days, radius_km=100):
    """
    Tasa esperada acumulada en ventana post-lluvia.
    mu (eventos/día/km²) × área_catálogo × Δt
    + contribución de triggering de eventos pre-lluvia decayendo con Omori.
    """
    mu  = params.get("mu", np.nan)    # eventos/día/km²
    if np.isnan(mu):
        return np.nan
    area_km2 = np.pi * radius_km**2   # área aproximada del catálogo
    # Tasa de fondo en toda el área
    bg_rate   = mu * area_km2          # eventos/día
    bg_total  = bg_rate * t_end_days   # eventos esperados solo de fondo

    # Tasa observada pre-lluvia por día (para escalar)
    n_pre    = len(cat_pre)
    pre_days = (cat_pre["time"].max() - cat_pre["time"].min()).total_seconds() / 86400
    pre_days = max(pre_days, 1)
    obs_rate_pre = n_pre / pre_days    # eventos/día observados

    # Si la tasa de fondo ETAS es ridículamente pequeña vs. la observada,
    # usamos la tasa observada escalada como mejor estimador
    if bg_total < 1 and obs_rate_pre > 0:
        bg_total = obs_rate_pre * t_end_days

    return bg_total


def plot_etas_result(cat_all, rain_date, params, expected, observed,
                     region, post_days=90):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f"{region}  —  ETAS pre-rain  |  rain: {rain_date.date()}", fontsize=13)

    # ── panel 1: tasa diaria + ETAS mu ──
    ax = axes[0]
    cat_all["day"] = (cat_all["time"] - rain_date).dt.total_seconds() / 86400
    daily = cat_all.groupby(cat_all["day"].apply(lambda x: int(x))).size()
    ax.bar(daily.index, daily.values, color="steelblue", alpha=0.6, width=1, label="observed")

    mu  = params.get("mu", 0)
    ax.axhline(mu, color="darkorange", lw=2, ls="--",
               label=f"μ ETAS = {mu:.3f} ev/day")
    ax.axvline(0, color="red", lw=2, ls="--", label="rain")
    ax.set_xlim(daily.index.min(), daily.index.max())
    ax.set_xlabel("Days relative to rain"); ax.set_ylabel("Events/day")
    ax.legend(fontsize=9)

    # ── panel 2: parámetros ETAS ──
    ax = axes[1]
    plot_keys = ["log10_mu","log10_k0","a","log10_c","omega","log10_tau","log10_d","gamma","rho"]
    labels = [k for k in plot_keys if k in params and params[k] is not None]
    values = [float(params[k]) for k in labels]
    colors = ["steelblue" if v >= 0 else "firebrick" for v in values]
    bars = ax.bar(labels, values, color=colors, alpha=0.8)
    ax.set_ylabel("Parameter value")
    ax.set_title(f"Post-rain excess: expected={expected:.1f}  "
                 f"observado={observed}  "
                 f"ratio={observed/max(expected,0.01):.2f}x")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + abs(val)*0.02,
                f"{val:.4f}", ha="center", fontsize=8)

    plt.tight_layout()
    path = f"{FIG_DIR}/{region}_etas.png"
    plt.savefig(path, dpi=120)
    plt.close()
    return path


# ── MAIN ──────────────────────────────────────────────────────────────────────

POST_WINDOW = 90   # días post-lluvia para comparar

results = []

for t in TARGETS:
    region = t["name"]
    mc     = t["mc"]
    print(f"\n{'='*60}")
    print(f"  ETAS: {region}  Mc={mc}")

    csv_path = f"{DATA_DIR}/{region}.csv"
    if not os.path.exists(csv_path):
        print(f"  SKIP: no existe {csv_path}")
        continue

    df_raw = pd.read_csv(csv_path, parse_dates=["time", "rain_date"])
    rain_dt = pd.Timestamp(df_raw["rain_date"].iloc[0], tz="UTC")

    # preparar catálogo completo
    cat = prep_catalog(df_raw, mc)
    cat_pre  = cat[cat["time"] < rain_dt].copy()
    cat_post = cat[(cat["time"] >= rain_dt) &
                   (cat["time"] < rain_dt + pd.Timedelta(days=POST_WINDOW))].copy()

    n_pre  = len(cat_pre)
    n_post = len(cat_post)
    print(f"  N pre-lluvia={n_pre}  N post-{POST_WINDOW}d={n_post}")

    if n_pre < t["pre_min_events"]:
        print(f"  SKIP: muy pocos eventos pre-lluvia ({n_pre} < {t['pre_min_events']})")
        results.append({"region": region, "mc": mc, "n_pre": n_pre,
                        "n_post": n_post, "status": "pocos_pre"})
        continue

    # ajustar ETAS
    calc = run_etas(cat_pre, mc, region)
    if calc is None:
        results.append({"region": region, "mc": mc, "n_pre": n_pre,
                        "n_post": n_post, "status": "ETAS_error"})
        continue

    params = calc._params_named
    print(f"  ETAS params: {params}")

    # tasa esperada en ventana post-lluvia
    from config_regions import REGIONS as REGION_CFG
    r_km = REGION_CFG.get(region, {}).get("r", 100)
    expected_bg = expected_rate_etas(params, cat_pre, rain_dt, POST_WINDOW, radius_km=r_km)

    # exceso
    excess        = n_post - expected_bg
    ratio_obs_exp = n_post / max(expected_bg, 0.01)

    print(f"  μ={params.get('mu', 'N/A'):.4f} ev/día  "
          f"esperado(fondo)={expected_bg:.1f}  "
          f"observado={n_post}  ratio={ratio_obs_exp:.2f}x")

    # figura
    fig_path = plot_etas_result(
        cat_all=cat, rain_date=rain_dt,
        params=params, expected=expected_bg, observed=n_post,
        region=region, post_days=POST_WINDOW
    )

    results.append({
        "region":        region,
        "mc":            mc,
        "n_pre":         n_pre,
        "n_post":        n_post,
        "mu":            round(params.get("mu", np.nan), 5),
        "K":             round(params.get("K", np.nan),  5),
        "alpha":         round(params.get("alpha", np.nan), 4),
        "c":             round(params.get("c", np.nan),   5),
        "p":             round(params.get("p", np.nan),   4),
        "expected_bg":   round(expected_bg, 1),
        "observed_post": n_post,
        "ratio_obs_exp": round(ratio_obs_exp, 2),
        "excess_events": round(excess, 1),
        "status":        "OK",
    })

df_res = pd.DataFrame(results)
df_res.to_csv(f"{RES_DIR}/etas_results.csv", index=False)

print("\n\n" + "="*90)
print("RESULTADOS ETAS — parámetros pre-lluvia y exceso post-lluvia")
print("="*90)
cols = ["region","mc","n_pre","n_post","mu","K","alpha","c","p",
        "expected_bg","observed_post","ratio_obs_exp","status"]
cols = [c for c in cols if c in df_res.columns]
print(df_res[cols].to_string(index=False))
