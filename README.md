# rainseismicity

Pipeline for **"A Hydrological Regime Framework for Rainfall-Triggered Seismicity: Global Screening with Impulse Contrast"**  
*Ortega et al. (2026), Geophysical Research Letters*

---

## Overview

This repository contains all analysis scripts used to screen 25 regions worldwide for rainfall-triggered seismic swarms. The central contribution is the **Impulse Contrast (IC)** index, which classifies rainfall events into hydrological regimes (I–IV) and predicts the detectability of Darcy-type pore-pressure diffusion signals.

---

## Pipeline Structure

Run scripts in order:

| Script | Description |
|--------|-------------|
| `01_download_isc.py` | Download seismicity catalogs from ISC/USGS/EMSC |
| `02_bayesian_switchpoint.py` | Bayesian Poisson switchpoint (rate change after rain) |
| `03_zaliapin_clustering.py` | Zaliapin–Ben-Zion NND clustering + b-value |
| `04_summary_table.py` | Compile region summary |
| `05_etas_prerain.py` | ETAS background rate (R = N_obs/N_ETAS) |
| `06_climate_indices.py` | ERA5 rainfall, API, IC, SPI per region |
| `07_master_table_full.py` | Master results table |
| `08_spacetime_depth_plots.py` | Space–time–depth diagnostic figures |
| `09_diffusion_r2t.py` | r²–t diffusion analysis (static) |
| `10_world_map.py` | Figure 1 — world map of regions |
| `11_publication_figures.py` | Figures 2–4 (regime panel, results, diffusion) |
| `13_ic_sensitivity.py` | IC threshold sensitivity analysis (Figure S0) |
| `14_seismicity_movies.py` | MP4 movies per region (depth, magnitude) |
| `15_subregion_movie.py` | Interactive subregion selector → MP4 |
| `16_pick_origin.py` | Interactive origin picker (X₀,Y₀,Z₀,t₀) |
| `17_pick_diffusivity.py` | Interactive r²–t slider → D_chosen |
| `18_diffusion_quality_metrics.py` | F_in, ρ̃, D_regression per subcatalog |
| `19_metrics_figure.py` | Figure S — diffusion quality metrics |
| `21_murcia_ign_analysis.py` | Supplemental: Murcia Spain (IGN catalog) |
| `22_calabria_ingv_analysis.py` | Supplemental: Calabria Italy (INGV catalog) |

### Configuration

All region parameters (coordinates, rain dates, Mc, f_inf) are in `config_regions.py`.

---

## Installation

```bash
conda env create -f environment.yml
conda activate rainseismo
```

---

## Data

Seismicity catalogs are downloaded automatically by `01_download_isc.py` from:
- ISC Bulletin: http://www.isc.ac.uk
- USGS ComCat: https://earthquake.usgs.gov
- EMSC: https://www.seismicportal.eu

Precipitation data (ERA5) are retrieved via [Open-Meteo](https://open-meteo.com) API (no key required).

National agency catalogs:
- INGV (Calabria): http://webservices.ingv.it/fdsnws/event/1/
- IGN (Murcia): https://www.ign.es/web/ign/portal/sis-catalogo-sismicidad

---

## Repository Structure

```
rainseismicity/
├── config_regions.py          # Region parameters (25 + 2 supplemental)
├── 01_download_isc.py         # ... pipeline scripts (see table above)
├── environment.yml            # Conda environment
└── README.md
```

Data, figures, and results are excluded from the repository (see `.gitignore`).  
Download catalogs with `01_download_isc.py` before running subsequent scripts.

---

## Citation

```
Ortega, R., Carciumaru, D. (2026).
A hydrological regime framework for rainfall-triggered seismicity: Global screening
with Impulse Contrast. Geophysical Research Letters.
https://doi.org/[to-be-added]
```

---

## License

MIT License — see [LICENSE](LICENSE) file.
