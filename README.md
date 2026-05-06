# Boom: Trajectory Unknown Challenge
For more information and scoring metrics, see the Challenge Website: https://www.freelancer.com/boom

The challenge has two parts: 
- **Forward prediction**: Mandatory for your submission to be accepted; build a physics-based, data-driven predictive model to predict ejecta outcomes given impact scenarios.
- **Inverse design** Optional but provides extra points in your score; using your trained model and an efficient search algorithm, propose 20 impact scenarios with ejecta outcomes that satisfy a given set of constraints.

Please read the following challenge descriptions carefully.

### Data submission files in this repository

| Output | Location |
|--------|----------|
| Forward prediction submission (`prediction_submission.csv`) | [`forward_prediction/prediction_submission.csv`](forward_prediction/prediction_submission.csv) |
| Inverse design results (inputs + model-predicted ejecta per solution) | [`inverse_predict/inverse_predictions.csv`](inverse_predict/inverse_predictions.csv) |

---

## Reproduction and environment (this codebase)

### Run and reproduction steps

This repository is meant to be run **from Jupyter** (JupyterLab or the notebook UI in VS Code / Cursor). Train, evaluate, export models, and build submissions by executing the notebooks top to bottom unless a notebook says otherwise.

**1. Environment**

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For reproducible builds, install once then record exact pins (for example `pip freeze > requirements-lock.txt`) and document the Python minor version you used.

**2. Start Jupyter**

From the **repository root**:

```bash
jupyter lab
# or: jupyter notebook
```

Open the `.ipynb` files below from that session (or open them in your editor’s Jupyter view with the workspace folder set to the repo root). Each notebook defines `data_dir`, `model_dir`, and output paths near the top—adjust only if you relocate data or artifacts.

**3. Train / reproduce forward models (notebooks)**

| Notebook location | What it does |
|-------------------|--------------|
| [`forward_notebooks/boom_model_training_*.ipynb`](forward_notebooks/) | Per-target training notebooks (mostly **linear** variants); models and `z_log` transforms are saved under paths like `models_linear/` as set in each notebook. |
| [`forward_notebook_with_lgbm/boom_model_training_*.ipynb`](forward_notebook_with_lgbm/) | Per-target notebooks with **LightGBM**; outputs typically under `models/` (see each notebook’s save cells). |

Run the cells in order through training, CV (if present), and the **model export** cells so `joblib` files exist before prediction.

**4. Predict / submission example (notebooks)**

| Notebook | Use when |
|----------|----------|
| [`forward_notebooks/predict.ipynb`](forward_notebooks/predict.ipynb) | Loading **`models_linear`** (linear-only stacks exported from `forward_notebooks/` training notebooks). |
| [`forward_notebook_with_lgbm/predict.ipynb`](forward_notebook_with_lgbm/predict.ipynb) | Loading **`models`** (LightGBM per-target exports from `forward_notebook_with_lgbm/`). |

These notebooks read `../forward_prediction/test.csv` (relative to the notebook folder), apply the same feature geometry/scaling as training, and write a CSV in the challenge submission column layout. Create any output directory referenced in the notebook if it does not exist.

[`boom_model_training_lin_LGBM.ipynb`](boom_model_training_lin_LGBM.ipynb) at the repo root uses **`forward_prediction/`** (no `../`) and often saves a **single** `models.joblib` bundle for all targets. That layout differs from the per-target `joblib` files expected by the two `predict.ipynb` variants above—keep training and prediction in the **same** notebook family, or add your own inference cells that match your export format.

**5. Other notebooks**

- [`pysr_physics_discovery.ipynb`](pysr_physics_discovery.ipynb) — symbolic regression (optional **Julia** / PySR stack).
- [`reverse_prediction/inverse_predict.ipynb`](reverse_prediction/inverse_predict.ipynb) — inverse design search using [`reverse_prediction/config.json`](reverse_prediction/config.json) and official constraints.

**CLI scripts (optional)**

Legacy scripts [`train.py`](train.py) and [`predict.py`](predict.py) exist but **are not** the primary workflow; prefer the notebooks above for reproduction.

**Large model artifacts**

If trained artifacts exceed comfortable Git limits, store them with [**Git LFS**](https://git-lfs.com/) or host them on **Google Drive** (or another object store), and document the download URL and expected filenames (for example `models.joblib`) in this README or a short `MODELS.md`.

---

### Software requirements

| Item | Notes |
|------|--------|
| **Dependency list** | [`requirements.txt`](requirements.txt) — includes **`jupyter`** for running notebooks; package versions are not pinned in-repo (freeze after install for exact reproduction). |
| **Python** | Use a **64-bit** Python **3.10+** (project has been used with **3.14** in development). Match training and inference Python versions when loading `joblib` pickles. |
| **OS** | **macOS** and **Linux** are typical. **Windows** is plausible but LightGBM wheels and shell paths may differ. |
| **CUDA / GPU** | **Not required.** Training uses **LightGBM** and **scikit-learn** on **CPU** by default. No CUDA-specific packages are listed in `requirements.txt`. |
| **macOS + LightGBM** | If `lightgbm` fails to load with an OpenMP error, install **libomp** (for example `brew install libomp` on Apple Silicon/Intel macOS). |
| **Optional: PySR** | [`pysr`](https://github.com/MilesCranmer/PySR) installs a Julia-backed stack; follow PySR docs for a compatible **Julia** install if you run `pysr_physics_discovery.ipynb`. |

There is no `pyproject.toml` in this repository; dependency management is **pip + `requirements.txt`**.

---

### Hardware requirements

| Aspect | Guidance |
|--------|-----------|
| **GPU** | None required for the default sklearn + LightGBM pipeline. |
| **RAM** | Roughly **4 GB** system RAM is sufficient for the bundled CSV scales; **8 GB+** is comfortable for notebooks, plotting, and SHAP. |
| **Training time** | Highly machine-dependent: often **minutes** on a modern laptop CPU for full CV + final fit at the provided train sizes (not profiled in-repo — replace this sentence with your measured wall time, for example *“~15 minutes on Apple M2, 16 GB RAM”*). |
| **Inference** | Scoring **forward_prediction/test.csv** is lightweight (seconds on CPU for tree models). |
| **Cloud costs** | Optional; document here if you train on a paid instance (instance type, region, approximate USD). |

---

### External dependencies

| Dependency | Role | Source / access |
|------------|------|------------------|
| **Boom challenge CSVs** | Official train / test scenarios and labels | Bundled under [`forward_prediction/`](forward_prediction/) in this repo (loaded via each notebook’s `data_dir`, typically `../forward_prediction` from `forward_notebooks/`). |
| **Inverse design specs** | Bounds and submission layout | [`inverse_design/constraints.json`](inverse_design/constraints.json), [`inverse_design/design_submission_template.csv`](inverse_design/design_submission_template.csv). |
| **Challenge rules & scoring** | Task definition | [Challenge website](https://www.freelancer.com/boom) (external; no API key). |
| **Pre-trained models** | None required | This pipeline trains from the provided CSVs only; any uploaded `models.joblib` is **your** artifact, not a third-party checkpoint. |
| **PySR / Julia** | Optional symbolic regression | PySR documentation and Julia downloads (external toolchains). |
| **Public APIs** | None used by the bundled notebooks | Add your own cells if you call external APIs. |

---

## 1) Forward Prediction

### Data description
**Impact parameters:**
Each impact scenario is **partially** described by 8 parameters:
- energy - Impact energy 
- angle_rad - Impact angle from horizon (in radians)
- coupling - Energy transfer efficiency between asteroid and surface
- strength - Material strength
- porosity - Material porosity
- gravity - Surface gravity
- atmosphere - Atmospheric density at the impact altitude
- shape_factor - Fragment irregularity (a higher value indicates that the material tends to fracture into highly irregular shards) 

**Ejecta Outcomes:**
The aftermath of each impact event is described by 6 statistical measures: 

- **P80** - Fragment diameter (mm) below which 80% of the total ejected mass lies
  - `P80 = d` such that `Σ(m_i | d_i ≤ d) = 0.8 × M_total`

- **fines_frac** - Fraction of total ejecta mass contributed by fragments smaller than 40mm diameter
  - `fines_frac = Σ(m_i | d_i < 40mm) / M_total`

- **oversize_frac** - Fraction of total ejecta mass contributed by fragments larger than 120mm diameter
  - `oversize_frac = Σ(m_i | d_i > 120mm) / M_total`

- **R95** - Landing distance (m) below which 95% of the total ejected mass lies
  - `R95 = r` such that `Σ(m_i | r_i ≤ r) = 0.95 × M_total`

- **R50_fines** - Median landing distance (m) for fragments smaller than 40mm (mass-weighted)
  - `R50_fines = r` such that `Σ(m_i | d_i < 40mm, r_i ≤ r) = 0.5 × Σ(m_i | d_i < 40mm)`

- **R50_oversize** - Median landing distance (m) for fragments larger than 120mm (mass-weighted)
  - `R50_oversize = r` such that `Σ(m_i | d_i > 120mm, r_i ≤ r) = 0.5 × Σ(m_i | d_i > 120mm)`

Notation:
- `d_i` = diameter of fragment i
- `r_i` = landing distance of fragment i  
- `m_i` = mass of fragment i (∝ d_i³)
- `M_total` = Σm_i (total ejected mass)

### Files provided
- `forward_prediction/train.csv` (impact scenarios for training)
- `forward_prediction/train_labels.csv` (ejecta outcomes for training)
- `forward_prediction/test.csv` (impact scenarios for scoring)
- `forward_prediction/prediction_submission_template.csv` (submission template)

### Submission format
Submit `prediction_submission.csv` to your repository with the exact columns:
- `scenario_id`
- `P80`
- `fines_frac`
- `oversize_frac`
- `R95`
- `R50_fines`
- `R50_oversize`

Note:
- `scenario_id` must match the row index in `forward_prediction/test.csv` (0-based).
- One row per test scenario.

---

## 2) Inverse Design

Constraints:
- Ejecta fragment diameter: `96 ≤ P80 ≤ 101 mm`
- Ejecta range: `R95 ≤ 175 m`
- Input parameters must be within specified bounds (see `inverse_design/constraints.json` for input bounds)

### Files Provided
- `inverse_design/constraints.json` (constraints on input and output parameters)
- `inverse_design/design_submission_template.csv` (submission template)

### Submission format
Submit `design_submission.csv` to your repository with the exact columns:
- `submission_id`
- `energy`
- `angle_rad`
- `coupling`
- `strength`
- `porosity`
- `gravity`
- `atmosphere`
- `shape_factor`
