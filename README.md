# ZNO Outcomes: Leakage‑aware Modeling (2016–2021)

End‑to‑end, leakage‑aware modeling on Ukrainian standardized tests (ZNO).
The pipeline ingests multi‑year raw files, harmonizes schema, enforces a temporal split
(train: 2016–2020 → test: 2021), trains baselines and ensembles with early stopping,
and produces diagnostics, fairness slices with confidence intervals, feature importance,
and artifacts under `artifacts/`.

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)

See [CONTRIBUTING.md](CONTRIBUTING.md) for style, tests, and PR flow.

## Quick start
```bash
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

# Optional: install in editable mode. If this fails, skip it — the notebook adds project root to sys.path.
pip install -e . || echo "editable install skipped"
```

1. Put raw ZNO CSV/XLSX files into `data/`. Filenames **must** include the year, e.g. `OpenData2018.csv`.
2. Launch Jupyter Lab:
    ```bash
    jupyter lab
    ```
3. Open `notebooks/ZNO_Score_Analysis_and_Prediction.ipynb` and run all cells.

## Data expectations
- **File naming:** contains a year token like `2018`, `2021`.
- **Formats supported:** `;`‑delimited CSV/TXT with encodings `cp1251` / `utf‑8` / `latin1`, plus XLS/XLSX.
- **Common columns:** `year`, `eoname`, `areaname`, `regname`, `*ball12`, `*teststatus`, `birth`, `sextypename`, `tername` / `tertypename`.
- **Target:** `average_test_score` built from `*ball12` (zeros treated as missing).

## Highlights
- Robust ingestion and schema harmonization across years and encodings
- Leakage‑aware target and strict temporal split (2016–2020 → 2021)
- Baselines (Dummy), ensembles (RandomForest/XGBoost with early stopping)
- Diagnostics: residuals, calibration, deciles
- Fairness slices with bootstrap 95% CIs
- Feature importance: XGB gain or grouped permutation
- Artifacts saved to `artifacts/`

### Holdout (2021) results
| Model | R² | RMSE | MAE |
|------:|---:|-----:|----:|
| Dummy (median) | -0.012 | 2.190 | 1.795 |
| RandomForest   |  0.226 | 1.915 | 1.573 |
| XGBoost        |  0.236 | 1.903 | 1.562 |

See `artifacts/metrics.json` and the saved `pipeline_*.joblib`.

## Repo tree
```
zno-ml-project/
├─ configs/
│  └─ default.yaml
├─ notebooks/
│  └─ ZNO_Score_Analysis_and_Prediction.ipynb
├─ src/
│  ├─ __init__.py
│  └─ utils.py
├─ tests/
│  ├─ test_data_contract.py
│  └─ test_utils.py
├─ artifacts/        # outputs created at runtime
├─ data/             # put raw files here (ignored by git)
├─ .github/workflows/ci.yml
├─ Makefile
├─ README.md
├─ CONTRIBUTING.md
├─ LICENSE
└─ bootstrap_zno_project.py
```

## Configuration
Edit `configs/default.yaml` to toggle FAST runs and set the random seed:
```yaml
FAST: true
DROP_ZERO_NOSCORE: true
RANDOM_STATE: 42
```

## Testing
```bash
pytest -q
```

## Makefile shortcuts
```bash
make setup     # venv + deps + editable install
make lint      # ruff + black --check + isort --check-only
make format    # black + isort
make test      # pytest
make notebook  # open Jupyter Lab
make clean     # remove caches and build artifacts
```

## Troubleshooting
- **Jupyter not found:** `pip install jupyterlab notebook pyzmq`
- **pyzmq Cython backend import error:** `pip install --upgrade pyzmq`
- **XGBoost wheel missing:** `pip install xgboost` (use Python 3.10/3.11 if build issues occur)
- **Stop Jupyter (Lab/Notebook):** press `Ctrl+C` in the terminal and confirm with `y`

---
© 2025 MIT License
