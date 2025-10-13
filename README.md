# ZNO Outcomes: Leakage‑aware Modeling (2016–2021)

End‑to‑end, leakage‑aware modeling on Ukrainian standardized tests (ZNO).
The pipeline ingests multi‑year raw files, harmonizes schema, enforces a temporal split
(train: 2016–2020 → test: 2021), trains baselines and ensembles with early stopping,
and produces diagnostics, fairness slices with confidence intervals, feature importance,
and artifacts under `artifacts/`.

![CI](https://github.com/<your-username>/zno-ml-project/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)


## Quick start
```bash
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

1. Put raw ZNO CSV/XLSX files into `data/`. Filenames **must** include the year, e.g. `OpenData2018.csv`.
2. Launch Jupyter Lab:
    ```bash
    jupyter lab
    ```
3. Open `notebooks/ZNO_Score_Analysis_and_Prediction.ipynb` and run all cells.

## Highlights
- Robust ingestion and schema harmonization across years and encodings
- Leakage‑aware target and strict temporal split (2016–2020 → 2021)
- Baselines (Dummy), ensembles (RandomForest/XGBoost with early stopping)
- Diagnostics: residuals, calibration, deciles
- Fairness slices with bootstrap 95% CIs
- Feature importance: XGB gain or grouped permutation
- Artifacts saved to `artifacts/`

## Configuration
Edit `configs/default.yaml` to toggle FAST runs and set the random seed:
```yaml
FAST: true
DROP_ZERO_NOSCORE: true
RANDOM_STATE: 42
```

## Testing
```bash
pytest
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

---
© 2025 MIT License

