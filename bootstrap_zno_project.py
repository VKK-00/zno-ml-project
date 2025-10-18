"""
Bootstrap a reproducible ML project for ZNO outcomes with a ready-to-run notebook.

Usage:
    python -X utf8 bootstrap_zno_project.py
"""
from __future__ import annotations

import datetime
import textwrap
from pathlib import Path

import nbformat as nbf

# ----------------------------- scaffold dirs -----------------------------
ROOT = Path(".").resolve()
for p in [
    ROOT/"notebooks",
    ROOT/"src",
    ROOT/"data",
    ROOT/"artifacts",
    ROOT/"tests",
    ROOT/"configs",
    ROOT/".github"/"workflows",
]:
    p.mkdir(parents=True, exist_ok=True)

# ----------------------------- .gitignore -----------------------------
(ROOT/".gitignore").write_text(textwrap.dedent("""\
    __pycache__/
    *.py[cod]
    .ipynb_checkpoints/
    **/.ipynb_checkpoints/*
    .venv/
    .env
    .DS_Store
    data/
    !data/.gitkeep
    artifacts/
    !artifacts/.gitkeep
    .vscode/
    .idea/
"""), encoding="utf-8")
(ROOT/"data"/".gitkeep").write_text("", encoding="utf-8")
(ROOT/"artifacts"/".gitkeep").write_text("", encoding="utf-8")

# ----------------------------- requirements -----------------------------
(ROOT/"requirements.txt").write_text(textwrap.dedent("""\
    pandas>=2.2
    numpy>=1.26
    scipy>=1.10
    scikit-learn>=1.4
    matplotlib>=3.8
    xgboost>=2.0
    shap>=0.46
    pyyaml>=6.0.1
    tqdm>=4.66
    chardet>=5.2
    nbformat>=5.10
    nbclient>=0.10
    jinja2>=3.1
    joblib>=1.3
    openpyxl>=3.1
    xlrd==1.2.0
    packaging>=23.2
    ruff>=0.5
    black>=24.3
    isort>=5.13
    pytest>=8.0
"""), encoding="utf-8")

# ----------------------------- pyproject -----------------------------
(ROOT/"pyproject.toml").write_text(textwrap.dedent("""\
    [project]
    name = "zno-ml-project"
    version = "0.1.0"
    description = "ZNO outcomes: robust ETL, temporal split, models (Ridge/RF/XGB), fairness & importance."
    requires-python = ">=3.10"

    [tool.black]
    line-length = 100
    target-version = ["py310"]

    [tool.isort]
    profile = "black"
    line_length = 100

    [tool.ruff]
    line-length = 100
    select = ["E", "F", "I"]
    ignore = ["E501"]

    [tool.pytest.ini_options]
    addopts = "-q"
    testpaths = ["tests"]
"""), encoding="utf-8")

# ----------------------------- config defaults -----------------------------
(ROOT/"configs"/"default.yaml").write_text(textwrap.dedent("""\
    FAST: true
    DROP_ZERO_NOSCORE: true
    RANDOM_STATE: 42
"""), encoding="utf-8")

# ----------------------------- README -----------------------------
year = datetime.datetime.now().year
(ROOT/"README.md").write_text(textwrap.dedent(f"""\
    # ZNO Outcomes: Leakage‑aware Modeling (2016–2021)

    End‑to‑end, leakage‑aware modeling on Ukrainian standardized tests (ZNO).
    The pipeline ingests multi‑year raw files, harmonizes schema, enforces a temporal split
    (train: 2016–2020 → test: 2021), trains baselines and ensembles with early stopping,
    and produces diagnostics, fairness slices with confidence intervals, feature importance,
    and artifacts under `artifacts/`.

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
    © {year} MIT License
"""), encoding="utf-8")

# ----------------------------- CONTRIBUTING -----------------------------
(ROOT/"CONTRIBUTING.md").write_text(textwrap.dedent("""\
    # Contributing

    Thanks for your interest in improving this project. The goal is to keep it interview‑ready,
    reproducible, and educational.

    ## Environment
    - Python 3.10+
    - Create a virtual environment and install dependencies from `requirements.txt`.
    - Install the package in editable mode: `pip install -e .`

    ## Style and quality
    - Code must be readable and well‑commented.
    - Follow Black (line length 100), isort, and Ruff (E, F, I).
    - Prefer explicit, descriptive variable names; avoid abbreviations that are not common in ML.

    ## Testing
    - Add or update unit tests under `tests/`.
    - Keep tests fast and deterministic.
    - Run `pytest` before submitting changes.

    ## Data privacy
    - Do not commit raw data. `.gitignore` excludes the `data/` folder by default.
    - Mask or aggregate any sensitive fields before sharing examples.

    ## Pull requests
    - Keep PRs focused and small.
    - Provide a clear description and screenshots where relevant.
    - Ensure CI passes: lint, format, and tests.
"""), encoding="utf-8")

# ----------------------------- Makefile -----------------------------
(ROOT/"Makefile").write_text(textwrap.dedent(r"""\
    .PHONY: setup lint format test notebook clean

    VENV?=.venv

    setup:
    \tpython -m venv $(VENV)
    \t# Windows PowerShell: . $(VENV)/Scripts/Activate.ps1 ; pip install -r requirements.txt ; pip install -e .
    \t# macOS/Linux:
    \t# source $(VENV)/bin/activate ; pip install -r requirements.txt ; pip install -e .

    lint:
    \truff check .
    \tblack --check --line-length 100 .
    \tisort --check-only --line-length 100 .

    format:
    \tblack --line-length 100 .
    \tisort --line-length 100 .

    test:
    \tpytest -q

    notebook:
    \tjupyter lab

    clean:
    \trm -rf __pycache__ .pytest_cache .ruff_cache .ipynb_checkpoints
    \tfind . -name "*.pyc" -delete
"""), encoding="utf-8")

# ----------------------------- LICENSE -----------------------------
(ROOT/"LICENSE").write_text(textwrap.dedent("""\
    MIT License

    Copyright (c) 2025 Volodymyr Ksienich

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""), encoding="utf-8")

# ----------------------------- src package -----------------------------
(ROOT/"src"/"__init__.py").write_text(textwrap.dedent("""\
    from .utils import (
        log, discover_data_files, load_year_file, to_float12, build_average_12,
        merge_math_strat, enrich_age, normalize_language_missing, finalize_teststatus,
        post_harmonize, attach_school_history_oot, RANDOM_STATE,
        safe_sample, na_report_by_year, numeric_summary, cat_cardinality,
        target_by_group, top_n_categories, completeness_table, set_matplotlib_cyrillic,
    )

    __all__ = [
        "log","discover_data_files","load_year_file","to_float12","build_average_12",
        "merge_math_strat","enrich_age","normalize_language_missing","finalize_teststatus",
        "post_harmonize","attach_school_history_oot","RANDOM_STATE","safe_sample",
        "na_report_by_year","numeric_summary","cat_cardinality","target_by_group",
        "top_n_categories","completeness_table","set_matplotlib_cyrillic",
    ]
"""), encoding="utf-8")

# Full, commented utils.py (matching the user's provided implementation)
(ROOT/"src"/"utils.py").write_text(textwrap.dedent(r'''\
    from __future__ import annotations

    import re
    from pathlib import Path
    from typing import Iterable, List, Optional, Tuple, Union

    import numpy as np
    import pandas as pd


    # ---------------------------- basics ----------------------------

    def log(msg: str) -> None:
        """Lightweight project logger for consistent console messages."""
        print(f"[utils] {msg}")


    RANDOM_STATE: int = 42
    POSSIBLE_ENCODINGS: Tuple[str, ...] = ("cp1251", "utf-8", "latin1")
    BALL12_SUFFIX = "ball12"

    # year-agnostic light canonicalization of header prefixes
    CANONICAL_MAP = {r"^fra": "fr", r"^spa": "sp"}


    def canonicalize_columns(columns: Iterable[str]) -> List[str]:
        """Normalize column names: lowercase, strip whitespace, collapse spaces, and apply simple canonical maps."""
        out = []
        for c in columns:
            col = re.sub(r"\s+", "", str(c).strip().lower())
            for pat, repl in CANONICAL_MAP.items():
                if re.match(pat, col):
                    col = re.sub(pat, repl, col)
            out.append(col)
        return out


    def _year_aware_rename(cols: List[str], year: int) -> List[str]:
        """Handle 'ukr' vs 'uml' naming drift across years."""
        # keep 'uml*' for <=2020, 'ukr*' otherwise
        if year <= 2020:
            return [col.replace("ukr", "uml") if "ukr" in col else col for col in cols]
        return cols


    def to_float12(series: pd.Series) -> pd.Series:
        """Parse 12-point scores from messy text to float, mapping '0' to 0.0 and missing tokens to NaN."""
        return (
            series.astype(str)
            .str.replace(",", ".", regex=False)
            .str.replace(" ", "", regex=False)
            .replace({"nan": np.nan, "None": np.nan, "": np.nan})
            .astype(float)
        )


    def _collapse_duplicate_columns_inplace(df: pd.DataFrame) -> pd.DataFrame:
        """Collapse duplicate-named columns. For *ball12 columns, take row-wise max after zero→NaN masking."""
        if not df.columns.duplicated().any():
            return df
        dup_names = pd.Index(df.columns)[df.columns.duplicated()].unique().tolist()
        for name in dup_names:
            same = [c for c in df.columns if c == name]
            if len(same) < 2:
                continue
            if name.endswith(BALL12_SUFFIX):
                vals = df[same].apply(to_float12).where(lambda x: x != 0, np.nan)
                df[name] = vals.max(axis=1, skipna=True)
            for col in same[1:]:
                df.drop(columns=col, inplace=True)
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep="first")]
        return df


    def load_year_file(path: Path, year: int) -> pd.DataFrame:
        """Read a single year's raw CSV/TXT/XLS(X), guess encodings for CSV, canonicalize headers, add `year`."""
        suf = path.suffix.lower()
        if suf in (".csv", ".txt"):
            last_err = None
            df = None
            for enc in POSSIBLE_ENCODINGS:
                try:
                    df = pd.read_csv(path, delimiter=";", encoding=enc, low_memory=False)
                    break
                except Exception as e:
                    last_err = e
            if df is None:
                raise last_err  # type: ignore
        elif suf in (".xls", ".xlsx"):
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file type: {path}")

        cols = canonicalize_columns(df.columns)
        cols = _year_aware_rename(cols, year)
        df.columns = cols
        df["year"] = year
        df = _collapse_duplicate_columns_inplace(df)
        return df


    def find_year_from_name(name: str) -> Optional[int]:
        """Extract a year token from a filename like 'OpenData2018.csv'."""
        m = re.search(r"(20(1[6-9]|2[0-9]))", name)
        return int(m.group(1)) if m else None


    def discover_data_files(data_dir: Union[str, Path]) -> list[tuple[Path, int]]:
        """Return sorted list of (path, year) pairs for all files in `data_dir` whose filenames contain a year."""
        data_dir = Path(data_dir)
        pairs: list[tuple[Path, int]] = []
        for f in data_dir.glob("*.*"):
            y = find_year_from_name(f.name)
            if y is not None:
                pairs.append((f, y))
        return sorted(pairs, key=lambda x: x[1])


    # ---------------------------- core harmonization ----------------------------

    def build_average_12(df: pd.DataFrame) -> pd.Series:
        """Leakage-safe target: mean of available *ball12 columns per row, treating 0 as missing."""
        cols = [c for c in df.columns if c.endswith(BALL12_SUFFIX)]
        if not cols:
            return pd.Series(np.nan, index=df.index, name="average_test_score")
        tmp = df[cols].apply(to_float12)
        tmp = tmp.where(~(tmp == 0), np.nan)
        try:
            tmp = tmp.groupby(level=0, axis=1).max()
        except Exception:
            tmp = tmp.loc[:, ~tmp.columns.duplicated()]
        return tmp.mean(axis=1, skipna=True).fillna(0.0).rename("average_test_score")


    def merge_math_strat(df: pd.DataFrame) -> pd.DataFrame:
        """Prefer stratified math score when available, otherwise fall back to general math score."""
        if "mathstball12" in df.columns and "mathball12" in df.columns:
            a = to_float12(df["mathstball12"]).fillna(0)
            b = to_float12(df["mathball12"]).fillna(0)
            df["mathball12"] = np.where(a > 0, a, b)
        return df


    def enrich_age(df: pd.DataFrame) -> pd.DataFrame:
        """Add 'age' column as `year - birth` when both are available."""
        if "birth" in df.columns and "year" in df.columns:
            with np.errstate(invalid="ignore"):
                df["age"] = df["year"].astype(float) - df["birth"].astype(float)
        return df


    def ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Guarantee presence of expected high-level features across all years."""
        for col in ["classprofilename", "classlangname", "tertypename"]:
            if col not in df.columns:
                df[col] = np.nan
        return df


    def derive_tertypename(df: pd.DataFrame) -> pd.DataFrame:
        """
        Harmonize settlement type to a unified set across years:
            'місто' | 'селище міського типу' | 'селище, село' | 'unknown'
        """
        CANON_MAP = {
            "місто": "місто",
            "м.": "місто",
            "город": "місто",
            "смт": "селище міського типу",
            "селище міського типу": "селище міського типу",
            "село": "селище, село",
            "с.": "селище, село",
            "селище": "селище, село",
            "селище, село": "селище, село",
        }

        if "tertypename" not in df.columns:
            df["tertypename"] = pd.NA

        s = pd.Series(df["tertypename"], copy=False).astype("string").str.lower().str.strip()
        s = s.replace(CANON_MAP)
        df["tertypename"] = s

        if "tername" in df.columns:
            tn = df["tername"].astype("string").str.lower().str.strip()
            is_city = tn.str.contains(r"(^|\s)(м\.|місто)\b", regex=True)
            is_urban = tn.str.contains(r"\bсмт\b", regex=True)
            is_rural = tn.str.contains(r"(^|\s)(с\.|село|селище)\b", regex=True)

            fill = pd.Series(index=df.index, dtype="string")
            fill[is_city] = "місто"
            fill[is_urban] = "селище міського типу"
            fill[is_rural] = "селище, село"

            df["tertypename"] = df["tertypename"].fillna(fill)

        df["tertypename"] = df["tertypename"].fillna("unknown").astype("string")
        return df


    def normalize_language_missing(df: pd.DataFrame) -> pd.DataFrame:
        """Fill NA language fields with 'unknown' and coerce to pandas string dtype."""
        lang_cols = [c for c in df.columns if c.endswith("lang") or "langname" in c]
        if not lang_cols:
            return df
        for c in lang_cols:
            df[c] = pd.Series(df[c], copy=False).astype("string").fillna("unknown")
        return df


    # Patterns to normalize language names (lowercased)
    LANG_NORMALIZATION_PATTERNS: list[tuple[str, str]] = [
        (r"^\s*(укр|україн(ська)?).*", "українська"),
        (r"^\s*(рос|рус|росій(ська)?).*", "російська"),
        (r"^\s*(англ|english|англій(ська)?).*", "англійська"),
        (r"^\s*(нім|нем|німецьк(а)?).*", "німецька"),
        (r"^\s*(франц|француз(ька)?).*", "французька"),
        (r"^\s*(ісп|исп|іспанськ(а)?).*", "іспанська"),
        (r"^\s*(румун|rumun).*", "румунська"),
        (r"^\s*(польськ|pol).*", "польська"),
        (r"^\s*(угор|мадяр).*", "угорська"),
        (r"^\s*(болгар).*", "болгарська"),
        (r"^\s*(грецьк|грец).*", "грецька"),
    ]

    SUBJECT_PREFIXES = ["uml", "ukr", "hist", "math", "phys", "chem", "bio", "geo", "eng", "fr", "deu", "sp", "rus"]


    def subject_language_columns(df: pd.DataFrame) -> list[str]:
        """Return columns like 'histlang', 'mathlang', ... that reflect exam language."""
        cols: list[str] = []
        for p in SUBJECT_PREFIXES:
            c = f"{p}lang"
            if c in df.columns:
                cols.append(c)
        return cols


    def derive_testlanguage(df: pd.DataFrame, *, out_col: str = "testlanguage") -> pd.DataFrame:
        """
        Build one row-level 'testlanguage' from all '*lang' / '*langname' columns.
        If still 'unknown', fall back to the mode across per-subject '*lang' columns.
        """
        # If already present with any non-null values, keep it (user override)
        if out_col in df.columns and df[out_col].notna().any():
            return df

        lang_cols = [c for c in df.columns if c.endswith("lang") or c.endswith("langname")]
        if not lang_cols:
            df[out_col] = pd.Series("unknown", index=df.index, dtype="string")
            return df

        # 1) normalize all candidate columns
        langs = df[lang_cols].copy().astype("string")
        langs = langs.apply(lambda s: s.str.strip().str.lower())
        langs = langs.replace({"unknown": pd.NA, "nan": pd.NA, "none": pd.NA, "": pd.NA})
        for pat, repl in LANG_NORMALIZATION_PATTERNS:
            langs = langs.replace(pat, repl, regex=True)

        # 2) row-wise mode
        mode_df = langs.mode(axis=1, dropna=True)
        out = mode_df[0].astype("string")

        # 3) fallback: per-subject '*lang' columns only (less noisy than classlangname)
        subs = subject_language_columns(df)
        if subs:
            subs_df = (
                df[subs]
                .astype("string")
                .apply(lambda s: s.str.strip().str.lower())
                .replace({"unknown": pd.NA, "nan": pd.NA, "none": pd.NA, "": pd.NA})
            )
            for pat, repl in LANG_NORMALIZATION_PATTERNS:
                subs_df = subs_df.replace(pat, repl, regex=True)
            subs_mode = subs_df.mode(axis=1, dropna=True)
            out = out.fillna(subs_mode[0])

        df[out_col] = out.fillna("unknown").astype("string")
        return df


    def finalize_teststatus(df: pd.DataFrame) -> pd.DataFrame:
        """Derive per-subject 'has score' / 'no score' flags aligned with *ball12 values."""
        for name in [c for c in df.columns if c.endswith(BALL12_SUFFIX)]:
            status_col = name.replace(BALL12_SUFFIX, "teststatus")
            if status_col in df.columns:
                obj = df.loc[:, name]
                if isinstance(obj, pd.DataFrame):
                    conv = obj.apply(to_float12).where(lambda x: x != 0, np.nan)
                    vals = conv.max(axis=1, skipna=True).fillna(0)
                else:
                    vals = to_float12(obj).fillna(0)
                    vals = vals.where(vals != 0, np.nan).fillna(0)
                df[status_col] = np.where(vals > 0, "has score", "no score")
        return df


    # ---------------------------- DPA / participation ----------------------------

    def _any_notna(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
        """Return a boolean Series: whether any of the given columns is non-null per row."""
        acc = pd.Series(False, index=df.index)
        for c in cols:
            if c in df.columns:
                acc = acc | df[c].notna()
        return acc


    def _has_score_col(df: pd.DataFrame, prefix: str) -> pd.Series:
        """Check whether a subject has a positive score, inferring from teststatus or *ball12 when needed."""
        st = f"{prefix}teststatus"
        b12 = f"{prefix}ball12"
        if st in df.columns:
            return df[st].astype(str).str.lower().eq("has score")
        if b12 in df.columns:
            s = to_float12(df[b12])
            return s.fillna(0).gt(0)
        return pd.Series(False, index=df.index)


    def derive_dpa_flags(df: pd.DataFrame) -> pd.DataFrame:
        """
        Year-aware flags for participation & required DPA bundles.
            * 2018–2020: required = UML + (MATH or HIST)
            * 2021+:     required = (UKR or UML) + MATH + (HIST or any foreign)
        """
        took = {p: _has_score_col(df, p) for p in SUBJECT_PREFIXES}

        took_ukr_any = took["ukr"] | took["uml"]
        took_foreign_any = took["eng"] | took["fr"] | took["deu"] | took["sp"]

        all_taken = pd.concat([t.astype(int) for t in took.values()], axis=1)
        n_tests_taken = all_taken.sum(axis=1)
        n_foreign_taken = (
            took["eng"].astype(int) + took["fr"].astype(int) + took["deu"].astype(int) + took["sp"].astype(int)
        )

        y = df["year"].astype(int) if "year" in df.columns else pd.Series(0, index=df.index)
        m_pre2021 = y.le(2020)

        req_pre2021_met = took_ukr_any & (took["math"] | took["hist"])
        req_2021_met = took_ukr_any & took["math"] & (took["hist"] | took_foreign_any)
        dpa_required_done = np.where(m_pre2021, req_pre2021_met, req_2021_met).astype(bool)

        count_pre2021 = took_ukr_any.astype(int) + ((took["math"] | took["hist"]).astype(int))
        count_2021 = took_ukr_any.astype(int) + took["math"].astype(int) + ((took["hist"] | took_foreign_any).astype(int))
        dpa_required_count_met = np.where(m_pre2021, count_pre2021, count_2021)

        df = df.copy()
        for p in ["ukr", "uml", "math", "hist", "phys", "chem", "bio", "geo", "eng", "fr", "deu", "sp", "rus"]:
            df[f"took_{p}"] = took[p].astype(int)
        df["took_ukr_any"] = took_ukr_any.astype(int)
        df["took_foreign_any"] = took_foreign_any.astype(int)
        df["n_tests_taken"] = n_tests_taken.astype(int)
        df["n_foreign_taken"] = n_foreign_taken.astype(int)
        df["dpa_required_done"] = dpa_required_done.astype(int)
        df["dpa_required_count_met"] = dpa_required_count_met.astype(int)
        return df


    def post_harmonize(df: pd.DataFrame) -> pd.DataFrame:
        """One pass after raw ingestion — year-agnostic, leakage-safe derived features."""
        df = ensure_feature_columns(df)
        df = derive_tertypename(df)
        df = normalize_language_missing(df)
        df = derive_testlanguage(df)
        df = derive_dpa_flags(df)
        return df


    # ---------------------------- OOT school history (+ fallbacks) ----------------------------

    def attach_school_history_oot(
        df: pd.DataFrame,
        *,
        target_col: str = "average_test_score",
        id_col: str = "eoname",
        area_col: str = "areaname",
        region_col: str = "regname",
        ref_year: int = 2021,
    ) -> pd.DataFrame:
        """
        Attach past-years (strictly < ref_year) history to ref_year rows:
            - school_hist_mean / school_hist_count by school (id_col)
            - if missing → fill from area (areaname), else from region (regname)
        """
        if df.empty or target_col not in df.columns or "year" not in df.columns:
            return df

        out = df.copy()
        past_mask = (out["year"] < ref_year) & (out[target_col] > 0)

        # keys with sentinels to keep NA buckets
        school_key = out[id_col].fillna("__unknown__") if id_col in out.columns else pd.Series("__unknown__", index=out.index)
        area_key = out[area_col].fillna("__unknown__") if area_col in out.columns else pd.Series("__unknown__", index=out.index)
        region_key = out[region_col].fillna("__unknown__") if region_col in out.columns else pd.Series("__unknown__", index=out.index)

        # school-level history
        hist_school = (
            out.loc[past_mask, [target_col]]
            .groupby(school_key[past_mask], dropna=False)
            .agg(school_hist_mean=(target_col, "mean"), school_hist_count=(target_col, "size"))
            .reset_index(names="school_key")
        )

        # area-level history
        hist_area = (
            out.loc[past_mask, [target_col]]
            .groupby(area_key[past_mask], dropna=False)
            .agg(area_hist_mean=(target_col, "mean"), area_hist_count=(target_col, "size"))
            .reset_index(names="area_key")
        )

        # region-level history
        hist_region = (
            out.loc[past_mask, [target_col]]
            .groupby(region_key[past_mask], dropna=False)
            .agg(region_hist_mean=(target_col, "mean"), region_hist_count=(target_col, "size"))
            .reset_index(names="region_key")
        )

        # attach to 2021 rows only
        m_ref = out["year"].eq(ref_year)
        if not m_ref.any():
            log(f"attach_school_history_oot: no rows for year {ref_year}, skipping.")
            return out

        join = pd.DataFrame(
            {
                "school_key": school_key[m_ref].to_numpy(),
                "area_key": area_key[m_ref].to_numpy(),
                "region_key": region_key[m_ref].to_numpy(),
            },
            index=out.index[m_ref],
        )
        join = join.merge(hist_school, on="school_key", how="left")
        join = join.merge(hist_area, on="area_key", how="left")
        join = join.merge(hist_region, on="region_key", how="left")

        # fill school_* with area_* then region_* (counts fill the same way)
        school_mean_filled = (
            join["school_hist_mean"].fillna(join["area_hist_mean"]).fillna(join["region_hist_mean"])
        )
        school_count_filled = (
            join["school_hist_count"].fillna(join["area_hist_count"]).fillna(join["region_hist_count"])
        )

        # write back to the 2021 rows
        out.loc[m_ref, "school_hist_mean"]  = school_mean_filled.to_numpy()
        out.loc[m_ref, "school_hist_count"] = school_count_filled.to_numpy()

        # ---- safe, explicit counts for logging ----
        area_fills = int(((join["school_hist_count"].isna()) & (join["area_hist_count"].notna())).sum())
        region_fills = int((
            (join["school_hist_count"].isna())
            & (join["area_hist_count"].isna())
            & (join["region_hist_count"].notna())
        ).sum())

        log(
            "[enrich] OOT school history attached to "
            f"{int(m_ref.sum())} rows | "
            f"school coverage: {int(join['school_hist_count'].notna().sum())} | "
            f"filled via area: {area_fills} | "
            f"filled via region: {region_fills}"
        )
        return out


    # ---------------------------- light EDA helpers used in notebook ----------------------------

    def safe_sample(df: pd.DataFrame, n: int = 300_000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
        """Return at most n rows as a reproducible sample; return df if already small."""
        if len(df) <= n:
            return df
        return df.sample(n=n, random_state=random_state)


    def na_report_by_year(df: pd.DataFrame, top_k: int = 25) -> pd.DataFrame:
        """Report top missing columns per year with counts and percentages."""
        if "year" not in df.columns:
            raise ValueError("na_report_by_year: 'year' column is required.")
        out = []
        for y, g in df.groupby("year"):
            cnt = g.isna().sum().sort_values(ascending=False)
            pct = (cnt / g.shape[0] * 100).round(2)
            top = pd.DataFrame({"year": y, "na_count": cnt, "na_pct": pct})
            # carry the column name explicitly
            top = top.head(top_k)
            top = top.assign(column=top.index.tolist()).reset_index(drop=True)
            out.append(top[["year", "column", "na_count", "na_pct"]])
        return pd.concat(out, ignore_index=True)


    def numeric_summary(df: pd.DataFrame, cols: Optional[list[str]] = None) -> pd.DataFrame:
        """Describe numeric columns with robust percentiles and missingness."""
        if cols is None:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not cols:
            return pd.DataFrame()
        desc = df[cols].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
        desc["missing_pct"] = (1 - df[cols].notna().mean()).values * 100
        return desc


    def cat_cardinality(df: pd.DataFrame, cols: Optional[list[str]] = None) -> pd.DataFrame:
        """Return cardinalities for categorical columns (object/string)."""
        if cols is None:
            cols = [c for c in df.columns if str(df[c].dtype).startswith(("object", "string"))]
        rows = []
        for c in cols:
            vc = df[c].astype("string").value_counts(dropna=True)
            rows.append({"column": c, "cardinality": int(vc.shape[0])})
        return pd.DataFrame(rows).sort_values("cardinality", ascending=False, ignore_index=True)


    def target_by_group(df: pd.DataFrame, target: str = "average_test_score", group: str = "year") -> pd.DataFrame:
        """Aggregate target by a categorical group with standard stats."""
        if target not in df.columns or group not in df.columns:
            return pd.DataFrame()
        return (df.groupby(group)[target]
                    .agg(["count", "mean", "std", "median", "min", "max"])
                    .sort_index()
                    .reset_index())


    def top_n_categories(df: pd.DataFrame, col: str, n: int = 15) -> pd.DataFrame:
        """Show top-n categories by frequency with share in percent."""
        vc = df[col].astype("string").value_counts(dropna=False).head(n)
        out = vc.to_frame("count")
        out["pct"] = (out["count"] / len(df) * 100).round(2)
        out.index.name = col
        return out.reset_index()


    def completeness_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Return completeness ratios (0..1) for selected columns."""
        res = {c: float(df[c].notna().mean()) for c in cols if c in df.columns}
        return (pd.Series(res, name="completeness")
                    .sort_values(ascending=False)
                    .reset_index()
                    .rename(columns={"index": "column"}))


    def set_matplotlib_cyrillic(
        prefer: tuple[str, ...] = (
            "Noto Sans", "DejaVu Sans", "Arial Unicode MS", "Liberation Sans", "Arial", "FreeSans"
        ),
        verbose: bool = False,
        silence_findfont: bool = True,
    ) -> None:
        """
        Configure Matplotlib to render Cyrillic cleanly without a flood of findfont warnings.
        Picks the first actually installed font from `prefer` and uses it.
        """
        import warnings
        import matplotlib as mpl
        from matplotlib import font_manager as fm

        # All available font family names on this system
        available = {f.name for f in fm.fontManager.ttflist}

        # Keep only preferred fonts that are installed
        installed = [name for name in prefer if name in available]
        if not installed:
            # DejaVu Sans ships with Matplotlib — safe default
            installed = ["DejaVu Sans"]

        # Configure: one primary family + fallbacks (only installed ones)
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = installed
        mpl.rcParams["axes.unicode_minus"] = False  # render minus as a normal glyph

        # Optionally suppress findfont warnings
        if silence_findfont:
            warnings.filterwarnings("ignore", message=".*findfont:.*")

        if verbose:
            print(f"[utils] Cyrillic font = {installed[0]} | candidates used: {installed}")
    '''), encoding="utf-8")

# ----------------------------- tests -----------------------------
(ROOT/"tests"/"test_data_contract.py").write_text(textwrap.dedent("""\
    from pathlib import Path
    from src.utils import discover_data_files, load_year_file

    def test_contract_smoke():
        pairs = discover_data_files(Path('data'))
        if not pairs:
            return
        p, year = pairs[0]
        df = load_year_file(p, year)
        assert isinstance(df.shape, tuple)
"""), encoding="utf-8")

(ROOT/"tests"/"test_utils.py").write_text(textwrap.dedent("""\
    import numpy as np
    import pandas as pd
    from src.utils import (
        build_average_12, derive_testlanguage, attach_school_history_oot,
        derive_dpa_flags
    )

    def toy_df():
        # two years, two schools, simple scores
        df = pd.DataFrame({
            "year": [2019, 2020, 2021, 2021],
            "eoname": ["A","A","A","B"],
            "areaname": ["area1","area1","area1","area2"],
            "regname": ["reg1","reg1","reg1","reg2"],
            "umlball12": [6, 8, np.nan, 7],
            "histball12": [np.nan, 10, 9, np.nan],
            "mathball12": [5, np.nan, 6, 3],
            "umlteststatus": ["has score","has score","has score","has score"],
            "mathteststatus": ["has score","no score","has score","has score"],
            "histteststatus": ["no score","has score","has score","no score"],
        })
        df["average_test_score"] = build_average_12(df)
        return df

    def test_build_average_12():
        df = toy_df()
        assert "average_test_score" in df.columns
        assert df["average_test_score"].iloc[0] > 0

    def test_derive_testlanguage_mode_and_unknown():
        df = toy_df()
        df["englang"] = ["англійська", None, None, None]
        df = derive_testlanguage(df)
        assert (df["testlanguage"] == "англійська").iloc[0]
        assert df["testlanguage"].notna().all()

    def test_attach_school_history_no_leakage():
        df = toy_df()
        out = attach_school_history_oot(df, ref_year=2021)
        # history only for 2021 rows
        assert out.loc[out["year"] < 2021, ["school_hist_mean","school_hist_count"]].isna().all().all()
        # school A has history; B should fallback via area/reg or remain NA
        assert pd.notna(out.loc[(out["year"]==2021) & (out["eoname"]=="A"), "school_hist_mean"]).all()

    def test_derive_dpa_flags_shapes():
        df = toy_df()
        out = derive_dpa_flags(df)
        for col in ["took_uml","took_math","took_hist","took_foreign_any","n_tests_taken","dpa_required_done"]:
            assert col in out.columns
"""), encoding="utf-8")

# ----------------------------- CI workflow -----------------------------
(ROOT/".github"/"workflows"/"ci.yml").write_text(textwrap.dedent("""\
    name: CI
    on: [push, pull_request]
    jobs:
        lint-test:
        runs-on: ubuntu-latest
        steps:
            - uses: actions/checkout@v4
            - uses: actions/setup-python@v5
            with:
                python-version: '3.11'
            - run: python -m pip install -U pip
            - run: pip install -U -r requirements.txt
            - run: pip install -e .
            - run: ruff check .
            - run: black --check --line-length 100 .
            - run: isort --check-only --line-length 100 .
            - run: pytest -q
"""), encoding="utf-8")

# ----------------------------- Notebook -----------------------------
nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(
        "# ZNO Outcomes: Analysis & ML Prediction (2016–2021)\n"
        "Notebook goal: End-to-end, leakage-aware modeling of Ukrainian standardized tests (ZNO) across 2016–2021.\n\n"
        "The pipeline ingests multi-year raw files, harmonizes schema, enforces a temporal split "
        "(train: 2016–2020 → test: 2021), trains baselines and tree ensembles with early stopping, "
        "then produces diagnostics, fairness slices with CI, feature importance, and artifacts.\n\n"
        "All artifacts are saved under `artifacts/`."
    ),
    nbf.v4.new_markdown_cell("## ====================== Cell 1: Setup, imports, project root, config ======================"),
    nbf.v4.new_code_cell(textwrap.dedent(r'''
        from __future__ import annotations

        # ---- Standard library ----
        import json
        import sys
        import time
        import warnings
        from inspect import signature
        from pathlib import Path
        from time import perf_counter
        from typing import Dict, Tuple

        # ---- Third-party ----
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import sklearn
        import yaml
        from joblib import dump
        from packaging import version
        from scipy import sparse
        from scipy.stats import loguniform
        from sklearn.base import clone
        from sklearn.compose import ColumnTransformer
        from sklearn.dummy import DummyRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge, SGDRegressor  # SGD is imported for parity with prior variants
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import GroupKFold, RandomizedSearchCV, train_test_split
        from sklearn.pipeline import Pipeline, make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        # ---- XGBoost with EarlyStopping compatibility handling ----
        try:
            from xgboost import XGBRegressor
            try:
                # XGBoost ≥1.7 exposes a callback-based EarlyStopping
                from xgboost.callback import EarlyStopping as XGBEarlyStopping
            except Exception:
                XGBEarlyStopping = None
            HAS_XGB = True
        except Exception:
            HAS_XGB = False
            XGBEarlyStopping = None

        def add_project_root_to_syspath(hops: int = 3):
            """
            Walk up at most `hops` directories from the current working directory and
            insert the first folder that contains a `src` directory to `sys.path`.
            """
            here = Path.cwd().resolve()
            candidates = [here]
            for _ in range(hops):
                candidates.append(candidates[-1].parent)

            for base in candidates:
                if (base / "src").exists():
                    if str(base) not in sys.path:
                        sys.path.insert(0, str(base))
                    print(f"[bootstrap] Project root set to: {base}")
                    return base

            raise RuntimeError(
                "Could not find a folder with `src` within the first "
                f"{hops} parent hops from {here}"
            )

        PROJECT_ROOT = add_project_root_to_syspath()

        # ---- Project utilities (feature engineering, ingestion, helpers) ----
        import src.utils as utils
        from importlib import reload
        reload(utils)
        from src.utils import (
            log, discover_data_files, load_year_file, build_average_12,
            merge_math_strat, enrich_age, finalize_teststatus, post_harmonize,
            safe_sample, na_report_by_year, numeric_summary, cat_cardinality,
            target_by_group, top_n_categories, completeness_table, set_matplotlib_cyrillic,
            attach_school_history_oot, RANDOM_STATE
        )

        # Configure Matplotlib to render Cyrillic cleanly (no font spam in logs)
        set_matplotlib_cyrillic(verbose=True)

        # ---- Pandas display knobs for wide tables ----
        pd.options.display.max_columns = 200
        pd.options.display.width = 180
        warnings.filterwarnings("ignore")

        # ---- Data & artifact directories ----
        DATA_DIR = next((d for d in [PROJECT_ROOT/'data', PROJECT_ROOT.parent/'data'] if d.exists()),
                        PROJECT_ROOT/'data')
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        ARTIFACTS_DIR = PROJECT_ROOT/'artifacts'
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        # ---- Runtime knobs (defaults) ----
        FAST = True                # toggle False for a slower, thorough run
        DROP_ZERO_NOSCORE = True   # drop rows with no positive scores
        RANDOM_STATE = 42
        np.random.seed(RANDOM_STATE)

        # ---- Optional config override: configs/default.yaml ----
        CFG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
        if CFG_PATH.exists():
            cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
            FAST = bool(cfg.get("FAST", FAST))
            DROP_ZERO_NOSCORE = bool(cfg.get("DROP_ZERO_NOSCORE", DROP_ZERO_NOSCORE))
            RANDOM_STATE = int(cfg.get("RANDOM_STATE", RANDOM_STATE))
            np.random.seed(RANDOM_STATE)
            log(f"[config] Loaded {CFG_PATH} -> FAST={FAST}, DROP_ZERO_NOSCORE={DROP_ZERO_NOSCORE}, RS={RANDOM_STATE}")

        print("Project root:", PROJECT_ROOT)
        print("Data dir    :", DATA_DIR)
        print("Artifacts   :", ARTIFACTS_DIR)
        print("FAST mode   :", FAST)
    ''')),
    nbf.v4.new_markdown_cell("## ====================== Cell 2: Ingestion & year-aware harmonization ======================"),
    nbf.v4.new_code_cell(textwrap.dedent(r'''
        # Reads all raw files in data/ whose filenames contain a year, applies canonicalization,
        # merges subject variants, builds a leakage-safe target, derives age, normalizes teststatus,
        # and harmonizes derived features including settlement and testing language.

        pairs = discover_data_files(DATA_DIR)
        if not pairs:
            print("⚠️ Put raw ZNO files into data/ — filenames must include the year, e.g. OpenData2018.csv")

        frames = []
        t0 = perf_counter()
        for path, year in pairs:
            df = load_year_file(path, year)      # encoding-safe read + canonical headers
            df = merge_math_strat(df)            # unify math score variants
            df["average_test_score"] = build_average_12(df)  # leakage-aware 12-point average
            df = enrich_age(df)                  # 'age' = year - birth
            df = finalize_teststatus(df)         # robust 'has score' vs 'no score'
            df = post_harmonize(df)              # year-agnostic derived features (e.g., tertypename, testlanguage)
            frames.append(df)
        data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        # Attach out-of-time (pre-2021) school history to 2021 rows only, with area/region fallbacks.
        data = attach_school_history_oot(
            data,
            target_col="average_test_score",
            id_col="eoname",
            area_col="areaname",
            region_col="regname",
            ref_year=2021,
        )

        # Quick peek at participation / DPA flags (sanity check)
        if not data.empty:
            display(data.filter(regex="^took_|^n_|^dpa_required").head())

        print(f"[done] ingestion+harmonization in {perf_counter() - t0:,.1f}s | shape={getattr(data, 'shape', None)}")
        display(data.head(3))
    ''')),
    nbf.v4.new_markdown_cell("## ====================== Cell 3: Exploratory Data Analysis (EDA) ======================"),
    nbf.v4.new_code_cell(textwrap.dedent(r'''
        # High-level structure
        set_matplotlib_cyrillic()  # ensure Cyrillic labels render
        print("Shape:", data.shape)
        print("Years:", sorted(data["year"].dropna().unique().tolist()))
        print("Rows by year:\n", data["year"].value_counts().sort_index())

        # Year-wise summary of the target
        display(target_by_group(data, "average_test_score", "year"))

        # Cardinality of important categoricals (drives OHE width)
        display(cat_cardinality(data, cols=["sextypename","regname","areaname","tertypename","testlanguage"]).head(10))

        # Top missing fields per year (quick completeness audit)
        miss = na_report_by_year(data, top_k=20)
        display(miss.head(60))

        # Numeric summaries for a focused subset of columns used downstream
        num_cols_for_summary = [
            c for c in ["average_test_score", "age", "n_tests_taken", "n_foreign_taken",
                        "dpa_required_done", "dpa_required_count_met",
                        "school_hist_mean", "school_hist_count"] if c in data.columns
        ]
        display(numeric_summary(data, cols=num_cols_for_summary))

        # Target distribution and simple conditioning (sampled for performance)
        S = safe_sample(data, n=500_000, random_state=RANDOM_STATE)
        if S.empty:
            raise ValueError("DataFrame is empty after ingestion—check data paths and file patterns.")

        fig, ax = plt.subplots(1, 3, figsize=(15, 4))

        # Global histogram
        axs = ax[0]
        axs.hist(S["average_test_score"].dropna(), bins=60)
        axs.set_title("Average score — global")
        axs.set_xlabel("average_test_score"); axs.set_ylabel("count")

        # Boxplot by year
        axs = ax[1]
        S.boxplot(column="average_test_score", by="year", ax=axs)
        axs.set_title("Score by year"); axs.set_xlabel("year"); axs.set_ylabel("score")
        axs.figure.suptitle("")

        # Boxplot by gender
        axs = ax[2]
        if "sextypename" in S.columns:
            S.boxplot(column="average_test_score", by="sextypename", ax=axs)
            axs.set_title("Score by gender"); axs.set_xlabel("sextypename"); axs.set_ylabel("score")
            axs.figure.suptitle("")
        else:
            axs.set_visible(False)

        plt.tight_layout()
        plt.show()

        # Settlement & test language quick tables (top categories)
        if "tertypename" in data.columns:
            display(top_n_categories(data.dropna(subset=["average_test_score"]), "tertypename", n=5))
        if "testlanguage" in data.columns:
            display(top_n_categories(data.dropna(subset=["average_test_score"]), "testlanguage", n=8))

        # Cross-tab of settlement type frequencies by year (spot schema drifts)
        print(data.pivot_table(index="year", columns="tertypename", values="outid", aggfunc="size", fill_value=0))

        # Participation & DPA completion over time
        took_cols = [c for c in data.columns if c.startswith("took_") and not c.endswith("_any")]
        cols_to_show = sorted(set(took_cols + ["dpa_required_done"]) & set(data.columns))
        if cols_to_show:
            by_year = data.groupby("year")[cols_to_show].mean().sort_index()
            display((by_year * 100).round(1).tail())

            # DPA completion rate bar plot
            if "dpa_required_done" in data.columns:
                share = data.groupby("year")["dpa_required_done"].mean().sort_index()
                share.plot(kind="bar", figsize=(6, 3))
                plt.title("DPA required — completion rate by year")
                plt.ylabel("share")
                plt.tight_layout()
                plt.show()

        # Correlations among numeric features used for modeling (leakage-safe)
        num_corr_cols = [c for c in [
            "average_test_score","age","year",
            "n_tests_taken","n_foreign_taken",
            "dpa_required_done","dpa_required_count_met",
            "school_hist_mean","school_hist_count"
        ] if c in data.columns]
        if num_corr_cols:
            corr = data[num_corr_cols].corr()
            fig, ax = plt.subplots(figsize=(6,5))
            im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
            ax.set_xticks(range(len(num_corr_cols))); ax.set_xticklabels(num_corr_cols, rotation=45, ha="right")
            ax.set_yticks(range(len(num_corr_cols))); ax.set_yticklabels(num_corr_cols)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.title("Numeric correlations")
            plt.tight_layout()
            plt.show()

        # Regions (top-10 by volume): simple descriptive stats
        if "regname" in data.columns:
            top_regs = data["regname"].value_counts().head(10).index
            tbl = (data.loc[data["regname"].isin(top_regs)]
                        .groupby("regname")["average_test_score"]
                        .agg(["count","mean","std"]).sort_values("count", ascending=False))
            display(tbl)
    ''')),
    nbf.v4.new_markdown_cell("## ====================== Cell 4: Feature selection (leakage-safe) + temporal split ======================"),
    nbf.v4.new_code_cell(textwrap.dedent(r'''
        target = "average_test_score"
        if data.empty or target not in data.columns:
            X = pd.DataFrame()
            y = pd.Series(dtype=float, name=target)
        else:
            # Block any leakage from subject-level outcomes or statuses
            score_cols  = [c for c in data.columns if c.endswith("ball12")]
            status_cols = [c for c in data.columns if "teststatus" in c]
            banned = set(score_cols + status_cols + [target])

            # Candidate features robust across years
            categorical_candidates = [
                "sextypename","regname","areaname","tername","tertypename","regtypename",
                "classprofilename","classlangname","testlanguage",
            ]
            numeric_candidates = [
                "age", "year","n_tests_taken", "n_foreign_taken",
                "dpa_required_done", "dpa_required_count_met",
                "school_hist_mean", "school_hist_count"
            ]

            cat_features = [c for c in categorical_candidates if c in data.columns and c not in banned]
            num_features = [c for c in numeric_candidates if c in data.columns and c not in banned]

            X = data[cat_features + num_features].copy()
            y = data[target].astype(float).copy()

            # Enforce consistent dtype for categoricals (string → OHE later)
            for c in cat_features:
                X[c] = X[c].astype("string")
            print(f"[info] features → cat={len(cat_features)} | num={len(num_features)}")

            # Drop rows with no positive scores at all (often noisy outliers)
            if DROP_ZERO_NOSCORE:
                tst_cols = [c for c in data.columns if c.endswith("teststatus")]
                if tst_cols:
                    has_any_score = data[tst_cols].eq("has score").any(axis=1)
                    keep = has_any_score & (y > 0)
                    dropped = int((~keep).sum())
                    X, y = X.loc[keep], y.loc[keep]
                    print(f"[filter] dropped {dropped:,} strict no-score rows")

        # Temporal split if 2021 exists; else, random split for robustness
        if not X.empty and "year" in data.columns and (data["year"] == 2021).any():
            m_train = (data.loc[X.index, "year"] < 2021).values
            m_test  = (data.loc[X.index, "year"] == 2021).values
            X_train, y_train = X.loc[m_train], y.loc[m_train]
            X_test,  y_test  = X.loc[m_test],  y.loc[m_test]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE
            )

        # Optional groups for GroupKFold audits downstream (by school)
        groups_train = data.loc[X_train.index, "eoname"].astype(str) if "eoname" in data.columns else None
        groups_test  = data.loc[X_test.index, "eoname"].astype(str)  if "eoname" in data.columns else None
        print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    ''')),
    nbf.v4.new_markdown_cell("## ====================== Cell 5: Robust preprocessing (categoricals → OHE, numerics → scaled) ======================"),
    nbf.v4.new_code_cell(textwrap.dedent(r'''
        # Lock feature lists (defensive recomputation)
        cat_features = [c for c in ["sextypename","regname","areaname","tername","tertypename",
                                    "regtypename","classprofilename","classlangname","testlanguage"]
                        if c in X_train.columns]
        num_features = [c for c in ["age","year","n_tests_taken","n_foreign_taken",
                                    "dpa_required_done","dpa_required_count_met",
                                    "school_hist_mean", "school_hist_count"]
                        if c in X_train.columns]

        # Type hygiene: numerics → numeric dtype
        for c in num_features:
            X_train.loc[:, c] = pd.to_numeric(X_train[c], errors="coerce")
            X_test.loc[:,  c] = pd.to_numeric(X_test[c],  errors="coerce")

        # Categorical NA handling: fill sentinel to avoid pd.NA downstream in OHE
        SENTINEL = "__MISSING__"
        for df_ in (X_train, X_test):
            for c in cat_features:
                df_.loc[:, c] = df_[c].astype(object)
            df_.loc[:, cat_features] = df_[cat_features].where(
                pd.notna(df_[cat_features]), SENTINEL
            )

        # Numeric transformer: impute + scale (no mean centering on sparse)
        numeric_tf = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler(with_mean=False)),
        ])

        # OneHotEncoder compatibility across sklearn versions
        enc_sig = signature(OneHotEncoder.__init__)
        use_sparse_output = "sparse_output" in enc_sig.parameters
        ohe_kwargs = dict(handle_unknown="ignore")
        if use_sparse_output:
            ohe_kwargs["sparse_output"] = True
        else:
            ohe_kwargs["sparse"] = True

        # Size control: clip extremely rare categories in FAST mode
        min_freq = 0.02 if FAST else 20
        if "min_frequency" in enc_sig.parameters:
            ohe_kwargs["min_frequency"] = min_freq

        # Categorical transformer: OHE only (we already imputed)
        categorical_tf = Pipeline([
            ("onehot", OneHotEncoder(**ohe_kwargs)),
        ])

        # ColumnTransformer: join numeric and categorical pipelines
        preprocess = ColumnTransformer(
            transformers=[
                ("num", numeric_tf, num_features),
                ("cat", categorical_tf, cat_features),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        # Fit/transform → keep arrays float32, preserve sparse when available
        print("[stage] fitting preprocessing ...")
        t0 = perf_counter()
        Xtr = preprocess.fit_transform(X_train)
        Xte = preprocess.transform(X_test)
        Xtr = (Xtr.tocsr() if sparse.issparse(Xtr) else np.asarray(Xtr)).astype(np.float32)
        Xte = (Xte.tocsr() if sparse.issparse(Xte) else np.asarray(Xte)).astype(np.float32)
        print(f"[done] preprocessing in {perf_counter()-t0:,.1f}s | "
                f"train {X_train.shape}→{Xtr.shape} | test {X_test.shape}→{Xte.shape}")

        # Transformed feature names (best-effort across sklearn versions)
        try:
            feat_names_trans = preprocess.get_feature_names_out()
        except Exception:
            feat_names_trans = np.array([f"f{i}" for i in range(Xtr.shape[1])])
    ''')),
    nbf.v4.new_markdown_cell("## ====================== Cell 6: Models + early stopping for XGB + evaluation helpers ======================"),
    nbf.v4.new_code_cell(textwrap.dedent(r'''
        def eval_metrics(y_true, y_pred) -> Dict[str, float]:
            """Compute MAE, RMSE, R² for arrays or array-like inputs."""
            y_true = np.asarray(y_true, dtype=float)
            y_pred = np.asarray(y_pred, dtype=float)
            try:
                rmse = mean_squared_error(y_true, y_pred, squared=False)
            except TypeError:
                # For older sklearn versions without squared=
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            return {
                "MAE": float(mean_absolute_error(y_true, y_pred)),
                "RMSE": float(rmse),
                "R2": float(r2_score(y_true, y_pred)),
            }

        def _fit_xgb_compat(est, Xtr, ytr, random_state=RANDOM_STATE):
            """Fit XGBRegressor with broad EarlyStopping compatibility across versions.
            Falls back to plain .fit if callbacks or kwargs are unsupported."""
            Xtr0, Xval, ytr0, yval = train_test_split(Xtr, ytr, test_size=0.10, random_state=random_state)

            tried_kwargs = [
                {"early_stopping_rounds": 50, "eval_set": [(Xval, yval)], "verbose": False},
                {"early_stopping_rounds": 50, "eval_set": [(Xval, yval)]},  # alternative without verbose
                {"callbacks": [XGBEarlyStopping(rounds=50, save_best=True)] if XGBEarlyStopping else None,
                    "eval_set": [(Xval, yval)], "verbose": False},
                {"eval_set": [(Xval, yval)], "verbose": False},
                {"eval_set": [(Xval, yval)]},
                {},
            ]

            for kw in tried_kwargs:
                kw = {k: v for k, v in kw.items() if v is not None}
                try:
                    est.fit(Xtr0, ytr0, **kw)
                    return est
                except TypeError:
                    continue

            est.fit(Xtr, ytr)
            return est

        def fit_and_eval(name: str, est, Xtr, ytr, Xte, yte):
            """Clone, fit, predict, and print metrics for a given estimator.
            Returns (name, metrics_dict, fitted_estimator)."""
            start = perf_counter()
            est = clone(est)

            if "XGBRegressor" in est.__class__.__name__:
                est = _fit_xgb_compat(est, Xtr, ytr, random_state=RANDOM_STATE)
            else:
                est.fit(Xtr, ytr)

            preds = est.predict(Xte)

            try:
                rmse = mean_squared_error(yte, preds, squared=False)
            except TypeError:
                rmse = np.sqrt(mean_squared_error(yte, preds))

            metrics = {
                "MAE": float(mean_absolute_error(yte, preds)),
                "RMSE": float(rmse),
                "R2": float(r2_score(yte, preds)),
            }
            dur = perf_counter() - start
            print(f"[model] {name:12s} in {dur:,.1f}s | R2={metrics['R2']:.3f}  RMSE={metrics['RMSE']:.3f}")
            return name, metrics, est

        # ---- Model zoo (FAST-friendly defaults; increase capacity in SLOW mode) ----
        models = {
            "dummy_median": DummyRegressor(strategy="median"),
            "rf": RandomForestRegressor(
                n_estimators=150 if FAST else 500,
                max_depth=12 if FAST else 24,
                max_features=0.5 if FAST else 1.0,
                min_samples_leaf=2,
                max_samples=0.5 if FAST else 0.8,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        }
        if HAS_XGB:
            models["xgb"] = XGBRegressor(
                n_estimators=600 if FAST else 900,
                learning_rate=0.07 if FAST else 0.05,
                max_depth=5 if FAST else 7,
                min_child_weight=3 if FAST else 7,
                subsample=0.8 if FAST else 1,
                colsample_bytree=0.9,
                reg_lambda=1.0 if FAST else 10,
                reg_alpha=0.0,
                tree_method="hist",
                max_bin=256,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                eval_metric="rmse",
                verbosity=0,
            )

        # ---- Model tuning prep (time-aware CV) ---- 
        DO_TUNING = not FAST  # toggle
        def tune_model(name, base_estimator, preprocess, X_df, y, random_state=RANDOM_STATE):
            """RandomizedSearchCV wrapper around a Pipeline(preprocess, model) with GroupKFold on 'year'.
            Returns (best_fitted_pipeline, best_params_dict)."""
            pipe = Pipeline([("preprocess", preprocess), ("model", clone(base_estimator))])
            groups = X_df["year"].astype(int) if "year" in X_df.columns else pd.Series(0, index=X_df.index)

            n_iter = 12 if FAST else 35
            cv = GroupKFold(n_splits=3 if FAST else 5)

            if name == "xgb":
                param_distributions = {
                    "model__n_estimators": np.linspace(300, 1200, 10, dtype=int),
                    "model__max_depth": [4,5,6,7,8],
                    "model__learning_rate": loguniform(1e-2, 2e-1),
                    "model__subsample": [0.6, 0.8, 1.0],
                    "model__colsample_bytree": [0.6, 0.8, 1.0],
                    "model__min_child_weight": [1,3,5,7,10],
                    "model__reg_lambda": [0.1, 1, 5, 10],
                    "model__reg_alpha": [0.0, 0.1, 0.5],
                }
            elif name == "rf":
                param_distributions = {
                    "model__n_estimators": np.linspace(200, 800, 7, dtype=int),
                    "model__max_depth": [None, 12, 16, 20, 24],
                    "model__min_samples_leaf": [1, 2, 4, 8],
                    "model__max_features": [0.3, 0.5, 0.7, 1.0],
                    "model__max_samples": [0.5, 0.8, None],
                }
            else:
                # Nothing to tune: fit the baseline pipeline directly
                pipe.fit(X_df, y)
                return pipe, {}

            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=param_distributions,
                n_iter=n_iter,
                scoring="neg_root_mean_squared_error",
                cv=cv.split(X_df, y, groups=groups),
                random_state=random_state,
                n_jobs=-1,
                verbose=1 if not FAST else 0,
                refit=True,
            )
            search.fit(X_df, y)
            best_pipe = search.best_estimator_
            print(f"[tune] {name}: best CV RMSE={-search.best_score_:.3f}")
            return best_pipe, search.best_params_
    ''')),
    nbf.v4.new_markdown_cell("## ====================== Cell 7: Train sequentially, pick best, persist pipeline ======================"),
    nbf.v4.new_code_cell(textwrap.dedent(r'''
        print("[stage] training models ...")
        results = [fit_and_eval(n, m, Xtr, y_train, Xte, y_test) for n, m in models.items()]
        print("[done] all models trained.")

        # Aggregate metrics and pick best by R²
        metrics_all = {}
        best_name, best_r2, best_est = None, -1e9, None
        for name, m, est in results:
            metrics_all[name] = m
            if m["R2"] > best_r2:
                best_name, best_r2, best_est = name, m["R2"], est

        # Build final pipeline (optionally tuned)
        final_pipeline = None
        best_label = best_name
        if DO_TUNING and best_name in ("xgb", "rf"):
            tuned_pipe, tuned_params = tune_model(best_name, models[best_name], clone(preprocess), X_train.copy(), y_train)
            # Evaluate tuned pipeline on raw X_test (preprocess inside the pipeline)
            tuned_pred = tuned_pipe.predict(X_test)
            try:
                tuned_rmse = mean_squared_error(y_test, tuned_pred, squared=False)
            except TypeError:
                tuned_rmse = np.sqrt(mean_squared_error(y_test, tuned_pred))
            tuned_r2 = r2_score(y_test, tuned_pred)
            print(f"[tuned] {best_name}: Test R2={tuned_r2:.3f} RMSE={tuned_rmse:.3f}")
            if tuned_r2 >= metrics_all[best_name]["R2"]:
                final_pipeline = tuned_pipe
                best_label = f"{best_name}_tuned"
                print("[select] tuned pipeline chosen for artifact.")
            else:
                print("[select] tuned underperforms on holdout; keep untuned.")

        # If not tuned or tuned worse, wrap best_est with a fresh preprocessing pipeline
        if final_pipeline is None:
            final_pipeline = Pipeline([("preprocess", clone(preprocess)), ("model", clone(best_est))])
            final_pipeline.fit(X_train, y_train)

        # Holdout predictions (kept for consistent downstream diagnostics)
        y_pred = final_pipeline.predict(X_test)

        # Persist metrics and full pipeline artifact
        (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics_all, ensure_ascii=False, indent=2), encoding="utf-8")
        dump(final_pipeline, ARTIFACTS_DIR / f"pipeline_{best_label}.joblib")
        print(f"[artifact] full pipeline saved to artifacts/pipeline_{best_label}.joblib")
    ''')),
    nbf.v4.new_markdown_cell("## ====================== Cell 8: Residual diagnostics + fairness slices (with 95% CIs) ======================"),
    nbf.v4.new_code_cell(textwrap.dedent(r'''
        # Residual plots using the final pipeline predictions to ensure consistency with artifacts
        if X_test.shape[0] > 0:
            resid = y_test - y_pred
            fig, ax = plt.subplots(2, 1, figsize=(7, 8))
            ax[0].scatter(y_pred[:30_000], resid[:30_000], s=2, alpha=0.25)
            ax[0].axhline(0, color="gray", lw=1)
            ax[0].set_title("Residuals vs. predictions")
            ax[0].set_xlabel("Predicted average_test_score")
            ax[0].set_ylabel("Residual")
            ax[1].hist(resid, bins=60)
            ax[1].set_title("Residual distribution")
            ax[1].set_xlabel("Residual"); ax[1].set_ylabel("Count")
            plt.tight_layout(); plt.savefig(ARTIFACTS_DIR/"residuals.png", dpi=140, bbox_inches="tight"); plt.show()

        def _metric_triplet(y_true, y_pred):
            """Helper: return (MAE, RMSE, R²) for a slice."""
            y_true = np.asarray(y_true, float)
            y_pred = np.asarray(y_pred, float)
            err = y_true - y_pred
            mae  = float(np.abs(err).mean())
            rmse = float(np.sqrt((err**2).mean()))
            r2   = float(r2_score(y_true, y_pred))
            return mae, rmse, r2

        def _bootstrap_cis(y_true, y_pred, B=None, rng=None):
            """Percentile bootstrap confidence intervals for (MAE, RMSE, R²).
            FAST uses B=500; SLOW uses B=2000."""
            if B is None:
                B = 500 if FAST else 2000
            if rng is None:
                rng = np.random.RandomState(RANDOM_STATE)

            y_true = np.asarray(y_true, float)
            y_pred = np.asarray(y_pred, float)
            n = y_true.shape[0]
            idx = np.arange(n)

            maes, rmses, r2s = [], [], []
            for _ in range(B):
                bs = rng.choice(idx, size=n, replace=True)
                m, r, r2 = _metric_triplet(y_true[bs], y_pred[bs])
                maes.append(m); rmses.append(r); r2s.append(r2)

            q = (2.5, 97.5)
            mae_ci  = (float(np.percentile(maes, q[0])),  float(np.percentile(maes, q[1])))
            rmse_ci = (float(np.percentile(rmses, q[0])), float(np.percentile(rmses, q[1])))
            r2_ci   = (float(np.percentile(r2s, q[0])),   float(np.percentile(r2s, q[1])))
            return mae_ci, rmse_ci, r2_ci

        def fairness_report(y_true, y_pred, data_slice, name):
            """Compute slice-level metrics and 95% bootstrap CIs for categorical group values.
            Slices with n < 50 are skipped to avoid volatile estimates.
            Returns a nested dict: {name: {group_value: {count, MAE, RMSE, R2, *_CI95}}}"""
            s = pd.Series(data_slice).astype("string")
            y_true = np.asarray(y_true, dtype=float)
            y_pred = np.asarray(y_pred, dtype=float)

            out = {}
            for val in s.dropna().unique().tolist():
                mask = s.eq(val).fillna(False).to_numpy()
                n = int(mask.sum())
                if n < 50:
                    continue

                yt, yp = y_true[mask], y_pred[mask]
                mae, rmse, r2 = _metric_triplet(yt, yp)
                mae_ci, rmse_ci, r2_ci = _bootstrap_cis(yt, yp)

                out[str(val)] = {
                    "count": n,
                    "MAE": mae, "RMSE": rmse, "R2": r2,
                    "MAE_CI95": mae_ci, "RMSE_CI95": rmse_ci, "R2_CI95": r2_ci,
                }
            return {name: out}

        # Build fairness slices across selected demographics and operational flags
        fair = {}
        if X_test.shape[0] > 0:
            # gender
            if "sextypename" in data.columns:
                fair.update(
                    fairness_report(y_test, y_pred, data.loc[X_test.index, "sextypename"], "gender")
                )
            # settlement type
            if "tertypename" in data.columns:
                fair.update(
                    fairness_report(y_test, y_pred, data.loc[X_test.index, "tertypename"], "settlement")
                )
            # primary language of testing
            if "testlanguage" in data.columns:
                fair.update(
                    fairness_report(y_test, y_pred, data.loc[X_test.index, "testlanguage"], "testlanguage")
                )
            # DPA completion flag (mapped to yes/no)
            if "dpa_required_done" in data.columns:
                dpa_slice = data.loc[X_test.index, "dpa_required_done"].map({0: "no", 1: "yes"}).astype("string")
                fair.update(
                    fairness_report(y_test, y_pred, dpa_slice, "dpa_required_done")
                )
            # Regions (top-10 by frequency; others dropped)
            if "regname" in data.columns:
                regs = data.loc[X_test.index, "regname"].astype("string")
                top10 = regs.value_counts().head(10).index
                regs_top = regs.where(regs.isin(top10), other=pd.NA)
                fair.update(
                    fairness_report(y_test, y_pred, regs_top, "region_top10")
                )

        # Persist fairness report as JSON artifact
        (ARTIFACTS_DIR / "fairness_slices.json").write_text(
            json.dumps(fair, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Add MAE gaps versus global holdout MAE (helps spot over/under-performance by group)
        global_mae = float(np.abs(y_test - y_pred).mean())
        for group, stats in fair.items():
            for k, v in stats.items():
                v["MAE_gap"] = round(v["MAE"] - global_mae, 3)

        # ---- Diagnostics (deciles & calibration) ----
        diag = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred})
        diag["true_decile"] = pd.qcut(diag["y_true"], q=10, labels=False, duplicates="drop")
        diag["pred_decile"] = pd.qcut(diag["y_pred"], q=10, labels=False, duplicates="drop")
        diag["abs_err"] = np.abs(diag["y_true"] - diag["y_pred"])

        # Error vs true deciles
        by_true = diag.groupby("true_decile").agg(
            y_true_mean=("y_true","mean"),
            mae=("abs_err","mean"),
            rmse=("abs_err", lambda s: float(np.sqrt(((diag.loc[s.index, 'y_true'] - diag.loc[s.index, 'y_pred'])**2).mean())))
        ).reset_index()

        # Calibration: ideal line ~ y_true_mean ≈ y_pred_mean per predicted decile
        by_pred = diag.groupby("pred_decile").agg(
            y_true_mean=("y_true","mean"),
            y_pred_mean=("y_pred","mean"),
        ).reset_index()

        fig, ax = plt.subplots(1, 2, figsize=(10,4))
        ax[0].plot(by_true["y_true_mean"], by_true["mae"], marker="o")
        ax[0].set_title("MAE vs true-score deciles")
        ax[0].set_xlabel("Mean true score per decile"); ax[0].set_ylabel("MAE")

        ax[1].plot(by_pred["y_pred_mean"], by_pred["y_true_mean"], marker="o")
        lims = [min(ax[1].get_xlim()+ax[1].get_ylim()), max(ax[1].get_xlim()+ax[1].get_ylim())]
        ax[1].plot(lims, lims, "--", lw=1)  # ideal line
        ax[1].set_title("Calibration: mean true vs mean predicted")
        ax[1].set_xlabel("Mean predicted (per decile)"); ax[1].set_ylabel("Mean true (per decile)")

        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR/"diagnostics_deciles_calibration.png", dpi=140, bbox_inches="tight")
        plt.show()

        # 2021-by-slice quick tables (examples)
        if {"tertypename","sextypename"}.issubset(data.columns):
            test_idx = X_test.index
            slices = {}
            for col in ["tertypename","sextypename","testlanguage"]:
                g = data.loc[test_idx, col].astype("string")
                rows = []
                for v, mask in g.groupby(g):
                    m = mask.index
                    yt = y_test.loc[m].values; yp = pd.Series(y_pred, index=y_test.index).loc[m].values
                    mae, rmse, r2 = _metric_triplet(yt, yp)
                    rows.append((str(v), len(m), mae, rmse, r2))
                slices[col] = (pd.DataFrame(rows, columns=[col,"count","MAE","RMSE","R2"])
                                    .sort_values("count", ascending=False))
                display(slices[col])
    ''')),
    nbf.v4.new_markdown_cell("## ====================== Cell 9: GroupKFold audit by school ======================"),
    nbf.v4.new_code_cell(textwrap.dedent(r'''
        def group_cv_r2(estimator, X, y, groups, n_splits=5):
            """Compute mean, std, and fold-level R² across GroupKFold splits."""
            uniq = np.unique(groups)
            if len(uniq) < n_splits:
                raise ValueError(f"Not enough groups for GroupKFold: groups={len(uniq)}, splits={n_splits}")
            gkf = GroupKFold(n_splits=n_splits)
            scores = []
            for tr, va in gkf.split(X, y, groups):
                m = clone(estimator)
                m.fit(X[tr], y.iloc[tr])
                p = m.predict(X[va])
                scores.append(r2_score(y.iloc[va], p))
            scores = np.asarray(scores)
            return scores.mean(), scores.std(ddof=1), scores

        try:
            if groups_train is not None:
                # Use the best model if available; otherwise a simple Ridge fallback
                est_for_audit = best_est if best_est is not None else Ridge(alpha=1.0, random_state=RANDOM_STATE).fit(Xtr, y_train)
                mean_r2, std_r2, fold_r2 = group_cv_r2(est_for_audit, Xtr, y_train, groups_train.values,
                                                        n_splits=3 if FAST else 5)
                print(f"[GroupKFold by eoname] R2 mean±sd: {mean_r2:.3f} ± {std_r2:.3f} | folds={np.round(fold_r2, 3)}")
            else:
                print("No group column (eoname) available — skipping GroupKFold.")
        except ValueError as e:
            print(f"GroupKFold skipped: {e}")
    ''')),
    nbf.v4.new_markdown_cell("## ====================== Cell 10: Feature importance ======================"),
    nbf.v4.new_code_cell(textwrap.dedent(r'''
        from sklearn.metrics import mean_absolute_error

        def families_from_feature_names(names: np.ndarray) -> dict[str, list[int]]:
            """Group one-hot encoded columns into families by base field, e.g.:
            'sextypename_<cat1>', 'sextypename_<cat2>' → 'sextypename': [idx1, idx2]."""
            fam2idx: dict[str, list[int]] = {}
            for j, full in enumerate(names):
                base = str(full)
                # With verbose_feature_names_out=False, ColumnTransformer yields 'num__age' or 'sextypename_<cat>'
                if "__" in base:
                    base = base.split("__", 1)[1]
                base = base.split("_", 1)[0]  # 'sextypename_<cat>' → 'sextypename'
                fam2idx.setdefault(base, []).append(j)
            return fam2idx

        def grouped_perm_importance(estimator, X, y, names, n_repeats=5, random_state=RANDOM_STATE, max_n=20000):
            """Permutation importance that shuffles feature families together to respect one-hot structure.
            Returns (importance_df, n_used, n_repeats)."""
            rng = np.random.RandomState(random_state)
            # Sample for tractability on large matrices
            n = min(max_n, X.shape[0])
            idx = rng.choice(X.shape[0], size=n, replace=False)
            Xs = X[idx]
            ys = y.iloc[idx] if hasattr(y, "iloc") else y[idx]
            # Force dense to satisfy permutation_importance contracts across sklearn versions
            Xd = Xs.toarray().astype(np.float32) if sparse.issparse(Xs) else np.asarray(Xs, dtype=np.float32)

            fam2idx = families_from_feature_names(names)
            baseline = mean_absolute_error(ys, estimator.predict(Xd))
            rows = []
            for fam, cols in fam2idx.items():
                deltas = []
                for _ in range(n_repeats):
                    perm = rng.permutation(n)
                    Xp = Xd.copy()
                    Xp[:, cols] = Xd[perm][:, cols]
                    mae = mean_absolute_error(ys, estimator.predict(Xp))
                    deltas.append(mae - baseline)
                deltas = np.asarray(deltas, float)
                rows.append((fam, float(deltas.mean()), float(deltas.std(ddof=1))))
            imp = (pd.DataFrame(rows, columns=["family", "importance_mean", "importance_std"])
                        .sort_values("importance_mean", ascending=False)
                        .reset_index(drop=True))
            return imp, n, n_repeats

        # 1) XGB native importance (if applicable)
        imp_df = None
        source = None
        if best_name == "xgb":
            try:
                booster = best_est.get_booster()
                gain_map = booster.get_score(importance_type="gain")  # {feature_name: gain}
                # XGBoost names features as f0, f1, ... — map to transformed feature names
                mapped = []
                for k, v in gain_map.items():
                    try:
                        idx = int(k[1:])  # 'f123' → 123
                    except Exception:
                        continue
                    if 0 <= idx < len(feat_names_trans):
                        mapped.append((feat_names_trans[idx], float(v)))
                imp_df = (pd.DataFrame(mapped, columns=["feature", "importance"])
                            .sort_values("importance", ascending=False)
                            .reset_index(drop=True))
                source = "xgboost-gain"
            except Exception as e:
                print(f"XGB importance failed ({e}) — falling back to permutation.")

        # 2) Fallback: grouped permutation importance on transformed features
        if imp_df is None:
            imp_df, n_used, n_repeats = grouped_perm_importance(
                best_est, Xte, y_test, feat_names_trans,
                n_repeats=5 if FAST else 10,
                random_state=RANDOM_STATE,
                max_n=15000 if FAST else 30000
            )
            imp_df.rename(columns={"family": "feature", "importance_mean": "importance"}, inplace=True)
            source = "grouped-permutation"

        print(f"[importance] source = {source}")
        display(imp_df.head(20))

        # Save CSV and plot top-15 for reporting
        imp_file = ARTIFACTS_DIR / ("feature_importance_top.csv" if source == "xgboost-gain"
                                        else "feature_importance_families_grouped.csv")
        imp_df.to_csv(imp_file, index=False, encoding="utf-8")
        top = imp_df.head(15).iloc[::-1]
        plt.figure(figsize=(15, 6))
        plt.barh(top["feature"], top["importance"])
        plt.title(f"Top-15 feature{' families' if source!='xgboost-gain' else ''} by {source}")
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / ("feature_importance_top15.png" if source == "xgboost-gain"
                                        else "feature_importance_families_grouped_top15.png"),
                    dpi=140, bbox_inches="tight")
        plt.show()
    ''')),
    nbf.v4.new_markdown_cell("## ====================== Cell 11: Final summary block ======================"),
    nbf.v4.new_code_cell(textwrap.dedent(r'''
        summary = {
            "data_shape": tuple(data.shape) if hasattr(data, "shape") else None,
            "train_shape": tuple(X_train.shape),
            "test_shape": tuple(X_test.shape),
            "features": {"categorical": cat_features, "numeric": num_features},
            "metrics": metrics_all,
            "best_model": best_name,
            "artifacts": {
                "metrics": str((ARTIFACTS_DIR / "metrics.json").resolve()),
                "pipeline": str((ARTIFACTS_DIR / f"pipeline_{best_label}.joblib").resolve()),
                "residuals_png": str((ARTIFACTS_DIR / "residuals.png").resolve()),
                "importance_csv": str((ARTIFACTS_DIR / ("feature_importance_top.csv"
                                                        if best_name == "xgb"
                                                        else "feature_importance_families_grouped.csv")).resolve()),
            },
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    ''')),
]

with open(ROOT/"notebooks"/"ZNO_Score_Analysis_and_Prediction.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)


print("✅ Project scaffolded. Open notebooks/ZNO_Score_Analysis_and_Prediction.ipynb and run.")
