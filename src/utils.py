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
