import numpy as np
import pandas as pd

from src.utils import (
    attach_school_history_oot,
    build_average_12,
    derive_dpa_flags,
    derive_testlanguage,
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

