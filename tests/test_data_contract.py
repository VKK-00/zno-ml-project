from pathlib import Path

from src.utils import discover_data_files, load_year_file

def test_contract_smoke():
    pairs = discover_data_files(Path('data'))
    if not pairs:
        return
    p, year = pairs[0]
    df = load_year_file(p, year)
    assert isinstance(df.shape, tuple)

