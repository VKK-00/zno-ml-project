\
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
