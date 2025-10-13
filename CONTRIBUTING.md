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
