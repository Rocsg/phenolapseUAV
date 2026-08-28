# Contributing to phenolapseUAV

Contributions are welcome through GitHub issues and pull requests.

## Development setup

1. Fork and clone the repository.
2. Create and activate a Python 3.10–3.12 virtual environment.
3. Install development dependencies with `python -m pip install -e ".[dev]"`.
4. Run `python -m pytest` and `python -m ruff check .` before submitting.

For automatic AROSICS/OpenCV workflows, install `.[automatic,dev]` instead.
Keep UI strings, comments, documentation, and public APIs in English. Add tests
for fixes and new behavior. Do not include private imagery or generated outputs.

By submitting a contribution, you agree that it is distributed under the
GNU General Public License version 3.
