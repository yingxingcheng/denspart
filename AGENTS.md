# Repository Guidelines

## Project Structure & Module Organization

DensPart uses a `src/` layout. Core partitioning models are in `src/denspart/` (`mbis.py`,
`lisa.py`, `hirshfeld.py`, `hirshfeld_i.py`, `avh.py`, and `spline.py`); command-line dispatch is
in `__main__.py`. External-format converters belong in `src/denspart/adapters/`, and small output
utilities live in `src/denspart/utils/`. Tests mirror features under `tests/`; shared numerical
input is stored in `tests/density-water.npz`. Do not commit caches, egg-info changes, or generated
calculation archives.

## Build, Test & Development Commands

Develop with Python 3.10 or newer in an isolated environment:

```bash
python -m pip install -e .
pytest -q
ruff check src/ tests/
black --check src/ tests/
pre-commit run --all-files
```

The `qc-grid` development package is required by the partitioning tests; optional adapter tests
may additionally require IOData, GBasis, ASE, or GPAW. Run a focused test while iterating, for
example `pytest tests/test_spline.py -q`, followed by the complete suite before committing.

## Coding Style & Scientific Contracts

Use four-space indentation, a 100-character line limit, `snake_case` functions, and
`CapWords` classes. Keep NumPy arrays explicit about shape and units, and validate atomic numbers,
electron populations, charge-state completeness, and density normalization at input boundaries.
Package-neutral basis schemas use the `aim-*` identifiers; continue accepting documented
`denspart-*` identifiers for backward compatibility. Hirshfeld, Hirshfeld-I, and AVH share the
spline pro-atom representation, but their coefficient-update rules must remain distinct. Do not
make GPAW or other large adapters mandatory runtime dependencies.

## Testing, Commits & Pull Requests

Add focused regression tests for optimization, spline interpolation, state selection, CLI
validation, tensor/index changes, and legacy inputs. Numerical assertions should use justified
tolerances and test conservation as well as convergence. Commit messages are short and
imperative, such as `support shared AIM basis schemas`. Pull requests should describe the
algorithmic or compatibility impact, list tests run, link relevant issues, and call out changes
to public CLI options, schemas, or reference results.
