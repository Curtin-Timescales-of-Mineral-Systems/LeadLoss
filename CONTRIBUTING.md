# Contributing

Thank you for considering a contribution.

## Scope

This repository contains:

- application code (to be added/maintained over time), and
- a manuscript reproduction bundle under `papers/2025-peak-picking/`.

To keep the manuscript workflow reproducible, please avoid changes that break the ability to regenerate the tables/figures without clearly documenting the new requirements.

## Suggested workflow

1. Create a feature branch.
2. Make small, focused commits.
3. If you modify manuscript scripts, run the relevant reproduction commands from:
   - `papers/2025-peak-picking/README.md`
4. Open a pull request with:
   - a clear summary of changes,
   - links to any related issues,
   - notes on whether outputs changed.

## Repository hygiene

- Do not commit:
  - `__pycache__/`, `*.pyc`, `.DS_Store`, or virtual environments.
  - large, machine-specific intermediate outputs unless they are required for reproduction.
- If you add large artefacts (data, binaries), consider using a release asset or an external archive (e.g., Zenodo) and document it.
