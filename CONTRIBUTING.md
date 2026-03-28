# Contributing

Thank you for considering a contribution.

## Scope

This repository contains the application code and tests for the public LeadLoss release.
Manuscript assets are distributed separately from the code repository.

## Suggested workflow

1. Create a feature branch.
2. Make small, focused commits.
3. Run the relevant tests before opening a pull request.
4. Open a pull request with:
   - a clear summary of changes,
   - links to any related issues,
   - notes on whether user-facing behaviour changed.

## Repository hygiene

- Do not commit:
  - `__pycache__/`, `*.pyc`, `.DS_Store`, or virtual environments.
  - large, machine-specific intermediate outputs unless they are required for reproduction.
- If you add large artefacts (data, binaries), consider using a release asset or an external archive (e.g., Zenodo) and document it.
