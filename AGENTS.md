# Repository Guidelines

## Project Structure & Module Organization
- Source code lives under `src/` (e.g., `src/aipatent_api/`).
- Tests are in `tests/` with mirrors of package modules (e.g., `tests/test_routes/`).
- Developer utilities/scripts go in `scripts/` (one-task, self-documented).
- Configuration and assets: `pyproject.toml`, `.env(.example)`, `docs/`, `infra/` as applicable.
- Typical modules: `src/aipatent_api/routes/`, `services/`, `models/`, `core/` (config, logging), `schemas/` (pydantic), `clients/` (external APIs).

## Build, Test, and Development Commands
- Install: `make install` (preferred) or `pip install -e .` from the repo root.
- Run dev API: `make dev` or `uvicorn aipatent_api.main:app --reload`.
- Lint/format: `make lint` and `make fmt` or `ruff check .` and `black .`.
- Tests: `make test` or `pytest -q` (see coverage below).
- List all tasks: `make help` (if `Makefile` is present).

## Coding Style & Naming Conventions
- Python 3.x, 4-space indentation, 120-char soft wrap.
- Use type hints everywhere; public functions/classes require docstrings.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE` for constants, `kebab-case` for files/scripts.
- Tools: Black (format), Ruff (lint/imports), MyPy (types). Keep imports sorted and unused code removed.

## Testing Guidelines
- Framework: `pytest` with `pytest-cov`.
- Structure: `tests/test_*.py`; group by feature (e.g., `tests/test_routes/test_patents.py`).
- Fixtures: `tests/conftest.py` for app/client fixtures and factories.
- Coverage: target ≥ 90% statements; run `pytest --cov=src/aipatent_api --cov-report=term-missing`.

## Commit & Pull Request Guidelines
- Commits: use Gitmoji. Pattern: "<emoji> <scope>: <subject>" or "<emoji> <subject>"; present tense, no trailing period.
- Examples: `✨ routes: add search endpoint`, `🐛 prevent 500 on empty query`, `🧪 cover pagination in search tests`, `📝 docs: add auth example`, `⬆️ deps: bump uvicorn to 0.30.0`.
- Cheat sheet: `✨` feature, `🐛` fix, `🧪` tests, `📝` docs, `♻️` refactor, `🔧` config, `🔒` security, `⬆️` deps, `🚑` hotfix, `🚀` release/deploy.
- PRs: include emoji in title, clear summary, linked issues (`Closes #123`), curl examples for API changes, and update docs/changelogs.
- Requirements: CI green (lint, type-check, tests), review from at least one maintainer, and no secrets in diffs.

## Security & Configuration
- Use environment variables; never commit secrets. Provide `.env.example` and load via `python-dotenv` if needed.
- Validate all inputs (pydantic models) and handle timeouts/retries for external clients.
- Prefer least-privilege keys and separate dev/test/prod configs.
