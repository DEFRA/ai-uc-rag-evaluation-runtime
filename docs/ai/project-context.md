# Project Context for AI Agents

## Purpose of the System

The `ai-uc-rag-evaluation-runtime` is a Python-based backend service that is an example implementation of RAG system. The service is built using the FastAPI framework.

## Architecture Overview

- `app/...` – Main application logic (FastAPI, LangChain/LangGraph agents).
  - `app/entrypoints/...` – Application entry points (e.g., `fastapi.py`).
  - `app/common/...` – Shared utilities (logging, metrics, HTTP client).
- `tests/...` – Comprehensive pytest suite (unit and integration tests).
- `scripts/...` – Command-line utilities and helper scripts (e.g., security checks).
- `docs/...` – Documentation including AI guidelines and system prompts.

## Key Constraints

- **Dependency Management**: All execution uses `uv` for dependency management and virtual environment handling.
- **Task Management**: All common development tasks (linting, testing) use `taskipy` as defined in `pyproject.toml`.
- **Code Quality**: All code must pass `ruff` for linting and formatting.
- **Testing**: All tests must use `pytest` function-style patterns.
- **Python Version**: The project requires Python >= 3.12.

## Example Files to Reference

- `app/entrypoints/fastapi.py` – Main application entry point.
- `tests/entrypoints/test_fastapi.py` – Example of a pytest test file.
- `pyproject.toml` – Project configuration and dependencies.

