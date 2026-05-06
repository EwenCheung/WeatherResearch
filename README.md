# Weather Research

Research workspace for AI-DOP-style observation-to-initial-condition weather
forecasting.

## Environment

This repo uses `uv` for Python dependency and environment management.

```bash
uv sync
```

Useful optional dependency sets:

```bash
uv sync --extra notebooks
uv sync --extra notebooks --extra aifs
uv sync --extra weatherbench
uv sync --group dev
```

Run Python through the managed environment:

```bash
uv run python -m weather_research
```

Add dependencies with `uv add`, for example:

```bash
uv add xarray
uv add --optional aifs earthkit-data
uv add --group dev pytest
```

Phase 0 notebook walkthroughs should use `notebooks` first. Add `aifs` when
running the AIFS baseline notebook, and add `weatherbench` when evaluating
forecast Zarr outputs.

## Repository Layout

```text
src/weather_research/  Python package code
notebook/              Project-owned learning and experiment notebooks
data/                  Local datasets, ignored by git
artifacts/             Generated outputs, ignored by git
checkpoints/           Model checkpoints, ignored by git
```
