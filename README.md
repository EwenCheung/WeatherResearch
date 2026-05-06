# Weather Research CLI Cheatsheet

AI-DOP research workspace for converting Aardvark model-ready observations into
an IC-like physical weather state.

Current implemented pipeline:

```text
sample_data_final.pkl
-> Aardvark observation sample
-> pretrained Aardvark encoder
-> 24-variable IC-like physical weather state
-> xarray/Zarr output
-> table view
```

Phase 1 stops at `RawObservation -> IC`. Forecast rollout and decoding are later
phases.

## Environment

Use `uv` for Python dependency and environment control.

```bash
uv sync --extra notebooks --extra aifs --group dev
```

Run commands through the managed environment:

```bash
uv run --no-sync weather-research --help
```

Device behavior for Phase 1:

```python
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
```

## Main Commands

Show CLI help:

```bash
uv run --no-sync weather-research --help
```

Show package version:

```bash
uv run --no-sync weather-research --version
```

## 1. Generate IC-Like Zarr

Convert Aardvark's sample observation pickle into an IC-like Zarr dataset.

```bash
uv run --no-sync weather-research raw-to-ic \
  --device auto \
  --output outputs/aardvark_initial_condition.zarr \
  --time 2019-01-01T00:00:00
```

Default input:

```text
Reference-Repo-aardvark-weather-public-main/data/sample_data_final.pkl
```

Use a custom sample path:

```bash
uv run --no-sync weather-research raw-to-ic \
  --sample /path/to/sample_data_final.pkl \
  --output outputs/aardvark_initial_condition.zarr
```

Force CPU:

```bash
uv run --no-sync weather-research raw-to-ic \
  --device cpu \
  --output outputs/aardvark_initial_condition.zarr
```

Expected successful output:

```text
Wrote IC-like Zarr: outputs/aardvark_initial_condition.zarr
Variables: 24
Dims: {'time': 1, 'latitude': 121, 'longitude': 240}
```

## 2. View IC As A Table

Print the first 20 grid rows:

```bash
uv run --no-sync weather-research show-ic-table \
  --input outputs/aardvark_initial_condition.zarr \
  --rows 20
```

Print all rows:

```bash
uv run --no-sync weather-research show-ic-table \
  --input outputs/aardvark_initial_condition.zarr \
  --rows 0
```

Save full table to CSV:

```bash
uv run --no-sync weather-research show-ic-table \
  --input outputs/aardvark_initial_condition.zarr \
  --rows 20 \
  --output-csv outputs/aardvark_initial_condition_table.csv
```

The table columns are:

```text
time, latitude, longitude,
u10, v10, t2m, mslp,
z200, z500, z700, z850,
q200, q500, q700, q850,
t200, t500, t700, t850,
u200, u500, u700, u850,
v200, v500, v700, v850
```

Each row is one grid point at `t=0`. Satellite and station observations are not
shown as separate columns after encoding because the encoder fuses them into a
gridded physical weather state.

## 3. Inspect Zarr In Python

```bash
uv run --no-sync python - <<'PY'
import xarray as xr

ds = xr.open_zarr("outputs/aardvark_initial_condition.zarr")
print(ds)
print(float(ds["2m_temperature"].mean()))
PY
```

## 4. Tests And Checks

Fast tests:

```bash
uv run --no-sync pytest -q
```

Type check:

```bash
uv run --no-sync pyright src tests
```

Run the real Aardvark encoder integration test:

```bash
INTEGRATION_TESTS=1 uv run --no-sync pytest \
  tests/test_aardvark_initial_conditions.py::test_generate_initial_condition_from_file_integration \
  -q
```

## Important Files

```text
PLAN.md                                      Living project roadmap
src/weather_research/__main__.py             CLI entry point
src/weather_research/aardvark_observations.py
src/weather_research/aardvark_initial_conditions.py
src/weather_research/weather_state_schema.py
src/weather_research/weather_state_xarray.py
src/weather_research/weather_state_tables.py
tests/                                       Unit and integration tests
notebook/                                    Teaching notebooks
```

Generated outputs are ignored by git:

```text
outputs/
```
