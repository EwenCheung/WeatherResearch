# AI-DOP Weather Research Plan

## Current Goal

Build the first working AI-DOP pipeline in small learning phases.

Phase 1 only builds:

```text
RawObservation -> IC-like physical state
```

For Phase 1, `sample_data_final.pkl` is our "raw observation sample" for
development. Technically, it is already Aardvark-preprocessed model-ready
observation data, not untouched satellite or station source files.

## Project Direction

The long-term pipeline is:

```text
RawObservation
-> Observation preprocessing
-> Initial Condition / IC-like state
-> Forecast model
-> Forecast state at configurable lead time
-> Decode to physical fields or station outputs
-> Evaluation
```

Phase 1 stops at the IC-like state. Forecast rollout, decoding, and evaluation
belong to later phases.

## Phase 1: RawObservation -> IC

### Goal

Use Aardvark's pretrained encoder to convert `sample_data_final.pkl` into an
IC-like gridded physical state.

### Input

```text
Reference-Repo-aardvark-weather-public-main/data/sample_data_final.pkl
```

### Output

An xarray/Zarr dataset containing Aardvark's 24-variable IC-like state.

### Variables

```text
u10, v10, t2m, mslp,
z200, z500, z700, z850,
q200, q500, q700, q850,
t200, t500, t700, t850,
u200, u500, u700, u850,
v200, v500, v700, v850
```

### Implementation

Build Python modules first, notebook later:

- `phase1_schema.py`
- `phase1_observation.py`
- `phase1_ic.py`
- `phase1_xarray.py`

The notebook layer should explain and visualize the working pipeline after the
module implementation is stable.

### Tests

- Load sample pickle.
- Confirm required sections exist.
- Run Aardvark encoder.
- Confirm IC output shape is `(121, 240, 24)`.
- Convert to xarray.
- Save and reopen Zarr.
- Confirm no NaN values.

## Phase 2: IC -> Forecast

Use the IC-like state as input to the forecast processor.

Lead time must be configurable:

- `lead_hours=6`
- `lead_hours=24`
- `lead_hours=240`
- etc.

Do not hard-code 10 days. Ten days is one possible run configuration, not the
pipeline definition.

## Phase 3: Decode Forecast

Decode forecasted state into:

- gridded physical fields
- station forecast outputs
- optional user-facing condition summaries

Rain/no-rain is not Phase 1 because current Aardvark variables do not directly
include precipitation.

## Development Environment

Use `uv` for dependency control.

Default approach:

- keep `pyproject.toml`
- keep `uv.lock`
- rebuild `.venv` only when the environment is broken

Clean rebuild command:

```bash
uv sync --extra notebooks --extra aifs --group dev
```

Do not remove `pyproject.toml` or `uv.lock` unless the dependency setup becomes
clearly wrong.

## Notebook Policy

Development can be done in Python modules and tests first.

Jupyter notebooks are for:

- teaching
- visualization
- final walkthroughs

After Phase 1 code works, create or update a notebook for easier review.

## Current Assumptions

- Phase 1 uses Aardvark, not AIFS.
- Phase 1 IC means Aardvark IC-like 24-variable physical state.
- AIFS Appendix A 94-field IC is a later phase.
- No new model training is required in Phase 1.
