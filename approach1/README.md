# Approach 1: Singapore Station Observations

This approach starts with raw Singapore station observations from data.gov.sg:

https://data.gov.sg/datasets?query=weather&resultId=1459

For now, only the realtime station APIs are included. Radar and Himawari
satellite data can be added later.

## Supported Endpoints

- Air temperature
- Rainfall
- Relative humidity
- Wind speed
- Wind direction

Each endpoint gives a different physical measurement, so the cleaned data keeps
the variable name and unit instead of collapsing everything into only
`[value, lat, lon]`.

## Usage

```python
from approach1.station_api import retrieve_all_station_observations
from approach1.station_cleaning import station_observations_to_tensor

df = retrieve_all_station_observations("2026-05-07T16:00:00+08:00")
station_tensor = station_observations_to_tensor(df)

features = station_tensor.features
mask = station_tensor.mask
coords = station_tensor.coords
station_ids = station_tensor.station_ids
feature_names = station_tensor.feature_names
```

The cleaned long table contains:

```text
requested_time, observed_time, station_id, station_name, latitude, longitude,
variable, value, unit, reading_type
```

The model-ready feature matrix is:

```text
[n_stations, n_features]
```

with these features:

```text
latitude
longitude
hour_sin
hour_cos
dayofyear_sin
dayofyear_cos
temperature_c
relative_humidity_pct
rainfall_5min_mm
rainfall_present
log1p_rainfall_5min_mm
wind_speed_mps
wind_dir_sin
wind_dir_cos
wind_u_mps
wind_v_mps
```

Missing or invalid values are filled with `0.0` in `features` and marked as
`0` in `mask`. Valid observed values are marked as `1` in `mask`.

Station id and station name are not model features. They stay in the table only
for row alignment and debugging. Units and reading types are also metadata: they
are used to validate meaning and perform conversions, not passed directly as
string/categorical model inputs.

Latitude and longitude are included as raw model features. The raw coordinates
are also available in `coords` for inspection and mapping.

Rainfall also gets `rainfall_present` and `log1p_rainfall_5min_mm` features.
Wind speed is converted from knots to m/s. Wind direction is encoded as sine and
cosine because direction is circular. When both wind speed and direction exist,
`wind_u_mps = wind_speed_mps * wind_dir_sin` and
`wind_v_mps = wind_speed_mps * wind_dir_cos` are derived.
