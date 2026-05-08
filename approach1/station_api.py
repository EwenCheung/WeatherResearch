"""Retrieve and normalize Singapore realtime weather station observations."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

BASE_URL = "https://api-open.data.gov.sg/v2/real-time/api"
REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "weather-research/0.1",
}

AIR_TEMPERATURE = "air_temperature"
RAINFALL = "rainfall"
RELATIVE_HUMIDITY = "relative_humidity"
WIND_SPEED = "wind_speed"
WIND_DIRECTION = "wind_direction"

ENDPOINTS: dict[str, str] = {
    AIR_TEMPERATURE: "air-temperature",
    RAINFALL: "rainfall",
    RELATIVE_HUMIDITY: "relative-humidity",
    WIND_SPEED: "wind-speed",
    WIND_DIRECTION: "wind-direction",
}

NORMALIZED_COLUMNS = [
    "requested_time",
    "observed_time",
    "station_id",
    "station_name",
    "latitude",
    "longitude",
    "variable",
    "value",
    "unit",
    "reading_type",
]


def retrieve_air_temperature(date_time: str | datetime) -> pd.DataFrame:
    """Retrieve air temperature readings for a requested Singapore-local time."""
    return _retrieve_endpoint(AIR_TEMPERATURE, date_time)


def retrieve_rainfall(date_time: str | datetime) -> pd.DataFrame:
    """Retrieve 5-minute rainfall readings for a requested Singapore-local time."""
    return _retrieve_endpoint(RAINFALL, date_time)


def retrieve_relative_humidity(date_time: str | datetime) -> pd.DataFrame:
    """Retrieve relative humidity readings for a requested Singapore-local time."""
    return _retrieve_endpoint(RELATIVE_HUMIDITY, date_time)


def retrieve_wind_speed(date_time: str | datetime) -> pd.DataFrame:
    """Retrieve 10-minute mean wind speed readings for a requested time."""
    return _retrieve_endpoint(WIND_SPEED, date_time)


def retrieve_wind_direction(date_time: str | datetime) -> pd.DataFrame:
    """Retrieve 10-minute mean wind direction readings for a requested time."""
    return _retrieve_endpoint(WIND_DIRECTION, date_time)


def retrieve_all_station_observations(date_time: str | datetime) -> pd.DataFrame:
    """Retrieve all supported station observations as one normalized table."""
    tables = [
        retrieve_air_temperature(date_time),
        retrieve_rainfall(date_time),
        retrieve_relative_humidity(date_time),
        retrieve_wind_speed(date_time),
        retrieve_wind_direction(date_time),
    ]
    return pd.concat(tables, ignore_index=True)[NORMALIZED_COLUMNS]


def normalize_station_payload(
    payload: dict[str, Any],
    variable: str,
    requested_time: str | datetime,
) -> pd.DataFrame:
    """Convert one data.gov.sg realtime payload into normalized long rows."""
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("Expected payload['data'] to be a dictionary.")

    stations = data.get("stations", [])
    readings = data.get("readings", [])
    if not isinstance(stations, list) or not isinstance(readings, list):
        raise ValueError("Expected stations and readings to be lists.")

    station_lookup = {
        station["id"]: station
        for station in stations
        if isinstance(station, dict) and "id" in station
    }
    unit = data.get("readingUnit")
    reading_type = data.get("readingType")
    requested = _format_date_time(requested_time)
    rows: list[dict[str, object]] = []

    for reading in readings:
        if not isinstance(reading, dict):
            continue
        observed_time = reading.get("timestamp")
        values = reading.get("data", [])
        if not isinstance(values, list):
            continue

        for item in values:
            if not isinstance(item, dict):
                continue
            station_id = item.get("stationId")
            station = station_lookup.get(station_id)
            if station is None:
                continue
            location = station.get("location", {})
            rows.append(
                {
                    "requested_time": requested,
                    "observed_time": observed_time,
                    "station_id": station_id,
                    "station_name": station.get("name"),
                    "latitude": location.get("latitude"),
                    "longitude": location.get("longitude"),
                    "variable": variable,
                    "value": item.get("value"),
                    "unit": unit,
                    "reading_type": reading_type,
                }
            )

    return pd.DataFrame(rows, columns=NORMALIZED_COLUMNS)


def _retrieve_endpoint(variable: str, date_time: str | datetime) -> pd.DataFrame:
    payload = _fetch_json(ENDPOINTS[variable], date_time)
    return normalize_station_payload(payload, variable, date_time)


def _fetch_json(endpoint: str, date_time: str | datetime) -> dict[str, Any]:
    query = urlencode({"date": _format_api_date(date_time)})
    url = f"{BASE_URL}/{endpoint}?{query}"
    request = Request(url, headers=REQUEST_HEADERS)
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _format_date_time(date_time: str | datetime) -> str:
    if isinstance(date_time, datetime):
        return date_time.isoformat()
    if isinstance(date_time, str):
        return date_time
    raise TypeError("date_time must be a string or datetime.")


def _format_api_date(date_time: str | datetime) -> str:
    formatted = _format_date_time(date_time)
    if len(formatted) > 19:
        return formatted[:19]
    return formatted
