from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from weather_research import __version__
from weather_research.aardvark_initial_conditions import (
    DevicePreference,
    generate_aardvark_initial_condition_from_file,
)
from weather_research.aardvark_observations import DEFAULT_AARDVARK_SAMPLE_PATH
from weather_research.weather_state_tables import open_initial_condition_table
from weather_research.weather_state_xarray import save_initial_condition_zarr


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="weather-research",
        description="AI-DOP weather research utilities.",
    )
    parser.add_argument("--version", action="store_true", help="Print package version.")
    subparsers = parser.add_subparsers(dest="command")

    raw_to_ic = subparsers.add_parser(
        "raw-to-ic",
        help="Convert the Aardvark observation sample into an IC-like Zarr dataset.",
    )
    raw_to_ic.add_argument(
        "--sample",
        type=Path,
        default=DEFAULT_AARDVARK_SAMPLE_PATH,
        help="Path to Aardvark sample_data_final.pkl.",
    )
    raw_to_ic.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/aardvark_initial_condition.zarr"),
        help="Output Zarr path.",
    )
    raw_to_ic.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Compute device. Auto chooses CUDA if available, else CPU.",
    )
    raw_to_ic.add_argument(
        "--time",
        default=None,
        help="Optional output timestamp, for example 2019-01-01T00:00:00.",
    )

    show_ic = subparsers.add_parser(
        "show-ic-table",
        help="Show a lat/lon table from an IC-like Zarr dataset.",
    )
    show_ic.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/aardvark_initial_condition.zarr"),
        help="Input IC-like Zarr path.",
    )
    show_ic.add_argument(
        "--rows",
        type=int,
        default=20,
        help="Number of table rows to print. Use 0 to print all rows.",
    )
    show_ic.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV path for saving the full IC table.",
    )

    args = parser.parse_args()
    if args.version or args.command is None:
        print(f"weather-research {__version__}")
        return

    if args.command == "raw-to-ic":
        ic = generate_aardvark_initial_condition_from_file(
            args.sample,
            device_preference=args.device,
        )
        output_time = np.datetime64(args.time) if args.time else None
        dataset = save_initial_condition_zarr(
            ic,
            args.output,
            time=output_time,
            overwrite=True,
        )
        print(f"Wrote IC-like Zarr: {args.output}")
        print(f"Variables: {len(dataset.data_vars)}")
        print(f"Dims: {dict(dataset.sizes)}")
        return

    if args.command == "show-ic-table":
        max_rows = None if args.rows == 0 else args.rows
        table = open_initial_condition_table(args.input, max_rows=max_rows)
        print(table.to_string(index=False))
        if args.output_csv is not None:
            full_table = open_initial_condition_table(args.input, max_rows=None)
            full_table.to_csv(args.output_csv, index=False)
            print(f"Wrote IC table CSV: {args.output_csv}")
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
