from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr


def interval_label(summary: dict[str, object]) -> str:
    interval = summary["global_attrs"].get("interval_minutes")
    return f"{interval}-minute" if interval not in (None, "", "None") else "interval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exploratory data analysis on a weather NetCDF dataset."
    )
    parser.add_argument(
        "--input",
        default="nyc_mrms_30min.nc",
        help="Path to the NetCDF file to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/weather_eda",
        help="Directory for EDA outputs.",
    )
    parser.add_argument(
        "--results-dir",
        default="",
        help="Directory for tabular outputs.",
    )
    parser.add_argument(
        "--graphs-dir",
        default="",
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top intervals and entities to include in summaries.",
    )
    parser.add_argument(
        "--export-raw-csv",
        action="store_true",
        help="Export the flattened dataset as CSV alongside summary tables.",
    )
    return parser.parse_args()


def rounded_frame(frame: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    rounded = frame.copy()
    float_columns = rounded.select_dtypes(include=["float", "float32", "float64"]).columns
    rounded[float_columns] = rounded[float_columns].round(decimals)
    return rounded


def build_summary(dataset: xr.Dataset, variable: str, top_n: int) -> dict[str, object]:
    data_array = dataset[variable]
    frame = data_array.to_dataframe().reset_index()
    value_name = variable
    dimension_columns = list(data_array.dims)
    entity_columns = [column for column in dimension_columns if column != "time"]
    key_columns = [column for column in dimension_columns if column in frame.columns]

    times = pd.to_datetime(dataset["time"].values)
    time_deltas = times.to_series().sort_values().diff().dropna()
    values = frame[value_name].to_numpy(dtype=float, copy=False)
    positive_mask = values > 0

    summary: dict[str, object] = {
        "dataset_shape": dict(dataset.sizes),
        "dims": dimension_columns,
        "entity_columns": entity_columns,
        "variable": variable,
        "dtype": str(data_array.dtype),
        "global_attrs": {key: str(value) for key, value in dataset.attrs.items()},
        "variable_attrs": {key: str(value) for key, value in data_array.attrs.items()},
        "time_start": times.min(),
        "time_end": times.max(),
        "time_steps": len(times),
        "time_delta_minutes": sorted(
            {(delta.total_seconds() / 60.0) for delta in time_deltas.to_list()}
        ),
        "records": len(frame),
        "duplicate_keys": int(frame.duplicated(subset=key_columns).sum()),
        "missing_values": int(frame[value_name].isna().sum()),
        "non_finite_values": int((~np.isfinite(values)).sum()),
        "negative_values": int((values < 0).sum()),
        "zero_values": int((values == 0).sum()),
        "positive_values": int(positive_mask.sum()),
        "wet_share": float(positive_mask.mean()),
        "distinct_values": int(frame[value_name].nunique(dropna=True)),
        "all_time_slices_identical": bool(
            ((data_array.diff("time").fillna(0)) == 0).all().item()
        )
        if "time" in data_array.dims and dataset.sizes.get("time", 0) > 1
        else True,
        "coord_ranges": {},
        "coordinate_outliers": None,
        "coordinate_outlier_count": 0,
    }

    for coord_name in ("lat", "lon", "latitude", "longitude", "grid_lat", "grid_lon"):
        if coord_name in dataset.coords:
            summary["coord_ranges"][coord_name] = (
                float(dataset[coord_name].min().item()),
                float(dataset[coord_name].max().item()),
            )

    describe = frame[value_name].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99])
    summary["distribution"] = {key: float(value) for key, value in describe.items()}

    if positive_mask.any():
        non_zero = frame.loc[positive_mask, value_name]
        summary["non_zero_distribution"] = {
            key: float(value)
            for key, value in non_zero.describe(
                percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]
            ).items()
        }
    else:
        summary["non_zero_distribution"] = None

    per_time = (
        frame.groupby("time")[value_name]
        .agg(["count", "sum", "mean", "median", "max", "std"])
        .fillna(0.0)
        .reset_index()
    )
    active_by_time = (
        frame.groupby("time")[value_name]
        .apply(lambda series: int((series > 0).sum()))
        .reset_index(name="active_entities")
    )
    per_time = per_time.merge(active_by_time, on="time", how="left")
    per_time["pct_active_entities"] = per_time["active_entities"] / per_time["count"]
    per_time = per_time.sort_values("time").reset_index(drop=True)
    summary["per_time"] = per_time

    daily = per_time.copy()
    daily["date"] = pd.to_datetime(daily["time"]).dt.date
    daily = (
        daily.groupby("date")[["sum", "active_entities"]]
        .agg({"sum": "sum", "active_entities": "sum"})
        .reset_index()
        .rename(
            columns={
                "sum": "daily_precipitation_total",
                "active_entities": "daily_active_entity_count",
            }
        )
    )
    summary["daily"] = daily.sort_values("date").reset_index(drop=True)

    top_intervals = (
        per_time.sort_values(["sum", "max"], ascending=[False, False])
        .head(top_n)
        .reset_index(drop=True)
        .rename(columns={"sum": "total_precipitation"})
    )
    summary["top_intervals"] = top_intervals

    if entity_columns:
        group_columns = entity_columns.copy()
        aggregate_map = {
            value_name: ["count", "sum", "mean", "max"],
        }
        entity_summary = frame.groupby(group_columns).agg(aggregate_map)
        entity_summary.columns = ["observation_count", "total_precipitation", "mean_precipitation", "max_precipitation"]
        entity_summary = entity_summary.reset_index()

        wet_counts = (
            frame.assign(is_wet=frame[value_name] > 0)
            .groupby(group_columns)["is_wet"]
            .sum()
            .reset_index(name="wet_intervals")
        )
        entity_summary = entity_summary.merge(wet_counts, on=group_columns, how="left")
        entity_summary["pct_wet_intervals"] = (
            entity_summary["wet_intervals"] / entity_summary["observation_count"]
        )

        if "zone_id" in entity_summary.columns:
            extra_columns = [
                column
                for column in ("grid_lat", "grid_lon", "latitude", "longitude")
                if column in frame.columns
            ]
            if extra_columns:
                zone_lookup = (
                    frame[["zone_id", *extra_columns]]
                    .drop_duplicates(subset=["zone_id"])
                    .reset_index(drop=True)
                )
                entity_summary = entity_summary.merge(zone_lookup, on="zone_id", how="left")

        entity_summary = entity_summary.sort_values(
            ["total_precipitation", "max_precipitation"], ascending=[False, False]
        ).reset_index(drop=True)
        if "zone_id" in entity_summary.columns and {"latitude", "longitude"}.issubset(entity_summary.columns):
            coordinate_outliers = entity_summary[
                (~entity_summary["latitude"].between(40.4, 41.1))
                | (~entity_summary["longitude"].between(-74.3, -73.6))
                | entity_summary["latitude"].isna()
                | entity_summary["longitude"].isna()
            ].copy()
            summary["coordinate_outliers"] = coordinate_outliers.reset_index(drop=True)
            summary["coordinate_outlier_count"] = int(len(coordinate_outliers))
        summary["entity_summary"] = entity_summary
        summary["top_entities"] = entity_summary.head(top_n).reset_index(drop=True)
    else:
        summary["entity_summary"] = None
        summary["top_entities"] = None

    summary["frame"] = frame
    return summary


def save_csv_outputs(summary: dict[str, object], results_dir: Path, export_raw_csv: bool) -> list[Path]:
    csv_dir = results_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    per_time_path = csv_dir / "per_time_summary.csv"
    rounded_frame(summary["per_time"]).to_csv(per_time_path, index=False)
    outputs.append(per_time_path)

    daily_path = csv_dir / "daily_summary.csv"
    rounded_frame(summary["daily"]).to_csv(daily_path, index=False)
    outputs.append(daily_path)

    top_intervals_path = csv_dir / "top_intervals.csv"
    rounded_frame(summary["top_intervals"]).to_csv(top_intervals_path, index=False)
    outputs.append(top_intervals_path)

    if summary["entity_summary"] is not None:
        entity_path = csv_dir / "entity_summary.csv"
        rounded_frame(summary["entity_summary"]).to_csv(entity_path, index=False)
        outputs.append(entity_path)

        top_entities_path = csv_dir / "top_entities.csv"
        rounded_frame(summary["top_entities"]).to_csv(top_entities_path, index=False)
        outputs.append(top_entities_path)

        if summary["coordinate_outliers"] is not None:
            outlier_path = csv_dir / "coordinate_outliers.csv"
            rounded_frame(summary["coordinate_outliers"]).to_csv(outlier_path, index=False)
            outputs.append(outlier_path)

    if export_raw_csv:
        raw_path = csv_dir / "weather_data_flat.csv"
        rounded_frame(summary["frame"]).to_csv(raw_path, index=False)
        outputs.append(raw_path)

    return outputs


def save_plots(summary: dict[str, object], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    figure_paths: list[str] = []

    per_time = summary["per_time"]
    variable = summary["variable"]
    daily = summary["daily"]
    frame = summary["frame"]

    timeline_path = output_dir / "interval_totals.png"
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(per_time["time"], per_time["sum"], color="#2563eb", linewidth=1.5)
    ax.set_title(f"Total Precipitation per {interval_label(summary).title()}")
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{variable} total")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(timeline_path, dpi=180)
    plt.close(fig)
    figure_paths.append(timeline_path.name)

    daily_path = output_dir / "daily_totals.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=daily, x="date", y="daily_precipitation_total", ax=ax, color="#0f766e")
    ax.set_title("Daily Precipitation Totals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total precipitation")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(daily_path, dpi=180)
    plt.close(fig)
    figure_paths.append(daily_path.name)

    histogram_path = output_dir / "non_zero_distribution.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    non_zero = frame.loc[frame[variable] > 0, variable]
    if non_zero.empty:
        ax.text(0.5, 0.5, "No positive precipitation values present", ha="center", va="center")
        ax.set_axis_off()
    else:
        sns.histplot(non_zero, bins=40, ax=ax, color="#1d4ed8")
        ax.set_title("Distribution of Non-Zero Precipitation Values")
        ax.set_xlabel(variable)
    fig.tight_layout()
    fig.savefig(histogram_path, dpi=180)
    plt.close(fig)
    figure_paths.append(histogram_path.name)

    if summary["top_entities"] is not None:
        entity_plot_path = output_dir / "top_entities.png"
        fig, ax = plt.subplots(figsize=(10, 4))
        top_entities = summary["top_entities"].copy()
        entity_label = (
            top_entities["zone_id"].astype(str)
            if "zone_id" in top_entities.columns
            else top_entities.index.astype(str)
        )
        sns.barplot(
            x=entity_label,
            y=top_entities["total_precipitation"],
            ax=ax,
            color="#9333ea",
        )
        ax.set_title("Top Entities by Total Precipitation")
        ax.set_xlabel("Zone ID" if "zone_id" in top_entities.columns else "Entity")
        ax.set_ylabel("Total precipitation")
        fig.tight_layout()
        fig.savefig(entity_plot_path, dpi=180)
        plt.close(fig)
        figure_paths.append(entity_plot_path.name)

    if summary["entity_summary"] is not None and "zone_id" in summary["entity_summary"].columns:
        zones = gpd.read_file("../zones/data/taxi_zones/taxi_zones.shp")
        zones["LocationID"] = zones["LocationID"].astype(int)
        zone_weather = zones.merge(
            summary["entity_summary"],
            left_on="LocationID",
            right_on="zone_id",
            how="left",
        )

        total_map_path = output_dir / "zone_total_precipitation_choropleth.png"
        fig, ax = plt.subplots(figsize=(11, 11))
        zone_weather.plot(
            column="total_precipitation",
            cmap="Blues",
            linewidth=0.25,
            edgecolor="#303030",
            legend=True,
            missing_kwds={"color": "#efefef", "label": "No data"},
            ax=ax,
        )
        ax.set_title("NYC Taxi Zones: Total Weekly Precipitation")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(total_map_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(total_map_path.name)

        max_map_path = output_dir / "zone_peak_precipitation_choropleth.png"
        fig, ax = plt.subplots(figsize=(11, 11))
        zone_weather.plot(
            column="max_precipitation",
            cmap="YlGnBu",
            linewidth=0.25,
            edgecolor="#303030",
            legend=True,
            missing_kwds={"color": "#efefef", "label": "No data"},
            ax=ax,
        )
        ax.set_title(f"NYC Taxi Zones: Peak {interval_label(summary).title()} Precipitation")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(max_map_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(max_map_path.name)

    return figure_paths


def print_outputs(csv_outputs: list[Path], figure_names: list[str], figures_dir: Path) -> None:
    print("CSV outputs:")
    for path in csv_outputs:
        print(f"  {path}")
    print("Figure outputs:")
    for name in figure_names:
        print(f"  {figures_dir / name}")


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.input)
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir) if args.results_dir else output_dir
    figures_dir = Path(args.graphs_dir) if args.graphs_dir else output_dir / "figures"

    dataset = xr.open_dataset(dataset_path)
    variable = next(iter(dataset.data_vars))
    summary = build_summary(dataset, variable, args.top_n)
    csv_outputs = save_csv_outputs(summary, results_dir, export_raw_csv=args.export_raw_csv)
    figure_names = save_plots(summary, figures_dir)
    print_outputs(csv_outputs, figure_names, figures_dir)


if __name__ == "__main__":
    main()
