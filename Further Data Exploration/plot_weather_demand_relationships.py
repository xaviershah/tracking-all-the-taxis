from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "combined_2024_with_features_with_weather.parquet"
DEFAULT_OUTPUT_DIR = BASE_DIR

def main() -> None:
    input_path = DEFAULT_INPUT.resolve()
    output_dir = DEFAULT_OUTPUT_DIR.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    temp_df = con.execute(
        """
        WITH hourly_zone AS (
            SELECT
                time_bucket(INTERVAL '1 hour', pickup_datetime) AS hour_slot,
                AVG(TRY_CAST(HourlyDryBulbTemperature AS DOUBLE)) AS temp_c,
                COUNT(*)::DOUBLE AS hourly_pickups
            FROM read_parquet(?)
            WHERE pickup_datetime >= TIMESTAMP '2024-01-01'
              AND pickup_datetime < TIMESTAMP '2025-01-01'
              AND pickup_datetime IS NOT NULL
              AND pu_zone_id IS NOT NULL
            GROUP BY pu_zone_id, 1
        )
        SELECT
            CONCAT(
                CAST(FLOOR(temp_c / 5.0) * 5 AS INTEGER),
                ' to ',
                CAST(FLOOR(temp_c / 5.0) * 5 + 5 AS INTEGER),
                'C'
            ) AS temp_bin,
            AVG(hourly_pickups) AS avg_hourly_pickups,
            AVG(temp_c) AS mean_temp_c
        FROM hourly_zone
        WHERE temp_c IS NOT NULL
        GROUP BY 1
        ORDER BY mean_temp_c
        """,
        [str(input_path)],
    ).fetchdf()

    precip_df = con.execute(
        """
        WITH hourly_zone AS (
            SELECT
                time_bucket(INTERVAL '1 hour', pickup_datetime) AS hour_slot,
                AVG(TRY_CAST(HourlyPrecipitation AS DOUBLE)) AS precip,
                COUNT(*)::DOUBLE AS hourly_pickups
            FROM read_parquet(?)
            WHERE pickup_datetime >= TIMESTAMP '2024-01-01'
              AND pickup_datetime < TIMESTAMP '2025-01-01'
              AND pickup_datetime IS NOT NULL
              AND pu_zone_id IS NOT NULL
            GROUP BY pu_zone_id, 1
        )
        SELECT
            CASE
                WHEN precip <= 0 THEN 'No rain'
                WHEN precip <= 0.10 THEN 'Light rain'
                ELSE 'Heavy rain'
            END AS precip_bin,
            AVG(hourly_pickups) AS avg_zone_hourly_pickups
        FROM hourly_zone
        WHERE precip IS NOT NULL
        GROUP BY 1
        ORDER BY CASE
            WHEN precip_bin = 'No rain' THEN 1
            WHEN precip_bin = 'Light rain' THEN 2
            ELSE 3
        END
        """,
        [str(input_path)],
    ).fetchdf()

    wind_df = con.execute(
        """
        WITH hourly_zone AS (
            SELECT
                time_bucket(INTERVAL '1 hour', pickup_datetime) AS hour_slot,
                AVG(TRY_CAST(HourlyWindSpeed AS DOUBLE)) AS wind_speed,
                COUNT(*)::DOUBLE AS hourly_pickups
            FROM read_parquet(?)
            WHERE pickup_datetime >= TIMESTAMP '2024-01-01'
              AND pickup_datetime < TIMESTAMP '2025-01-01'
              AND pickup_datetime IS NOT NULL
              AND pu_zone_id IS NOT NULL
            GROUP BY pu_zone_id, 1
        )
        SELECT
            CASE
                WHEN wind_speed < 8 THEN 'Low wind'
                WHEN wind_speed < 16 THEN 'Medium wind'
                WHEN wind_speed < 25 THEN 'Heavy wind'
                ELSE 'Extreme wind'
            END AS wind_bin,
            AVG(hourly_pickups) AS avg_zone_hourly_pickups
        FROM hourly_zone
        WHERE wind_speed IS NOT NULL
        GROUP BY 1
        ORDER BY CASE
            WHEN wind_bin = 'Low wind' THEN 1
            WHEN wind_bin = 'Medium wind' THEN 2
            WHEN wind_bin = 'Heavy wind' THEN 3
            ELSE 4
        END
        """,
        [str(input_path)],
    ).fetchdf()

    daily_df = con.execute(
        """
        SELECT
            CAST(pickup_datetime AS DATE) AS pickup_date,
            COUNT(*)::BIGINT AS total_pickups,
            AVG(TRY_CAST(HourlyDryBulbTemperature AS DOUBLE)) AS avg_temp_c
        FROM read_parquet(?)
        WHERE pickup_datetime >= TIMESTAMP '2024-01-01'
          AND pickup_datetime < TIMESTAMP '2025-01-01'
          AND pickup_datetime IS NOT NULL
        GROUP BY 1
        ORDER BY 1
        """,
        [str(input_path)],
    ).fetchdf()
    con.close()

    daily_df["pickup_date"] = pd.to_datetime(daily_df["pickup_date"])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(temp_df["temp_bin"], temp_df["avg_hourly_pickups"], marker="o", linewidth=2)
    ax.set_title("Average Hourly Pickup Demand by Temperature Bin (2024)")
    ax.set_xlabel("Temperature bin")
    ax.set_ylabel("Average pickups per zone-hour")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(output_dir / "weather_temp_bins_avg_demand.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(precip_df["precip_bin"], precip_df["avg_zone_hourly_pickups"])
    ax.set_title("Average Zone Demand by Precipitation Intensity (2024)")
    ax.set_xlabel("Precipitation bin")
    ax.set_ylabel("Average pickups per zone-hour")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(output_dir / "weather_precip_bins_avg_demand.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(wind_df["wind_bin"], wind_df["avg_zone_hourly_pickups"])
    ax.set_title("Average Zone Demand by Wind Speed Intensity (2024)")
    ax.set_xlabel("Wind speed bin")
    ax.set_ylabel("Average pickups per zone-hour")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(output_dir / "weather_wind_bins_avg_demand.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(15, 6))
    ax2 = ax1.twinx()
    ax1.plot(daily_df["pickup_date"], daily_df["total_pickups"], color="#1f77b4", linewidth=1.4)
    ax2.plot(daily_df["pickup_date"], daily_df["avg_temp_c"], color="#d62728", linewidth=1.5, alpha=0.85)
    ax1.set_title("Daily Pickup Volume vs Daily Average Temperature (2024)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Total daily pickups", color="#1f77b4")
    ax2.set_ylabel("Average temperature (C)", color="#d62728")
    ax1.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(output_dir / "daily_pickups_vs_temp_dual_axis_2024.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("Done. Plots saved in:", output_dir)


if __name__ == "__main__":
    main()
