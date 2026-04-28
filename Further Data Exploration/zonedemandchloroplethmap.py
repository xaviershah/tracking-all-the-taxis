from pathlib import Path

import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "combined_2024_with_features_with_weather.parquet"
DEFAULT_OUTPUT_PNG = BASE_DIR / "nyc_pickups_avg_per_30min_choropleth.png"
DEFAULT_ZONES_FILE = BASE_DIR / "taxi_zones" / "taxi_zones.shp"


def compute_zone_summary(dataset_path: Path):
    con = duckdb.connect()
    df = con.execute(
        """
        WITH slot_counts AS (
            SELECT
                CAST(pu_zone_id AS INTEGER) AS zone_id,
                time_bucket(INTERVAL '30 minutes', pickup_datetime) AS slot_30min,
                COUNT(*) AS pickup_count
            FROM read_parquet(?)
            WHERE pu_zone_id IS NOT NULL
              AND pickup_datetime IS NOT NULL
            GROUP BY 1, 2
        )
        SELECT
            zone_id,
            AVG(pickup_count)::DOUBLE AS avg_pickups_per_30min
        FROM slot_counts
        GROUP BY 1
        ORDER BY 1
        """,
        [str(dataset_path)],
    ).fetchdf()
    con.close()
    return df


def load_zones(path: Path):
    gdf = gpd.read_file(path)
    gdf["zone_id"] = gdf["LocationID"].astype(int)
    return gdf


def save_map_png(zones_with_data, output_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 12))
    zones_with_data.plot(
        column="avg_pickups_per_30min",
        cmap="YlOrRd",
        linewidth=0.2,
        edgecolor="black",
        legend=True,
        legend_kwds={"label": "Average pickups per 30-minute slot"},
        ax=ax,
    )
    ax.set_title("NYC Taxi Demand by Zone\nAverage pickups per 30-minute slot", fontsize=14, pad=14)
    ax.set_axis_off()
    plt.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    dataset_path = DEFAULT_DATASET.resolve()
    output_png = DEFAULT_OUTPUT_PNG.resolve()
    zones_path = DEFAULT_ZONES_FILE.resolve()

    zone_summary = compute_zone_summary(dataset_path)
    zones = load_zones(zones_path)

    merged = zones.merge(zone_summary, on="zone_id", how="left")
    merged["avg_pickups_per_30min"] = merged["avg_pickups_per_30min"].fillna(0.0)

    save_map_png(merged, output_png)
    print(f"Done. PNG saved to: {output_png}")


if __name__ == "__main__":
    main()
