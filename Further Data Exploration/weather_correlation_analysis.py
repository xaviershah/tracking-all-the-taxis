from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "combined_2024_with_features_with_weather.parquet"
OUTPUT_DIR = BASE_DIR

WEATHER_COLS = [
    "HourlyDewPointTemperature",
    "HourlyDryBulbTemperature",
    "HourlyPrecipitation",
    "HourlyWetBulbTemperature",
    "HourlyWindDirection",
    "HourlyWindSpeed",
    "DailyAverageDewPointTemperature",
    "DailyAverageDryBulbTemperature",
    "DailyAverageWetBulbTemperature",
    "DailyAverageWindSpeed",
    "DailyCoolingDegreeDays",
    "DailyDepartureFromNormalAverageTemperature",
    "DailyHeatingDegreeDays",
    "DailyMaximumDryBulbTemperature",
    "DailyMinimumDryBulbTemperature",
    "DailyPeakWindDirection",
    "DailyPeakWindSpeed",
    "DailyPrecipitation",
    "DailySnowDepth",
    "DailySnowfall",
]


def build_zone_slot_table(con: duckdb.DuckDBPyConnection, input_path: Path) -> pd.DataFrame:
    weather_aggregates = ",\n            ".join(
        [f'AVG(TRY_CAST("{c}" AS DOUBLE)) AS "w_{c}"' for c in WEATHER_COLS]
    )
    sql = f"""
        SELECT
            CAST(pu_zone_id AS INTEGER) AS zone_id,
            time_bucket(INTERVAL '30 minutes', pickup_datetime) AS slot_30min,
            COUNT(*)::BIGINT AS pickup_count,
            {weather_aggregates}
        FROM read_parquet(?)
        WHERE pickup_datetime >= TIMESTAMP '2024-01-01'
          AND pickup_datetime < TIMESTAMP '2025-01-01'
          AND pickup_datetime IS NOT NULL
          AND pu_zone_id IS NOT NULL
        GROUP BY 1, 2
    """
    return con.execute(sql, [str(input_path)]).fetchdf()


def spearman_with_pvalues(df: pd.DataFrame, demand_col: str, feature_cols: list[str]) -> pd.DataFrame:
    y = df[demand_col].to_numpy(dtype=float)
    rows = []
    for col in feature_cols:
        x = df[col].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            rho, pval = np.nan, np.nan
        else:
            rho, pval = stats.spearmanr(x[mask], y[mask])
        rows.append({"feature": col, "spearman_rho": rho, "p_value": pval, "n": int(mask.sum())})
    return pd.DataFrame(rows).sort_values("spearman_rho", ascending=True, na_position="last")


def plot_spearman(spearman_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * len(spearman_df))))
    y = np.arange(len(spearman_df))
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in spearman_df["spearman_rho"]]
    ax.barh(y, spearman_df["spearman_rho"], color=colors, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(spearman_df["feature"], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spearman rho")
    ax.set_title("Spearman correlation: pickups vs weather features (2024)")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    input_path = INPUT_PATH.resolve()
    out_dir = OUTPUT_DIR.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    zone_slot = build_zone_slot_table(con, input_path)
    con.close()

    feature_cols = [f"w_{c}" for c in WEATHER_COLS]
    spearman_df = spearman_with_pvalues(zone_slot, "pickup_count", feature_cols)
    spearman_df["feature"] = spearman_df["feature"].str.replace("^w_", "", regex=True)

    spearman_csv = out_dir / "weather_spearman_zone_slot_vs_demand.csv"
    spearman_png = out_dir / "weather_spearman_zone_slot_ranked.png"
    spearman_df.to_csv(spearman_csv, index=False)
    plot_spearman(spearman_df, spearman_png)

    print(f"Spearman table: {spearman_csv}")
    print(f"Spearman chart: {spearman_png}")


if __name__ == "__main__":
    main()
