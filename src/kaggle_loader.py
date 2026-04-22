from pathlib import Path

import pandas as pd

from config import KAGGLE_DATA_DIR
from process import finalize_results



class KaggleDataError(RuntimeError):
    pass


KAGGLE_DRIVER_NUMBERS = {
    "albon": 23,
    "alonso": 14,
    "antonelli": 12,
    "bearman": 87,
    "bortoleto": 5,
    "bottas": 77,
    "de_vries": 21,
    "doohan": 7,
    "gasly": 10,
    "hadjar": 6,
    "hamilton": 44,
    "hulkenberg": 27,
    "kevin_magnussen": 20,
    "lawson": 30,
    "leclerc": 16,
    "max_verstappen": 1,
    "norris": 4,
    "ocon": 31,
    "perez": 11,
    "piastri": 81,
    "ricciardo": 3,
    "russell": 63,
    "sainz": 55,
    "sargeant": 2,
    "stroll": 18,
    "tsunoda": 22,
    "zhou": 24,
}

# AI generated: initial draft of this script was created with the help of AI assistance
# and then reviewed and refactored by Rui Chen. The final version was manually edited.

def load_kaggle_dataset(years, data_dir=KAGGLE_DATA_DIR):
    data_dir = Path(data_dir)

    required_files = ["races.csv", "qualifying.csv", "results.csv", "drivers.csv"]
    for filename in required_files:
        if not (data_dir / filename).exists():
            raise KaggleDataError(f"Missing {filename} in {data_dir}")

    races = pd.read_csv(data_dir / "races.csv")
    qualifying = pd.read_csv(data_dir / "qualifying.csv")
    results = pd.read_csv(data_dir / "results.csv")
    drivers = pd.read_csv(data_dir / "drivers.csv")

    circuits = pd.DataFrame()
    if (data_dir / "circuits.csv").exists():
        circuits = pd.read_csv(data_dir / "circuits.csv")

    year_column = "year" if "year" in races.columns else "season"
    race_name_column = "name" if "name" in races.columns else "race_name"
    race_id_column = "raceId" if "raceId" in races.columns else "race_id"
    circuit_id_column = "circuitId" if "circuitId" in races.columns else "circuit_id"

    races = races[pd.to_numeric(races[year_column], errors="coerce").isin(years)].copy()
    if races.empty:
        return pd.DataFrame()

    qual_race_column = "raceId" if "raceId" in qualifying.columns else "race_id"
    qual_driver_column = "driverId" if "driverId" in qualifying.columns else "driver_id"
    qual_df = qualifying[[qual_race_column, qual_driver_column, "position"]].copy()
    qual_df.columns = ["race_id", "driver_id", "qualifying_position"]

    result_race_column = "raceId" if "raceId" in results.columns else "race_id"
    result_driver_column = "driverId" if "driverId" in results.columns else "driver_id"
    result_position_column = "positionOrder" if "positionOrder" in results.columns else "position_order"
    race_df = results[[result_race_column, result_driver_column, result_position_column]].copy()
    race_df.columns = ["race_id", "driver_id", "race_position"]

    driver_id_column = "driverId" if "driverId" in drivers.columns else "driver_id"
    driver_number_column = None
    for column in ["number", "driver_number", "permanentNumber"]:
        if column in drivers.columns:
            driver_number_column = column
            break

    code_column = "code" if "code" in drivers.columns else None
    first_name_column = "forename" if "forename" in drivers.columns else "givenName"
    last_name_column = "surname" if "surname" in drivers.columns else "familyName"

    merged = pd.merge(qual_df, race_df, on=["race_id", "driver_id"], how="inner")
    merged = pd.merge(
        merged,
        races[[race_id_column, year_column, race_name_column, circuit_id_column]],
        left_on="race_id",
        right_on=race_id_column,
        how="inner",
    )

    driver_columns = [driver_id_column, first_name_column, last_name_column]
    if driver_number_column is not None:
        driver_columns.append(driver_number_column)
    if code_column is not None:
        driver_columns.append(code_column)

    merged = pd.merge(
        merged,
        drivers[driver_columns],
        left_on="driver_id",
        right_on=driver_id_column,
        how="left",
    )

    merged.rename(
        columns={
            driver_number_column: "driver_number",
            code_column: "name_acronym",
            year_column: "year",
            race_name_column: "circuit",
        },
        inplace=True,
    )

    if "driver_number" not in merged.columns:
        merged["driver_number"] = merged["driver_id"].map(KAGGLE_DRIVER_NUMBERS)
    if "name_acronym" not in merged.columns:
        merged["name_acronym"] = pd.NA

    merged["full_name"] = (
        merged[first_name_column].fillna("").str.strip()
        + " "
        + merged[last_name_column].fillna("").str.strip()
    ).str.strip()

    if not circuits.empty and circuit_id_column in circuits.columns and "name" in circuits.columns:
        lookup = circuits[[circuit_id_column, "name"]].copy()
        lookup.columns = [circuit_id_column, "circuit_name"]
        merged = pd.merge(merged, lookup, on=circuit_id_column, how="left")
        merged["circuit"] = merged["circuit_name"].fillna(merged["circuit"])
        merged.drop(columns=["circuit_name"], inplace=True)

    merged["meeting_key"] = merged["race_id"]

    keep = [
        "driver_number",
        "qualifying_position",
        "race_position",
        "full_name",
        "name_acronym",
        "year",
        "circuit",
        "meeting_key",
    ]

    return finalize_results(merged[keep].copy())
