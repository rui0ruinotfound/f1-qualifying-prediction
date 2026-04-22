import pandas as pd

from config import FASTF1_CACHE_DIR
from process import finalize_results


class FastF1DataError(RuntimeError):
    pass


def load_results_table(results, position_column, include_names=False):
    if results is None or len(results) == 0:
        columns = ["driver_number", position_column]
        if include_names:
            columns.extend(["full_name", "name_acronym"])
        return pd.DataFrame(columns=columns)

    df = results.copy()
    keep = ["DriverNumber", "Position"]

    if include_names:
        keep.extend(["FullName", "Abbreviation"])

    keep = [column for column in keep if column in df.columns]
    df = df[keep].copy()
    df.rename(
        columns={
            "DriverNumber": "driver_number",
            "Position": position_column,
            "FullName": "full_name",
            "Abbreviation": "name_acronym",
        },
        inplace=True,
    )
    return df

# AI generated: initial draft of this script was created with the help of AI assistance
# and then reviewed and refactored by Rui Chen. The final version was manually edited.

def load_fastf1_multiple_seasons(years):
    try:
        import fastf1
    except ModuleNotFoundError as exc:
        raise FastF1DataError("FastF1 is not installed") from exc

    FASTF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(FASTF1_CACHE_DIR))

    all_rows = []

    for year in years:
        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as exc:
            raise FastF1DataError(f"Could not load FastF1 schedule for {year}: {exc}") from exc

        for event in schedule.itertuples(index=False):
            round_number = getattr(event, "RoundNumber", None)
            if round_number is None or pd.isna(round_number) or round_number <= 0:
                continue

            circuit = getattr(event, "EventName", None) or getattr(event, "Location", "Unknown")
            meeting_key = f"fastf1_{year}_{int(round_number)}"

            try:
                qualifying = fastf1.get_session(year, int(round_number), "Q")
                race = fastf1.get_session(year, int(round_number), "R")
                qualifying.load(laps=False, telemetry=False, weather=False, messages=False)
                race.load(laps=False, telemetry=False, weather=False, messages=False)
            except Exception as exc:
                print(f"  Skipped FastF1 {circuit} ({year}): {exc}")
                continue

            qual_df = load_results_table(
                qualifying.results,
                "qualifying_position",
                include_names=True,
            )
            race_df = load_results_table(race.results, "race_position")

            merged = pd.merge(qual_df, race_df, on="driver_number", how="inner")
            merged["year"] = year
            merged["circuit"] = circuit
            merged["meeting_key"] = meeting_key

            if not merged.empty:
                all_rows.append(merged)
                print(f"  Loaded FastF1: {circuit} ({year})")

    if not all_rows:
        return pd.DataFrame()

    return finalize_results(pd.concat(all_rows, ignore_index=True))
