"""
load_results.py
Loads and merges qualifying and race result data for analysis.
Combines data from the OpenF1 API across multiple races and seasons.
"""

import pandas as pd
from src.openf1_api import OpenF1Client


def load_season_pairs(year: int, client: OpenF1Client | None = None) -> pd.DataFrame:
    """
    For a given season year, fetch all qualifying + race pairs and merge them.

    Args:
        year:   F1 season year (2022–2025)
        client: Optional shared OpenF1Client instance

    Returns:
        DataFrame with columns: year, circuit, meeting_key, driver_number,
        full_name, qualifying_position, race_position
    """
    if client is None:
        client = OpenF1Client()

    print(f"Fetching sessions for {year}...")
    all_sessions = client.get_sessions(year=year)

    if not all_sessions:
        print(f"  No sessions found for {year}.")
        return pd.DataFrame()

    race_sessions = [s for s in all_sessions if s.get("session_name") == "Race"]
    qual_sessions = [s for s in all_sessions if s.get("session_name") == "Qualifying"]

    if not race_sessions or not qual_sessions:
        print(f"  Incomplete session data for {year}.")
        return pd.DataFrame()

    # Index qualifying sessions by meeting_key for fast lookup
    qual_by_meeting: dict[int, dict] = {
        s["meeting_key"]: s for s in qual_sessions if "meeting_key" in s
    }

    records: list[pd.DataFrame] = []

    for race in race_sessions:
        meeting_key = race.get("meeting_key")
        race_session_key = race.get("session_key")
        circuit = race.get("circuit_short_name", "Unknown")

        qual = qual_by_meeting.get(meeting_key)
        if qual is None:
            continue

        qual_session_key = qual["session_key"]

        # ── Fetch position data ──────────────────────────────────────────────
        raw_qual = client.get_position(session_key=qual_session_key)
        raw_race = client.get_position(session_key=race_session_key)

        if not raw_qual or not raw_race:
            continue

        qual_df = _last_position(raw_qual, "qualifying_position")
        race_df = _last_position(raw_race, "race_position")

        # ── Fetch driver names ───────────────────────────────────────────────
        raw_drivers = client.get_drivers(session_key=race_session_key)
        drivers_df = pd.DataFrame(raw_drivers) if raw_drivers else pd.DataFrame()

        # ── Merge ────────────────────────────────────────────────────────────
        merged = pd.merge(qual_df, race_df, on="driver_number", how="inner")

        if not drivers_df.empty and "driver_number" in drivers_df.columns:
            name_cols = [c for c in ("driver_number", "full_name", "name_acronym")
                         if c in drivers_df.columns]
            merged = pd.merge(merged, drivers_df[name_cols], on="driver_number", how="left")

        merged["year"] = year
        merged["circuit"] = circuit
        merged["meeting_key"] = meeting_key

        records.append(merged)
        print(f"  Loaded: {circuit} ({year})")

    if not records:
        return pd.DataFrame()

    return pd.concat(records, ignore_index=True)


def load_multiple_seasons(years: list[int], client: OpenF1Client | None = None) -> pd.DataFrame:
    """
    Load and combine data across multiple seasons.

    Args:
        years:  List of season years, e.g. [2022, 2023, 2024]
        client: Optional shared OpenF1Client instance

    Returns:
        Combined DataFrame for all seasons, with position columns as int
    """
    if client is None:
        client = OpenF1Client()

    all_data = [load_season_pairs(year, client=client) for year in years]
    all_data = [df for df in all_data if not df.empty]

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    combined.dropna(subset=["qualifying_position", "race_position"], inplace=True)
    combined["qualifying_position"] = combined["qualifying_position"].astype(int)
    combined["race_position"] = combined["race_position"].astype(int)
    return combined


# ── Internal helpers ─────────────────────────────────────────────────────────

def _last_position(records: list[dict], col_name: str) -> pd.DataFrame:
    """
    Given a list of position records (time-series from OpenF1), return one
    row per driver containing their last recorded position.

    Args:
        records:  Raw list of dicts from client.get_position()
        col_name: Name to give the position column in the output DataFrame

    Returns:
        DataFrame with columns ['driver_number', col_name]
    """
    df = pd.DataFrame(records)
    if df.empty or "driver_number" not in df.columns:
        return pd.DataFrame(columns=["driver_number", col_name])

    if "date" in df.columns:
        df = df.sort_values("date")

    df = (
        df.groupby("driver_number")["position"]
        .last()
        .reset_index()
        .rename(columns={"position": col_name})
    )
    return df