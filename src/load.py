
import pandas as pd

from config import QUALIFYING_SESSION_NAME, RACE_SESSION_NAME
from openf1_api import OpenF1APIError, OpenF1Client
from process import merge_session_results, session_position

# AI generated: initial draft of this script was created with the help ofAI assistance
# and then reviewed and refactored by Rui Chen. The final version was manually edited.
def load_season_pairs(year, client=None):
    """Fetch and merge qualifying and race position data for one season."""
    if client is None:
        client = OpenF1Client()

    print(f"Fetching sessions for {year}...")
    all_sessions = client.get_sessions(year=year)

    if not all_sessions:
        print(f"  No sessions found for {year}.")
        return pd.DataFrame()

    race_sessions = [
        s for s in all_sessions if s.get("session_name") == RACE_SESSION_NAME
    ]
    qual_sessions = [
        s for s in all_sessions if s.get("session_name") == QUALIFYING_SESSION_NAME
    ]

    if not race_sessions or not qual_sessions:
        print(f"  Incomplete session data for {year}.")
        return pd.DataFrame()

    qual_by_meeting = {
        s["meeting_key"]: s for s in qual_sessions if "meeting_key" in s
    }

    records = []

    for race in race_sessions:
        meeting_key = race.get("meeting_key")
        race_session_key = race.get("session_key")
        circuit = race.get("circuit_short_name", "Unknown")

        qual = qual_by_meeting.get(meeting_key)
        if qual is None:
            continue

        qual_session_key = qual["session_key"]

        try:
            raw_qual = client.get_session_result(session_key=qual_session_key)
            raw_race = client.get_session_result(session_key=race_session_key)
        except OpenF1APIError as exc:
            print(f"  Skipped {circuit} ({year}): {exc}")
            continue

        if not raw_qual or not raw_race:
            continue

        qual_df = session_position(raw_qual, "qualifying_position")
        race_df = session_position(raw_race, "race_position")

        try:
            raw_drivers = client.get_drivers(session_key=race_session_key)
            drivers_df = pd.DataFrame(raw_drivers) if raw_drivers else pd.DataFrame()
        except OpenF1APIError as exc:
            print(f"  Loaded {circuit} ({year}) without driver names: {exc}")
            drivers_df = pd.DataFrame()

        merged = merge_session_results(
            qual_df, race_df, drivers_df, year, circuit, meeting_key
        )

        records.append(merged)
        print(f"  Loaded: {circuit} ({year})")

    if not records:
        return pd.DataFrame()

    return pd.concat(records, ignore_index=True)


def load_multiple_seasons(years, client=None):
    """Load the requested seasons and combine them into one table."""
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
