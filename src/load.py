import pandas as pd

from config import QUALIFYING_SESSION_NAME, RACE_SESSION_NAME
from openf1_api import OpenF1APIError, OpenF1Client
from process import finalize_results, merge_session_results, session_position

# AI generated: initial draft of this script was created with the help ofAI assistance
# and then reviewed and refactored by Rui Chen. The final version was manually edited.


def load_season_pairs(year, client=None):
    if client is None:
        client = OpenF1Client()

    print(f"Fetching sessions for {year}...")
    sessions = client.get_sessions(year=year)

    if not sessions:
        print(f"  No sessions found for {year}.")
        return pd.DataFrame()

    race_sessions = [row for row in sessions if row.get("session_name") == RACE_SESSION_NAME]
    qualifying_sessions = [
        row for row in sessions if row.get("session_name") == QUALIFYING_SESSION_NAME
    ]

    if not race_sessions or not qualifying_sessions:
        print(f"  Incomplete session data for {year}.")
        return pd.DataFrame()

    qualifying_by_meeting = {
        row["meeting_key"]: row for row in qualifying_sessions if "meeting_key" in row
    }

    all_rows = []

    for race in race_sessions:
        meeting_key = race.get("meeting_key")
        qualifying = qualifying_by_meeting.get(meeting_key)
        if qualifying is None:
            continue

        circuit = race.get("circuit_short_name", "Unknown")

        try:
            qualifying_results = client.get_session_result(session_key=qualifying["session_key"])
            race_results = client.get_session_result(session_key=race["session_key"])
        except OpenF1APIError as exc:
            print(f"  Skipped {circuit} ({year}): {exc}")
            continue

        if not qualifying_results or not race_results:
            continue

        qual_df = session_position(qualifying_results, "qualifying_position")
        race_df = session_position(race_results, "race_position")

        try:
            drivers = client.get_drivers(session_key=race["session_key"])
            drivers_df = pd.DataFrame(drivers)
        except OpenF1APIError:
            drivers_df = pd.DataFrame()

        merged = merge_session_results(qual_df, race_df, drivers_df, year, circuit, meeting_key)
        if not merged.empty:
            all_rows.append(merged)
            print(f"  Loaded: {circuit} ({year})")

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


def load_multiple_seasons(years, client=None):
    if client is None:
        client = OpenF1Client()

    frames = [load_season_pairs(year, client=client) for year in years]
    frames = [frame for frame in frames if not frame.empty]

    if not frames:
        return pd.DataFrame()

    return finalize_results(pd.concat(frames, ignore_index=True))
