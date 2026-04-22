# AI generated: initial draft of this script was created with the help ofAI assistance
# and then reviewed and refactored by Rui Chen. The final version was manually edited.
import pandas as pd


def session_position(records, col_name):
    """Keep the final classified position for each driver in one session."""
    df = pd.DataFrame(records)
    if df.empty or "driver_number" not in df.columns:
        return pd.DataFrame(columns=["driver_number", col_name])

    df = df[["driver_number", "position"]].dropna().copy()
    df = df.rename(columns={"position": col_name})
    return df


def merge_session_results(qual_df, race_df, drivers_df, year, circuit, meeting_key):
    """Merge qualifying, race, and driver data into one analysis table."""
    merged = pd.merge(qual_df, race_df, on="driver_number", how="inner")

    if not drivers_df.empty and "driver_number" in drivers_df.columns:
        name_cols = [
            col for col in ("driver_number", "full_name", "name_acronym")
            if col in drivers_df.columns
        ]
        merged = pd.merge(merged, drivers_df[name_cols], on="driver_number", how="left")

    merged["year"] = year
    merged["circuit"] = circuit
    merged["meeting_key"] = meeting_key
    return merged
