import pandas as pd


RESULT_COLUMNS = [
    "driver_number",
    "qualifying_position",
    "race_position",
    "full_name",
    "name_acronym",
    "year",
    "circuit",
    "meeting_key",
]


def session_position(records, column_name):
    df = pd.DataFrame(records)
    if df.empty or "driver_number" not in df.columns:
        return pd.DataFrame(columns=["driver_number", column_name])

    df = df[["driver_number", "position"]].dropna().copy()
    df.rename(columns={"position": column_name}, inplace=True)
    return df


def merge_session_results(qual_df, race_df, drivers_df, year, circuit, meeting_key):
    merged = pd.merge(qual_df, race_df, on="driver_number", how="inner")

    if not drivers_df.empty and "driver_number" in drivers_df.columns:
        keep = [
            column
            for column in ["driver_number", "full_name", "name_acronym"]
            if column in drivers_df.columns
        ]
        merged = pd.merge(merged, drivers_df[keep], on="driver_number", how="left")

    merged["year"] = year
    merged["circuit"] = circuit
    merged["meeting_key"] = meeting_key
    return finalize_results(merged)


# AI generated: initial draft of this script was created with the help of AI assistance
# and then reviewed and refactored by Rui Chen. The final version was manually edited.

def finalize_results(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    df = df.copy()

    for column in RESULT_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    for column in ["qualifying_position", "race_position", "year"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "driver_number" in df.columns:
        numeric_driver_number = pd.to_numeric(df["driver_number"], errors="coerce")
        if numeric_driver_number.notna().all():
            df["driver_number"] = numeric_driver_number.astype(int)

    df.dropna(subset=["qualifying_position", "race_position"], inplace=True)
    df = df[RESULT_COLUMNS].copy()
    df["qualifying_position"] = df["qualifying_position"].astype(int)
    df["race_position"] = df["race_position"].astype(int)

    if df["year"].notna().all():
        df["year"] = df["year"].astype(int)

    return df.reset_index(drop=True)
