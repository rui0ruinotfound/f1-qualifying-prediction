import pandas as pd


RESULT_COLUMNS = [
    "driver_number",
    "qualifying_position",
    "race_position",
    "pit_stop_count",
    "rainfall",
    "track_temperature",
    "circuit_type",
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


STREET_CIRCUIT_KEYWORDS = {
    "baku",
    "marina_bay",
    "jeddah",
    "las vegas",
    "monaco",
    "monte carlo",
    "vegas",
    "singapore",
}

HYBRID_CIRCUIT_KEYWORDS = {
    "albert_park",
    "melbourne",
    "miami",
    "montreal",
    "villeneuve",
}


def classify_circuit_type(circuit):
    if pd.isna(circuit):
        return pd.NA

    name = str(circuit).lower()
    if any(keyword in name for keyword in STREET_CIRCUIT_KEYWORDS):
        return "street"
    if any(keyword in name for keyword in HYBRID_CIRCUIT_KEYWORDS):
        return "hybrid"
    return "permanent"


def pit_stop_counts(records):
    df = pd.DataFrame(records)
    if df.empty or "driver_number" not in df.columns:
        return pd.DataFrame(columns=["driver_number", "pit_stop_count"])

    df = df[["driver_number"]].copy()
    counts = df.groupby("driver_number").size().reset_index(name="pit_stop_count")
    return counts


def merge_session_results(
    qual_df,
    race_df,
    drivers_df,
    year,
    circuit,
    meeting_key,
    pit_df=None,
):
    merged = pd.merge(qual_df, race_df, on="driver_number", how="inner")

    if not drivers_df.empty and "driver_number" in drivers_df.columns:
        keep = [
            column
            for column in ["driver_number", "full_name", "name_acronym"]
            if column in drivers_df.columns
        ]
        merged = pd.merge(merged, drivers_df[keep], on="driver_number", how="left")

    if pit_df is not None and not pit_df.empty:
        merged = pd.merge(merged, pit_df, on="driver_number", how="left")
    if "pit_stop_count" not in merged.columns:
        merged["pit_stop_count"] = 0
    merged["pit_stop_count"] = merged["pit_stop_count"].fillna(0)

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

    for column in [
        "qualifying_position",
        "race_position",
        "pit_stop_count",
        "rainfall",
        "track_temperature",
        "year",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "driver_number" in df.columns:
        numeric_driver_number = pd.to_numeric(df["driver_number"], errors="coerce")
        if numeric_driver_number.notna().all():
            df["driver_number"] = numeric_driver_number.astype(int)

    df.dropna(subset=["qualifying_position", "race_position"], inplace=True)
    df = df[RESULT_COLUMNS].copy()
    df["qualifying_position"] = df["qualifying_position"].astype(int)
    df["race_position"] = df["race_position"].astype(int)
    if df["pit_stop_count"].notna().any():
        df["pit_stop_count"] = df["pit_stop_count"].fillna(0).astype(int)
    if df["year"].notna().all():
        df["year"] = df["year"].astype(int)

    return df.reset_index(drop=True)
