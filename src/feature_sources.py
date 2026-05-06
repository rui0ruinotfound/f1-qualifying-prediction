import pandas as pd

from config import FASTF1_CACHE_DIR, KAGGLE_DATA_DIR
from process import classify_circuit_type


OPENF1_CIRCUIT_TO_KAGGLE_ID = {
    "Sakhir": "bahrain",
    "Jeddah": "jeddah",
    "Melbourne": "albert_park",
    "Baku": "baku",
    "Miami": "miami",
    "Imola": "imola",
    "Monte Carlo": "monaco",
    "Catalunya": "catalunya",
    "Montreal": "villeneuve",
    "Spielberg": "red_bull_ring",
    "Silverstone": "silverstone",
    "Hungaroring": "hungaroring",
    "Spa-Francorchamps": "spa",
    "Zandvoort": "zandvoort",
    "Monza": "monza",
    "Singapore": "marina_bay",
    "Suzuka": "suzuka",
    "Lusail": "losail",
    "Austin": "americas",
    "Mexico City": "rodriguez",
    "Interlagos": "interlagos",
    "Las Vegas": "vegas",
    "Yas Marina Circuit": "yas_marina",
    "Shanghai": "shanghai",
}


FASTF1_EVENT_TO_KAGGLE_ID = {
    "Bahrain Grand Prix": "bahrain",
    "Saudi Arabian Grand Prix": "jeddah",
    "Australian Grand Prix": "albert_park",
    "Azerbaijan Grand Prix": "baku",
    "Miami Grand Prix": "miami",
    "Emilia Romagna Grand Prix": "imola",
    "Monaco Grand Prix": "monaco",
    "Spanish Grand Prix": "catalunya",
    "Canadian Grand Prix": "villeneuve",
    "Austrian Grand Prix": "red_bull_ring",
    "British Grand Prix": "silverstone",
    "Hungarian Grand Prix": "hungaroring",
    "Belgian Grand Prix": "spa",
    "Dutch Grand Prix": "zandvoort",
    "Italian Grand Prix": "monza",
    "Singapore Grand Prix": "marina_bay",
    "Japanese Grand Prix": "suzuka",
    "Qatar Grand Prix": "losail",
    "United States Grand Prix": "americas",
    "Mexico City Grand Prix": "rodriguez",
    "São Paulo Grand Prix": "interlagos",
    "Las Vegas Grand Prix": "vegas",
    "Abu Dhabi Grand Prix": "yas_marina",
    "Chinese Grand Prix": "shanghai",
}

# AI generated: initial draft of this script was created with the help of AI assistance
# and then reviewed and refactored by Rui Chen. The final version was manually edited.

def add_circuit_id(df):
    df = df.copy()
    df["circuit_id"] = df["circuit"].map(OPENF1_CIRCUIT_TO_KAGGLE_ID)
    return df


def load_kaggle_circuit_features(years, data_dir=KAGGLE_DATA_DIR):
    races_path = data_dir / "races.csv"
    if not races_path.exists():
        return pd.DataFrame(columns=["year", "circuit_id", "circuit_type"])

    races = pd.read_csv(races_path)
    year_column = "year" if "year" in races.columns else "season"
    circuit_id_column = "circuitId" if "circuitId" in races.columns else "circuit_id"

    races = races[pd.to_numeric(races[year_column], errors="coerce").isin(years)].copy()
    if races.empty or circuit_id_column not in races.columns:
        return pd.DataFrame(columns=["year", "circuit_id", "circuit_type"])

    features = races[[year_column, circuit_id_column]].copy()
    features.columns = ["year", "circuit_id"]
    features["year"] = features["year"].astype(int)
    features["circuit_type"] = features["circuit_id"].map(classify_circuit_type)
    return features.drop_duplicates(subset=["year", "circuit_id"])


def summarize_weather(weather_data):
    if weather_data is None or len(weather_data) == 0:
        return {"rainfall": pd.NA, "track_temperature": pd.NA}

    rainfall = pd.to_numeric(weather_data.get("Rainfall"), errors="coerce")
    track_temperature = pd.to_numeric(weather_data.get("TrackTemp"), errors="coerce")

    return {
        "rainfall": int((rainfall.fillna(0) > 0).any()) if rainfall.notna().any() else pd.NA,
        "track_temperature": (
            track_temperature.mean() if track_temperature.notna().any() else pd.NA
        ),
    }

# AI generated: initial draft of this script was created with the help of AI assistance
# and then reviewed and refactored by Rui Chen. The final version was manually edited.

def load_fastf1_weather_features(years):
    try:
        import fastf1
    except ModuleNotFoundError:
        print("  Skipped FastF1 weather features: FastF1 is not installed")
        return pd.DataFrame(columns=["year", "circuit_id", "rainfall", "track_temperature"])

    FASTF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(FASTF1_CACHE_DIR))

    rows = []
    for year in years:
        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as exc:
            print(f"  Skipped FastF1 weather schedule for {year}: {exc}")
            continue

        for event in schedule.itertuples(index=False):
            round_number = getattr(event, "RoundNumber", None)
            if round_number is None or pd.isna(round_number) or round_number <= 0:
                continue

            event_name = getattr(event, "EventName", None)
            circuit_id = FASTF1_EVENT_TO_KAGGLE_ID.get(event_name)
            if circuit_id is None:
                continue

            try:
                race = fastf1.get_session(year, int(round_number), "R")
                race.load(laps=False, telemetry=False, weather=True, messages=False)
            except Exception as exc:
                print(f"  Skipped FastF1 weather for {event_name} ({year}): {exc}")
                continue

            try:
                weather = summarize_weather(race.weather_data)
            except Exception as exc:
                print(f"  Skipped FastF1 weather for {event_name} ({year}): {exc}")
                continue

            rows.append(
                {
                    "year": int(year),
                    "circuit_id": circuit_id,
                    "rainfall": weather["rainfall"],
                    "track_temperature": weather["track_temperature"],
                }
            )
            print(f"  Loaded FastF1 weather: {event_name} ({year})")

    if not rows:
        return pd.DataFrame(columns=["year", "circuit_id", "rainfall", "track_temperature"])

    return pd.DataFrame(rows)


def enrich_openf1_with_external_features(df, years):
    enriched = add_circuit_id(df)

    circuit_features = load_kaggle_circuit_features(years)
    if not circuit_features.empty:
        enriched = pd.merge(
            enriched.drop(columns=["circuit_type"], errors="ignore"),
            circuit_features,
            on=["year", "circuit_id"],
            how="left",
        )

    weather_features = load_fastf1_weather_features(years)
    if not weather_features.empty:
        enriched = pd.merge(
            enriched.drop(columns=["rainfall", "track_temperature"], errors="ignore"),
            weather_features,
            on=["year", "circuit_id"],
            how="left",
        )

    return enriched.drop(columns=["circuit_id"], errors="ignore")
