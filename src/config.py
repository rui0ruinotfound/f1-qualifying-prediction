from pathlib import Path

from dotenv import load_dotenv

# AI generated: initial draft of this script was created with the help ofAI assistance
# and then reviewed and refactored by Rui Chen. The final version was manually edited.

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# Prefer the project-root .env, but keep src/.env as a fallback for compatibility.
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(SRC_DIR / ".env")

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
KAGGLE_DATA_DIR = DATA_DIR / "kaggle"
FASTF1_CACHE_DIR = DATA_DIR / "fastf1_cache"

OPENF1_BASE_URL = "https://api.openf1.org/v1"
SEASONS = [2023, 2024, 2025]

REQUEST_TIMEOUT = 30
REQUEST_PAUSE_SECONDS = 1.0
MAX_RETRIES = 4
RETRY_DELAY_SECONDS = 10

MERGED_RESULTS_FILE = "merged_results.csv"
AVG_FINISH_FILE = "avg_finish_by_grid.csv"
SOURCE_SUMMARY_FILE = "source_comparison_summary.csv"

QUALIFYING_SESSION_NAME = "Qualifying"
RACE_SESSION_NAME = "Race"
