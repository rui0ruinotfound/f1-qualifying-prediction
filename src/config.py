from pathlib import Path

from dotenv import load_dotenv



# AI generated: initial draft of this script was created with the help ofAI assistance
# and then reviewed and refactored by Rui Chen. The final version was manually edited.
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

# project configuration
DATA_DIR = "../data"
RESULTS_DIR = "../results"

# data source configuration
FASTF1_DOCS_URL = "https://docs.fastf1.dev"
OPENF1_BASE_URL = "https://api.openf1.org/v1"
OPENF1_POSITION_URL = "https://api.openf1.org/v1/positions"
ERGAST_KAGGLE_URL = (
    "https://www.kaggle.com/datasets/rockyt07/formula-1-championships-1950-2025"
)
REQUEST_TIMEOUT = 30
SEASONS = [2023, 2024, 2025]
REQUEST_PAUSE_SECONDS = 1.0
MAX_RETRIES = 4
RETRY_DELAY_SECONDS = 10

# output configuration
MERGED_RESULTS_FILE = "merged_results.csv"
AVG_FINISH_FILE = "avg_finish_by_grid.csv"

# session labels used in OpenF1
QUALIFYING_SESSION_NAME = "Qualifying"
RACE_SESSION_NAME = "Race"
