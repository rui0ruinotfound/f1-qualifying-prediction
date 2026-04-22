# F1 Qualifying Prediction

**DSCI 510 - Spring 2026 | Rui Chen**

This project analyzes whether qualifying position can predict race results in recent Formula 1 seasons under the current ground-effect regulation era. The original project scope covered the 2022-2025 seasons, and the current OpenF1-based implementation analyzes the 2023-2025 seasons because OpenF1 historical coverage starts in 2023.

## Data sources

| # | Name | Type | URL |
|---|------|------|-----|
| 1 | OpenF1 API | REST API (JSON) | https://api.openf1.org/v1/positions |

The current submitted implementation uses OpenF1 session, session result, and driver data. Because OpenF1 historical coverage starts in 2023, the current implementation runs on the 2023-2025 seasons.

## Results

The current run of the project loaded 1,208 race-driver records from the 2023-2025 seasons. The Pearson correlation between qualifying position and race finishing position was 0.7497, which suggests a strong positive relationship between grid slot and race outcome.

Pole position converted to a race win in 42 of 65 races, for a pole-to-win rate of 64.62%. The linear regression model estimated `race_position = 0.6877 * qualifying_position + 2.4485`, with R-squared = 0.5621, MAE = 2.6405, RMSE = 3.4669, cross-validated MAE = 2.6484, and cross-validated RMSE = 3.4778. These results support the idea that qualifying position is an important predictor of finishing position, although race-day factors still create meaningful variation.

Chart outputs are saved in the `results/` folder as `.png` files together with `summary_statistics.csv` and `avg_finish_by_grid.csv`.

## Installation

- Install packages with `pip install -r requirements.txt`
- No API key is required for OpenF1
- The current submitted code uses `pandas`, `numpy`, `scikit-learn`, `python-dotenv`, and `matplotlib`

## Running analysis

From the `src/` directory run:

`python main.py`

Results will appear in the `results/` folder. Downloaded or merged data will be stored in the `data/` folder.

## AI Usage Disclosure

Generative AI tools were used to assist with portions of this project. Specifically, OpenAI Codex/ChatGPT was used as a coding assistant for selected implementation, debugging, documentation, and configuration support.

All code sections, comments, documentation, or configuration content that were generated with AI should be labeled in the source files with the comment format `# AI generated:` as required by the course policy.

## Project structure



f1-qualifying-prediction/
├── src/
│ ├── init.py
│ ├── openf1_api.py # OpenF1 API data fetching
│ ├── load_results.py # Data loading and merging
│ ├── analysis.py # EDA and regression modeling
│ └── main.py # Entry point
├── doc/ # Progress report (PDF)
├── data/ # Local data (gitignored)
├── results/ # Output files (gitignored)
├── tests.py # Test suite
├── requirements.txt
└── .gitignore
```

## Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd f1-qualifying-prediction

# Install dependencies
pip install -r requirements.txt
Running the Project
# Run full analysis pipeline
python -m src.main

# Run tests
python tests.py
Analysis Methods
Exploratory Data Analysis: Correlation between grid position and finishing position, pole-to-win conversion rate, and average finishing position per grid slot
Linear Regression: Modeling finishing position as a function of qualifying position, evaluated with MAE and RMSE
Notes
Do not commit .env, data/, or results/ directories — they are gitignored
API data is fetched live; no local data files are committed
