# F1 Qualifying Prediction

**DSCI 510 – Spring 2026 | Rui Chen**

## Project Summary

This project investigates whether qualifying position can predict race finishing position in Formula 1 during the 2022–2025 seasons. These seasons share the same ground-effect aerodynamic regulations, providing a consistent technical environment for analysis.

## Research Question

> Can qualifying (grid) position reliably predict race finishing position in the 2022–2025 F1 seasons?

## Data Sources

| # | Name | Type | URL |
|---|------|------|-----|
| 1 | OpenF1 API | REST API (JSON) | https://api.openf1.org/v1 |
| 2 | FastF1 Python Library | Python API | https://docs.fastf1.dev |
| 3 | Ergast / Kaggle Dataset | CSV file | https://www.kaggle.com/datasets/rockyt07/formula1-championships-1950-2025 |

At this stage, only the OpenF1 API is fully implemented. Other data sources will be integrated in future iterations.

## Project Structure



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