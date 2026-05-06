# F1 Race Result Factor Analysis

**DSCI 510 - Spring 2026 | Rui Chen**

AI generated: initial draft of this README.md was created with the help of AI assistance
and then reviewed and refactored by Rui Chen. The final version was manually edited.

## Introduction

This project studies whether Formula 1 qualifying position and selected race-context features can predict race finishing position. The original proposal scope was 2022-2025, but the implemented analysis focuses on 2023-2025 because OpenF1 historical coverage begins in 2023. The final modeling dataset combines features from three sources: OpenF1, FastF1, and Kaggle/Ergast.

Research question: How well does qualifying position predict Formula 1 race finishing position, and do race-context factors such as pit stops, weather, and circuit type add explanatory value?

## Data sources

| # | Name | Role in project | Type | URL |
|---|------|-----------------|------|-----|
| 1 | OpenF1 API | Qualifying position, finishing position, pit stop count, driver info | REST API (JSON) | https://api.openf1.org/v1/ |
| 2 | FastF1 | Race weather features: rainfall and track temperature | Python library / historical F1 session data | https://docs.fastf1.dev/ |
| 3 | Formula 1 Championships (1950-2025) | Race and circuit metadata used to engineer circuit type | Kaggle CSV dataset | https://www.kaggle.com/datasets/rockyt07/formula-1-championships-1950-2025 |

The three source paths are implemented in the project and are standardized into the same merged analysis schema:

- `OpenF1`: uses `sessions`, `session_result`, `drivers`, and `pit`
- `FastF1`: uses race-session weather data
- `Kaggle/Ergast`: uses local race and circuit metadata

The final merged table uses the following columns:

- `driver_number`
- `qualifying_position`
- `race_position`
- `pit_stop_count`
- `rainfall`
- `track_temperature`
- `circuit_type`
- `full_name`
- `name_acronym`
- `year`
- `circuit`
- `meeting_key`

## Source comparison

All three data sources support the same main conclusion: qualifying position has a positive relationship with race finishing position. The expanded combined dataset then adds race-context factors to test whether they improve or explain variation beyond the qualifying-only baseline.

- `OpenF1` is the most complete and most important source in this project. It covers 69 races and 1,208 driver-race records from 2023-2025, with Pearson correlation `0.7497`.
- `FastF1` also shows a positive relationship and serves as a useful validation source. The current local run produced 958 driver-race records across 48 races, with Pearson correlation `0.6649`.
- `Kaggle` also shows a positive relationship, with Pearson correlation `0.6751`, but its recent qualifying and results coverage is incomplete, so the current usable sample is only 299 driver-race records across 15 races.

Because of these differences in coverage, OpenF1 is used as the base driver-race table. FastF1 and Kaggle/Ergast are then merged into that table for weather and circuit-type features used in the final combined model.

## Analysis

- Summary statistics
- Pearson correlation between qualifying position and race finishing position
- Pole-to-win conversion rate
- Average finishing position by grid slot
- Simple linear regression using qualifying position to predict race position
- Individual feature regressions for pit stops, weather, and circuit type
- Combined regression model using qualifying position plus selected additional features
- Strategy scenario simulation using the combined model with hypothetical pit/weather/circuit conditions. Tire-compound sequences are included only as scenario labels, not as model predictors, to avoid using post-race information.
- Evaluation with `R-squared`, `MAE`, `RMSE`, and cross-validated error metrics
- Linear regression diagnostics for model applicability: residual plots, Q-Q plot, Durbin-Watson statistic, Breusch-Pagan test, and residual normality test

## Summary of the results

The main combined analysis loaded 1,208 race-driver records from the 2023-2025 seasons. The Pearson correlation between qualifying position and race finishing position was `0.7497`, which suggests a strong positive relationship between starting position and race outcome.

Pole position converted to a race win in `42 of 65` races, for a pole-to-win rate of `64.62%`. The linear regression model estimated:

`race_position = 0.6877 * qualifying_position + 2.4485`

Main evaluation metrics:

- `R-squared = 0.5621`
- `MAE = 2.6405`
- `RMSE = 3.4669`
- `Cross-validated MAE = 2.6484`
- `Cross-validated RMSE = 3.4778`

These results suggest that qualifying position is the strongest single predictor in this project. Additional factors such as pit stops, rainfall, track temperature, and circuit type are included to study race context, but they add only a small improvement over the qualifying-only baseline in the current dataset.


## Project structure

```
f1-qualifying-prediction/
├── data/
│   └── .gitkeep
├── doc/
│   ├── Rui_Chen_progress_report.pdf
│   └── Rui_Chen_presentation.pdf
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── openf1_api.py
│   ├── load.py
│   ├── fastf1_loader.py
│   ├── feature_sources.py
│   ├── kaggle_loader.py
│   ├── process.py
│   ├── analyze.py
│   ├── main.py
│   ├── tests.py
│   └── results.ipynb
├── requirements.txt
├── .env.example
└── .gitignore
```

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

Package requirements include `pandas`, `numpy`, `scikit-learn`, `scipy`, `python-dotenv`, `matplotlib`, and `fastf1`.

Additional setup notes:

- OpenF1 does not require an API key
- OpenF1 and FastF1 may require internet access at runtime
- Kaggle data should be downloaded locally into `data/kaggle/`

## Data fetching and required credentials

The pipeline fetches or loads data in the following ways:

- **OpenF1:** fetched automatically from the public OpenF1 API when running `python src/main.py --source openf1` or `python src/main.py --source combined`. No API key is required.
- **FastF1:** fetched automatically through the FastF1 Python package when running `python src/main.py --source fastf1` or `python src/main.py --source combined`. No API key is required, but internet access is needed on the first run so FastF1 can download and cache session data.
- **Kaggle/Ergast CSV files:** download the Formula 1 Championships dataset from Kaggle and place the extracted CSV files in `data/kaggle/`, or set `KAGGLE_DATA_DIR` in `.env` to another local folder. The required files are `races.csv`, `qualifying.csv`, `results.csv`, and `drivers.csv`; `circuits.csv` is optional but recommended for circuit metadata. A Kaggle account or Kaggle API token may be needed to download the dataset from Kaggle, but the code does not require Kaggle credentials after the CSV files are downloaded locally.

To download the Kaggle data manually, open the dataset page at https://www.kaggle.com/datasets/rockyt07/formula-1-championships-1950-2025, click **Download**, unzip the file, and copy the CSV files into:

```bash
data/kaggle/
```

The folder should contain at least:

```text
data/kaggle/races.csv
data/kaggle/qualifying.csv
data/kaggle/results.csv
data/kaggle/drivers.csv
data/kaggle/circuits.csv
```

If using the Kaggle CLI instead, configure Kaggle credentials outside this repository and run:

```bash
mkdir -p data/kaggle
kaggle datasets download -d rockyt07/formula-1-championships-1950-2025 -p data/kaggle --unzip
```

Do not put API keys or Kaggle tokens in this repository. If using a local environment file, copy `.env.example` to `.env` and only store local paths there.

## How to run

From the project root, install the Python dependencies:

```bash
pip install -r requirements.txt
```

If using a custom Kaggle data folder, copy `.env.example` to `.env` and set `KAGGLE_DATA_DIR` to that local folder. If the Kaggle CSV files are placed in the default `data/kaggle/` folder, no `.env` file is required.

Download the Kaggle dataset using either the manual download or Kaggle CLI instructions above, then make sure the required CSV files are available in `data/kaggle/`.

The recommended way to reproduce the main analysis is to run:

```bash
python src/main.py --source combined
```

This command uses OpenF1 as the base driver-race table, adds FastF1 weather features and Kaggle/Ergast circuit metadata when available, and saves generated outputs locally. It creates `data/` and `results/` folders on the user's machine. Actual data files and cache files are intentionally not included in the GitHub repository because the final submission guidelines say not to upload data files.

After running the command above, open `src/results.ipynb` and run the notebook cells to review the generated tables and charts. The notebook reads the generated `data/merged_results.csv` file by default so that it does not re-fetch the full API pipeline every time.

If you want the notebook itself to run the full pipeline from scratch, set this variable in the notebook's data-loading cell:

```python
RUN_PIPELINE_IN_NOTEBOOK = True
```

Other source-specific runs are also available:

```bash
python src/main.py --source openf1
python src/main.py --source fastf1
python src/main.py --source kaggle
python src/main.py --source all
```

`--source openf1` uses only OpenF1 data. `--source combined` is the final three-source modeling run. `--source all` runs the three source-specific analyses plus the combined analysis and writes a source comparison table. `--source kaggle` requires the Kaggle CSV files to be downloaded locally into `data/kaggle/`.

## Outputs

Typical generated outputs include:

- `data/merged_results.csv`
- `results/summary_statistics.csv`
- `results/avg_finish_by_grid.csv`
- `results/source_comparison_summary.csv`
- `results/feature_model_comparison.csv`
- `results/feature_importance.csv`
- `results/regression_diagnostics.csv`
- `results/strategy_scenarios.csv`
- `results/F1_Qualifying_scatterplot.png`
- `results/F1_Qualifying_linechart.png`
- `results/F1_Pole_to_Win_barchart.png`
- `results/F1_Predicted_vs_Actual_scatterplot.png`
- `results/F1_Finishing_Position_boxplot.png`
- `results/F1_Residuals_vs_Fitted.png`
- `results/F1_QQ_Plot.png`
- `results/F1_Residuals_by_Grid.png`
- `results/F1_Feature_Correlation_Heatmap.png`
- `results/F1_Feature_Importance.png`
- `results/F1_Combined_Actual_vs_Predicted.png`
- `results/F1_Strategy_Scenarios.png`

## Testing

The test file is located at `src/tests.py`.

Run tests with:

```bash
python src/tests.py
```

The test suite includes:

- OpenF1 session and endpoint checks
- data-loading helper checks
- correlation test
- regression test
- pole-to-win test
- source-normalization checks for FastF1 and Kaggle loaders
- regression diagnostics checks

## Notes

- `data/` is included only as an empty placeholder folder. Actual data files and caches should not be committed.
- `src/results.ipynb` is intentionally included as part of the project deliverable
- Kaggle recent-season qualifying/results coverage is incomplete, so it is supplementary for source comparison. Its race/circuit metadata is still used in the final modeling table to engineer `circuit_type`.

## AI Usage Disclosure

Generative AI tools were used to assist with parts of this project. OpenAI Codex/ChatGPT was used as a coding assistant for selected implementation, debugging, documentation, and configuration support.

All code sections, comments, documentation, or configuration content that were generated with AI are labeled with the comment format `# AI generated:` as required by the course policy.
