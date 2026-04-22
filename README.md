# F1 Qualifying Prediction

**DSCI 510 - Spring 2026 | Rui Chen**

This project studies whether Formula 1 qualifying position can predict race finishing position. The original proposal scope was 2022-2025, but the implemented analysis focuses on 2023-2025 because OpenF1 historical coverage begins in 2023. The project uses one main data source and two supplementary data sources so that the relationship can be checked across multiple datasets.

## Research question

Can qualifying position predict Formula 1 race finishing position?

## Data sources

| # | Name | Role in project | Type | URL |
|---|------|-----------------|------|-----|
| 1 | OpenF1 API | Main analysis source | REST API (JSON) | https://api.openf1.org/v1/ |
| 2 | FastF1 | Supplementary validation source | Python library / historical F1 session data | https://docs.fastf1.dev/ |
| 3 | Formula 1 Championships (1950-2025) | Supplementary validation source | Kaggle CSV dataset | https://www.kaggle.com/datasets/rockyt07/formula-1-championships-1950-2025 |

The three source paths are implemented in the project and are standardized into the same merged analysis schema:

- `OpenF1`: uses `sessions`, `session_result`, and `drivers`
- `FastF1`: uses qualifying and race session classifications
- `Kaggle`: uses local race, qualifying, results, and driver CSV tables

The final merged table uses the following columns:

- `driver_number`
- `qualifying_position`
- `race_position`
- `full_name`
- `name_acronym`
- `year`
- `circuit`
- `meeting_key`

## Source comparison

All three data sources support the same main conclusion: qualifying position has a positive relationship with race finishing position.

- `OpenF1` is the most complete and most important source in this project. It covers 69 races and 1,208 driver-race records from 2023-2025, with Pearson correlation `0.7497`.
- `FastF1` also shows a positive relationship and serves as a useful validation source. The current local run produced 958 driver-race records across 48 races, with Pearson correlation `0.6649`.
- `Kaggle` also shows a positive relationship, with Pearson correlation `0.6751`, but its recent qualifying and results coverage is incomplete, so the current usable sample is only 299 driver-race records across 15 races.

Because of these differences in coverage, OpenF1 is used as the main analysis source, while FastF1 and Kaggle are used as supporting datasets.

## Main results

The main OpenF1 analysis loaded 1,208 race-driver records from the 2023-2025 seasons. The Pearson correlation between qualifying position and race finishing position was `0.7497`, which suggests a strong positive relationship between starting position and race outcome.

Pole position converted to a race win in `42 of 65` races, for a pole-to-win rate of `64.62%`. The linear regression model estimated:

`race_position = 0.6877 * qualifying_position + 2.4485`

Main evaluation metrics:

- `R-squared = 0.5621`
- `MAE = 2.6405`
- `RMSE = 3.4669`
- `Cross-validated MAE = 2.6484`
- `Cross-validated RMSE = 3.4778`

These results suggest that qualifying position is an important predictor of race finishing position, although race-day strategy, incidents, weather, and reliability still create meaningful variation.

## Analysis methods

- Summary statistics
- Pearson correlation between qualifying position and race finishing position
- Pole-to-win conversion rate
- Average finishing position by grid slot
- Simple linear regression using qualifying position to predict race position
- Evaluation with `R-squared`, `MAE`, `RMSE`, and cross-validated error metrics

## Project structure

```
f1-qualifying-prediction/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── openf1_api.py
│   ├── load.py
│   ├── fastf1_loader.py
│   ├── kaggle_loader.py
│   ├── process.py
│   ├── analyze.py
│   ├── main.py
│   ├── tests.py
│   └── results.ipynb
├── doc/
├── data/
├── results/
├── requirements.txt
└── .gitignore
```

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

Package requirements include `pandas`, `numpy`, `scikit-learn`, `python-dotenv`, `matplotlib`, and `fastf1`.

Additional setup notes:

- OpenF1 does not require an API key
- OpenF1 and FastF1 may require internet access at runtime
- Kaggle data should be downloaded locally into `data/kaggle/`

## Running analysis

From the project root, run one of the following:

```bash
python src/main.py --source openf1
python src/main.py --source fastf1
python src/main.py --source kaggle
```

Results are saved to the `results/` folder, and merged data tables are saved to the `data/` folder.

## Outputs

Typical generated outputs include:

- `data/merged_results.csv`
- `results/summary_statistics.csv`
- `results/avg_finish_by_grid.csv`
- `results/source_comparison_summary.csv`
- `results/F1_Qualifying_scatterplot.png`
- `results/F1_Qualifying_linechart.png`
- `results/F1_Pole_to_Win_barchart.png`
- `results/F1_Predicted_vs_Actual_scatterplot.png`
- `results/F1_Finishing_Position_boxplot.png`

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

## Notes

- `data/` and `results/` are gitignored and should not be committed
- `src/results.ipynb` is intentionally included as part of the project deliverable
- Kaggle recent-season coverage is incomplete, so it is used as a supplementary source rather than the main source

## AI Usage Disclosure

Generative AI tools were used to assist with parts of this project. OpenAI Codex/ChatGPT was used as a coding assistant for selected implementation, debugging, documentation, and configuration support.

All code sections, comments, documentation, or configuration content that were generated with AI are labeled with the comment format `# AI generated:` as required by the course policy.
