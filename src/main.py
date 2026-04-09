"""
main.py
Entry point for the F1 Qualifying Prediction project.
Orchestrates data loading, EDA, and regression modeling.
"""

import os
from src.load_results import load_multiple_seasons
from src.analysis import (
    compute_correlation,
    pole_to_win_rate,
    average_finish_by_grid,
    run_linear_regression,
    summary_statistics,
)

# Seasons under the same ground-effect aero regulations
SEASONS = [2022, 2023, 2024]

# Directory to save any output data/results
RESULTS_DIR = "results"
DATA_DIR = "data"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("F1 Qualifying Position vs Race Result Analysis")
    print(f"Seasons: {SEASONS}")
    print("=" * 60)

    # Load data
    df = load_multiple_seasons(SEASONS)

    if df.empty:
        print("No data loaded. Check API connectivity.")
        return

    print(f"\nTotal race-driver records loaded: {len(df)}")
    print(f"Seasons covered: {sorted(df['year'].unique())}")

    # Save raw merged data locally (not committed to git)
    df.to_csv(os.path.join(DATA_DIR, "merged_results.csv"), index=False)
    print(f"\nRaw data saved to {DATA_DIR}/merged_results.csv")

    # Exploratory analysis
    print("\n--- Summary Statistics ---")
    summary_statistics(df)

    print("\n--- Correlation ---")
    compute_correlation(df)

    print("\n--- Pole to Win Rate ---")
    pole_to_win_rate(df)

    print("\n--- Average Finish by Grid Position ---")
    avg_df = average_finish_by_grid(df)
    print(avg_df.head(10).to_string(index=False))
    avg_df.to_csv(os.path.join(RESULTS_DIR, "avg_finish_by_grid.csv"), index=False)

    # Regression model
    print("\n--- Linear Regression Model ---")
    run_linear_regression(df)

    print("\nDone.")


if __name__ == "__main__":
    main()