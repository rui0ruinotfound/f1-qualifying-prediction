import os

from config import (
    AVG_FINISH_FILE,
    DATA_DIR,
    MERGED_RESULTS_FILE,
    RESULTS_DIR,
    SEASONS,
)
from analyze import (
    average_finish_by_grid,
    compute_correlation,
    plot_statistics,
    pole_to_win_rate,
    run_linear_regression,
    save_summary_table,
    summary_statistics,
)
from load import load_multiple_seasons
from openf1_api import OpenF1APIError

# AI generated: initial draft of this script was created with the help ofAI assistance
# and then reviewed and refactored by Rui Chen. The final version was manually edited.
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("F1 Qualifying Position vs Race Result Analysis")
    print(f"Seasons: {SEASONS}")
    print("=" * 60)

    try:
        df = load_multiple_seasons(SEASONS)
    except OpenF1APIError as exc:
        print("\nCould not load data from the OpenF1 API.")
        print(f"Reason: {exc}")
        print("Please check your internet connection and try again.")
        return

    if df.empty:
        print("No data loaded. Check API connectivity.")
        return

    print(f"\nTotal race-driver records loaded: {len(df)}")
    print(f"Seasons covered: {sorted(df['year'].unique())}")

    merged_path = os.path.join(DATA_DIR, MERGED_RESULTS_FILE)
    df.to_csv(merged_path, index=False)
    print(f"\nRaw data saved to {merged_path}")

    # Exploratory analysis
    print("\n--- Summary Statistics ---")
    summary_statistics(df)

    print("\n--- Correlation ---")
    corr = compute_correlation(df)

    print("\n--- Pole to Win Rate ---")
    pole_to_win_rate(df)

    print("\n--- Average Finish by Grid Position ---")
    avg_df = average_finish_by_grid(df)
    print(avg_df.head(10).to_string(index=False))
    avg_path = os.path.join(RESULTS_DIR, AVG_FINISH_FILE)
    avg_df.to_csv(avg_path, index=False)

    print("\n--- Linear Regression Model ---")
    regression_results = run_linear_regression(df)

    print("\n--- Summary Table ---")
    save_summary_table(df, avg_df, corr, regression_results, RESULTS_DIR)

    print("\n--- Visualizations ---")
    plot_statistics(df, avg_df, corr, regression_results, RESULTS_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
