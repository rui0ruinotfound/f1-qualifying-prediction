import argparse

from analyze import (
    average_finish_by_grid,
    compute_correlation,
    plot_statistics,
    plot_feature_analysis,
    plot_individual_feature_charts,
    plot_strategy_scenarios,
    pole_to_win_rate,
    run_linear_regression,
    run_feature_model_comparison,
    run_strategy_scenarios,
    save_feature_analysis_outputs,
    save_source_comparison,
    save_summary_table,
    save_strategy_scenarios,
    summary_statistics,
)
from config import (
    AVG_FINISH_FILE,
    DATA_DIR,
    MERGED_RESULTS_FILE,
    RESULTS_DIR,
    SEASONS,
    SOURCE_SUMMARY_FILE,
)
from diagnostics import regression_diagnostics
from fastf1_loader import FastF1DataError, load_fastf1_multiple_seasons
from feature_sources import enrich_openf1_with_external_features
from kaggle_loader import KaggleDataError, load_kaggle_dataset
from load import load_multiple_seasons
from openf1_api import OpenF1APIError



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["openf1", "fastf1", "kaggle", "combined", "all"],
        default="combined",
    )
    return parser.parse_args()

# AI generated: initial draft of this script was created with the help of AI assistance
# and then reviewed and refactored by Rui Chen. The final version was manually edited.

def analyze_source(source_name, df, data_path, results_path):
    print(f"\n=== {source_name.upper()} ANALYSIS ===")
    print(f"Total race-driver records loaded: {len(df)}")
    print(f"Seasons covered: {sorted(df['year'].unique())}")

    df.to_csv(data_path, index=False)
    print(f"Saved merged data to {data_path}")

    print("\n--- Summary Statistics ---")
    summary_statistics(df)

    print("\n--- Correlation ---")
    corr = compute_correlation(df)

    print("\n--- Pole to Win Rate ---")
    pole_rate = pole_to_win_rate(df)

    print("\n--- Average Finish by Grid Position ---")
    avg_df = average_finish_by_grid(df)
    print(avg_df.head(10).to_string(index=False))
    avg_df.to_csv(results_path / AVG_FINISH_FILE, index=False)

    print("\n--- Linear Regression Model ---")
    regression_results = run_linear_regression(df)

    print("\n--- Linear Regression Diagnostics ---")
    diagnostic_results = regression_diagnostics(df, results_path)
    print(diagnostic_results.to_string(index=False))

    print("\n--- Summary Table ---")
    save_summary_table(df, avg_df, corr, regression_results, results_path)

    print("\n--- Visualizations ---")
    plot_statistics(df, avg_df, corr, regression_results, results_path)

    print("\n--- Individual and Combined Feature Models ---")
    feature_comparison, feature_results = run_feature_model_comparison(df)
    if not feature_comparison.empty:
        print(feature_comparison.to_string(index=False))
        save_feature_analysis_outputs(feature_comparison, feature_results, results_path)
        plot_individual_feature_charts(df, results_path)
        plot_feature_analysis(df, feature_comparison, feature_results, results_path)

        print("\n--- Strategy Scenario Simulation ---")
        scenario_df = run_strategy_scenarios(df)
        if not scenario_df.empty:
            print(scenario_df.to_string(index=False))
            save_strategy_scenarios(scenario_df, results_path)
            plot_strategy_scenarios(scenario_df, results_path)

    return {
        "source": source_name,
        "records": len(df),
        "races": df["meeting_key"].nunique(),
        "correlation": round(corr, 4),
        "pole_to_win_rate": round(pole_rate, 4),
        "r_squared": round(regression_results["r_squared"], 4),
        "mae": round(regression_results["mae"], 4),
        "rmse": round(regression_results["rmse"], 4),
    }


def load_source(source_name):
    if source_name == "openf1":
        return load_multiple_seasons(SEASONS)
    if source_name == "combined":
        df = load_multiple_seasons(SEASONS)
        if df.empty:
            return df
        print("\nAdding FastF1 weather and Kaggle/Ergast circuit features...")
        return enrich_openf1_with_external_features(df, SEASONS)
    if source_name == "fastf1":
        return load_fastf1_multiple_seasons(SEASONS)
    if source_name == "kaggle":
        return load_kaggle_dataset(SEASONS)
    raise ValueError(source_name)

# AI generated: initial draft of this script was created with the help of AI assistance
# and then reviewed and refactored by Rui Chen. The final version was manually edited.

def main():
    args = parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("F1 Race Result Factor Analysis")
    print(f"Seasons: {SEASONS}")
    print(f"Selected source: {args.source}")
    print("=" * 60)

    source_names = (
        ["openf1", "fastf1", "kaggle", "combined"]
        if args.source == "all"
        else [args.source]
    )
    comparison_rows = []

    for source_name in source_names:
        print(f"\nLoading source: {source_name}")

        try:
            df = load_source(source_name)
        except (OpenF1APIError, FastF1DataError, KaggleDataError) as exc:
            print(f"Could not load {source_name}.")
            print(f"Reason: {exc}")
            if args.source != "all":
                return
            continue

        if df.empty:
            print(f"No data loaded for {source_name}.")
            if args.source != "all":
                return
            continue

        if args.source == "all":
            data_path = DATA_DIR / f"{source_name}_{MERGED_RESULTS_FILE}"
            results_path = RESULTS_DIR / source_name
        else:
            data_path = DATA_DIR / MERGED_RESULTS_FILE
            results_path = RESULTS_DIR

        results_path.mkdir(parents=True, exist_ok=True)
        comparison_rows.append(analyze_source(source_name, df, data_path, results_path))

    if len(comparison_rows) > 1:
        save_source_comparison(comparison_rows, RESULTS_DIR, SOURCE_SUMMARY_FILE)
        print(f"\nSaved source comparison to {RESULTS_DIR / SOURCE_SUMMARY_FILE}")

    print("\nDone.")


if __name__ == "__main__":
    main()
