
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


def compute_correlation(df):
    """Return the Pearson correlation between qualifying and race position."""
    corr = df["qualifying_position"].corr(df["race_position"])
    print(f"Pearson Correlation (qualifying vs race position): {corr:.4f}")
    return corr


def pole_to_win_rate(df):
    """Return how often a pole position turned into a race win."""
    pole_starts = df[df["qualifying_position"] == 1]
    if pole_starts.empty:
        return 0.0
    wins_from_pole = pole_starts[pole_starts["race_position"] == 1]
    rate = len(wins_from_pole) / len(pole_starts)
    print(f"Pole-to-win rate: {rate:.2%} ({len(wins_from_pole)}/{len(pole_starts)} races)")
    return rate


def average_finish_by_grid(df):
    """Summarize the average finishing position for each grid slot."""
    avg = (
        df.groupby("qualifying_position")["race_position"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "avg_race_position", "count": "num_races"})
        .reset_index()
    )
    return avg


# AI generated: initial draft of this script was created with the help ofAI assistance
# and then reviewed and refactored by Rui Chen. The final version was manually edited.
def run_linear_regression(df):
    """Fit a simple linear model using qualifying position to predict finish."""
    X = df[["qualifying_position"]].values
    y = df["race_position"].values

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r_squared = r2_score(y, y_pred)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
    cv_mae = -cv_scores.mean()
    cv_rmse_scores = cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_squared_error"
    )
    cv_rmse = np.sqrt(-cv_rmse_scores.mean())

    results = {
        "intercept": model.intercept_,
        "coefficient": model.coef_[0],
        "r_squared": r_squared,
        "mae": mae,
        "rmse": rmse,
        "cv_mae": cv_mae,
        "cv_rmse": cv_rmse,
        "predicted_values": y_pred,
    }

    print(
        "Linear regression model: "
        f"race_pos = {model.coef_[0]:.4f} * qual_pos + {model.intercept_:.4f}"
    )
    print(
        f"R-squared: {r_squared:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, "
        f"CV MAE (5-fold): {cv_mae:.4f}, CV RMSE (5-fold): {cv_rmse:.4f}"
    )
    return results


def summary_statistics(df):
    """Print and return descriptive statistics for the position columns."""
    stats = df[["qualifying_position", "race_position"]].describe()
    print("\nSummary Statistics:")
    print(stats)
    return stats


# AI generated: initial draft of this script was created with the help ofAI assistance
# and then reviewed and refactored by Rui Chen. The final version was manually edited.
def save_summary_table(df, avg_df, corr, regression_results, result_dir):
    """Save a CSV summary table for the current analysis run."""
    os.makedirs(result_dir, exist_ok=True)

    summary_rows = [
        {"section": "overall", "metric": "number_of_races", "value": df["meeting_key"].nunique()},
        {"section": "overall", "metric": "number_of_driver_race_observations", "value": len(df)},
        {"section": "overall", "metric": "mean_qualifying_position", "value": round(df["qualifying_position"].mean(), 4)},
        {"section": "overall", "metric": "mean_finishing_position", "value": round(df["race_position"].mean(), 4)},
        {"section": "overall", "metric": "correlation_qualifying_vs_finishing", "value": round(corr, 4)},
        {"section": "overall", "metric": "regression_coefficient", "value": round(regression_results["coefficient"], 4)},
        {"section": "overall", "metric": "regression_intercept", "value": round(regression_results["intercept"], 4)},
        {"section": "overall", "metric": "r_squared", "value": round(regression_results["r_squared"], 4)},
        {"section": "overall", "metric": "mae", "value": round(regression_results["mae"], 4)},
        {"section": "overall", "metric": "rmse", "value": round(regression_results["rmse"], 4)},
        {"section": "overall", "metric": "cv_mae", "value": round(regression_results["cv_mae"], 4)},
        {"section": "overall", "metric": "cv_rmse", "value": round(regression_results["cv_rmse"], 4)},
    ]

    for row in avg_df.itertuples(index=False):
        summary_rows.append(
            {
                "section": "average_finish_by_grid",
                "metric": "avg_race_position",
                "qualifying_position": int(row.qualifying_position),
                "value": round(row.avg_race_position, 4),
                "num_races": int(row.num_races),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{result_dir}/summary_statistics.csv", index=False)
    return summary_df


def plot_statistics(df, avg_df, corr, regression_results, result_dir):
    """Create and save chart outputs for the F1 qualifying analysis."""
    os.makedirs(result_dir, exist_ok=True)
    plt = _load_matplotlib()

    _plot_correlation_scatter(df, corr, regression_results, result_dir, plt)
    _plot_avg_finish_line(avg_df, result_dir, plt)
    _plot_pole_to_win_bar(df, result_dir, plt)
    _plot_predicted_vs_actual(df, regression_results, result_dir, plt)
    _plot_finish_boxplot(df, result_dir, plt)


def _load_matplotlib():
    import matplotlib.pyplot as plt

    return plt


def _plot_correlation_scatter(df, corr, regression_results, result_dir, plt):
    plt.figure(figsize=(10, 6))
    plt.scatter(
        df["qualifying_position"],
        df["race_position"],
        alpha=0.35,
        color="darkred",
        label="Race-driver observations",
    )

    x_values = np.arange(1, 21)
    y_values = (
        regression_results["coefficient"] * x_values + regression_results["intercept"]
    )
    plt.plot(x_values, y_values, color="navy", linewidth=2, label="Regression line")
    plt.text(
        13.2,
        3.2,
        f"r = {corr:.4f}\nR^2 = {regression_results['r_squared']:.4f}",
        bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.85},
    )
    plt.title("Qualifying Position vs Finishing Position - F1 Qualifying")
    plt.xlabel("Qualifying position")
    plt.ylabel("Finishing position")
    plt.xticks(range(1, 21))
    plt.yticks(range(1, 21))
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{result_dir}/F1_Qualifying_scatterplot.png", bbox_inches="tight")
    plt.close()


def _plot_avg_finish_line(avg_df, result_dir, plt):
    plt.figure(figsize=(10, 6))
    plt.plot(
        avg_df["qualifying_position"],
        avg_df["avg_race_position"],
        marker="o",
        color="steelblue",
        linewidth=2,
    )
    plt.title("Average Finishing Position by Grid Position - F1 Qualifying")
    plt.xlabel("Grid position")
    plt.ylabel("Average finishing position")
    plt.xticks(avg_df["qualifying_position"])
    plt.grid(True)
    plt.savefig(f"{result_dir}/F1_Qualifying_linechart.png", bbox_inches="tight")
    plt.close()


def _plot_pole_to_win_bar(df, result_dir, plt):
    comparison = []
    labels = ["Grid 1", "Grid 2", "Grid 3"]

    for grid_position in [1, 2, 3]:
        starts = df[df["qualifying_position"] == grid_position]
        wins = starts[starts["race_position"] == 1]
        rate = 0 if starts.empty else len(wins) / len(starts)
        comparison.append(rate * 100)

    plt.figure(figsize=(8, 6))
    plt.bar(
        labels,
        comparison,
        color=["seagreen", "royalblue", "gray"],
        edgecolor="black",
    )
    plt.title("Win Rate by Front Grid Position - F1 Qualifying")
    plt.ylabel("Win rate (%)")
    plt.grid(axis="y")
    plt.savefig(f"{result_dir}/F1_Pole_to_Win_barchart.png", bbox_inches="tight")
    plt.close()


def _plot_predicted_vs_actual(df, regression_results, result_dir, plt):
    predicted = regression_results["predicted_values"]
    actual = df["race_position"].values

    plt.figure(figsize=(8, 8))
    plt.scatter(actual, predicted, alpha=0.4, color="purple")
    plt.plot([1, 20], [1, 20], color="black", linestyle="--", linewidth=2)
    plt.title("Predicted vs Actual Finishing Position - F1 Qualifying")
    plt.xlabel("Actual finishing position")
    plt.ylabel("Predicted finishing position")
    plt.xticks(range(1, 21))
    plt.yticks(range(1, 21))
    plt.grid(True)
    plt.savefig(f"{result_dir}/F1_Predicted_vs_Actual_scatterplot.png", bbox_inches="tight")
    plt.close()


def _plot_finish_boxplot(df, result_dir, plt):
    grouped_positions = []
    labels = []

    for grid_position in sorted(df["qualifying_position"].unique()):
        grouped_positions.append(
            df.loc[df["qualifying_position"] == grid_position, "race_position"].values
        )
        labels.append(str(int(grid_position)))

    plt.figure(figsize=(12, 6))
    plt.boxplot(grouped_positions, tick_labels=labels, patch_artist=True)
    plt.title("Finishing Position by Grid Position - F1 Qualifying")
    plt.xlabel("Grid position")
    plt.ylabel("Finishing position")
    plt.grid(axis="y")
    plt.savefig(f"{result_dir}/F1_Finishing_Position_boxplot.png", bbox_inches="tight")
    plt.close()
