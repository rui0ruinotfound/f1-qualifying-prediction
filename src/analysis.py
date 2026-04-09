"""
analysis.py
Exploratory data analysis and linear regression modeling.
Examines the relationship between qualifying position and race finishing position.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_correlation(df: pd.DataFrame) -> float:
    """
    Compute Pearson correlation between qualifying and race position.

    Args:
        df: DataFrame with 'qualifying_position' and 'race_position' columns

    Returns:
        Pearson correlation coefficient
    """
    corr = df["qualifying_position"].corr(df["race_position"])
    print(f"Pearson Correlation (qualifying vs race position): {corr:.4f}")
    return corr


def pole_to_win_rate(df: pd.DataFrame) -> float:
    """
    Calculate the percentage of pole positions that converted to race wins.

    Args:
        df: Merged DataFrame

    Returns:
        Pole-to-win conversion rate (0.0 – 1.0)
    """
    pole_starts = df[df["qualifying_position"] == 1]
    if pole_starts.empty:
        return 0.0
    wins_from_pole = pole_starts[pole_starts["race_position"] == 1]
    rate = len(wins_from_pole) / len(pole_starts)
    print(f"Pole-to-win rate: {rate:.2%} ({len(wins_from_pole)}/{len(pole_starts)} races)")
    return rate


def average_finish_by_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average finishing position for each grid position.

    Args:
        df: Merged DataFrame

    Returns:
        DataFrame indexed by qualifying_position with mean race_position
    """
    avg = (
        df.groupby("qualifying_position")["race_position"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "avg_race_position", "count": "num_races"})
        .reset_index()
    )
    return avg


def run_linear_regression(df: pd.DataFrame) -> dict:
    """
    Fit a linear regression model: race_position ~ qualifying_position.
    Evaluates with 5-fold cross-validation.

    Args:
        df: Merged DataFrame

    Returns:
        Dictionary with model coefficients, MAE, RMSE, and CV scores
    """
    X = df[["qualifying_position"]].values
    y = df["race_position"].values

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
    cv_mae = -cv_scores.mean()

    results = {
        "intercept": model.intercept_,
        "coefficient": model.coef_[0],
        "mae": mae,
        "rmse": rmse,
        "cv_mae": cv_mae,
    }

    print(f"Linear Regression: race_pos = {model.coef_[0]:.4f} * qual_pos + {model.intercept_:.4f}")
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, CV MAE (5-fold): {cv_mae:.4f}")
    return results


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Print and return summary statistics for qualifying and race positions.

    Args:
        df: Merged DataFrame

    Returns:
        Summary stats DataFrame
    """
    stats = df[["qualifying_position", "race_position"]].describe()
    print("\nSummary Statistics:")
    print(stats)
    return stats