from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


def fit_simple_regression(df):
    model_df = df[["qualifying_position", "race_position"]].dropna().copy()
    x = model_df[["qualifying_position"]].to_numpy()
    y = model_df["race_position"].to_numpy()

    model = LinearRegression()
    model.fit(x, y)
    fitted = model.predict(x)
    residuals = y - fitted

    return model_df, fitted, residuals


def durbin_watson_statistic(residuals):
    residuals = np.asarray(residuals)
    denominator = np.sum(residuals**2)
    if denominator == 0:
        return np.nan
    return np.sum(np.diff(residuals) ** 2) / denominator

# AI generated: initial draft of this script was created with the help of AI assistance
# and then reviewed and refactored by Rui Chen. The final version was manually edited.

def breusch_pagan_test(fitted_values, residuals):
    fitted_values = np.asarray(fitted_values)
    residuals = np.asarray(residuals)
    n = len(residuals)

    if n < 3 or np.var(residuals) == 0:
        return np.nan, np.nan

    squared_residuals = residuals**2
    aux_x = np.column_stack([np.ones(n), fitted_values])
    coefficients = np.linalg.lstsq(aux_x, squared_residuals, rcond=None)[0]
    aux_predicted = aux_x @ coefficients
    ss_total = np.sum((squared_residuals - squared_residuals.mean()) ** 2)
    ss_residual = np.sum((squared_residuals - aux_predicted) ** 2)

    if ss_total == 0:
        return np.nan, np.nan

    r_squared = 1 - (ss_residual / ss_total)
    statistic = n * r_squared
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    return statistic, p_value


def normality_test(residuals):
    residuals = np.asarray(residuals)
    if len(residuals) < 8 or np.var(residuals) == 0:
        return "not_enough_variation", np.nan, np.nan

    if len(residuals) <= 5000:
        statistic, p_value = stats.shapiro(residuals)
        return "shapiro_wilk", statistic, p_value

    statistic, p_value = stats.jarque_bera(residuals)
    return "jarque_bera", statistic, p_value

# AI generated: initial draft of this script was created with the help of AI assistance
# and then reviewed and refactored by Rui Chen. The final version was manually edited.

def regression_diagnostics(df, result_dir):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    model_df, fitted, residuals = fit_simple_regression(df)
    bp_statistic, bp_p_value = breusch_pagan_test(fitted, residuals)
    normality_method, normality_statistic, normality_p_value = normality_test(residuals)
    dw_statistic = durbin_watson_statistic(residuals)

    diagnostics = pd.DataFrame(
        [
            {
                "check": "linearity",
                "method": "residuals_vs_fitted_plot",
                "statistic": np.nan,
                "p_value": np.nan,
                "interpretation": "Inspect whether residuals are centered around zero without a curved pattern.",
            },
            {
                "check": "independence",
                "method": "durbin_watson",
                "statistic": round(dw_statistic, 4),
                "p_value": np.nan,
                "interpretation": "Values near 2 suggest weak serial correlation; values far from 2 suggest autocorrelation.",
            },
            {
                "check": "homoscedasticity",
                "method": "breusch_pagan",
                "statistic": round(bp_statistic, 4) if pd.notna(bp_statistic) else np.nan,
                "p_value": round(bp_p_value, 4) if pd.notna(bp_p_value) else np.nan,
                "interpretation": "Small p-values suggest non-constant residual variance.",
            },
            {
                "check": "normality",
                "method": normality_method,
                "statistic": (
                    round(normality_statistic, 4)
                    if pd.notna(normality_statistic)
                    else np.nan
                ),
                "p_value": round(normality_p_value, 4) if pd.notna(normality_p_value) else np.nan,
                "interpretation": "Small p-values suggest residuals deviate from a normal distribution.",
            },
        ]
    )
    diagnostics.to_csv(result_dir / "regression_diagnostics.csv", index=False)
    plot_regression_diagnostics(model_df, fitted, residuals, result_dir)
    return diagnostics


def plot_regression_diagnostics(model_df, fitted, residuals, result_dir):
    result_dir = Path(result_dir)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.scatter(fitted, residuals, alpha=0.35, color="darkslateblue")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Residuals vs Fitted Values")
    plt.xlabel("Fitted finishing position")
    plt.ylabel("Residual")
    plt.grid(True, alpha=0.3)
    plt.savefig(result_dir / "F1_Residuals_vs_Fitted.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot of Regression Residuals")
    plt.savefig(result_dir / "F1_QQ_Plot.png", bbox_inches="tight")
    plt.close()

    plot_df = model_df.copy()
    plot_df["residual"] = residuals
    grouped = []
    labels = []
    for grid_position in sorted(plot_df["qualifying_position"].unique()):
        grouped.append(
            plot_df.loc[
                plot_df["qualifying_position"] == grid_position,
                "residual",
            ].values
        )
        labels.append(str(int(grid_position)))

    plt.figure(figsize=(12, 6))
    plt.boxplot(grouped, tick_labels=labels, patch_artist=True)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Residuals by Qualifying Position")
    plt.xlabel("Qualifying position")
    plt.ylabel("Residual")
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(result_dir / "F1_Residuals_by_Grid.png", bbox_inches="tight")
    plt.close()
