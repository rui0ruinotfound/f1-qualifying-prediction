import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".matplotlib_cache"),
)
os.environ.setdefault("MPLBACKEND", "Agg")


def compute_correlation(df):
    corr = df["qualifying_position"].corr(df["race_position"])
    print(f"Pearson correlation: {corr:.4f}")
    return corr


def pole_to_win_rate(df):
    pole = df[df["qualifying_position"] == 1]
    if pole.empty:
        return 0.0

    wins = pole[pole["race_position"] == 1]
    rate = len(wins) / len(pole)
    print(f"Pole-to-win rate: {rate:.2%} ({len(wins)}/{len(pole)})")
    return rate


def average_finish_by_grid(df):
    avg_df = (
        df.groupby("qualifying_position")["race_position"]
        .agg(["mean", "count"])
        .reset_index()
    )
    avg_df.columns = ["qualifying_position", "avg_race_position", "num_races"]
    return avg_df

# AI generated: initial draft of this script was created with the help of AI assistance
# and then reviewed and refactored by Rui Chen. The final version was manually edited.

def run_linear_regression(df):
    x = df[["qualifying_position"]].values
    y = df["race_position"].values

    model = LinearRegression()
    model.fit(x, y)

    predicted = model.predict(x)
    mae = mean_absolute_error(y, predicted)
    rmse = np.sqrt(mean_squared_error(y, predicted))
    r_squared = r2_score(y, predicted)

    cv_mae = -cross_val_score(model, x, y, cv=5, scoring="neg_mean_absolute_error").mean()
    cv_rmse = np.sqrt(
        -cross_val_score(model, x, y, cv=5, scoring="neg_mean_squared_error").mean()
    )

    print(
        f"Linear regression: race_pos = {model.coef_[0]:.4f} * qual_pos + {model.intercept_:.4f}"
    )
    print(
        f"R-squared: {r_squared:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, "
        f"CV MAE: {cv_mae:.4f}, CV RMSE: {cv_rmse:.4f}"
    )

    return {
        "intercept": model.intercept_,
        "coefficient": model.coef_[0],
        "r_squared": r_squared,
        "mae": mae,
        "rmse": rmse,
        "cv_mae": cv_mae,
        "cv_rmse": cv_rmse,
        "predicted_values": predicted,
    }


def summary_statistics(df):
    stats = df[["qualifying_position", "race_position"]].describe()
    print("\nSummary statistics:")
    print(stats)
    return stats


def save_summary_table(df, avg_df, corr, regression_results, result_dir):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    rows = [
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
        rows.append(
            {
                "section": "average_finish_by_grid",
                "metric": "avg_race_position",
                "qualifying_position": int(row.qualifying_position),
                "value": round(row.avg_race_position, 4),
                "num_races": int(row.num_races),
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(result_dir / "summary_statistics.csv", index=False)
    return summary_df


def save_source_comparison(rows, result_dir, filename):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(result_dir / filename, index=False)
    return comparison_df


def available_model_specs(df):
    specs = {
        "Qualifying only": ["qualifying_position"],
        "Pit stops only": ["pit_stop_count"],
        "Weather only": ["rainfall", "track_temperature"],
        "Circuit type only": ["circuit_type"],
        "Combined": [
            "qualifying_position",
            "pit_stop_count",
            "circuit_type",
            "rainfall",
            "track_temperature",
        ],
    }

    available_specs = {}
    for model_name, features in specs.items():
        available_features = [
            feature
            for feature in features
            if feature in df.columns and df[feature].notna().any()
        ]
        if model_name == "Combined":
            if "qualifying_position" in available_features and len(available_features) > 1:
                available_specs[model_name] = available_features
        elif len(available_features) == len(features):
            available_specs[model_name] = available_features

    return available_specs


def _feature_pipeline(df, features):
    categorical_features = [
        feature
        for feature in features
        if df[feature].dtype == "object" or str(df[feature].dtype) == "category"
    ]
    numeric_features = [feature for feature in features if feature not in categorical_features]

    transformers = []
    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regression", LinearRegression()),
        ]
    )
    return model


def _feature_names(model, features):
    preprocessor = model.named_steps["preprocessor"]
    names = []

    if "numeric" in preprocessor.named_transformers_:
        names.extend(preprocessor.transformers_[0][2])

    if "categorical" in preprocessor.named_transformers_:
        categorical_pipeline = preprocessor.named_transformers_["categorical"]
        onehot = categorical_pipeline.named_steps["onehot"]
        categorical_features = [
            feature
            for name, _, feature in preprocessor.transformers_
            if name == "categorical"
            for feature in feature
        ]
        names.extend(onehot.get_feature_names_out(categorical_features).tolist())

    return names or features

# AI generated: initial draft of this script was created with the help of AI assistance
# and then reviewed and refactored by Rui Chen. The final version was manually edited.

def evaluate_regression_model(df, features, model_name):
    model_df = df[features + ["race_position"]].dropna(subset=["race_position"]).copy()
    model_df = model_df.dropna(subset=features, how="all")

    x = model_df[features]
    y = model_df["race_position"]

    model = _feature_pipeline(model_df, features)
    model.fit(x, y)

    predicted = model.predict(x)
    mae = mean_absolute_error(y, predicted)
    rmse = np.sqrt(mean_squared_error(y, predicted))
    r_squared = r2_score(y, predicted)

    cv_folds = min(5, len(model_df))
    if cv_folds >= 2:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_mae = -cross_val_score(
            model,
            x,
            y,
            cv=cv,
            scoring="neg_mean_absolute_error",
        ).mean()
        cv_rmse = np.sqrt(
            -cross_val_score(
                model,
                x,
                y,
                cv=cv,
                scoring="neg_mean_squared_error",
            ).mean()
        )
    else:
        cv_mae = np.nan
        cv_rmse = np.nan

    coefficient_names = _feature_names(model, features)
    coefficients = model.named_steps["regression"].coef_
    importance = pd.DataFrame(
        {
            "feature": coefficient_names,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    ).sort_values("abs_coefficient", ascending=False)

    print(
        f"{model_name}: R-squared={r_squared:.4f}, MAE={mae:.4f}, "
        f"RMSE={rmse:.4f}, CV MAE={cv_mae:.4f}, CV RMSE={cv_rmse:.4f}"
    )

    return {
        "model_name": model_name,
        "features": features,
        "n_observations": len(model_df),
        "mae": mae,
        "rmse": rmse,
        "r_squared": r_squared,
        "cv_mae": cv_mae,
        "cv_rmse": cv_rmse,
        "predicted_values": predicted,
        "actual_values": y.values,
        "importance": importance,
    }


def run_feature_model_comparison(df):
    specs = available_model_specs(df)
    results = []

    for model_name, features in specs.items():
        results.append(evaluate_regression_model(df, features, model_name))

    comparison = pd.DataFrame(
        [
            {
                "model": result["model_name"],
                "features": ", ".join(result["features"]),
                "n_observations": result["n_observations"],
                "mae": round(result["mae"], 4),
                "rmse": round(result["rmse"], 4),
                "r_squared": round(result["r_squared"], 4),
                "cv_mae": round(result["cv_mae"], 4),
                "cv_rmse": round(result["cv_rmse"], 4),
            }
            for result in results
        ]
    )
    return comparison, results


def feature_correlations(df):
    features = [
        feature
        for feature in [
            "qualifying_position",
            "pit_stop_count",
            "rainfall",
            "track_temperature",
            "circuit_type",
        ]
        if feature in df.columns and df[feature].notna().any()
    ]
    encoded = pd.get_dummies(df[features + ["race_position"]], drop_first=False)
    return encoded.corr(numeric_only=True)


def save_feature_analysis_outputs(comparison_df, model_results, result_dir):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(result_dir / "feature_model_comparison.csv", index=False)

    importance_rows = []
    for result in model_results:
        importance = result["importance"].copy()
        importance.insert(0, "model", result["model_name"])
        importance_rows.append(importance)

    if importance_rows:
        importance_df = pd.concat(importance_rows, ignore_index=True)
        importance_df.to_csv(result_dir / "feature_importance.csv", index=False)
    else:
        importance_df = pd.DataFrame()

    return importance_df


def plot_feature_analysis(df, comparison_df, model_results, result_dir):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    corr = feature_correlations(df)
    plt.figure(figsize=(10, 8))
    image = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation Heatmap for Model Features")
    plt.tight_layout()
    plt.savefig(result_dir / "F1_Feature_Correlation_Heatmap.png", bbox_inches="tight")
    plt.close()

    combined = next(
        (result for result in model_results if result["model_name"] == "Combined"),
        model_results[-1] if model_results else None,
    )
    if combined is None:
        return

    importance = combined["importance"].head(10).sort_values("abs_coefficient")
    plt.figure(figsize=(10, 6))
    plt.barh(importance["feature"], importance["abs_coefficient"], color="teal")
    plt.title("Feature Importance in Final Combined Model")
    plt.xlabel("Absolute coefficient after preprocessing")
    plt.tight_layout()
    plt.savefig(result_dir / "F1_Feature_Importance.png", bbox_inches="tight")
    plt.close()

    actual = combined["actual_values"]
    predicted = combined["predicted_values"]
    plt.figure(figsize=(8, 8))
    plt.scatter(actual, predicted, alpha=0.4, color="darkorange")
    plt.plot([1, 20], [1, 20], color="black", linestyle="--", linewidth=2)
    plt.title("Actual vs Predicted Finishing Position - Combined Model")
    plt.xlabel("Actual finishing position")
    plt.ylabel("Predicted finishing position")
    plt.xticks(range(1, 21))
    plt.yticks(range(1, 21))
    plt.grid(True)
    plt.savefig(result_dir / "F1_Combined_Actual_vs_Predicted.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(comparison_df["model"], comparison_df["rmse"], color="slateblue")
    plt.title("Model RMSE Comparison")
    plt.ylabel("RMSE")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(result_dir / "F1_Model_Comparison_RMSE.png", bbox_inches="tight")
    plt.close()


def plot_individual_feature_charts(df, result_dir):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    numeric_features = [
        ("pit_stop_count", "Pit Stops"),
        ("rainfall", "Rain Flag"),
        ("track_temperature", "Track Temperature"),
    ]
    for feature, label in numeric_features:
        if feature not in df.columns or not df[feature].notna().any():
            continue

        plt.figure(figsize=(9, 6))
        plt.scatter(df[feature], df["race_position"], alpha=0.35)
        plt.title(f"{label} vs Finishing Position")
        plt.xlabel(label)
        plt.ylabel("Finishing position")
        plt.grid(True)
        plt.savefig(result_dir / f"F1_{feature}_scatterplot.png", bbox_inches="tight")
        plt.close()

    if "circuit_type" in df.columns and df["circuit_type"].notna().any():
        grouped = [
            group["race_position"].values
            for _, group in df.dropna(subset=["circuit_type"]).groupby("circuit_type")
        ]
        labels = [
            circuit_type
            for circuit_type, _ in df.dropna(subset=["circuit_type"]).groupby("circuit_type")
        ]
        plt.figure(figsize=(8, 6))
        plt.boxplot(grouped, tick_labels=labels, patch_artist=True)
        plt.title("Finishing Position by Circuit Type")
        plt.xlabel("Circuit type")
        plt.ylabel("Finishing position")
        plt.grid(axis="y")
        plt.savefig(result_dir / "F1_Circuit_Type_boxplot.png", bbox_inches="tight")
        plt.close()


def run_strategy_scenarios(df):
    features = [
        "qualifying_position",
        "pit_stop_count",
        "circuit_type",
        "rainfall",
        "track_temperature",
    ]
    available_features = [
        feature
        for feature in features
        if feature in df.columns and df[feature].notna().any()
    ]
    if "qualifying_position" not in available_features or len(available_features) < 2:
        return pd.DataFrame()

    model_df = df[available_features + ["race_position"]].dropna(subset=["race_position"]).copy()
    model_df = model_df.dropna(subset=available_features, how="all")

    model = _feature_pipeline(model_df, available_features)
    model.fit(model_df[available_features], model_df["race_position"])

    default_values = {}
    for feature in available_features:
        if model_df[feature].dtype == "object" or str(model_df[feature].dtype) == "category":
            default_values[feature] = model_df[feature].mode().iloc[0]
        else:
            default_values[feature] = model_df[feature].median()

    scenarios = [
        # Tire strategy is a scenario label only. Model inputs remain pit/weather/circuit features
        # so that actual post-race compound usage is not used as a predictor.
        {
            "scenario": "P2 dry race, one stop",
            "tire_strategy_assumption": "Medium-Hard",
            "qualifying_position": 2,
            "pit_stop_count": 1,
            "rainfall": 0,
            "circuit_type": "permanent",
        },
        {
            "scenario": "P2 dry race, two stops",
            "tire_strategy_assumption": "Medium-Hard-Soft",
            "qualifying_position": 2,
            "pit_stop_count": 2,
            "rainfall": 0,
            "circuit_type": "permanent",
        },
        {
            "scenario": "P10 dry race, one stop",
            "tire_strategy_assumption": "Hard-Medium",
            "qualifying_position": 10,
            "pit_stop_count": 1,
            "rainfall": 0,
            "circuit_type": "permanent",
        },
        {
            "scenario": "P10 wet race, two stops",
            "tire_strategy_assumption": "Intermediate-Medium-Hard",
            "qualifying_position": 10,
            "pit_stop_count": 2,
            "rainfall": 1,
            "circuit_type": "permanent",
        },
        {
            "scenario": "P5 street circuit, one stop",
            "tire_strategy_assumption": "Medium-Hard",
            "qualifying_position": 5,
            "pit_stop_count": 1,
            "rainfall": 0,
            "circuit_type": "street",
        },
        {
            "scenario": "P5 street circuit, three stops",
            "tire_strategy_assumption": "Soft-Medium-Hard-Soft",
            "qualifying_position": 5,
            "pit_stop_count": 3,
            "rainfall": 0,
            "circuit_type": "street",
        },
        {
            "scenario": "P6 wet hybrid circuit, two stops",
            "tire_strategy_assumption": "Intermediate-Medium-Hard",
            "qualifying_position": 6,
            "pit_stop_count": 2,
            "rainfall": 1,
            "circuit_type": "hybrid",
        },
        {
            "scenario": "P6 wet hybrid circuit, three stops",
            "tire_strategy_assumption": "Wet-Intermediate-Medium-Hard",
            "qualifying_position": 6,
            "pit_stop_count": 3,
            "rainfall": 1,
            "circuit_type": "hybrid",
        },
    ]

    scenario_df = pd.DataFrame(scenarios)
    for feature, value in default_values.items():
        if feature not in scenario_df.columns:
            scenario_df[feature] = value
        else:
            scenario_df[feature] = scenario_df[feature].fillna(value)

    prediction_input = scenario_df[available_features]
    scenario_df["predicted_finish_position"] = model.predict(prediction_input)
    scenario_df["predicted_finish_position"] = scenario_df[
        "predicted_finish_position"
    ].round(2)

    display_columns = [
        "scenario",
        "tire_strategy_assumption",
    ] + available_features + ["predicted_finish_position"]
    return scenario_df[display_columns]


def save_strategy_scenarios(scenario_df, result_dir):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    if not scenario_df.empty:
        scenario_df.to_csv(result_dir / "strategy_scenarios.csv", index=False)
    return scenario_df


def plot_strategy_scenarios(scenario_df, result_dir):
    if scenario_df.empty:
        return

    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    plot_df = scenario_df.sort_values("predicted_finish_position", ascending=False)
    plt.figure(figsize=(10, 6))
    plt.barh(
        plot_df["scenario"],
        plot_df["predicted_finish_position"],
        color="seagreen",
    )
    plt.title("Strategy Scenario Simulation")
    plt.xlabel("Predicted finishing position")
    plt.xlim(1, max(20, plot_df["predicted_finish_position"].max() + 1))
    plt.grid(axis="x", alpha=0.35)
    plt.tight_layout()
    plt.savefig(result_dir / "F1_Strategy_Scenarios.png", bbox_inches="tight")
    plt.close()

# AI generated: initial draft of this script was created with the help of AI assistance
# and then reviewed and refactored by Rui Chen. The final version was manually edited.

def plot_statistics(df, avg_df, corr, regression_results, result_dir):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.scatter(
        df["qualifying_position"],
        df["race_position"],
        alpha=0.35,
        color="darkred",
        label="Race-driver observations",
    )
    x_values = np.arange(1, 21)
    y_values = regression_results["coefficient"] * x_values + regression_results["intercept"]
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
    plt.savefig(result_dir / "F1_Qualifying_scatterplot.png", bbox_inches="tight")
    plt.close()

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
    plt.savefig(result_dir / "F1_Qualifying_linechart.png", bbox_inches="tight")
    plt.close()

    labels = ["Grid 1", "Grid 2", "Grid 3"]
    win_rates = []
    for grid_position in [1, 2, 3]:
        starts = df[df["qualifying_position"] == grid_position]
        wins = starts[starts["race_position"] == 1]
        rate = 0 if starts.empty else len(wins) / len(starts)
        win_rates.append(rate * 100)

    plt.figure(figsize=(8, 6))
    plt.bar(labels, win_rates, color=["seagreen", "royalblue", "gray"], edgecolor="black")
    plt.title("Win Rate by Front Grid Position - F1 Qualifying")
    plt.ylabel("Win rate (%)")
    plt.grid(axis="y")
    plt.savefig(result_dir / "F1_Pole_to_Win_barchart.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.scatter(df["race_position"].values, regression_results["predicted_values"], alpha=0.4)
    plt.plot([1, 20], [1, 20], color="black", linestyle="--", linewidth=2)
    plt.title("Predicted vs Actual Finishing Position - F1 Qualifying")
    plt.xlabel("Actual finishing position")
    plt.ylabel("Predicted finishing position")
    plt.xticks(range(1, 21))
    plt.yticks(range(1, 21))
    plt.grid(True)
    plt.savefig(result_dir / "F1_Predicted_vs_Actual_scatterplot.png", bbox_inches="tight")
    plt.close()

    grouped = []
    labels = []
    for grid_position in sorted(df["qualifying_position"].unique()):
        grouped.append(df.loc[df["qualifying_position"] == grid_position, "race_position"].values)
        labels.append(str(int(grid_position)))

    plt.figure(figsize=(12, 6))
    plt.boxplot(grouped, tick_labels=labels, patch_artist=True)
    plt.title("Finishing Position by Grid Position - F1 Qualifying")
    plt.xlabel("Grid position")
    plt.ylabel("Finishing position")
    plt.grid(axis="y")
    plt.savefig(result_dir / "F1_Finishing_Position_boxplot.png", bbox_inches="tight")
    plt.close()
