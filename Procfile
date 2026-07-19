"""Train and compare loan eligibility classification models.

This script builds a baseline-versus-tuned evaluation flow with:
- preprocessing
- cross validation
- feature selection
- GridSearchCV hyperparameter tuning
- optional SMOTE for imbalanced classes
- before/after metric comparison
- best-model persistence
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    display_dataset_info,
    encode_target,
    handle_missing_values,
    load_dataset,
    split_features_target,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "dataset" / "loan_data.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "model.pkl"
REPORT_PATH = PROJECT_ROOT / "model" / "training_results.csv"
SUMMARY_PATH = PROJECT_ROOT / "model" / "training_summary.json"
PLOTS_DIR = PROJECT_ROOT / "model" / "plots"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 2


def build_preprocessor() -> ColumnTransformer:
    """Create the preprocessing block used by every model."""

    one_hot_kwargs = {"handle_unknown": "ignore"}
    try:
        OneHotEncoder(sparse_output=False)
        one_hot_kwargs["sparse_output"] = False
    except TypeError:
        one_hot_kwargs["sparse"] = False

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(**one_hot_kwargs)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_model_specs() -> Dict[str, Dict[str, object]]:
    """Define the candidate models and their tuning grids."""

    return {
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "grid": {
                "classifier__criterion": ["gini"],
                "classifier__max_depth": [3, 5],
                "classifier__min_samples_split": [2],
                "classifier__min_samples_leaf": [1],
            },
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            "grid": {
                "classifier__n_estimators": [100],
                "classifier__max_depth": [None, 6],
                "classifier__min_samples_split": [2],
                "classifier__min_samples_leaf": [1],
            },
        },
        "K-Nearest Neighbors": {
            "estimator": KNeighborsClassifier(),
            "grid": {
                "classifier__n_neighbors": [3, 5],
                "classifier__weights": ["distance"],
                "classifier__p": [2],
            },
        },
        "XGBoost": {
            "estimator": XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "grid": {
                "classifier__n_estimators": [100],
                "classifier__max_depth": [3, 4],
                "classifier__learning_rate": [0.1],
                "classifier__subsample": [0.8],
                "classifier__colsample_bytree": [0.8],
                "classifier__min_child_weight": [1],
                "classifier__gamma": [0],
            },
        },
    }


def build_baseline_pipeline(estimator: object) -> Pipeline:
    """Create the baseline pipeline for a model."""

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", estimator),
        ]
    )


def build_tuned_pipeline(estimator: object, use_smote: bool, smote_neighbors: int | None) -> ImbPipeline:
    """Create the tuned pipeline with feature selection and optional SMOTE."""

    smote_step: object = (
        SMOTE(random_state=RANDOM_STATE, k_neighbors=smote_neighbors)
        if use_smote and smote_neighbors is not None
        else "passthrough"
    )

    return ImbPipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("selector", SelectKBest(score_func=mutual_info_classif)),
            ("smote", smote_step),
            ("classifier", estimator),
        ]
    )


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, object]:
    """Calculate standard classification metrics."""

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "Confusion Matrix": confusion_matrix(y_true, y_pred),
        "Classification Report": classification_report(y_true, y_pred, zero_division=0),
    }


def cross_validate_pipeline(pipeline: Pipeline, x_train: pd.DataFrame, y_train: pd.Series, cv: StratifiedKFold) -> Dict[str, float]:
    """Run cross validation and return averaged metrics."""

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }
    scores = cross_validate(
        pipeline,
        x_train,
        y_train,
        scoring=scoring,
        cv=cv,
        return_train_score=True,
        n_jobs=1,
    )

    return {
        "Train Accuracy": float(scores["train_accuracy"].mean()),
        "Train Precision": float(scores["train_precision"].mean()),
        "Train Recall": float(scores["train_recall"].mean()),
        "Train F1": float(scores["train_f1"].mean()),
        "CV Accuracy": float(scores["test_accuracy"].mean()),
        "CV Precision": float(scores["test_precision"].mean()),
        "CV Recall": float(scores["test_recall"].mean()),
        "CV F1": float(scores["test_f1"].mean()),
    }


def determine_smote_settings(y_train: pd.Series) -> Tuple[bool, int | None]:
    """Detect class imbalance and choose a safe SMOTE configuration."""

    class_counts = y_train.value_counts()
    if len(class_counts) < 2:
        return False, None

    majority_ratio = class_counts.max() / class_counts.sum()
    minority_count = int(class_counts.min())

    if majority_ratio >= 0.60 and minority_count > 1:
        smote_neighbors = max(1, min(5, minority_count - 1))
        return True, smote_neighbors

    return False, None


def tune_model(
    estimator: object,
    grid: Dict[str, List[object]],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    use_smote: bool,
    smote_neighbors: int | None,
    selector_k_values: List[object],
) -> GridSearchCV:
    """Tune a model using GridSearchCV."""

    pipeline = build_tuned_pipeline(estimator, use_smote=use_smote, smote_neighbors=smote_neighbors)
    param_grid = dict(grid)
    param_grid["selector__k"] = selector_k_values

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
        },
        refit="f1",
        cv=cv,
        n_jobs=1,
        return_train_score=True,
    )
    search.fit(x_train, y_train)
    return search


def plot_model_comparison(results: pd.DataFrame, best_model_name: str) -> None:
    """Plot before-and-after charts for the four core metrics."""

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    figure, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    model_positions = np.arange(len(results))
    width = 0.35

    for axis, metric in zip(axes, metrics):
        before_values = results[f"Baseline {metric}"].to_numpy()
        after_values = results[f"Tuned {metric}"].to_numpy()

        before_bars = axis.bar(
            model_positions - width / 2,
            before_values,
            width,
            label="Before",
            color="#94a3b8",
            edgecolor="#334155",
        )
        after_bars = axis.bar(
            model_positions + width / 2,
            after_values,
            width,
            label="After",
            color="#1459c9",
            edgecolor="#0b2f66",
        )

        best_index = results.index[results["Model"] == best_model_name][0]
        after_bars[best_index].set_color("#22c55e")
        after_bars[best_index].set_edgecolor("#166534")
        after_bars[best_index].set_linewidth(2.2)

        axis.set_title(f"{metric} Before vs After", fontweight="bold")
        axis.set_xticks(model_positions)
        axis.set_xticklabels(results["Model"], rotation=20)
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
        axis.legend(frameon=False)

        for bars in [before_bars, after_bars]:
            for bar in bars:
                height = bar.get_height()
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.02,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    figure.suptitle(f"Loan Model Performance Comparison - Best Model: {best_model_name}", fontsize=16, fontweight="bold")
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    chart_path = PLOTS_DIR / "model_comparison.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def prepare_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    """Load, clean, split, and encode the dataset."""

    raw_dataframe = load_dataset(DATASET_PATH)
    display_dataset_info(raw_dataframe)
    cleaned_dataframe = handle_missing_values(raw_dataframe)
    features, target = split_features_target(cleaned_dataframe, TARGET_COLUMN)
    encoded_target, _ = encode_target(target)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        encoded_target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=encoded_target,
    )

    return x_train, x_test, y_train, y_test, encoded_target


def tune_and_compare_models() -> Tuple[pd.DataFrame, str, object]:
    """Train baseline and tuned models, compare them, and select the best pipeline."""

    x_train, x_test, y_train, y_test, encoded_target = prepare_data()
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    model_specs = build_model_specs()
    use_smote, smote_neighbors = determine_smote_settings(y_train)

    preprocessor = build_preprocessor()
    preprocessor.fit(x_train)
    transformed_feature_count = int(preprocessor.transform(x_train.iloc[: min(len(x_train), 20)]).shape[1])
    selector_candidates = sorted(
        {
            candidate
            for candidate in [4, 6, 8, 10, min(12, transformed_feature_count)]
            if candidate <= transformed_feature_count
        }
    )
    selector_candidates.append("all")

    comparison_rows: List[Dict[str, object]] = []
    tuned_models: Dict[str, object] = {}

    for model_name, spec in model_specs.items():
        print(f"\nTraining baseline model: {model_name}")
        baseline_pipeline = build_baseline_pipeline(spec["estimator"])
        baseline_cv = cross_validate_pipeline(baseline_pipeline, x_train, y_train, cv)
        baseline_pipeline.fit(x_train, y_train)
        baseline_predictions = baseline_pipeline.predict(x_test)
        baseline_holdout = evaluate_predictions(y_test, baseline_predictions)

        print(f"Tuning model with GridSearchCV: {model_name}")
        search = tune_model(
            estimator=spec["estimator"],
            grid=spec["grid"],
            x_train=x_train,
            y_train=y_train,
            cv=cv,
            use_smote=use_smote,
            smote_neighbors=smote_neighbors,
            selector_k_values=selector_candidates,
        )
        best_pipeline = search.best_estimator_
        tuned_models[model_name] = best_pipeline
        tuned_predictions = best_pipeline.predict(x_test)
        tuned_holdout = evaluate_predictions(y_test, tuned_predictions)

        best_index = int(search.best_index_)
        tuned_train_f1 = float(search.cv_results_["mean_train_f1"][best_index])
        tuned_cv_f1 = float(search.cv_results_["mean_test_f1"][best_index])
        generalization_gap = max(0.0, tuned_train_f1 - tuned_cv_f1)
        selection_score = (
            0.45 * tuned_holdout["Accuracy"]
            + 0.45 * tuned_holdout["F1 Score"]
            + 0.10 * tuned_cv_f1
            - 0.20 * generalization_gap
        )

        comparison_rows.append(
            {
                "Model": model_name,
                "Baseline Accuracy": baseline_holdout["Accuracy"],
                "Baseline Precision": baseline_holdout["Precision"],
                "Baseline Recall": baseline_holdout["Recall"],
                "Baseline F1 Score": baseline_holdout["F1 Score"],
                "Tuned Accuracy": tuned_holdout["Accuracy"],
                "Tuned Precision": tuned_holdout["Precision"],
                "Tuned Recall": tuned_holdout["Recall"],
                "Tuned F1 Score": tuned_holdout["F1 Score"],
                "Baseline CV Accuracy": baseline_cv["CV Accuracy"],
                "Baseline CV Precision": baseline_cv["CV Precision"],
                "Baseline CV Recall": baseline_cv["CV Recall"],
                "Baseline CV F1": baseline_cv["CV F1"],
                "Tuned CV Accuracy": float(search.cv_results_["mean_test_accuracy"][best_index]),
                "Tuned CV Precision": float(search.cv_results_["mean_test_precision"][best_index]),
                "Tuned CV Recall": float(search.cv_results_["mean_test_recall"][best_index]),
                "Tuned CV F1": tuned_cv_f1,
                "Generalization Gap": generalization_gap,
                "Selection Score": selection_score,
                "Best Params": json.dumps(search.best_params_),
                "Use SMOTE": use_smote,
            }
        )

        print(f"{model_name} baseline holdout accuracy: {baseline_holdout['Accuracy']:.4f}")
        print(f"{model_name} tuned holdout accuracy: {tuned_holdout['Accuracy']:.4f}")
        print(f"{model_name} tuned holdout F1: {tuned_holdout['F1 Score']:.4f}")
        print(f"{model_name} best params: {search.best_params_}")

    comparison_table = pd.DataFrame(comparison_rows).sort_values(
        by=["Selection Score", "Tuned Accuracy", "Tuned F1 Score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    best_model_name = str(comparison_table.iloc[0]["Model"])
    best_model = tuned_models[best_model_name]

    joblib.dump(best_model, MODEL_PATH)
    comparison_table.to_csv(REPORT_PATH, index=False)

    summary = {
        "best_model": best_model_name,
        "best_tuned_accuracy": float(comparison_table.iloc[0]["Tuned Accuracy"]),
        "best_tuned_f1": float(comparison_table.iloc[0]["Tuned F1 Score"]),
        "use_smote": bool(comparison_table.iloc[0]["Use SMOTE"]),
        "generalization_gap": float(comparison_table.iloc[0]["Generalization Gap"]),
        "target_accuracy_goal": 0.90,
        "target_accuracy_met": bool(comparison_table.iloc[0]["Tuned Accuracy"] >= 0.90),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    return comparison_table, best_model_name, best_model


def main() -> None:
    """Run the complete model training workflow."""

    comparison_table, best_model_name, _ = tune_and_compare_models()
    plot_model_comparison(comparison_table, best_model_name)

    print("\nBefore-and-after comparison table:")
    print(comparison_table.to_string(index=False))
    print(f"\nBest model: {best_model_name}")
    print(f"Saved best model to: {MODEL_PATH}")
    print(f"Saved comparison table to: {REPORT_PATH}")
    print(f"Saved summary to: {SUMMARY_PATH}")
    print(f"Saved comparison chart to: {PLOTS_DIR / 'model_comparison.png'}")


if __name__ == "__main__":
    main()
