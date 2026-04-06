from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
DEADLINE = pd.Timestamp("2026-01-18 18:00:00")
DATASET_FILE = "DataSet.csv"


@dataclass(frozen=True)
class ModelArtifacts:
    model: Pipeline
    feature_columns: list[str]


def map_time_category(hour: int) -> str:
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    if 21 <= hour < 24:
        return "night"
    return "late_night"


def load_and_prepare_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S")
    df = df.sort_values("Timestamp").drop_duplicates(
        subset=["Student Full Name (As per University Records)"],
        keep="first",
    )

    # Target: whether actual submission was after deadline.
    df["late"] = (df["Timestamp"] > DEADLINE).astype(int)

    # Leakage-safe features: no direct deadline delta and no unique student identifiers.
    df["submission_hour"] = df["Timestamp"].dt.hour
    df["submission_weekday"] = df["Timestamp"].dt.weekday
    df["is_weekend"] = (df["submission_weekday"] >= 5).astype(int)
    df["time_category"] = df["submission_hour"].apply(map_time_category)
    df["section"] = df["B.Tech Section"].fillna("Unknown")

    return df


def build_pipeline() -> Pipeline:
    numeric_features = ["submission_hour", "submission_weekday", "is_weekend"]
    categorical_features = ["time_category", "section"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def evaluate_with_cv(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    min_class_count = int(y.value_counts().min())
    n_splits = max(2, min(5, min_class_count))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }

    cv_results = cross_validate(
        model,
        x,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        error_score="raise",
    )
    return {metric: float(np.mean(values)) for metric, values in cv_results.items() if metric.startswith("test_")}


def evaluate_holdout(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    return {
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "average_precision": average_precision_score(y_test, y_proba),
    }


def compare_against_baseline(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series) -> float:
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)
    baseline_pred = baseline.predict(x_test)
    return balanced_accuracy_score(y_test, baseline_pred)


def train_model(dataset_path: Path) -> Tuple[ModelArtifacts, pd.DataFrame]:
    df = load_and_prepare_data(dataset_path)

    feature_columns = [
        "submission_hour",
        "submission_weekday",
        "is_weekend",
        "time_category",
        "section",
    ]

    x = df[feature_columns]
    y = df["late"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_pipeline()
    model.fit(x_train, y_train)

    cv_scores = evaluate_with_cv(model, x_train, y_train)
    holdout_scores = evaluate_holdout(model, x_test, y_test)
    baseline_bal_acc = compare_against_baseline(x_train, y_train, x_test, y_test)

    print("=" * 60)
    print("DATA OVERVIEW")
    print("=" * 60)
    print(f"Rows after de-duplication: {len(df)}")
    print(f"Late rate: {y.mean():.2%} ({int(y.sum())}/{len(y)})")
    print(f"Train/Test size: {len(x_train)}/{len(x_test)}")

    print("\n" + "=" * 60)
    print("STRATIFIED CROSS-VALIDATION (TRAINING SET)")
    print("=" * 60)
    for metric_name, value in sorted(cv_scores.items()):
        print(f"{metric_name.replace('test_', '')}: {value:.4f}")

    print("\n" + "=" * 60)
    print("HOLDOUT EVALUATION")
    print("=" * 60)
    for metric_name, value in holdout_scores.items():
        print(f"{metric_name}: {value:.4f}")

    print(f"\nBaseline balanced_accuracy (most_frequent): {baseline_bal_acc:.4f}")

    y_test_pred = model.predict(x_test)
    print("\nClassification report (holdout):")
    print(classification_report(y_test, y_test_pred, target_names=["On Time", "Late"], zero_division=0))
    print("Confusion matrix (holdout):")
    print(confusion_matrix(y_test, y_test_pred))

    return ModelArtifacts(model=model, feature_columns=feature_columns), df


def predict_late_probability(model_artifacts: ModelArtifacts, timestamp_str: str, section: str) -> Tuple[int, float]:
    timestamp = pd.to_datetime(timestamp_str, format="%d/%m/%Y %H:%M:%S")

    features = {
        "submission_hour": timestamp.hour,
        "submission_weekday": timestamp.weekday(),
        "is_weekend": int(timestamp.weekday() >= 5),
        "time_category": map_time_category(timestamp.hour),
        "section": section,
    }

    input_df = pd.DataFrame([features])[model_artifacts.feature_columns]
    prediction = int(model_artifacts.model.predict(input_df)[0])
    probability_late = float(model_artifacts.model.predict_proba(input_df)[0][1])
    return prediction, probability_late


def main() -> None:
    dataset_path = Path(__file__).resolve().parent / DATASET_FILE
    artifacts, _ = train_model(dataset_path)

    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)
    test_cases = [
        ("18/01/2026 14:00:00", "D2"),
        ("18/01/2026 19:00:00", "D2"),
        ("19/01/2026 10:00:00", "D1"),
    ]

    for timestamp_str, section in test_cases:
        label, p_late = predict_late_probability(artifacts, timestamp_str, section)
        status = "LATE" if label == 1 else "ON TIME"
        confidence = max(p_late, 1 - p_late)
        print(f"{timestamp_str} ({section}) -> {status} | confidence={confidence:.1%} | p_late={p_late:.1%}")


if __name__ == "__main__":
    main()