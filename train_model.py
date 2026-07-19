"""Prediction helpers for the Smart Lender project."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.preprocessing import FEATURE_COLUMNS, build_feature_schema, coerce_numeric_inputs

MODEL_PATH = PROJECT_ROOT / "model" / "model.pkl"


def load_artifacts() -> object:
    """Load the trained prediction pipeline from disk."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run python model/train_model.py first."
        )

    return joblib.load(MODEL_PATH)


def load_dataset_schema() -> pd.DataFrame:
    """Return the input schema as a one-row DataFrame template for the app."""

    return pd.DataFrame(
        [
            {
                "name": field["name"],
                "label": field["label"],
                "kind": field["kind"],
                "options": ",".join(field.get("options", [])),
            }
            for field in build_feature_schema()
        ]
    )


def prepare_input_frame(input_data: Dict[str, object]) -> pd.DataFrame:
    """Convert raw input values into a single-row DataFrame."""

    normalized_input = coerce_numeric_inputs(input_data)
    return pd.DataFrame([normalized_input], columns=FEATURE_COLUMNS)


def decode_prediction(predicted_value: object) -> str:
    """Convert the numeric class prediction into a human-readable label."""

    try:
        numeric_value = int(predicted_value)
    except Exception:
        numeric_value = 0

    return "Approved" if numeric_value == 1 else "Rejected"


def predict_from_input(input_data: Dict[str, object]) -> Dict[str, object]:
    """Predict loan eligibility from a raw input mapping."""

    model = load_artifacts()
    feature_frame = prepare_input_frame(input_data)

    predicted_class = model.predict(feature_frame)[0]
    probabilities = model.predict_proba(feature_frame)[0]
    approval_label = decode_prediction(predicted_class)
    class_index = int(np.where(model.classes_ == predicted_class)[0][0])
    probability_score = float(probabilities[class_index])

    return {
        "status": approval_label,
        "probability": probability_score,
        "predicted_class": int(predicted_class),
    }


def predict_loan_eligibility() -> None:
    """Run a small CLI flow for manual predictions."""

    schema = build_feature_schema()
    user_input: Dict[str, object] = {}

    print("\nLoan Eligibility Prediction")
    print("Enter applicant details below.\n")

    for field in schema:
        name = field["name"]
        label = field["label"]
        if field["kind"] == "numeric":
            while True:
                raw_value = input(f"Enter {label}: ").strip()
                try:
                    user_input[name] = float(raw_value)
                    break
                except ValueError:
                    print("Please enter a valid numeric value.")
        else:
            options = field.get("options", [])
            prompt_text = ", ".join(options)
            while True:
                raw_value = input(f"Enter {label} ({prompt_text}): ").strip()
                if raw_value in options:
                    user_input[name] = raw_value
                    break
                print(f"Invalid option. Choose one of: {prompt_text}")

    prediction = predict_from_input(user_input)
    print("\nPrediction Result:")
    print(f"Status: {prediction['status']}")
    print(f"Probability Score: {prediction['probability']:.4f}")


def main() -> None:
    """Entry point for command-line execution."""

    predict_loan_eligibility()


if __name__ == "__main__":
    main()
