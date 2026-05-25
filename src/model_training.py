"""
model_training.py

Handles:
1. Model training
2. Evaluation
3. Saving artifacts
"""

import os
import json
import joblib

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.preprocessing import (
    LabelEncoder
)


def encode_labels(y):
    """
    Encode sentiment labels.
    """

    encoder = LabelEncoder()

    encoded_y = (
        encoder.fit_transform(y)
    )

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        encoder,
        "models/label_encoder.pkl"
    )

    return encoded_y, encoder


def train_model(
    x_train,
    y_train
):
    """
    Train Logistic Regression.
    """

    model = (
        LogisticRegression(
            max_iter=1000,
            C=0.5,
            random_state=42
        )
    )

    model.fit(
        x_train,
        y_train
    )

    print(
        "Model trained successfully."
    )

    return model

def evaluate_model(
    model,
    x_test,
    y_test
):
    """
    Evaluate model.
    """

    predictions = model.predict(
        x_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    metrics = {
        "accuracy":
            round(accuracy, 4),

        "precision":
            round(precision, 4),

        "recall":
            round(recall, 4),

        "f1_score":
            round(f1, 4),

        "confusion_matrix":
            cm.tolist()
    }

    print(metrics)

    return metrics


def save_model(
    model,
    path=(
        "models/"
        "sentiment_model.pkl"
    )
):
    """
    Save trained model.
    """

    joblib.dump(
        model,
        path
    )

    print("Model saved.")


def save_metrics(
    metrics,
    path=(
        "models/"
        "model_metrics.json"
    )
):
    """
    Save metrics JSON.
    """

    with open(
        path,
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print(
        "Metrics saved."
    )