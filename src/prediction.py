"""
prediction.py
"""

import joblib
import numpy as np

from src.text_cleaning import (
    clean_text
)


MODEL_PATH = (
    "models/"
    "sentiment_model.pkl"
)

VECTORIZER_PATH = (
    "models/"
    "tfidf_vectorizer.pkl"
)

ENCODER_PATH = (
    "models/"
    "label_encoder.pkl"
)


def load_artifacts():
    """
    Load saved files.
    """

    model = joblib.load(
        MODEL_PATH
    )

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    encoder = joblib.load(
        ENCODER_PATH
    )

    return (
        model,
        vectorizer,
        encoder
    )


def predict_sentiment(
    review
):
    """
    Predict sentiment.
    """

    try:

        (
            model,
            vectorizer,
            encoder
        ) = load_artifacts()

        # Clean text
        cleaned_review = (
            clean_text(review)
        )

        transformed_review = (
            vectorizer.transform(
                [cleaned_review]
            )
        )

        prediction = (
            model.predict(
                transformed_review
            )[0]
        )

        probabilities = (
            model.predict_proba(
                transformed_review
            )[0]
        )

        confidence = (
            np.max(
                probabilities
            ) * 100
        )

        # Smart confidence calibration
        neutral_words = [
            "okay",
            "fine",
            "average",
            "decent",
            "slightly",
            "but"
        ]

        if any(
            word in cleaned_review
            for word in neutral_words
        ):
            confidence *= 0.82

        elif confidence > 95:
            confidence -= 10

        elif confidence > 90:
            confidence -= 5

        label = (
            encoder
            .inverse_transform(
                [prediction]
            )[0]
        )

        return (
            label.capitalize(),
            round(
                confidence,
                2
            ),
            probabilities
        )

    except Exception as error:

        print(
            f"Prediction error: "
            f"{error}"
        )

        return (
            None,
            None,
            None
        )