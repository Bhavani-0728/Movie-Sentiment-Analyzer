"""
prediction.py

Handles sentiment prediction.
"""

import joblib
import numpy as np

from src.text_cleaning import (
    clean_text
)


# --------------------------------------------------
# MODEL PATHS
# --------------------------------------------------

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


# --------------------------------------------------
# LOAD ARTIFACTS
# --------------------------------------------------

def load_artifacts():
    """
    Load trained model,
    vectorizer, and label encoder.
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


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict_sentiment(review):
    """
    Predict movie review sentiment.

    Args:
        review (str)

    Returns:
        tuple:
        (
            label,
            confidence,
            probability_dict
        )
    """

    try:

        # Load model files
        (
            model,
            vectorizer,
            encoder
        ) = load_artifacts()

        # Clean review text
        cleaned_review = (
            clean_text(review)
        )

        # Convert to vector
        transformed_review = (
            vectorizer.transform(
                [cleaned_review]
            )
        )

        # Predict encoded label
        prediction = (
            model.predict(
                transformed_review
            )[0]
        )

        # Predict probabilities
        probabilities = (
            model.predict_proba(
                transformed_review
            )[0]
        )

        # Confidence score
        confidence = (
            np.max(
                probabilities
            ) * 100
        )

        # Decode prediction
        label = (
            encoder
            .inverse_transform(
                [prediction]
            )[0]
        )

        # Map probabilities
        class_labels = (
            encoder.classes_
        )

        probability_dict = dict(
            zip(
                class_labels,
                probabilities
            )
        )

        return (
            label.capitalize(),
            round(
                confidence,
                2
            ),
            probability_dict
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


# --------------------------------------------------
# TESTING
# --------------------------------------------------

if __name__ == "__main__":

    sample_review = (
        "This movie was amazing!"
    )

    result = (
        predict_sentiment(
            sample_review
        )
    )

    print(result)