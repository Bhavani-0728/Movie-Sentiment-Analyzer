"""
feature_engineering.py
"""

import os
import joblib

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)


def create_tfidf_vectorizer():
    """
    Create TF-IDF vectorizer.
    """

    vectorizer = (
        TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
    )

    return vectorizer


def fit_transform_tfidf(
    x_train
):
    """
    Fit TF-IDF.
    """

    vectorizer = (
        create_tfidf_vectorizer()
    )

    x_train_tfidf = (
        vectorizer.fit_transform(
            x_train
        )
    )

    return (
        x_train_tfidf,
        vectorizer
    )


def transform_tfidf(
    vectorizer,
    x_test
):
    """
    Transform test data.
    """

    return (
        vectorizer.transform(
            x_test
        )
    )


def save_vectorizer(
    vectorizer,
    model_path=(
        "models/"
        "tfidf_vectorizer.pkl"
    )
):
    """
    Save vectorizer.
    """

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        vectorizer,
        model_path
    )

    print(
        "TF-IDF vectorizer saved."
    )


def vectorize_text(
    x_train,
    x_test
):
    """
    Full vectorization pipeline.
    """

    (
        x_train_tfidf,
        vectorizer
    ) = fit_transform_tfidf(
        x_train
    )

    x_test_tfidf = (
        transform_tfidf(
            vectorizer,
            x_test
        )
    )

    save_vectorizer(
        vectorizer
    )

    return (
        x_train_tfidf,
        x_test_tfidf,
        vectorizer
    )