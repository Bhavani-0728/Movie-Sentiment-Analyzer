"""
train.py

Complete training pipeline for
Movie Review Sentiment Analyzer.
"""

from sklearn.model_selection import (
    train_test_split
)

from src.data_preprocessing import (
    prepare_dataset
)

from src.feature_engineering import (
    vectorize_text
)

from src.model_training import (
    encode_labels,
    train_model,
    evaluate_model,
    save_model,
    save_metrics
)


def main():
    """
    Run complete training pipeline.
    """

    print("=" * 50)
    print("MOVIE SENTIMENT ANALYZER")
    print("=" * 50)

    # Step 1: Load + preprocess data
    dataframe = prepare_dataset()

    if dataframe is None:
        print(
            "Dataset loading failed."
        )
        return

    # Step 2: Features and target
    x = dataframe[
        "cleaned_review"
    ]

    y = dataframe[
        "sentiment"
    ]

    # Step 3: Encode labels
    y_encoded, encoder = (
        encode_labels(y)
    )

    # Step 4: Train test split
    (
        x_train,
        x_test,
        y_train,
        y_test
    ) = train_test_split(
        x,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    print(
        "\nTrain-test split completed."
    )

    # Step 5: TF-IDF vectorization
    (
        x_train_tfidf,
        x_test_tfidf,
        vectorizer
    ) = vectorize_text(
        x_train,
        x_test
    )

    print(
        "Feature engineering completed."
    )

    # Step 6: Model training
    model = train_model(
        x_train_tfidf,
        y_train
    )

    # Step 7: Evaluation
    metrics = evaluate_model(
        model,
        x_test_tfidf,
        y_test
    )

    print("\nModel Metrics")
    print("-" * 30)

    for key, value in (
        metrics.items()
    ):
        print(
            f"{key}: {value}"
        )

    # Step 8: Save artifacts
    save_model(model)

    save_metrics(metrics)

    print("\nTraining completed!")
    print(
        "All artifacts saved "
        "inside models folder."
    )


if __name__ == "__main__":
    main()