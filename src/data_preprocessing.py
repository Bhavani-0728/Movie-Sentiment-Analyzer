"""
data_preprocessing.py

This file handles:
1. Loading dataset
2. Handling missing values
3. Removing duplicates
4. Cleaning review text
5. Saving processed dataset
"""

import os
import pandas as pd

from src.text_cleaning import clean_text


def load_dataset(file_path):
    """
    Load dataset from CSV file.

    Args:
        file_path (str): Dataset path

    Returns:
        pd.DataFrame: Loaded dataset
    """

    try:
        dataframe = pd.read_csv(file_path)

        print("Dataset loaded successfully.")
        print(f"Shape: {dataframe.shape}")

        return dataframe

    except FileNotFoundError:
        print(f"Dataset not found: {file_path}")
        return None

    except Exception as error:
        print(f"Error loading dataset: {error}")
        return None


def handle_missing_values(dataframe):
    """
    Remove rows with null values.

    Args:
        dataframe (pd.DataFrame)

    Returns:
        pd.DataFrame
    """

    initial_shape = dataframe.shape

    dataframe = dataframe.dropna()

    print(
        f"Missing values removed: "
        f"{initial_shape[0] - dataframe.shape[0]}"
    )

    return dataframe


def remove_duplicates(dataframe):
    """
    Remove duplicate rows.

    Args:
        dataframe (pd.DataFrame)

    Returns:
        pd.DataFrame
    """

    initial_shape = dataframe.shape

    dataframe = dataframe.drop_duplicates()

    print(
        f"Duplicates removed: "
        f"{initial_shape[0] - dataframe.shape[0]}"
    )

    return dataframe


def preprocess_reviews(dataframe):
    """
    Clean review text.

    Args:
        dataframe (pd.DataFrame)

    Returns:
        pd.DataFrame
    """

    print("Cleaning review text...")

    dataframe["cleaned_review"] = (
        dataframe["review"]
        .astype(str)
        .apply(clean_text)
    )

    return dataframe


def save_processed_data(
    dataframe,
    output_path="data/processed/cleaned_reviews.csv"
):
    """
    Save cleaned dataset.

    Args:
        dataframe (pd.DataFrame)
        output_path (str)
    """

    try:
        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )

        dataframe.to_csv(
            output_path,
            index=False
        )

        print(
            f"Processed dataset saved to:\n"
            f"{output_path}"
        )

    except Exception as error:
        print(
            f"Error saving dataset: {error}"
        )


def prepare_dataset(
    input_path="data/raw/imdb_reviews.csv"
):
    """
    Complete preprocessing pipeline.

    Returns:
        pd.DataFrame
    """

    dataframe = load_dataset(input_path)

    if dataframe is None:
        return None

    dataframe = handle_missing_values(
        dataframe
    )

    dataframe = remove_duplicates(
        dataframe
    )

    dataframe = preprocess_reviews(
        dataframe
    )

    save_processed_data(dataframe)

    print("Data preprocessing complete.")

    return dataframe


# Testing
if __name__ == "__main__":

    dataset = prepare_dataset()

    if dataset is not None:
        print(dataset.head())