"""
text_cleaning.py

Handles NLP preprocessing.
"""

import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Download required resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


# Load stopwords
stop_words = set(
    stopwords.words("english")
)

# Preserve negation words
negation_words = {
    "not",
    "no",
    "nor",
    "never"
}

stop_words = (
    stop_words
    - negation_words
)


def remove_special_characters(text):
    """
    Remove special characters.
    """

    return re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )


def remove_stopwords(tokens):
    """
    Remove stopwords.
    """

    filtered_tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    return filtered_tokens


def lemmatize_text(tokens):
    """
    Lemmatize tokens.
    """

    lemmatized_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return lemmatized_tokens


def clean_text(text):
    """
    Complete text cleaning pipeline.
    """

    try:

        if not isinstance(
            text,
            str
        ):
            return ""

        # Lowercase
        text = text.lower()

        # Remove punctuation
        text = text.translate(
            str.maketrans(
                "",
                "",
                string.punctuation
            )
        )

        # Remove numbers
        text = re.sub(
            r"\d+",
            "",
            text
        )

        # Remove special chars
        text = (
            remove_special_characters(
                text
            )
        )

        # Tokenization
        tokens = (
            word_tokenize(
                text
            )
        )

        # Stopword removal
        tokens = (
            remove_stopwords(
                tokens
            )
        )

        # Lemmatization
        tokens = (
            lemmatize_text(
                tokens
            )
        )

        cleaned_text = (
            " ".join(tokens)
        )

        return cleaned_text

    except Exception as error:

        print(
            f"Cleaning error: "
            f"{error}"
        )

        return ""