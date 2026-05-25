"""
app.py

Streamlit application for
Movie Review Sentiment Analyzer.
"""

import json
import random
import streamlit as st

from src.prediction import predict_sentiment


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Outfit:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background-color: #0a0a0a;
        color: #f0ece4;
    }

    .block-container {
        padding: 3rem 2rem 5rem;
        max-width: 760px;
    }

    .film-strip {
        width: 100%;
        overflow: hidden;
        background: #111;
        border-top: 3px solid #e63946;
        border-bottom: 3px solid #e63946;
        padding: 6px 0;
        margin-bottom: 2.5rem;
        white-space: nowrap;
    }

    .film-strip-inner {
        display: inline-block;
        animation: ticker 18s linear infinite;
        font-family: 'Bebas Neue', sans-serif;
        font-size: 0.85rem;
        letter-spacing: 0.25em;
        color: #e63946;
    }

    @keyframes ticker {
        0% { transform: translateX(0); }
        100% { transform: translateX(-50%); }
    }

    .main-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: clamp(3rem, 8vw, 5.5rem);
        line-height: 0.92;
        color: #f0ece4;
        text-align: center;
        margin-bottom: 0.4rem;
    }

    .main-title span {
        color: #e63946;
    }

    .subtitle {
        text-align: center;
        color: #6b6560;
        font-size: 0.82rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }

    textarea {
        background: #0f0f0f !important;
        border: 1px solid #222 !important;
        border-radius: 8px !important;
        color: #f0ece4 !important;
    }

    div.stButton > button {
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# TOP TICKER
# --------------------------------------------------

st.markdown(
    """
    <div class="film-strip">
      <div class="film-strip-inner">
        🎬 NOW ANALYZING · SENTIMENT ENGINE ONLINE ·
        POSITIVE · NEGATIVE · NLP + ML ·
        TF-IDF · LOGISTIC REGRESSION ·
        🎬 NOW ANALYZING · SENTIMENT ENGINE ONLINE ·
      </div>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
    """
    <div class='main-title'>
        MOVIE<br>
        <span>REVIEW</span><br>
        ANALYZER
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='subtitle'>
        Predict movie review sentiment
        using NLP + Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# LOAD MODEL ACCURACY
# --------------------------------------------------

try:
    with open(
        "models/model_metrics.json",
        "r"
    ) as file:

        metrics = json.load(file)
        accuracy = metrics.get(
            "accuracy",
            "N/A"
        )

except Exception:
    accuracy = "N/A"


st.info(
    f"📈 Model Accuracy: {accuracy}"
)


# --------------------------------------------------
# EXAMPLE REVIEWS
# --------------------------------------------------

examples = [
    "This movie was fantastic! Amazing acting and story.",
    "Worst movie ever. Waste of time.",
    "The movie was decent but slightly boring.",
    "Absolutely loved this film. Highly recommend!"
]


if "review_text" not in st.session_state:
    st.session_state.review_text = ""


col1, col2 = st.columns(2)

with col1:
    if st.button("🎲 Example Review"):
        st.session_state.review_text = (
            random.choice(examples)
        )

with col2:
    if st.button("🗑 Clear Input"):
        st.session_state.review_text = ""


# --------------------------------------------------
# INPUT
# --------------------------------------------------

review = st.text_area(
    "Enter Movie Review",
    value=st.session_state.review_text,
    height=200,
    placeholder="Type review here..."
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("🔍 Predict Sentiment"):

    if review.strip() == "":
        st.warning(
            "Please enter a review."
        )

    else:

        (
            prediction,
            confidence,
            probabilities
        ) = predict_sentiment(review)

        if prediction:

            # Convert probabilities to %
            p_neg = (
                probabilities[0] * 100
            )

            p_pos = (
                probabilities[1] * 100
            )

            emoji = (
                "😊"
                if prediction.lower()
                == "positive"
                else "😞"
            )

            color = (
                "#52b788"
                if prediction.lower()
                == "positive"
                else "#e63946"
            )

            # Result card
            st.markdown("---")

            st.markdown(
                f"""
                <h2 style="
                    text-align:center;
                    color:{color};
                ">
                    {prediction} {emoji}
                </h2>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <p style="
                    text-align:center;
                    font-size:20px;
                    color:#f0ece4;
                ">
                    Confidence:
                    <strong>
                        {confidence:.2f}%
                    </strong>
                </p>
                """,
                unsafe_allow_html=True
            )

            st.write("### Positive 😊")
            st.progress(
                min(
                    float(p_pos / 100),
                    1.0
                )
            )

            st.caption(
                f"{p_pos:.2f}%"
            )

            st.write("### Negative 😞")
            st.progress(
                min(
                    float(p_neg / 100),
                    1.0
                )
            )

            st.caption(
                f"{p_neg:.2f}%"
            )