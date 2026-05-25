# 🎬 Movie Review Sentiment Analyzer — NLP-Based Sentiment Classification System

An NLP-powered web application that predicts movie review sentiment using traditional Machine Learning and Natural Language Processing techniques. Users can enter a movie review and instantly receive sentiment prediction, confidence score, and probability visualization through a modern Streamlit interface.

This project demonstrates an end-to-end NLP + Machine Learning pipeline using TF-IDF vectorization and Logistic Regression with a deployment-ready UI.

---

## 🌟 Features

- **🎭 Movie Review Sentiment Prediction** – Predict whether a review is Positive or Negative  
- **🧠 NLP Preprocessing Pipeline** – Tokenization, stopword removal, lemmatization, and text cleaning  
- **📊 Sentiment Confidence Score** – Displays prediction confidence percentage  
- **📈 Probability Visualization** – Positive vs Negative probability bars  
- **🎬 Interactive Streamlit UI** – Modern dark-themed movie-style interface  
- **⚡ Real-Time Prediction** – Instant review analysis  
- **🧩 Modular NLP Architecture** – Clean separation of preprocessing, feature engineering, training, and prediction  
- **📁 GitHub-Ready Structure** – Professional project organization  

---

## 📂 Project Structure

```txt
MOVIE-SENTIMENT-ANALYZER/
│
├── data/
│   ├── raw/
│   │   └── imdb_reviews.csv
│   │
│   └── processed/
│       └── cleaned_reviews.csv
│
├── models/
│   ├── sentiment_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoder.pkl
│   └── model_metrics.json
│
├── src/
│   ├── data_preprocessing.py
│   ├── text_cleaning.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── prediction.py
│
├── app.py
├── train.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git
- pip

---

### 1️⃣ Clone the Repository

### Option 1: Clone with Git

```bash
git clone <repository-url>
```

### Option 2: Download and extract the ZIP file

---

### 2️⃣ Install Dependencies

### Install all required Python packages

```bash
pip install -r requirements.txt
```

---

### 3️⃣ 📥 Dataset Setup

This project uses the **IMDb Movie Reviews Dataset**.

If the dataset is not included in the repository:

1. Download the IMDb Dataset from Kaggle.

2. Rename the file:

```txt
imdb_reviews.csv
```

3. Place the file inside:

```txt
data/raw/imdb_reviews.csv
```

4. Make sure the dataset contains:

```txt
review
sentiment
```

columns.

---

### 4️⃣ Train the Model

```bash
python train.py
```

This will generate:

- `sentiment_model.pkl`
- `tfidf_vectorizer.pkl`
- `label_encoder.pkl`
- `model_metrics.json`

inside the `models/` folder.

---

### 5️⃣ Run the Application

### Streamlit Frontend

```bash
streamlit run app.py
```

The app will open in your browser at:

```txt
http://localhost:8501
```

---

## 🧠 How It Works

1. User enters a movie review in the input area

2. NLP preprocessing is applied:

   - Lowercasing

   - Punctuation removal

   - Number removal

   - Tokenization

   - Stopword removal

   - Lemmatization

3. Cleaned text is converted into numerical features using **TF-IDF Vectorization**

4. **Logistic Regression model** predicts sentiment

5. Results are displayed as:

   - Sentiment label (Positive / Negative)

   - Confidence score

   - Probability visualization

---

## 🛠 Technical Details

### Technologies Used

- **Frontend**: Streamlit (Python Web Framework)

- **Machine Learning**: Scikit-learn (Logistic Regression)

- **NLP**: NLTK (Tokenization, Stopwords, Lemmatization)

- **Feature Extraction**: TF-IDF Vectorizer

- **Data Handling**: Pandas, NumPy

- **Model Persistence**: Joblib

- **Language**: Python 3.8+

---

## 🔧 Key Components

### 1. `src/data_preprocessing.py`

Data preprocessing module — Handles dataset loading, missing values, duplicate removal, text preprocessing, and saves cleaned data.

### 2. `src/text_cleaning.py`

NLP preprocessing module — Performs lowercase conversion, punctuation removal, tokenization, stopword removal, and lemmatization.

### 3. `src/feature_engineering.py`

Feature engineering module — Converts cleaned text into TF-IDF vectors and saves vectorizer.

### 4. `src/model_training.py`

Model training module — Trains Logistic Regression model, evaluates performance (Accuracy, Precision, Recall, F1-score), and saves trained artifacts.

### 5. `src/prediction.py`

Prediction module — Loads saved model and vectorizer, preprocesses user input, predicts sentiment, and returns confidence score.

### 6. `train.py`

Training pipeline entry script — Executes complete NLP workflow from preprocessing to model saving.

### 7. `app.py`

Streamlit frontend — Dark movie-themed UI with sentiment prediction, confidence score, and probability bars.

### 8. `models/sentiment_model.pkl`

Serialized trained machine learning model used for prediction.

### 9. `models/tfidf_vectorizer.pkl`

Saved TF-IDF vectorizer for transforming text input during prediction.

### 10. `models/model_metrics.json`

Stores model evaluation metrics.

---

## ⚠️ Troubleshooting

## Common Issues

### 1. **"Model file not found" error**

Make sure you trained the model before running the app:

```bash
python train.py
```

Ensure these files exist:

```txt
models/
```

- sentiment_model.pkl  
- tfidf_vectorizer.pkl  
- label_encoder.pkl  
- model_metrics.json  

---

### 2. **"Module not found" error**

Install dependencies:

```bash
pip install -r requirements.txt
```

Also create:

```txt
src/__init__.py
```

to avoid import issues.

---

### 3. **NLTK resource error**

Download required NLTK packages:

```python
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

---

### 4. **Streamlit app not opening**

Try manually opening:

```txt
http://localhost:8501
```

Or restart:

```bash
streamlit run app.py
```

---

### 5. **Incorrect sentiment prediction**

- Ensure dataset format is correct
- Retrain model if preprocessing changes were made
- Check if `imdb_reviews.csv` is placed correctly

---

### 6. **Prediction confidence too high**

This project includes confidence calibration for more realistic sentiment scores in mixed reviews.

---

## 🚀 Future Enhancements

🎭 Multi-class sentiment classification (Positive / Neutral / Negative)

🧠 Advanced NLP models (LSTM / Transformers)

🌐 Streamlit Cloud deployment

📊 Sentiment analytics dashboard

📈 Batch review analysis

🎬 Movie recommendation integration

🔐 User authentication system

---

## 📞 Support

If you encounter any issues or have questions:
    Phone Number : +91 9063197036
    Email : bhavanibhavya77@gmail.com