# Movie Genre Classification

A Python-based machine learning application that predicts the genre of a movie based on its plot or description using a Linear Support Vector Classifier (LinearSVC) and TF-IDF vectorization. The project includes a simple graphical user interface (GUI) built with Tkinter for easy interaction.

---

## Features

- Trains a LinearSVC model on movie plot descriptions to classify genres.
- Uses TF-IDF vectorization to convert text data into numerical features.
- Supports multi-class genre classification with 25+ genres.
- Provides a user-friendly Tkinter GUI to input movie descriptions and get genre predictions.
- Displays genre predictions along with relevant emojis for better visualization.
- Saves and loads the trained model and vectorizer using `joblib`.
- Achieves approximately 58% accuracy on the test dataset.

---

## Dataset

The model is trained on the [IMDB Genre Classification Dataset](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb) from Kaggle.

---

## How It Works

1. **Data Parsing:** Reads and parses training, test, and solution files.
2. **Data Cleaning:** Cleans and normalizes text data (lowercasing, removing special characters).
3. **Feature Extraction:** Combines movie titles and descriptions, then vectorizes using TF-IDF.
4. **Model Training:** Trains a LinearSVC classifier on the vectorized text.
5. **Prediction & Evaluation:** Predicts genres for test data and evaluates accuracy.
6. **GUI:** Allows users to input movie descriptions and get real-time genre predictions.

---

## Requirements

- Python 3.7+
- pandas
- scikit-learn
- joblib
- tkinter (usually included with Python)
- re (regular expressions)
- os

---

## Accuracy

The model achieves approximately 58.33% accuracy on the test dataset.

---

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/pari006/movie-genre-classification.git
   cd movie-genre-classification
