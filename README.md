# 🎬 Movie Genre Classification

A Python-based machine learning application that predicts the genre of a movie based on its plot or description using a **Linear Support Vector Classifier (LinearSVC)** and **TF-IDF vectorization**.  
The project also features a simple **Tkinter GUI** for easy interaction.

---

## ✨ Features

- Trains a **LinearSVC** model on movie plot descriptions to classify genres.  
- Uses **TF-IDF vectorization** to convert text data into numerical features.  
- Supports **multi-class genre classification** with 25+ genres.  
- Provides a **Tkinter GUI** for user-friendly interaction.  
- Displays predictions with relevant **genre emojis** 🎭.  
- Saves & loads the trained model/vectorizer using `joblib`.  

---

## 📂 Dataset

The model is trained on the [IMDB Genre Classification Dataset](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb).  

---

## ⚙️ How It Works

1. **Data Parsing** → Reads and parses training & test files.  
2. **Data Cleaning** → Normalizes text (lowercasing, removing special characters, stopwords).  
3. **Feature Extraction** → Combines movie titles & descriptions, vectorizes with TF-IDF.  
4. **Model Training** → Trains a **LinearSVC** classifier on text features.  
5. **Prediction & Evaluation** → Predicts genres for test data and evaluates performance.  
6. **GUI** → Enter a movie description → get real-time genre predictions.  

---

## 📊 Accuracy

- **Model**: Linear Support Vector Classifier (LinearSVC)  
- **Accuracy**: ~58.33% on the test dataset.  

🔧 **Possible Improvements:**  
- Hyperparameter tuning (C, loss, penalty).  
- Trying **deep learning models** (e.g., LSTMs, BERT).  
- Using more advanced embeddings (e.g., Word2Vec, GloVe).  

---

## 🛠️ Requirements

- Python 3.7+  
- Libraries:  
  - pandas  
  - scikit-learn  
  - joblib  
  - tkinter (comes pre-installed with Python)  
  - re (regex)  
  - os  

---

## 📦 Installation

1. Clone the repository:  

   ```bash
   git clone https://github.com/pari006/movie-genre-classification.git
   cd movie-genre-classification
