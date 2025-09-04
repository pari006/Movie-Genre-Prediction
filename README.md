# 🎬 Movie Genre Classification

A Python-based machine learning application that predicts the **genre of a movie** based on its plot or description.  
The project uses a **Linear Support Vector Classifier (LinearSVC)** with **TF-IDF vectorization** and provides a **Tkinter GUI** for easy interaction.

---

## ✨ Features

- 📖 Trains a **LinearSVC model** on movie plot descriptions.  
- 🔎 Uses **TF-IDF vectorization** to convert text into numerical features.  
- 🎭 Supports **multi-class genre classification** with 25+ genres.  
- 🖥️ **Tkinter GUI** to input movie descriptions and get instant predictions.  
- 😀 Shows predictions with **relevant emojis** for better visualization.  
- 💾 Saves & loads trained model/vectorizer using `joblib`.  
- 📊 Achieves **~58% accuracy** on the test dataset.  

---

## 📂 Dataset

- The model is trained on the **[IMDB Genre Classification Dataset](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)** from Kaggle.  
- Dataset includes **movie titles, descriptions, and genres**.  
- Preprocessing includes cleaning, normalization, and combining title + description.

---

## ⚙️ How It Works

1. **Data Parsing** → Reads training & test files.  
2. **Data Cleaning** → Lowercasing, removing special characters, etc.  
3. **Feature Extraction** → TF-IDF vectorization on text data.  
4. **Model Training** → Trains a `LinearSVC` classifier.  
5. **Evaluation** → Predicts test genres & computes accuracy.  
6. **GUI Application** → Tkinter app where users can input movie descriptions and get predictions in real time.  

---

## 🛠️ Requirements

- Python **3.7+**  
- pandas  
- scikit-learn  
- joblib  
- tkinter (comes pre-installed with Python)  
- re (regular expressions)  
- os  

Install dependencies:  

```bash
pip install pandas scikit-learn joblib
