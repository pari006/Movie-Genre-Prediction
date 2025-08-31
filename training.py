import pandas as pd
import re,os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import joblib

#Download dataset from here:
    #https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb
# ---------------------------
# 1. Parsing Functions
# ---------------------------
def parse_train_file(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ::: ", 3)
            if len(parts) == 4:
                rows.append(parts)
    return pd.DataFrame(rows, columns=["id", "title", "genre", "description"])

def parse_test_file(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ::: ", 2)
            if len(parts) == 3:
                rows.append(parts)
    return pd.DataFrame(rows, columns=["id", "title", "description"])

def parse_solution_file(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ::: ", 3)
            if len(parts) == 4:
                rows.append(parts)
    return pd.DataFrame(rows, columns=["id", "title", "genre_true", "description"])

# Replace with your actual file path
BASE_DIR = r"D:\Projects\Movie Genre Classification"
# ---------------------------
# 2. Load Data
# ---------------------------
train_file = os.path.join(BASE_DIR, "train_data.txt")
test_file = os.path.join(BASE_DIR, "test_data.txt")
solution_file = os.path.join(BASE_DIR, "test_data_solution.txt")

train_df = parse_train_file(train_file)
test_df = parse_test_file(test_file)
test_solution = parse_solution_file(solution_file)

# ---------------------------
# 3. Clean Data
# ---------------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

train_df["description"] = train_df["description"].astype(str).apply(clean_text)
test_df["description"] = test_df["description"].astype(str).apply(clean_text)

# Normalize genres & ids
train_df["genre"] = train_df["genre"].str.strip().str.lower()
test_solution["genre_true"] = test_solution["genre_true"].str.strip().str.lower()

train_df["id"] = train_df["id"].astype(str).str.strip()
test_df["id"] = test_df["id"].astype(str).str.strip()
test_solution["id"] = test_solution["id"].astype(str).str.strip()

# Drop duplicates & missing
train_df = train_df.drop_duplicates().dropna()
test_df = test_df.drop_duplicates().dropna()

# ---------------------------
# 4. Vectorization (use title+desc)
# ---------------------------
train_df["text"] = train_df["title"].astype(str) + " " + train_df["description"].astype(str)
test_df["text"] = test_df["title"].astype(str) + " " + test_df["description"].astype(str)

vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
X_train = vectorizer.fit_transform(train_df["text"])
y_train = train_df["genre"]
X_test = vectorizer.transform(test_df["text"])

# ---------------------------
# 5. Train Model
# ---------------------------
model = LinearSVC()
model.fit(X_train, y_train)

# ---------------------------
# 6. Predict
# ---------------------------
test_df["predicted_genre"] = model.predict(X_test)

# ---------------------------
# 7. Evaluate
# ---------------------------
results = test_df.merge(test_solution[["id", "genre_true"]], on="id", how="left")

accuracy = accuracy_score(results["genre_true"], results["predicted_genre"])
correct = (results["genre_true"] == results["predicted_genre"]).sum()

print(f"✅ Accuracy: {accuracy:.4f} ({correct}/{len(results)})")

# ---------------------------
# 8. Save Model & Vectorizer
# ---------------------------
joblib.dump(model, os.path.join(BASE_DIR, "genre_model.pkl"))
joblib.dump(vectorizer, os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
print("💾 Model and vectorizer saved successfully!")
