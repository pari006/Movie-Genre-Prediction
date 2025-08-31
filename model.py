import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import re
import joblib
import os

# ---------------------------
# 1. Clean text function
# ---------------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# ---------------------------
# 2. Load model & vectorizer
# ---------------------------
# 👇 Train once in your notebook, then save with joblib:
# joblib.dump(model, "genre_model.pkl")
# joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Replace with your actual file path
BASE_DIR = r"D:\Projects\Movie Genre Classification"

model = joblib.load(os.path.join(BASE_DIR, "genre_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))

def get_genre_emoji(genre):
    """Return appropriate emoji based on the predicted genre"""
    genre = genre.lower()
    emoji_mapping = {
        'drama': '🎭',            # Drama masks emoji
        'thriller': '🔪',         # Knife emoji for thriller
        'adult': '🍑',            # Peach emoji for adult content
        'documentary': '🎥',      # Movie camera emoji
        'comedy': '🤡',           # Clown/joker emoji for comedy
        'crime': '🚔',            # Police car emoji for crime
        'reality-tv': '📺',       # Television emoji for reality TV
        'horror': '👻',           # Ghost emoji for horror
        'sport': '⚽',            # Soccer ball emoji for sports
        'animation': '🎬',        # Clapper board emoji for animation
        'action': '💥',           # Explosion emoji for action
        'fantasy': '🐉',          # Dragon emoji for fantasy
        'short': '⏳',            # Hourglass emoji for short films
        'sci-fi': '👽',           # Alien emoji for sci-fi
        'music': '🎵',            # Music note emoji
        'adventure': '🗺️',        # Map emoji for adventure
        'talk-show': '🎤',        # Microphone emoji for talk shows
        'western': '🤠',          # Cowboy emoji
        'family': '👨‍👩‍👧‍👦',            # Family emoji
        'mystery': '🔍',          # Magnifying glass emoji for mystery
        'history': '🏛️',          # Classical building emoji
        'news': '📰',             # Newspaper emoji for news
        'biography': '📖',        # Book emoji for biography
        'romance': '💕',          # Hearts emoji for romance
        'game-show': '🎲',        # Game die emoji for game shows
        'musical': '🎶',          # Musical notes emoji
        'war': '⚔️',              # Crossed swords emoji for war
    }
    return emoji_mapping.get(genre, '🎬')  # Default to clapper board if genre not found

def predict_genre():
    """Function to predict the genre based on user input."""
    user_input = text_area.get("1.0", tk.END).strip()
    if not user_input:
        messagebox.showwarning("Warning", "Please enter a movie description first.")
        return
    
    cleaned = clean_text(user_input)
    X = vectorizer.transform([cleaned])
    prediction = model.predict(X)[0]
    # Get emoji for the predicted genre
    genre_emoji = get_genre_emoji(prediction)
    # Update result label with genre and emoji
    result_label.config(text=f"Predicted Genre: {prediction.title()} {genre_emoji}") 

# Create main window
root = tk.Tk()
root.title("Movie Genre Classifier")
root.geometry("700x650")
root.resizable(False, False)
root.configure(background='light cyan')

# Header
header_frame = ttk.Frame(root)
header_frame.pack(pady=10)
title_label = ttk.Label(header_frame, text="🎬 Movie Genre Classifier", font=('Algerian', 24), foreground="dark orange", background='light cyan')
title_label.pack()
subtitle_label = ttk.Label(header_frame, text="Enter a movie plot or description",font = ('Lucida Calligraphy',18), foreground="coral", background='light cyan')
subtitle_label.pack(pady=5)

# Text Input
text_area = tk.Text(root, height=10, width=35, 
                    font=('Comic Sans MS', 18), 
                    borderwidth=6, 
                    relief="ridge", 
                    highlightthickness=3, 
                    highlightbackground="lightgoldenrodyellow",
                    highlightcolor="light coral")
text_area.pack(pady=10)

# Predict Button
style = ttk.Style()
style.configure("Custom.TButton", font=('Monotype Corsiva', 22), foreground = "medium violet red")
predict_button = ttk.Button(root, text="Predict Genre", command=predict_genre, style="Custom.TButton")
predict_button.pack(pady=5)


# Result Display
result_label = ttk.Label(root, text="", font=('Modern No. 20', 22), foreground="chocolate2", background='light cyan')
result_label.pack(pady=10)

# Footer with accuracy information
footer_frame = ttk.Frame(root, padding=(10, 5))
footer_frame.pack(side=tk.BOTTOM, pady=10)
accuracy_label = ttk.Label(footer_frame, 
                          text="This model is 58.33% accurate", 
                          font=('MV Boli', 12), 
                          foreground="saddlebrown", background='light cyan')
accuracy_label.pack()

# Start the Tkinter event loop
root.mainloop()