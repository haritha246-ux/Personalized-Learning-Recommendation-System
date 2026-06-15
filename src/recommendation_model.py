import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

df = pd.read_csv("data/coursea_data.csv")

df["content"] = (
    df["course_title"].astype(str) + " " +
    df["course_organization"].astype(str) + " " +
    df["course_difficulty"].astype(str)
)

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["content"])

similarity = cosine_similarity(tfidf_matrix)

os.makedirs("models", exist_ok=True)

pickle.dump((similarity, df), open("models/similarity_model.pkl", "wb"))

print("Model trained and saved successfully")