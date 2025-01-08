import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load data (sampling for speed)
df = pd.read_csv("/kaggle/input/amazon-fine-food-reviews/Reviews.csv")

# Basic info
print("Shape:", df.shape)
print(df.head())

# Numeric columns
num_cols = ["Id", "HelpfulnessNumerator", "HelpfulnessDenominator", "Score", "Time"]

# Correlation matrix
corr_matrix = df[num_cols].corr()
plt.figure(figsize=(6,6))
sns.heatmap(corr_matrix, annot=False, cmap="viridis", square=True)
plt.title("Correlation Matrix for Reviews.csv")
plt.show()

# Pairplot
sns.set(style="whitegrid", context="notebook")
g = sns.pairplot(df[num_cols], kind="scatter", diag_kind="kde")
g.fig.suptitle("Scatter and Density Plot", y=1.02)
plt.show()

# Distribution plots for a few columns
plt.figure(figsize=(12, 4))
for i, col in enumerate(["HelpfulnessNumerator", "HelpfulnessDenominator", "Score"]):
    plt.subplot(1, 3, i+1)
    df[col].plot.hist(bins=20)
    plt.title(col)
    plt.xlabel(col)
plt.tight_layout()
plt.show()

# Create a binary sentiment label
df["Sentiment"] = df["Score"].apply(lambda x: 1 if x > 3 else 0)
df.dropna(subset=["Text"], inplace=True)
df.reset_index(inplace=True, drop=False)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[{}]".format(re.escape(string.punctuation)), "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["cleaned_text"] = df["Text"].apply(clean_text)
X = df["cleaned_text"]
y = df["Sentiment"]
row_ids = df["index"]

# Train/test split
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, row_ids, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words="english"
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(
    max_iter=300,
    solver="saga",
    random_state=42
)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# Generate a submission-like file for the test split
submission = pd.DataFrame({
    "RowID": idx_test,
    "PredictedSentiment": y_pred
})
submission.sort_values("RowID", inplace=True)
submission.to_csv("submission.csv", index=False)
print("Created submission.csv with predicted sentiments.")
