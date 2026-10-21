"""Re-run the sentiment pipeline from amazon_reviews_sentiment.py and rebuild the README figures.

Run from the repo root with Reviews.csv in data/:  python scripts/make_figures.py
Uses all 568,454 reviews, the same cleaning, split, TF-IDF settings and logistic regression as the main
script. Takes about 4 minutes and a few GB of RAM, mostly in the bigram vectorizer. Writes PNGs to docs/assets/.
"""
import os
import re
import string
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             precision_recall_curve, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "Reviews.csv")
OUT = os.path.join(ROOT, "docs", "assets")
os.makedirs(OUT, exist_ok=True)

BG, PANEL = "#0d0c0b", "#131211"
INK, INK2, INK3 = "#ede7de", "#aaa299", "#7a736c"
GRID = "#24221f"
EMBER, TEMPER, PURPLE, GREEN = "#d95926", "#3987e5", "#9085e9", "#199e70"
SERIES = [EMBER, TEMPER, PURPLE, GREEN, "#e8b44c", "#c0509c"]
HEAT = LinearSegmentedColormap.from_list("heat", ["#141110", "#2a1410", "#4b170c", "#7a1f0a", "#ad320c",
                                                  "#d9520f", "#f07a1e", "#f9a63b", "#fccf6b", "#fff0c2"])
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "semibold",
    "axes.titlesize": 12.5, "axes.titlelocation": "left", "axes.titlepad": 12, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "xtick.color": INK3, "ytick.color": INK3, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6, "ytick.major.pad": 6,
    "axes.axisbelow": True, "font.family": ["Segoe UI", "DejaVu Sans"], "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "legend.frameon": False, "legend.fontsize": 9.5, "legend.labelcolor": INK2,
    "figure.dpi": 110, "savefig.dpi": 200,
})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} KB)")


def caption(fig, text):
    fig.text(0.012, -0.015, text, color=INK3, fontsize=8.5, ha="left", va="top")


# same cleaning as amazon_reviews_sentiment.py
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[{}]".format(re.escape(string.punctuation)), "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    t0 = time.time()
    df = pd.read_csv(DATA, usecols=["Score", "Text"])  # the model only uses these two columns
    df["Sentiment"] = df["Score"].apply(lambda x: 1 if x > 3 else 0)
    df.dropna(subset=["Text"], inplace=True)
    df.reset_index(inplace=True, drop=False)
    df["cleaned_text"] = df["Text"].apply(clean_text)
    df["words"] = df["Text"].str.split().str.len()
    df.drop(columns="Text", inplace=True)  # keeps peak memory down during vectorizing
    idx_train, idx_test = train_test_split(df.index, test_size=0.2, random_state=42, stratify=df["Sentiment"])
    y_train, y_test = df.loc[idx_train, "Sentiment"], df.loc[idx_test, "Sentiment"]

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
    X_train = vectorizer.fit_transform(df.loc[idx_train, "cleaned_text"])
    X_test = vectorizer.transform(df.loc[idx_test, "cleaned_text"])
    print(f"vectorized in {time.time() - t0:.0f} s")
    model = LogisticRegression(max_iter=300, solver="saga", random_state=42)
    model.fit(X_train, y_train)
    print(f"trained in {time.time() - t0:.0f} s, iterations {model.n_iter_[0]}")
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, pred)
    p, r, f, s = precision_recall_fscore_support(y_test, pred)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    print(f"reviews {len(df)}  positive {int(df.Sentiment.sum())}  negative {int((df.Sentiment == 0).sum())}")
    print(f"train {len(idx_train)}  test {len(idx_test)}")
    print(f"accuracy {acc:.4f}")
    for lab, a, b, c, n in zip(["negative", "positive"], p, r, f, s):
        print(f"  {lab:9s} P {a:.4f}  R {b:.4f}  F1 {c:.4f}  n {n}")
    print("confusion (rows actual neg/pos):", cm.tolist())
    ap_neg = average_precision_score(1 - y_test, 1 - proba)
    print(f"average precision, negative class: {ap_neg:.4f}")

    test = df.loc[idx_test].assign(pred=pred)
    by_star = test.groupby("Score")["pred"].agg(["mean", "size"])
    for star, row in by_star.iterrows():
        print(f"  {star} stars: {int(row['size']):6d} test reviews, predicted positive {row['mean']:.3f}")
    seen = set(df.loc[idx_train, "cleaned_text"])
    in_train = test["cleaned_text"].isin(seen).values
    print(f"test reviews whose cleaned text also appears in train: {in_train.sum()} ({in_train.mean():.1%})")
    print(f"accuracy on the {(~in_train).sum()} unseen test reviews: {accuracy_score(y_test[~in_train], pred[~in_train]):.4f}")
    no3 = test["Score"].values != 3
    print(f"accuracy without 3-star reviews: {accuracy_score(y_test[no3], pred[no3]):.4f}")
    wrong = pred != y_test.values
    print(f"test errors {wrong.sum()}, of which on 3-star reviews: {(wrong & ~no3).sum()}")
    prec_c, rec_c, _ = precision_recall_curve(1 - y_test, 1 - proba)
    for target in (0.7, 0.8, 0.9):
        print(f"  negative class: best precision at recall >= {target}: {prec_c[rec_c >= target].max():.3f}")

    # hero: confusion matrix + what the model says for each star rating
    fig = plt.figure(figsize=(12.5, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3], wspace=0.3)
    ax = fig.add_subplot(gs[0])
    ax.imshow(cm, cmap=HEAT, vmin=0, vmax=cm.max(), aspect="auto")
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=15, fontweight="semibold",
                    color=BG if v > cm.max() * 0.55 else INK)
    ax.set_xticks([0, 1], ["negative", "positive"])
    ax.set_yticks([0, 1], ["negative\n(1 to 3 stars)", "positive\n(4 to 5 stars)"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"{len(idx_test):,} held-out reviews · accuracy {acc:.3f}")

    ax = fig.add_subplot(gs[1])
    stars = by_star.index.values
    share_pos = by_star["mean"].values
    ax.bar(stars, 1 - share_pos, 0.62, color=TEMPER, label="predicted negative")
    ax.bar(stars, share_pos, 0.62, bottom=1 - share_pos, color=EMBER, label="predicted positive")
    for st, sp in zip(stars, share_pos):
        ax.text(st, 1.02, f"{sp:.0%} pos", ha="center", color=INK2, fontsize=9)
    ax.set_xticks(stars, [f"{st} star" + ("s" if st > 1 else "") for st in stars])
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("share of test reviews")
    ax.set_title("Predictions by the review's star rating")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=2)
    fig.text(0.012, -0.075, "Stratified 80/20 split (random_state=42). Label = 1 if Score > 3, so 3-star reviews "
             "count as negative during training and evaluation.", color=INK3, fontsize=8.5, ha="left", va="top")
    save(fig, "hero.png")

    # data: star distribution + length by label
    words = df["words"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.6], "wspace": 0.25})
    ax = axes[0]
    counts = df["Score"].value_counts().sort_index()
    bars = ax.bar(counts.index, counts.values, 0.62, color=[TEMPER, TEMPER, TEMPER, EMBER, EMBER])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 5000, f"{v:,}", ha="center", color=INK2, fontsize=9)
    ax.set_xticks(counts.index, [f"{i}" for i in counts.index])
    ax.set_xlabel("star rating")
    ax.set_ylim(0, counts.max() * 1.13)
    ax.set_ylabel("reviews")
    ax.set_title(f"Star ratings, {len(df):,} reviews")
    ax.legend(handles=[Patch(color=TEMPER, label="labelled negative"), Patch(color=EMBER, label="labelled positive")],
              loc="upper left")
    ax = axes[1]
    bins = np.arange(0, 401, 8)
    for lab, name, col in [(0, "negative", TEMPER), (1, "positive", EMBER)]:
        w = words[df.Sentiment == lab]
        ax.hist(w.clip(upper=400), bins=bins, density=True, color=col, alpha=0.55,
                label=f"{name} (median {int(w.median())} words)")
    ax.set_xlabel("words per review (clipped at 400)")
    ax.set_ylabel("share of class")
    ax.set_title("Review length by label")
    ax.legend(loc="upper right")
    save(fig, "data.png")

    # tokens
    names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]
    order = np.argsort(coef)
    print("top negative:", ", ".join(names[order[:15]]))
    print("top positive:", ", ".join(names[order[-15:]][::-1]))
    n = 14
    idx = np.concatenate([order[:n], order[-n:]])
    vals = coef[idx]
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    yy = np.arange(len(idx))
    ax.barh(yy, vals, color=[TEMPER if v < 0 else EMBER for v in vals], height=0.72)
    ax.set_yticks(yy, names[idx])
    ax.axvline(0, color=INK3, lw=0.8)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("logistic regression weight")
    ax.set_title("Strongest weights, unigrams and bigrams")
    ax.legend(handles=[Patch(color=EMBER, label="toward positive"), Patch(color=TEMPER, label="toward negative")],
              loc="upper left")
    save(fig, "tokens.png")

    # precision-recall for the minority (negative) class
    prec, rec, thr = precision_recall_curve(1 - y_test, 1 - proba)
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.plot(rec, prec, color=TEMPER, lw=2)
    ax.scatter([r[0]], [p[0]], s=60, color=EMBER, edgecolor=INK, zorder=5)
    ax.annotate(f"default 0.5 threshold\nprecision {p[0]:.2f}, recall {r[0]:.2f}", (r[0], p[0]),
                xytext=(0.12, 0.55), color=INK2, fontsize=9, arrowprops=dict(arrowstyle="-", color=INK3, lw=0.8))
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="x")
    ax.set_xlabel("recall, negative reviews")
    ax.set_ylabel("precision, negative reviews")
    ax.set_title(f"Finding the negative reviews · average precision {ap_neg:.3f}")
    save(fig, "pr_curve.png")
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
