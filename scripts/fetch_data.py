"""Download the Fine Foods reviews from SNAP and write data/Reviews.csv in the Kaggle column layout.

Run from the repo root:  python scripts/fetch_data.py
Downloads finefoods.txt.gz (116 MB) from https://snap.stanford.edu/data/web-FineFoods.html and converts it to
the 10 columns of the Kaggle "Amazon Fine Food Reviews" CSV (568,454 rows, about 300 MB). Skip this if you
already have Reviews.csv from Kaggle; put it in data/ instead.
"""
import csv
import gzip
import os
import urllib.request

URL = "https://snap.stanford.edu/data/finefoods.txt.gz"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
COLUMNS = ["Id", "ProductId", "UserId", "ProfileName", "HelpfulnessNumerator", "HelpfulnessDenominator",
           "Score", "Time", "Summary", "Text"]
KEYS = {"product/productId": "ProductId", "review/userId": "UserId", "review/profileName": "ProfileName",
        "review/helpfulness": "helpfulness", "review/score": "Score", "review/time": "Time",
        "review/summary": "Summary", "review/text": "Text"}


def records(path):
    rec = {}
    with gzip.open(path, "rb") as f:
        for raw in f:
            line = raw.decode("latin-1").rstrip("\n")
            if not line.strip():
                if rec:
                    yield rec
                rec = {}
                continue
            key, _, value = line.partition(": ")
            if key in KEYS:
                rec[KEYS[key]] = value
    if rec:
        yield rec


def main():
    os.makedirs(DATA, exist_ok=True)
    gz = os.path.join(DATA, "finefoods.txt.gz")
    if not os.path.exists(gz):
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, gz)
    out = os.path.join(DATA, "Reviews.csv")
    n = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for n, rec in enumerate(records(gz), start=1):
            num, den = rec.get("helpfulness", "0/0").split("/")
            writer.writerow([n, rec.get("ProductId", ""), rec.get("UserId", ""), rec.get("ProfileName", ""),
                             int(num), int(den), int(float(rec.get("Score", "0"))), int(rec.get("Time", "0")),
                             rec.get("Summary", ""), rec.get("Text", "")])
    print(f"wrote {out} with {n:,} reviews")


if __name__ == "__main__":
    main()
