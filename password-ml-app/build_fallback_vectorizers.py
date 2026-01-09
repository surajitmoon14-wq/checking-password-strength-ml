from pathlib import Path
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / 'Password Strength.csv'

# Read up to N passwords for training
N = 50000
passwords = []
with open(CSV, newline='') as f:
    reader = csv.reader(f)
    header = next(reader)
    # assume first column is password
    for i, row in enumerate(reader):
        if i >= N:
            break
        if row:
            passwords.append(row[0])

if not passwords:
    raise SystemExit('No passwords found in CSV')

# Create vectorizers
vec_char_13 = TfidfVectorizer(analyzer='char', ngram_range=(1,3))
vec_char_13.fit(passwords)

# Save vectorizers in places where loader will prefer resaved files
candidates = [
    ROOT / 'tfidf_vectorizer_resaved.pkl',
    ROOT / 'tfidf_vectorizer_1_resaved.pkl',
    ROOT / 'tfidf_vectorizer (2)_resaved.pkl',
    ROOT / 'models' / 'new_vectorizer_resaved.pkl',
    ROOT / 'models' / 'length_vectorizer_resaved.pkl',
]
for p in candidates:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(vec_char_13, p)
        print('wrote', p)
    except Exception as e:
        print('skip', p, e)
