import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
df = pd.read_csv("data/challenge_2/fakenews_with_labels.csv", encoding='latin-1', on_bad_lines='skip')
EXPECTED_COLS = ['title', 'text', 'subject', 'date', 'label']
df.columns = EXPECTED_COLS
df['label'] = df['label'].astype(str).str.strip().str.upper()
df = df[df['label'].isin(['TRUE', 'FALSE'])]
df['target_num'] = df['label'].map({"FALSE": 0, "TRUE": 1})

# 2. Prepare Features
df['combined'] = df['title'].fillna('') + " " + df['text'].fillna('')
print("Vectorizing...")
vec = TfidfVectorizer(max_features=10000) # Slightly lower for fast CV testing
X = vec.fit_transform(df['combined'])
y = df['target_num']

# 3. Cross Validation
print("Running 5-Fold Cross Validation...")
model = LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1)
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"✅ 5-Fold CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
