# Challenge 2 - Fake News Detection
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from lightgbm import LGBMClassifier
import os
import sys

# Ensure preprocess.py can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import clean_text

# --- CONFIGURATION ---
TRAIN_PATH = "../data/challenge_2/fakenews_with_labels.csv"
TEST_PATH = "../data/challenge_2/FakeNews_no_labels.csv"
OUTPUT_PATH = "../outputs/FakeNews_no_labels.csv"

def main():
    print("Starting Challenge 2 Pipeline (Fake News Detection)...")

    # 1. Load Data
    print("Loading datasets...")
    try:
        df_train = pd.read_csv(TRAIN_PATH, encoding='latin-1', on_bad_lines='skip')
        df_test = pd.read_csv(TEST_PATH, encoding='latin-1', on_bad_lines='skip')
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check your file paths.")
        return

    # 2. Force Clean Column Names
    # Bypassing the 'title' artifact by explicitly renaming all 5 known columns
    EXPECTED_COLS = ['title', 'text', 'subject', 'date', 'label']
    df_train.columns = EXPECTED_COLS
    df_test.columns = EXPECTED_COLS

    # 3. Standardize and Map Labels
    print("Mapping labels...")
    # Convert any parsed booleans or misformatted strings to standard uppercase strings
    df_train['label'] = df_train['label'].astype(str).str.strip().str.upper()
    
    # Drop rows where the target might be missing or corrupted
    df_train = df_train[df_train['label'].isin(['TRUE', 'FALSE'])]
    
    LABEL_MAPPING = {"FALSE": 0, "TRUE": 1}
    REVERSE_MAPPING = {0: "FALSE", 1: "TRUE"}
    df_train['target_num'] = df_train['label'].map(LABEL_MAPPING)

    # 4. Feature Engineering (Combine Title + Text)
    print("Combining features and cleaning text (this may take a moment)...")
    df_train['title'] = df_train['title'].fillna('')
    df_train['text'] = df_train['text'].fillna('')
    df_test['title'] = df_test['title'].fillna('')
    df_test['text'] = df_test['text'].fillna('')

    # Fake news detection is highly dependent on sensationalist titles combined with the body
    df_train['combined_text'] = df_train['title'] + " " + df_train['text']
    df_test['combined_text'] = df_test['title'] + " " + df_test['text']

    df_train['cleaned_text'] = df_train['combined_text'].apply(clean_text)
    df_test['cleaned_text'] = df_test['combined_text'].apply(clean_text)

    # 5. Train/Validation Split
    X_train, X_val, y_train, y_val = train_test_split(
        df_train['cleaned_text'], 
        df_train['target_num'], 
        test_size=0.2, 
        random_state=42,
        stratify=df_train['target_num']
    )

    # 6. Feature Extraction (TF-IDF)
    print("Vectorizing combined text...")
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(df_test['cleaned_text'])

    # 7. Model Training
    print("Training LightGBM Classifier...")
    model = LGBMClassifier(
        n_estimators=300, 
        learning_rate=0.1, 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train_vec, y_train)

    # 8. Evaluation (Overall Accuracy)
    print("\n--- Validation Results ---")
    val_preds = model.predict(X_val_vec)
    accuracy = accuracy_score(y_val, val_preds)
    
    print(f"Overall Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_val, val_preds, target_names=["FALSE (Fake)", "TRUE (Real)"]))
    print("--------------------------\n")

    # 9. Final Predictions & Strict Reverse Mapping
    print("Generating final predictions...")
    test_preds_num = model.predict(X_test_vec)
    
    # Ensure exact string match for submission requirements
    df_test['label'] = [REVERSE_MAPPING[pred] for pred in test_preds_num]
    
    # Strip out the processing columns to maintain pristine original structure
    df_test = df_test.drop(columns=['combined_text', 'cleaned_text'])
    
    # 10. Export
    os.makedirs("../outputs", exist_ok=True)
    df_test.to_csv(OUTPUT_PATH, index=False)
    print(f"Success! Predictions saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
