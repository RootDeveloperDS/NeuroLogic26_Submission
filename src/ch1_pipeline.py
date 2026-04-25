# Challenge 1 - Disaster Response Classification
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from lightgbm import LGBMClassifier
import os
import sys

# Ensure preprocess.py can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import clean_text

# --- CONFIGURATION ---
TRAIN_PATH = "../data/challenge_1/Disaster_with_label.csv"
TEST_PATH = "../data/challenge_1/Disaster_no_label.csv"
OUTPUT_PATH = "../outputs/Disaster_no_label.csv"

def main():
    print("Starting Challenge 1 Pipeline...")

    # 1. Load Data and Standardize Columns
    print("Loading datasets...")
    try:
        df_train = pd.read_csv(TRAIN_PATH)
        df_test = pd.read_csv(TEST_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check your file paths.")
        return

    # Strip leading/trailing spaces from column names to prevent KeyErrors
    df_train.columns = df_train.columns.str.strip()
    df_test.columns = df_test.columns.str.strip()

    TEXT_COL = 'Tweet Text'
    LABEL_COL = 'label'

    # 2. Map Target Strings to Binary for Training
    print("Mapping labels...")
    label_mapping = {"Not Informative": 0, "Informative": 1}
    reverse_mapping = {0: "Not Informative", 1: "Informative"}
    
    # Drop rows where the label is missing in the training set
    df_train = df_train.dropna(subset=[LABEL_COL])
    df_train['target_num'] = df_train[LABEL_COL].map(label_mapping)

    # 3. Preprocess Text
    print("Cleaning text data...")
    df_train['cleaned_text'] = df_train[TEXT_COL].apply(clean_text)
    df_test['cleaned_text'] = df_test[TEXT_COL].apply(clean_text)

    # 4. Train/Validation Split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        df_train['cleaned_text'], 
        df_train['target_num'], 
        test_size=0.2, 
        random_state=42,
        stratify=df_train['target_num']
    )

    # 5. Feature Extraction (TF-IDF)
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(df_test['cleaned_text'])

    
    # 6. Model Training (Ensemble Architecture)
    print("Training Ensemble Classifier (LightGBM + LogReg + MNB)...")
    
    # Model A: LightGBM
    lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight='balanced', random_state=42, n_jobs=-1)
    
    # Model B: Logistic Regression (Excellent for sparse TF-IDF matrices)
    logreg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    
    # Model C: Naive Bayes (Standard baseline for NLP, fast and reliable)
    mnb = MultinomialNB()

    # Combine models. 'soft' voting uses predicted probabilities for better accuracy.
    ensemble = VotingClassifier(
        estimators=[('lgbm', lgbm), ('lr', logreg), ('mnb', mnb)],
        voting='soft'
    )
    
    ensemble.fit(X_train_vec, y_train)

    # 7. Evaluation (Macro F1-Score)
    print("\n--- Validation Results ---")
    val_preds = ensemble.predict(X_val_vec)
    macro_f1 = f1_score(y_val, val_preds, average='macro')
    
    print(f"Macro F1-Score: {macro_f1:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_val, val_preds, target_names=["Not Informative", "Informative"]))
    print("--------------------------\n")


    # 8. Final Predictions & Reverse Mapping
    print("Generating final predictions...")
    test_preds_num = ensemble.predict(X_test_vec)
    
    # Convert 0/1 back to exact original strings
    df_test[LABEL_COL] = [reverse_mapping[pred] for pred in test_preds_num]
    
    # Drop temporary processing columns to match original structure exactly
    df_test = df_test.drop(columns=['cleaned_text'])
    
    # 9. Export Data
    os.makedirs("../outputs", exist_ok=True)
    df_test.to_csv(OUTPUT_PATH, index=False)
    print(f"Success! Strict string-formatted predictions saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
