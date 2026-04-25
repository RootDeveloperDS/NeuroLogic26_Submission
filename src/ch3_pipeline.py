# Challenge 3 - Multilingual Toxicity Detection
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import os
import sys

# Ensure preprocess.py can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import clean_text

# --- CONFIGURATION ---
TRAIN_PATH = "../data/challenge_3/toxic_labeled.xlsx"
TEST_PATH = "../data/challenge_3/toxic_no_label_evaluation.xlsx"
OUTPUT_PATH = "../outputs/toxic_no_label_evaluation.csv" 

TEXT_COL = 'text'
LABEL_COL = 'label'

def main():
    print("Starting Challenge 3 Pipeline (Multilingual Toxicity)...")

    # 1. Load Data
    print("Loading datasets...")
    try:
        # Prioritizing Excel as per your initial folder structure
        df_train = pd.read_excel(TRAIN_PATH)
        df_test = pd.read_excel(TEST_PATH)
    except Exception as e:
        print(f"Notice: Could not read as Excel ({e}). Attempting CSV fallback...")
        df_train = pd.read_csv(TRAIN_PATH, encoding='utf-8')
        df_test = pd.read_csv(TEST_PATH, encoding='utf-8')

    # Drop any severely corrupted rows
    df_train = df_train.dropna(subset=[LABEL_COL, TEXT_COL])
    df_test[TEXT_COL] = df_test[TEXT_COL].fillna('')

    # Ensure labels are strictly integers (0 and 1)
    df_train[LABEL_COL] = df_train[LABEL_COL].astype(int)

    # 2. Preprocess Text
    print("Cleaning multilingual text data...")
    df_train['cleaned_text'] = df_train[TEXT_COL].apply(clean_text)
    df_test['cleaned_text'] = df_test[TEXT_COL].apply(clean_text)

    # 3. Train/Validation Split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        df_train['cleaned_text'], 
        df_train[LABEL_COL], 
        test_size=0.2, 
        random_state=42,
        stratify=df_train[LABEL_COL]
    )

    # 4. Feature Extraction (Character N-Grams for Multilingual Support)
    print("Vectorizing text (Optimized for Hindi + English)...")
    # 'char_wb' captures character sequences within word boundaries
    vectorizer = TfidfVectorizer(max_features=25000, analyzer='char_wb', ngram_range=(2, 5))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(df_test['cleaned_text'])

    # 5. Model Training (Ensemble Architecture)
    print("Training Ensemble Classifier (LightGBM + Logistic Regression)...")
    
    lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.1, random_state=42, n_jobs=-1)
    logreg = LogisticRegression(max_iter=1000, random_state=42)

    ensemble = VotingClassifier(
        estimators=[('lgbm', lgbm), ('lr', logreg)],
        voting='soft'
    )
    
    ensemble.fit(X_train_vec, y_train)

    # 6. Evaluation (ROC-AUC Score)
    print("\n--- Validation Results ---")
    # Soft voting allows us to directly extract predict_proba for ROC-AUC
    val_probs = ensemble.predict_proba(X_val_vec)[:, 1]
    val_preds = ensemble.predict(X_val_vec)
    
    roc_auc = roc_auc_score(y_val, val_probs)
    
    print(f"Mean ROC-AUC Score: {roc_auc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_val, val_preds, target_names=["0 (Non-toxic)", "1 (Toxic)"]))
    print("--------------------------\n")

    # 7. Final Predictions
    print("Generating strict binary predictions...")
    test_preds = ensemble.predict(X_test_vec)
    
    # 8. Export Data
    # Enforcing strict requested output
    df_test[LABEL_COL] = test_preds.astype(int)
    
    # Keeping only the essential columns to prevent structure violations
    df_submission = df_test[[TEXT_COL, LABEL_COL]]
    
    os.makedirs("../outputs", exist_ok=True)
    df_submission.to_csv(OUTPUT_PATH, index=False)
    print(f"Success! Predictions saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
