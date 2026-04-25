# 🚀 NeuroLogic '26: Global NLP Datathon Submission

This repository contains the complete, reproducible machine learning pipelines for all three NLP challenge tracks of the **NeuroLogic '26 Datathon**. 

The core architecture across all challenges is designed for rapid inference and memory efficiency, utilizing robust text preprocessing, **TF-IDF vectorization**, and highly optimized **LightGBM classifiers**.

---

## 📂 Repository Structure

```text
NeuroLogic26_Submission
├── README.md                 # Project documentation and run instructions
├── data/                     # Raw dataset directory
│   ├── challenge_1/          # Disaster tweets datasets
│   ├── challenge_2/          # Fake news datasets
│   └── challenge_3/          # Multilingual toxicity datasets
├── outputs/                  # Final prediction CSV files for submission
│   ├── Disaster_no_label.csv
│   ├── FakeNews_no_labels.csv
│   └── toxic_no_label_evaluation.csv
├── requirements.txt          # Python environment dependencies
├── results/                  # Visual proof of validation metrics
│   ├── ch1_f1_scores.png
│   ├── ch2_accuracy1.png
│   ├── ch2_accuracy2.png
│   ├── ch3_roc_auc1.png
│   └── ch3_roc_auc2.png
└── src/                      # Modular source code
    ├── __init__.py
    ├── preprocess.py         # Reusable text cleaning logic
    ├── ch1_pipeline.py       # Challenge 1 execution script
    ├── ch2_pipeline.py       # Challenge 2 execution script
    └── ch3_pipeline.py       # Challenge 3 execution script
```
## 🎯 Challenge Summaries & Performance Metrics
All models were evaluated internally using an **80/20 Stratified Train-Validation Split**.
### 🚨 Challenge 1: Real-Time Disaster Tweet Classification
 * **Objective:** Binary classification of social media text to identify real disaster events.
 * **Methodology:** Text normalization (removal of URLs, HTML, punctuation) followed by TF-IDF Vectorization (Word N-grams, max 15,000 features). Used LGBMClassifier with class_weight='balanced' to effectively counteract the heavy class imbalance between Informative and Not Informative tweets. Target strings were reverse-mapped to ensure exact submission compliance.
 * **Reported Metric:** **Macro F1-Score: 0.8174**
### 📰 Challenge 2: Fake News & Misinformation Detection
 * **Objective:** Classification of news articles as reliable or misleading (TRUE/FALSE) based on title and content.
 * **Methodology:** Concatenated article titles and body text to maximize semantic context. Bypassed dataset encoding anomalies using latin-1 and on_bad_lines='skip'. Applied TF-IDF Vectorization (Word N-grams, max 20,000 features) and trained a 300-estimator LightGBM model.
 * **Reported Metric:** **Overall Accuracy: 0.9967 (99.67%)**
### 🌐 Challenge 3: Multilingual Toxic Comment Classification
 * **Objective:** Binary classification identifying toxicity across multilingual text (English & Hindi).
 * **Methodology:** Utilized **Character-level N-gram TF-IDF Vectorization** (char_wb, n-gram range 2-5, max 25,000 features). This specific architectural choice effectively captures sub-word morphological roots, slang, and phonetic spelling variations across both Hindi and English simultaneously, avoiding the need for heavy, latency-inducing translation pipelines.
 * **Reported Metric:** **Mean ROC-AUC Score: 0.9841**
## ⚙️ Reproducibility & Execution Instructions
To reproduce these results from scratch, please follow these steps:
**1. Clone the repository and navigate to the project directory:**
```bash
git clone <https://github.com/RootDeveloperDS/NeuroLogic26_Submission>
cd NeuroLogic26_Submission
```
**2. Create and activate a virtual environment:**
 * **Windows:**
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```
 * **macOS/Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
**3. Install required dependencies:**
```bash
pip install -r requirements.txt
```
**4. Execute the pipelines:**
Ensure your terminal is in the root directory (NeuroLogic26_Submission). Run each script sequentially. The scripts will automatically load the data, train the models, output the validation metrics to the console, and generate the final predictions in the outputs/ folder.
```bash
python src/ch1_pipeline.py
python src/ch2_pipeline.py
python src/ch3_pipeline.py
```
