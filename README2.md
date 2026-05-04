# 🧠 NeuroLogic '26 — Beginner's Guide to This Project

> **Who is this for?** Anyone who is new to programming, data science, or machine learning and wants to understand what this project does, how it works, and why it was built this way — explained in plain, simple language.

---

## 📌 Table of Contents

1. [What Is This Project?](#-what-is-this-project)
2. [The Big Picture — How It All Works](#-the-big-picture--how-it-all-works)
3. [Folder Structure — What Lives Where](#-folder-structure--what-lives-where)
4. [The Code Files Explained](#-the-code-files-explained)
   - [preprocess.py — The Text Cleaner](#preprocesspy--the-text-cleaner)
   - [Challenge 1 — Disaster Tweet Classification](#-challenge-1-disaster-tweet-classification)
   - [Challenge 2 — Fake News Detection](#-challenge-2-fake-news-detection)
   - [Challenge 3 — Multilingual Toxic Comment Detection](#-challenge-3-multilingual-toxic-comment-detection)
5. [The Machine Learning Models Used](#-the-machine-learning-models-used)
6. [What is TF-IDF?](#-what-is-tf-idf-the-heart-of-it-all)
7. [Final Results](#-final-results)
8. [Why Not Use ChatGPT-style AI?](#-why-not-use-chatgpt-style-ai)
9. [How to Run It Yourself](#-how-to-run-it-yourself)

---

## 🧩 What Is This Project?

This is a **competition submission** for an NLP (Natural Language Processing) Datathon called **NeuroLogic '26**.

> **NLP** = Teaching computers to read and understand human language (text).  
> **Datathon** = A competition where teams compete to build the best ML models on given datasets.

The challenge had **3 tasks**, and this project solves all 3 of them:

| # | Task | Example |
|---|---|---|
| 1 | Read a tweet → Is it about a **real disaster**? | *"Wildfire spreading near downtown"* → ✅ Real disaster |
| 2 | Read a news article → Is it **fake or real**? | *"Aliens land on White House lawn"* → ❌ Fake |
| 3 | Read a comment (English or Hindi) → Is it **toxic**? | *"You are the worst!"* → ⚠️ Toxic |

---

## 🗺️ The Big Picture — How It All Works

Think of it like a factory assembly line:

```
📄 Raw Text (tweets, articles, comments)
         ↓
🧹 Step 1: Clean the text (remove junk like URLs, symbols)
         ↓
🔢 Step 2: Convert words into numbers (TF-IDF)
         ↓
🤖 Step 3: Feed numbers into ML models
         ↓
🏷️ Step 4: Get a prediction (category label)
         ↓
💾 Step 5: Save results to a CSV file
```

Every challenge follows this exact same flow. Only the data and some settings differ.

---

## 📁 Folder Structure — What Lives Where

```
NeuroLogic26_Submission/
│
├── README.md              ← Original technical documentation
├── README2.md             ← ⭐ This file (beginner-friendly guide)
├── requirements.txt       ← List of Python libraries needed
│
├── data/                  ← The raw datasets (input text + labels)
│   ├── challenge_1/       ← Disaster tweets
│   ├── challenge_2/       ← Fake news articles
│   └── challenge_3/       ← Multilingual toxic comments
│
├── src/                   ← All the Python code (the brains)
│   ├── preprocess.py      ← Shared text cleaner (used by all 3)
│   ├── ch1_pipeline.py    ← Full pipeline for Challenge 1
│   ├── ch2_pipeline.py    ← Full pipeline for Challenge 2
│   ├── ch3_pipeline.py    ← Full pipeline for Challenge 3
│   └── cv_check.py        ← Extra accuracy check for Challenge 2
│
├── outputs/               ← Final predictions (output CSV files)
│   ├── Disaster_no_label.csv
│   ├── FakeNews_no_labels.csv
│   └── toxic_no_label_evaluation.csv
│
└── results/               ← Proof screenshots / charts of model accuracy
    ├── executive_summary.png
    ├── ch1_f1_scores.png
    ├── ch2_accuracy2.png
    ├── ch2_cv_proof.png
    └── ch3_roc_auc.png
```

**Simple rule:** `data/` is what goes *in*, `outputs/` is what comes *out*, `src/` is where the magic happens.

---

## 🔧 The Code Files Explained

### `preprocess.py` — The Text Cleaner

This is a **shared helper function** used by all 3 challenge pipelines. Real-world text is messy — it has typos, emojis, URLs, HTML code, and symbols that confuse ML models. This file cleans all of that up.

Here is what `clean_text()` does, step by step:

```
Input:  "Check out http://news.com <b>BREAKING!</b> @CNN #Disaster flood!!!"
           ↓ lowercase
        "check out http://news.com <b>breaking!</b> @cnn #disaster flood!!!"
           ↓ remove URLs
        "check out  <b>breaking!</b> @cnn #disaster flood!!!"
           ↓ remove HTML tags
        "check out  breaking! @cnn #disaster flood!!!"
           ↓ remove @mentions and # symbols
        "check out  breaking!  cnn  disaster flood!!!"
           ↓ remove punctuation
        "check out  breaking   cnn  disaster flood"
           ↓ collapse extra spaces

Output: "check out breaking cnn disaster flood"
```

**Why clean?** ML models learn patterns from words. Cleaner text = more consistent patterns = a smarter, more accurate model.

---

### 🚨 Challenge 1: Disaster Tweet Classification

**File:** `src/ch1_pipeline.py`

**Goal:** Read a tweet and decide — is this about a *real disaster*, or just someone using dramatic language?

**Example:**
- `"The fire in my heart burns for you 🔥"` → ❌ Not a disaster (it's poetic)
- `"Massive wildfire forces 5000 evacuations in California"` → ✅ Real disaster

**How the pipeline works:**

| Step | What happens |
|---|---|
| **1. Load data** | Reads `Disaster_with_label.csv` — thousands of tweets already labeled "Informative" (real) or "Not Informative" (not real) |
| **2. Clean text** | Every tweet goes through `clean_text()` |
| **3. Split data** | 80% of tweets are used to *train* the model, 20% are kept aside to *test* how accurate it is |
| **4. TF-IDF** | Converts all words into numbers (up to 15,000 features, including 2-word phrases like "car crash") |
| **5. Train 3 models** | LightGBM + Logistic Regression + Naive Bayes all learn from the training data |
| **6. Ensemble** | Combines all 3 model predictions using *soft voting* (explained below) |
| **7. Evaluate** | Measures **Macro F1-Score** on the 20% test set → **0.8341** |
| **8. Predict & Save** | Runs on the unlabeled test file and saves predictions |

> **What is Macro F1-Score?**  
> It's a single number (0 to 1) that measures how well the model does on *both* classes equally — it won't give a high score just because the model is good at the bigger category. A score of 0.83 is very strong for this type of task.

**Special trick — Handling imbalanced data:**  
There are more "not disaster" tweets than "disaster" tweets in the dataset. If we ignore this, the model learns to just always say "not disaster" because that's the safe bet. To fix this, `class_weight='balanced'` is used — it tells the model: *"Treat every rare disaster tweet as more important during training."* No fake data is created; the math is just adjusted.

---

### 📰 Challenge 2: Fake News Detection

**File:** `src/ch2_pipeline.py`

**Goal:** Read a news article's title and body, decide — is it **TRUE (real news)** or **FALSE (fake news)**?

**How the pipeline works:**

| Step | What happens |
|---|---|
| **1. Load data** | Reads `fakenews_with_labels.csv` |
| **2. Combine features** | Joins the article **title + body text** into one big string — titles are important because fake news often has clickbait, sensationalist headlines |
| **3. Clean text** | Runs through `clean_text()` |
| **4. TF-IDF** | Up to 20,000 word features |
| **5. Train 1 model** | Just LightGBM — no ensemble needed here, the data is clean and well-structured |
| **6. Evaluate** | Measures **Accuracy** → **99.67%** |
| **7. Predict & Save** | Runs on the unlabeled test file, saves predictions |

> **What is Accuracy?**  
> Out of every 100 articles, the model correctly labels ~99–100 of them. That's extremely high.

**Extra step — `cv_check.py` (Cross-Validation):**  
When accuracy is 99.67%, a natural question is: *"Is the model actually that good, or did it just memorize the training data?"* To prove it isn't cheating, a **5-Fold Cross Validation** is run:

```
Full dataset
     ↓ Split into 5 equal chunks
┌─────┬─────┬─────┬─────┬─────┐
│  1  │  2  │  3  │  4  │  5  │
└─────┴─────┴─────┴─────┴─────┘

Round 1: Train on [2,3,4,5] → Test on [1] → Score?
Round 2: Train on [1,3,4,5] → Test on [2] → Score?
Round 3: Train on [1,2,4,5] → Test on [3] → Score?
...and so on

Average all 5 scores → Final CV Accuracy: 99.81%
```

This proves the model genuinely generalizes — it's not just memorizing.

---

### 🌐 Challenge 3: Multilingual Toxic Comment Detection

**File:** `src/ch3_pipeline.py`

**Goal:** Read a comment that could be in English, Hindi, or a mix of both ("Hinglish"), and decide — is it **toxic (1)** or **non-toxic (0)**?

**Example:**
- `"You are a wonderful person!"` → 0 (non-toxic)
- `"yaar tu bahut bekar hai"` → 1 (toxic, in Hindi)
- `"bro you're such a loser yaar"` → 1 (toxic, mixed English + Hindi)

**How the pipeline works:**

| Step | What happens |
|---|---|
| **1. Load data** | Reads `toxic_labeled.xlsx` (Excel format) |
| **2. Clean text** | Runs through `clean_text()` |
| **3. TF-IDF (special!)** | Uses **character n-grams** instead of whole words |
| **4. Train ensemble** | LightGBM + Logistic Regression with soft voting |
| **5. Evaluate** | Measures **ROC-AUC** → **0.9855** |
| **6. Predict & Save** | Runs on the unlabeled test file, saves predictions |

> **What is ROC-AUC?**  
> A score from 0 to 1 measuring how well the model separates toxic from non-toxic. 0.5 = random guessing. 1.0 = perfect. 0.98 is excellent.

**The Key Innovation — Character N-grams:**

Most NLP approaches split text into *words*. But when you have Hindi text mixed with English, word-splitting breaks down:

- `"yaar"` (Hindi for "friend") is one word
- `"tu"` (Hindi for "you") is another
- The model has never seen these "words" before

Instead of translating everything (which needs an internet API), this project uses **character-level n-grams** — it looks at tiny character sequences:

```
Word: "toxic"
Character 2-grams: "to", "ox", "xi", "ic"
Character 3-grams: "tox", "oxi", "xic"
Character 4-grams: "toxi", "oxic"
Character 5-grams: "toxic"
```

The pattern `"tox"` appears in English "toxic" and also in Hinglish slang. The model learns these shared character patterns without needing to know which language it's reading. **No internet. No translation API. No language detection.** Pure offline math.

---

## 🤖 The Machine Learning Models Used

Here's a beginner-friendly explanation of each model in the project:

### LightGBM (Light Gradient Boosting Machine)
Think of it as building **hundreds of decision trees**, where each new tree learns from the mistakes of the previous one. Very fast and very accurate for structured data.

```
Tree 1: "Does the tweet contain 'fire'?" → If yes → likely disaster
Tree 2: "Does it contain 'fire' AND 'evacuate'?" → even more likely disaster
Tree 3: ...and so on, 300 trees total
```

### Logistic Regression
The simplest model here. It draws a straight dividing line between two categories in the data. Surprisingly powerful for text, especially when combined with others.

### Naive Bayes
Counts how often certain words appear in each category. For example, if the word "wildfire" almost always appears in disaster tweets and rarely in non-disaster tweets, Naive Bayes learns that association.

### Soft-Voting Ensemble
Instead of asking "which single model is best?", you ask **all models** and average their *confidence scores*:

```
Tweet: "Flooding reported near downtown"

LightGBM says:  90% chance it's a disaster
Logistic Reg:   85% chance it's a disaster
Naive Bayes:    78% chance it's a disaster

Average:        84.3% → Label: DISASTER ✅
```

This is more reliable than any one model alone — like getting a second (and third) opinion from a doctor.

---

## 🔢 What is TF-IDF? (The Heart of It All)

Computers cannot understand text directly. They only understand numbers. **TF-IDF** is the bridge.

**TF = Term Frequency** — How often does this word appear in *this document*?  
**IDF = Inverse Document Frequency** — How rare is this word across *all documents*?

```
"the" appears in every single article → low IDF score → ignored
"wildfire" appears in only 5% of articles → high IDF score → important!
```

The result: each piece of text becomes a long row of numbers (a vector), where important words have high values and common/boring words have values near zero.

```
Tweet: "massive wildfire evacuation underway"
           ↓ TF-IDF
[0, 0, 0.45, 0, 0.67, 0, 0.52, 0, 0, ... 0.38, ...]
  ↑ "the"=0  ↑ "wildfire"=0.67  ↑ "evacuation"=0.52
```

These numbers are what the ML models actually train on.

---

## 📈 Final Results

| Challenge | Task | Metric Used | Score |
|---|---|---|---|
| 1 | Disaster Tweet Classification | Macro F1-Score | **0.8341** |
| 2 | Fake News Detection | Accuracy | **99.67%** |
| 3 | Multilingual Toxicity Detection | ROC-AUC | **0.9855** |

All results were measured on data the model had **never seen during training** (the held-out 20% validation set), making these honest, unbiased scores.

---

## ⚡ Why Not Use ChatGPT-style AI?

Modern NLP often uses giant models like BERT or GPT. So why didn't this project use them?

| Problem with LLMs | This project's solution |
|---|---|
| Need expensive GPU hardware | Runs on a normal laptop CPU |
| Slow — seconds per prediction | Fast — under 10 milliseconds |
| Cost money (API calls) | Completely free to run |
| Require internet connection | Works 100% offline |
| Send private user data to a server | All data stays local |

For a real-world system that needs to moderate thousands of comments per second, a 99ms BERT model would be unusable. A 5ms TF-IDF + LightGBM pipeline scales effortlessly.

---

## 🚀 How to Run It Yourself

> **Requirements:** Python 3.10 or higher, Git

**Step 1 — Get the code**
```bash
git clone https://github.com/RootDeveloperDS/NeuroLogic26_Submission.git
cd NeuroLogic26_Submission
```

**Step 2 — Set up a virtual environment** *(keeps your system Python clean)*

On Windows:
```cmd
python -m venv venv
venv\Scripts\activate
```

On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

**Step 3 — Install all libraries**
```bash
pip install -r requirements.txt
```

**Step 4 — Run the pipelines**
```bash
# Challenge 1: Disaster Tweet Classification
python src/ch1_pipeline.py

# Challenge 2: Fake News Detection
python src/ch2_pipeline.py

# [Optional] Verify Cross-Validation for Challenge 2
python src/cv_check.py

# Challenge 3: Multilingual Toxicity Detection
python src/ch3_pipeline.py
```

After running, check the `outputs/` folder — your prediction CSV files will be there. ✅

---

## 🧠 Key Takeaways

1. **Text is cleaned first** — garbage in, garbage out. Clean text = better models.
2. **TF-IDF converts text to numbers** — the bridge between human language and machine math.
3. **Ensemble models beat single models** — combining different learners reduces mistakes.
4. **Character n-grams handle multilingual text** — no translation needed.
5. **Lightweight is better for production** — fast, offline, cheap, and private.

---

<div align="center">

*This guide was written to make the project accessible to anyone — no prior ML knowledge required.*

**NeuroLogic '26 · Global NLP Datathon**

</div>
