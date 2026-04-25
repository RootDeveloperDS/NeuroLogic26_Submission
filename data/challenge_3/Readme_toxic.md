### 📢 Hackathon Advisory: Multilingual Toxicity Classification Task

#### 🧩 Dataset Source

This challenge uses the **Multilingual Toxicity Dataset** from Hugging Face.

* The dataset contains text samples across 2 languages, English and Hindi.
* Each sample is labeled for **binary toxicity classification**:

  * `0` → Non-toxic
  * `1` → Toxic

---

#### 📂 Dataset Structure

You are provided with:

1. **Training Dataset (CSV)**

   * Contains text + corresponding toxicity labels.
   * Label distribution in training data:

     ```
     0    4506
     1    4494
     ```

2. **Evaluation Dataset (no_label)(CSV)**

   * Contains only text (labels removed).

---

#### 🤖 Task Overview

* Train a classification model using the **training dataset**.
* Predict toxicity labels for the **evaluation dataset**.

---

#### ⚠️ Critical Advisory on Labels

* This is a **binary classification problem**.
* The only valid labels are:

  * `0` (non-toxic)
  * `1` (toxic)

✅ Your submission **must use exactly these values**.

❌ Submissions using:

* `toxic / non-toxic`
* `yes / no`
* `-1 / 1`
* any other encoding

will be **rejected or incorrectly evaluated**.

---

Following these rules strictly ensures your submission is evaluated correctly.
