# Fake News Detection Dataset (Binary Classification  Competition Use)

## Overview

This competition uses data derived from the **Fake and Real News Dataset**.

**We are not the creators or owners of the original dataset.**
The data has been sourced from Kaggle and **preprocessed into a single unified CSV file** for this competition.

The task is to classify news articles as **real or fake**.

---

## Dataset Source

Original dataset available at:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

 The provided dataset is reduced form of above dataset. 
---

## Task Definition

Participants must build a **binary classification model**:

### Labels

* **TRUE** ’ Real news
* **FALSE** ’ Fake news

---

## Dataset Statistics

The dataset is reduced and following are the details

| Label     | Count      |
| --------- | ---------- |
| FALSE     | 8455       |
| TRUE      | 15438      |
| **Total** | **23893**  |

## Dataset Format

The competition dataset is provided as a **single CSV file**.

### Columns

* `title`  Headline of the article
* `text`  Full news content
* `subject`  Topic/category
* `date`  Publication date
* `label`  Target label (`TRUE` or `FALSE`)

---

## Label Mapping (Optional for Modeling)

Participants may convert labels to numeric format:

| Label | Numeric |
| ----- | ------- |
| TRUE  | 1       |
| FALSE | 0       |

Example:

```python id="xnjmvx"
df["label_num"] = df["label"].map({"TRUE": 1, "FALSE": 0})
```

---

## Evaluation

Suggested metrics:


* Accuracy


---

## Important Disclaimer

* This dataset is **not originally created by the competition organizers**
* It has been **reformatted and labeled for competition use**
* Original credit belongs to the dataset creator on Kaggle

Participants must comply with:

* Kaggle dataset terms of use
* Any applicable licensing restrictions

---

## Citation

If you use this dataset, please cite the original source:

> Fake and Real News Dataset
> Available at: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

---

## Acknowledgment

We thank the original dataset creator for making this data publicly available.

---
