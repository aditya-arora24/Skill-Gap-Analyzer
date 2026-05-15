# Resume–Job Alignment ML System

A machine learning pipeline that determines whether a candidate's resume is a strong match for a job description. It combines semantic embeddings, keyword similarity, and structural skill signals — then evaluates against a hand-labeled gold standard to ensure results reflect real-world judgment.

---

## The Problem

Simple systems compare resumes and jobs using only semantic similarity (e.g., cosine similarity on SBERT embeddings). This works reasonably well but has a known flaw: semantically similar text doesn't guarantee skill alignment. A resume written in professional language scores high even when it lacks the specific skills a job requires.

This project fixes that by training classifiers on a richer feature set that captures what semantic similarity misses.

---

## Pipeline Overview

```
Raw Data (130 resumes, 43 jobs)
        │
        ▼
  1. Text Cleaning          — normalize, strip HTML, mask PII
        │
        ▼
  2. Skill Extraction       — 258 seed keywords + 7 contextual patterns
        │
        ▼
  3. SBERT Pairing          — Top-6 most similar jobs per resume → 9,495 pairs
        │
        ▼
  4. Weak Labeling          — skill_coverage quantiles (70th / 30th percentile)
        │
        ▼
  5. Feature Upgrade        — add tfidf_similarity + skill_imbalance
        │
        ▼
  6. Model Training         — LogReg + GBM, CV threshold tuning
        │
        ▼
  7. Gold Standard Eval     — 100 expert-labeled pairs, zero leakage
```

---

## Features

| Feature | Description |
|---|---|
| `semantic_similarity` | SBERT cosine similarity (`all-MiniLM-L6-v2`) — captures contextual meaning |
| `tfidf_similarity` | TF-IDF cosine similarity — captures exact keyword overlap |
| `num_resume_skills` | Count of skills extracted from resume (mean: 6.5) |
| `num_job_skills` | Count of skills extracted from job description (mean: 10.0) |
| `skill_imbalance` | `num_job_skills − num_resume_skills` — measures how underqualified a candidate is |

**Note:** `skill_coverage` and `skill_gap` are used only to generate weak training labels — they are deliberately excluded from model features to prevent label leakage.

---

## Models & Evaluation

Two classifiers are trained on the 9,495 weakly-labeled pairs and evaluated on the 100-sample expert-labeled gold set:

- **Logistic Regression** — fast, interpretable baseline
- **Gradient Boosting Classifier** — captures non-linear feature interactions

Each model's decision threshold is tuned independently using 5-fold cross-validated out-of-fold probabilities on training data (threshold range: 0.10–0.90, optimized for F1). Thresholds are never tuned on the gold set.

**Baselines compared:**

| Baseline | Threshold |
|---|---|
| SBERT Only | 0.50 |
| TF-IDF Only | 0.15 |
| Midsem Heuristic (0.6 × semantic + 0.4 × skill_coverage) | 0.50 |

---

## Leakage Prevention

This is the most critical design constraint in the project:

- `skill_coverage` is used to generate weak labels but is **never passed as a model feature**
- The `TfidfVectorizer` is fitted **only on training text**, then applied to the gold set without refitting
- The `StandardScaler` is fitted **only on training features**, then applied to the gold set
- The gold standard is used **only for final evaluation** — never for training or threshold selection
- Cross-validation for threshold tuning uses **out-of-fold probabilities only**

---

## Data

| Dataset | Rows | Description |
|---|---|---|
| `ml_ready_dataset.csv` | 9,495 | Training pairs with weak supervision labels and all 5 features |
| `gold_standard_final.csv` | 100 | Expert-labeled pairs (56 positive, 44 negative) |

**Source:** 130 resumes across 23 categories × 43 unique job titles. Top-6 SBERT pairing per resume.

---

## Project Structure

```
MLPR_PROJECT/
├── data/
│   ├── raw/                        # Original input files (excluded from git)
│   └── processed/                  # Derived datasets (excluded from git)
├── src/
│   ├── pipeline.py                 # Step 1–6: cleaning, extraction, pairing, weak labels
│   ├── upgrade_features.py         # Step 7: add tfidf_similarity + skill_imbalance
│   ├── create_gold_standard.py     # Sample 100 pairs for human labeling
│   ├── generate_labels.py          # Write hardcoded gold labels to CSV
│   ├── merge_gold_standard.py      # Merge labels into gold_standard_final.csv
│   ├── train_pipeline.py           # Data loading, splitting, scaling utilities
│   └── evaluate_models.py          # Train, tune thresholds, evaluate on gold set
├── models/                         # Saved model artifacts (excluded from git)
│   ├── scaler.pkl
│   ├── logreg_model.pkl
│   └── gbm_model.pkl
├── outputs/
│   ├── feature_distributions.png   # Feature histograms
│   └── feature_importance_final.png # GBM feature importances
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

All scripts are run from the `src/` directory. The pipeline has a fixed execution order:

```bash
cd src

# Build the training dataset (SBERT encoding — ~5 min)
python pipeline.py

# Add TF-IDF and skill imbalance features (leakage-safe)
python upgrade_features.py

# One-time: build the gold standard
python generate_labels.py
python create_gold_standard.py
python merge_gold_standard.py

# Train models, tune thresholds, evaluate on gold set, save artifacts
python evaluate_models.py
```

To re-run evaluation only (models already trained):

```bash
cd src
python evaluate_models.py
```

---

## Loading Saved Models

```python
import joblib

scaler = joblib.load("models/scaler.pkl")
logreg = joblib.load("models/logreg_model.pkl")
gbm    = joblib.load("models/gbm_model.pkl")

# Features must be in this order:
# [semantic_similarity, tfidf_similarity, num_resume_skills, num_job_skills, skill_imbalance]
X_scaled     = scaler.transform(X_new)
predictions  = gbm.predict(X_scaled)
```

---

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
sentence-transformers
torch
joblib
```
