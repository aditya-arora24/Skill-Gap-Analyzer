
# SkillMatch — Two-Stage Skill-Aware Resume–Job Matching System

> MLPR Endterm Project — Plaksha University (2026)

A hybrid retrieval-and-reranking machine learning pipeline for resume–job matching using semantic similarity, lexical similarity, ESCO-grounded skill reasoning, and active learning.

---

# Project Overview

Traditional resume–job matching systems rely heavily on semantic similarity or keyword overlap, often producing plausible-looking but structurally incorrect matches.

SkillMatch introduces a two-stage architecture:

1. **SBERT-based semantic retrieval**
2. **Hybrid ML reranking using semantic, lexical, skill-based, and structural features**

The project systematically investigates:
- Weak supervision
- LLM-supervised learning
- Self-training
- Active learning

under limited-label settings.

---

# Final Architecture

```text
Resume + Job Description
        ↓
Stage 1: SBERT Retrieval
        ↓
Top-K Candidate Pairs
        ↓
Stage 2: Hybrid ML Reranking
(Logistic Regression + 8 Features)
        ↓
Match Probability + Skill Gap Analysis
````

---

# Features Used

| Category    | Features                                                                              |
| ----------- | ------------------------------------------------------------------------------------- |
| Semantic    | embedding_similarity, title_similarity                                                |
| Lexical     | tfidf_similarity                                                                      |
| Skill-Based | skill_overlap, weighted_skill_score, num_missing_skills, avg_missing_skill_importance |
| Structural  | years_of_experience                                                                   |

---

# Methodology Evolution

| Strategy         | Description                            | F1 Score |
| ---------------- | -------------------------------------- | -------- |
| SBERT Baseline   | Semantic retrieval only                | 0.604    |
| Weak Supervision | Formula-based labels                   | 0.440    |
| LLM Supervised   | 400 LLM labels                         | 0.704    |
| Self-Training v1 | Confidence-threshold pseudo-labeling   | 0.564    |
| Self-Training v2 | Diversified percentile pseudo-labeling | 0.579    |
| Active Learning  | Uncertainty sampling                   | 0.769    |

### Key Finding

Confidence-based pseudo-labeling reinforced existing decision boundaries and degraded performance, while uncertainty-based active learning improved model discrimination under the same labeling budget.

---

# Leakage Diagnosis

Initial weak-label supervision produced severe leakage:

* Cross-validation F1: **0.99**
* Held-out test F1: **0.44**

### Root Cause

The same features used to generate heuristic labels were also used during training.

This motivated the transition toward:

* LLM-supervised labeling
* Independent evaluation
* Active learning

---

# Dataset

* 2,484 resumes
* 853 job descriptions
* 20,253 skill vocabulary entries
* 50,650 candidate pairs

### Sources

* Kaggle Resume Dataset
* Public Job Description Corpus
* ESCO + O*NET Skill Vocabulary

---

# Gold Standard Construction

500 resume–job pairs were labeled independently by:

* Claude
* GPT
* Gemini

Final labels were assigned using majority voting.

### Agreement Statistics

* 71% unanimous agreement
* 80.7% mean pairwise agreement
* Cohen’s κ = 0.61

---

# Final Results

## Best Active Learning Model

| Metric    | Value |
| --------- | ----- |
| F1 Score  | 0.769 |
| Precision | 0.667 |
| Recall    | 0.909 |

### Improvement Over Baseline

* +0.165 F1 over SBERT baseline

---

# Ablation Study

| Configuration       | F1 Score |
| ------------------- | -------- |
| SBERT Only          | 0.628    |
| Skill Features Only | 0.706    |
| Full Hybrid System  | 0.769    |

### Key Insight

Skill-aware structural features contribute independent predictive signal beyond semantic similarity alone.

---

# Repository Structure

```text
SkillMatch/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── gold/
│
├── src/
│   ├── preprocess_pipeline.py
│   ├── matching_pipeline.py
│   ├── build_final_dataset.py
│   ├── evaluate_models.py
│   ├── run_ablation_study.py
│   ├── active_learning.py
│   └── ...
│
├── outputs/
│   ├── feature_importance.png
│   ├── ablation_results.png
│   ├── confusion_matrix.png
│   └── feature_distributions.png
│
├── docs/
│   ├── presentation.pdf
│   ├── methodology_notes.md
│   └── literature_review.pdf
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Running the Project

## 1. Preprocess Data

```bash
python preprocess_pipeline.py
```

## 2. Generate Candidate Pairs

```bash
python matching_pipeline.py
```

## 3. Build Final Dataset

```bash
python build_final_dataset.py
```

## 4. Train and Evaluate Models

```bash
python evaluate_models.py
```

## 5. Run Ablation Study

```bash
python run_ablation_study.py
```

---

# Literature Positioning

Compared to prior resume-matching systems, this project introduces:

* Independent leakage diagnosis
* Multi-LLM consensus labeling
* Active learning under fixed label budget
* Empirical self-training failure analysis
* Hybrid reranking beyond semantic retrieval

---

# Limitations

* Gold labels are LLM-generated, not recruiter-generated
* Test set size remains limited
* English-only and IT-heavy corpus
* Real-world deployment not yet validated
* Bias and fairness auditing not yet performed

---

# Future Work

* Human recruiter validation
* Cross-domain transfer evaluation
* Bias auditing and fairness analysis
* Real-time deployment pipeline
* Longitudinal recommendation tracking

---

# Authors
Aditya Arora | Reya Saigal | Kuhuk Katiyar

MLPR Endterm Project Team
Plaksha University — 2026

```
```
