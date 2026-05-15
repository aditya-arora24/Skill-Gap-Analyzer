# Dataset Specifications

## Raw inputs

| File | Size | Source | Content |
|---|---|---|---|
| Resume (1).csv | 53.7 MB | Kaggle | 2,484 anonymized resumes |
| training_data.csv | 3.6 MB | Public | 853 job descriptions with structured `model_response` |
| Skills.xlsx | 3.2 MB | O*NET (U.S. DoL) | 35 generic competencies with importance scores |
| skills_en.csv | 9.3 MB | ESCO (European Commission) | 13,960 skill entries |
| broaderRelationsSkillPillar_en.csv | 4.9 MB | ESCO | 20,819 hierarchy edges |

## Processed dataset

| Artifact | Rows | Description |
|---|---|---|
| cleaned_resumes.parquet | 2,484 | Cleaned + skill-extracted resumes |
| cleaned_jobs.parquet | 853 | Cleaned + skill-extracted jobs |
| pair_features.parquet | 42,650 | Top-50 SBERT retrieval × 13 features |
| pair_features_diversified.parquet | 50,650 | + 5,000 A_mid + 2,000 B_xcat + 1,000 C_rand |
| gold_labels.csv | 500 | 3-LLM majority-vote labels with metadata |
| active_labels.csv | 200 | Active-learning uncertain-band labels |

## Test set

100 held-out pairs (22 positive, 78 negative), stratified 80/20 split with random_state=42.
Used identically across all six methodology comparisons.
