# Feature Importance — Three Methods on the Final Model

| Feature | Coefficient | \|coef\| | Permutation F1 drop | Ablation F1 drop |
|---|---|---|---|---|
| `embedding_similarity` | +0.979 | 0.979 | 0.250 | +0.063 |
| `skill_overlap` | +0.716 | 0.716 | 0.071 | -0.047 |
| `num_missing_skills` | -0.699 | 0.699 | 0.105 | +0.088 |
| `weighted_skill_score` | -0.697 | 0.697 | 0.053 | -0.015 |
| `tfidf_similarity` | +0.560 | 0.560 | 0.085 | +0.035 |
| `title_similarity` | +0.497 | 0.497 | 0.118 | +0.116 |
| `avg_missing_skill_importance` | -0.279 | 0.279 | 0.011 | -0.006 |
| `years_of_experience` | +0.004 | 0.004 | 0.000 | +0.000 |

Notes: skill_overlap and weighted_skill_score have near-equal opposite-sign coefficients — multicollinearity signature. years_of_experience contributes essentially nothing.
