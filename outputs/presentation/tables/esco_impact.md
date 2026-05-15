# ESCO Vocabulary Integration — Before vs After

| Metric | Before ESCO | After ESCO | Change |
|---|---|---|---|
| skill_overlap > 0 | 36% | 68% | +32 pp |
| years_of_experience > 0 | 39% | 48% | +9 pp |
| title_similarity unique | 7,286 | 31,254 | +23,968 |
| avg skills per resume | 3.40 | 13.20 | +9.80 |
| avg skills per job | 1.70 | 5.96 | +4.26 |


## Vocabulary sizes

| Component | Skills |
|---|---|
| TECH_SKILLS (curated) | 355 |
| ESCO knowledge (preferredLabel) | 3,208 |
| ESCO altLabels added | 16,570 |
| ESCO hiddenLabels added | 190 |
| **Total merged vocabulary** | **20,253** |

Filter applied during ESCO integration: skillType=knowledge, length ≥ 4 chars, ≥ 50% alphabetic, not in 90-word stopword/verb blocklist.
