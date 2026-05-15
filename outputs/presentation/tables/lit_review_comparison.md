# Literature Comparison Table

| Dimension | P1 (Dash et al., IJEDR 2025) | P2 (Daberao et al., GITCON 2025) | P3 (Sribharathi et al., ICSSS 2025) | **Our Project** |
|---|---|---|---|---|
| Skill extraction | BERT/RoBERTa NER fine-tuned | spaCy + Levenshtein | TF-IDF + BERT | **ESCO vocabulary (20k) + flashtext** |
| Skill taxonomy | ESCO + O*NET | None (string distance) | None (BERT vectors) | **ESCO + O*NET, depth-weighted importance** |
| Match scoring | Sentence-BERT cosine | 7-dim feature vector + Random Forest | BERT cosine + supervised classifier | **8-feature Logistic Regression** |
| Label source | Undisclosed | K-Means cluster labels (flagged weakness) | Undisclosed | **3-LLM consensus majority vote** |
| Inter-rater agreement disclosed | No | N/A | No | **Yes: 71% unanimous, 80.7% pairwise, κ=0.61** |
| Reported metric | 0.90 accuracy (RoBERTa) | 0.72 accuracy (vs K-Means labels) | 0.918 accuracy | **0.769 F1 on stratified LLM gold** |
| Methodology comparisons | BERT vs RoBERTa | RF vs XGBoost vs ANN | BERT vs TF-IDF baseline | **5 methodologies on same test** |
| Leakage detection | Not addressed | Not addressed | Not addressed | **Empirical: CV 0.99 → test 0.44** |
| Self-training tested | No | No | No | **Yes — 2 variants, both fail** |
| Active learning tested | No | No | No | **Yes — +6.5 F1 lift** |
| Production deployment | Docker/K8s stack | Not demonstrated | 200+ user pilot | **Prototype dashboard only** |
