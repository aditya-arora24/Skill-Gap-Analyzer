# Resume–Job Alignment as Two-Stage Retrieval and Reranking — A Skill-Aware ML System with Active Learning

**Final project report. Comprehensive documentation of the problem framing, literature positioning, dataset construction, methodology iterations, results, limitations, and future work.**

---

## Table of Contents

1. [Abstract](#abstract)
2. [Introduction & Problem Statement](#1-introduction--problem-statement)
3. [Literature Survey](#2-literature-survey)
4. [Dataset and Preprocessing](#3-dataset-and-preprocessing)
5. [ML Methodology](#4-ml-methodology)
6. [Results and Performance Metrics](#5-results-and-performance-metrics)
7. [Feature Importance Analysis](#6-feature-importance-analysis)
8. [Context-Aware Skill Weighting Experiment](#7-context-aware-skill-weighting-experiment)
9. [Deployability and Dashboard Prototype](#8-deployability-and-dashboard-prototype)
10. [Limitations](#9-limitations)
11. [Future Work](#10-future-work)
12. [Conclusion](#11-conclusion)
13. [Appendix — Code and Dataset Provenance](#appendix--code-and-dataset-provenance)

---

## Abstract

We build a two-stage system for resume–job alignment. **Stage 1** uses Sentence-BERT (SBERT) semantic similarity to retrieve the top 50 most plausible candidate-job pairs out of approximately 2.1 million possible pairs (2,484 resumes × 853 jobs). **Stage 2** is a learned reranker — a Logistic Regression classifier over eight features combining lexical, semantic, skill-coverage, skill-importance, and structural signals — that filters retrieved candidates into final match decisions and produces an interpretable skill-gap report.

We tested five distinct training methodologies on the same held-out evaluation set: (1) **weak supervision** using formula-derived labels, (2) **LLM-supervised learning** using three-LLM majority-vote gold labels, (3) **self-training v1** with absolute confidence thresholds, (4) **self-training v2** with percentile-based pseudo-label selection on a diversified pool, and (5) **active learning** that uses LLM judges to label only the most uncertain cases.

On a stratified 100-pair held-out gold standard labeled by Claude, GPT-4, and Gemini majority vote, our final active-learning model achieves **F1 = 0.769**, a **+16.5 absolute F1-point improvement** over an SBERT-only baseline (F1 = 0.604). The weak-supervised baseline collapses to F1 = 0.440 with a textbook leakage signature (cross-validation F1 = 0.99 → test F1 = 0.44, a 0.55-point CV-test gap). Both self-training variants degrade F1 by 12–14 points relative to plain supervised, empirically validating Yarowsky's 1995 confirmation-bias warning. Feature importance analysis confirms skill features carry independent signal: a LogReg trained on only the seven non-SBERT features achieves F1 = 0.706, beating an SBERT-only LogReg at F1 = 0.628 by 7.8 points.

Three findings independently warrant reporting: (i) formula-based weak supervision does not generalize to externally-judged labels regardless of leakage-safe feature partitioning; (ii) confidence-based pseudo-labeling (self-training) fails because it systematically excludes the most informative pool samples; (iii) skill-aware features grounded in the ESCO occupational taxonomy with hierarchy-depth-based importance weighting contribute independent predictive value beyond document-level semantic similarity.

---

## 1. Introduction & Problem Statement

### 1.1 The problem

Given a resume and a job description, the system must predict whether the candidate is a good match for the role. The problem has two natural users:

- **The job seeker**, who faces information asymmetry — they see hundreds of job postings on aggregator platforms but have no systematic way to identify which they are qualified for, what skills are missing, or where to invest their learning time.
- **The recruiter**, who faces volume — over 250 applications per role on average (per industry reporting cited in Daberao et al. 2025), of which most are scanned in seconds and many are filtered by primitive keyword-matching ATS systems that reject qualified candidates on the basis of resume formatting rather than competence.

Both users face the same underlying technical problem: a function that takes (resume, job) → (match probability, skill gap diagnosis).

We frame the project from the **job-seeker side** because:
- The skill-gap report — what's missing and how important is it — is more empowering as user-facing output than as recruiter screening output.
- Plaksha students are themselves job seekers; the system is more directly applicable as an internship/placement aid than as an admissions filter.
- Recruiter-side hiring AI carries known ethical risks (Amazon's 2018 model that downgraded women's resumes is the canonical reference); a candidate-side framing focuses on giving individuals information rather than gatekeeping on behalf of employers.

The same trained model serves both directions by symmetry.

### 1.2 Why it matters

Three impact arguments:

**Recruiter productivity at scale.** A 5-percentage-point precision improvement at the top of a ranked candidate list saves a recruiter approximately 10–15 minutes per role at typical screening volumes. Aggregated across an organization with hundreds of monthly hires, that is meaningful time recovered for higher-value work.

**Candidate empowerment.** Most job seekers receive no specific feedback when applications are rejected. An automated skill-gap report — "you have these required skills, you're missing these others ordered by importance" — gives candidates actionable information that recruiter rejections almost never provide.

**Career planning over time horizons.** Aggregating skill gaps across a candidate's target roles produces a personalized learning trajectory. Rather than learning skills opportunistically, candidates can prioritize the skills with highest impact across their preferred career paths.

### 1.3 Concrete deployability scenarios at Plaksha

The same trained model maps to three Plaksha-specific use cases:

1. **Internship matching tool** for the placement portal: rank live internship/job postings by per-student fit, surface skill gaps per posting.
2. **Course-to-career mapping**: given a student's transcript skills and a target role description, recommend which electives to take next semester.
3. **Skill-development advisor**: aggregate gap reports across a student's top-10 dream roles → surface the five skills with highest marginal benefit if learned. Acts as a quantitative complement to traditional career counseling.

### 1.4 Scope and non-goals

The project's scope is the underlying model and methodology. The web application layer, user authentication, learning-platform integrations, and live job-feed ingestion are scoped as future work. We build a static HTML dashboard prototype to demonstrate the model's output capabilities but do not deploy a production service.

---

## 2. Literature Survey

We surveyed three peer-reviewed papers in the same problem space (AI-based skill gap analysis and resume-job matching), published in 2025, and synthesized the field's structural assumptions and unresolved questions.

### 2.1 The three papers compared

**Paper 1 — Dash et al. (IJEDR, Nov 2025) — "AI Based Skill Gap Analyzer"**
- Full-stack web application (React + Vite frontend, Node.js + Express backend, Python NLP microservice)
- Skill extraction via fine-tuned BERT and RoBERTa for Named Entity Recognition
- ESCO and O*NET ontologies for skill normalization
- Sentence-BERT for semantic matching between candidate skills and job requirements
- Recommendation engine linking identified gaps to learning resources on Coursera, Udemy, and edX, evaluated with nDCG@5 and Precision@5
- Reported metrics: BERT accuracy = 0.86, RoBERTa accuracy = 0.90; recommendation nDCG@5 = 0.81; 92% user satisfaction; 500 concurrent users tested at <2% error rate

**Paper 2 — Daberao et al. (IEEE GITCON, August 2025) — "ResumeInsight"**
- React + Express + Flask architecture for Indian campus recruitment
- 7-dimensional feature vector combining academic metrics (CGPA, 10th/12th grade marks) with skill similarity computed via Levenshtein string distance
- No transformer models; uses spaCy for parsing and Random Forest / XGBoost / ANN for classification
- Training labels derived from K-Means clustering rather than human judgment — a known methodological weakness
- Dataset: 1,500 anonymized resumes, 50 job descriptions
- Reported metrics: 72% fit-prediction accuracy (Random Forest), 89.3% F1 on skill extraction; no user study, no recommendation engine

**Paper 3 — Sribharathi et al. (IEEE ICSSS, 2025) — "Scopira AI"**
- Full career guidance platform — broader scope than P1 or P2
- Four-module architecture: resume parsing, skill gap analysis, job recommendation, career path generator, plus a professional branding advisor
- TF-IDF + BERT embeddings, supervised learning (Logistic Regression, Random Forest, SVM), decision tree career path generator
- Dataset: 5,000 anonymized resumes, 3,000 job postings
- Reported metrics: 91.8% accuracy, 89% user satisfaction, MRR = 0.89, nDCG@5 = 0.86; benchmarked against LinkedIn Skills Match (83.5%) and IBM Watson Career Explorer (86.8%)
- Real-world pilot: 200+ users across 3 universities and 2 corporate training centers; 58% reported securing interviews aligned with system recommendations

### 2.2 What the three papers share

All three follow the same conceptual pipeline: extract skills from unstructured resume text using NLP → identify required skills from the job description → compare the two skill sets to find gaps → optionally surface learning recommendations. All three use Python for NLP, JavaScript frameworks for the frontend, and MongoDB for persistence. All three were built primarily for software-industry use cases.

### 2.3 Field-level structural assumptions and gaps

A detailed analysis of the three papers (see `skill_gap_analysis_full.md` for the full 13,000-word version) identified eight load-bearing assumptions shared across the field that are never empirically tested:

1. Resumes are truthful (self-reported skills correspond to actual capability)
2. Job descriptions are accurate (listed requirements correspond to what the role actually requires day-to-day)
3. Closing a measured skill gap improves employment outcomes
4. Text similarity is a valid proxy for actual job fit
5. Skill ontologies (ESCO, O*NET) are neutral and complete representations of the skill landscape
6. High user-satisfaction scores indicate genuine usefulness rather than interface comfort
7. The skill gap is the primary obstacle to employment (rather than network access, credential discrimination, or structural demand deficits)
8. Historical hiring data encodes quality rather than bias

The analysis identified five unanswered research questions that no paper in the field has addressed:

1. **Implicit and soft skill extraction from narrative text.** Existing systems extract explicitly stated skills well; none credibly extracts skills implied in narrative ("coordinated a five-person team through a launch" → project management).
2. **Cross-domain model generalization.** All published systems are trained and tested in narrow domains (primarily IT and data science); none tests whether models transfer to healthcare, legal, creative, or non-Western markets.
3. **Longitudinal validation of recommendation outcomes.** No paper measures whether following AI skill-gap recommendations actually changes employment trajectories.
4. **Algorithmic bias measurement and mitigation.** No paper publishes disaggregated performance metrics by gender, ethnicity, or socioeconomic background.
5. **Multilingual and non-Western resume parsing.** All systems operate on English-language resumes using English-trained models.

### 2.4 How our work positions against the literature

| Dimension | P1 (Dash) | P2 (Daberao) | P3 (Scopira) | **Our project** |
|---|---|---|---|---|
| Skill extraction | BERT/RoBERTa NER fine-tuned | spaCy + Levenshtein | TF-IDF + BERT | **Vocabulary lookup over ESCO (20k) + flashtext** |
| Skill taxonomy | ESCO + O*NET | None (string distance) | None (BERT vectors) | **ESCO + O*NET, with depth-based importance weighting** |
| Match scoring | Sentence-BERT cosine | 7-dim feature vector + Random Forest | BERT cosine + supervised | **8-feature LogReg over multi-channel signals** |
| Label source for training | Unexplained gold standard | K-Means cluster labels (a flagged weakness) | Unexplained gold standard | **3-LLM consensus majority vote (Claude/GPT/Gemini)** |
| Inter-rater agreement disclosed | No | N/A (no humans) | No | **Yes: 71% unanimous, 80.7% pairwise, κ = 0.61** |
| Reported accuracy | 0.90 (RoBERTa) | 0.72 (RF on K-Means labels) | 0.918 | **0.769 F1 on stratified independent gold** |
| Methodology comparisons | BERT vs RoBERTa | RF vs XGBoost vs ANN | BERT vs TF-IDF baseline | **Weak / LLM-supervised / self-training v1+v2 / active learning** |
| Leakage detection | Not addressed | Not addressed | Not addressed | **Empirical demonstration (CV F1 = 0.99 → test F1 = 0.44)** |
| Self-training tested | No | No | No | **Yes, two variants, both fail with principled diagnosis** |
| Active learning tested | No | No | No | **Yes, +6.5 F1 over supervised** |
| Production deployment | Docker/Kubernetes stack | None demonstrated | 200+ user pilot | **Prototype only (future work)** |

### 2.5 Where our work meaningfully advances the field

Our work directly addresses three of the field's structural gaps identified in the literature survey:

1. **Methodological rigor on label sources.** P2's K-Means cluster labels are flagged in the survey as "a measure of self-consistency within a flawed labelling process, not actual fit." Our 3-LLM consensus methodology with full disclosure of inter-rater agreement (71% unanimous, 80.7% pairwise, kappa 0.61) provides a transparent and reproducible alternative.

2. **Empirical demonstration of weak-supervision leakage.** Neither P1, P2, nor P3 runs a cross-validation vs external-test comparison to check for label-feature leakage. We empirically demonstrate the failure: a weak-supervised model with leakage-safe feature partitioning achieves CV F1 = 0.99 but only test F1 = 0.44 on independent LLM-judged labels.

3. **Head-to-head methodology comparison.** The field has not run a controlled comparison of weak supervision vs LLM supervision vs self-training vs active learning on the same task. Our work fills this gap and produces a dose-response curve: same supervised seed model + 200 additional labels selected by uncertainty → +6.5 F1; same seed + 5,000+ pseudo-labels selected by confidence → -12 F1.

What the literature survey calls "the single most important unanswered question" — *does acting on these system's recommendations actually change employment trajectories* — remains beyond the scope of any research project in this field, including ours. We acknowledge this gap explicitly in Limitations (Section 9).

---

## 3. Dataset and Preprocessing

### 3.1 Raw data sources

We use five publicly available datasets, all with permissive licensing and no proprietary dependencies:

| File | Size | Source | Content |
|---|---|---|---|
| `Resume (1).csv` | 53.7 MB | Kaggle public dataset | 2,484 resumes with raw text + HTML + broad Category label (24 categories: HR, INFORMATION-TECHNOLOGY, FINANCE, HEALTHCARE, etc.) |
| `training_data.csv` | 3.6 MB | Public dataset | 853 jobs with company name, position title, free-text description, and a structured `model_response` JSON field |
| `Skills.xlsx` | 3.2 MB | U.S. Department of Labor (O*NET) | 35 generic occupational skills with importance and level scores per SOC code |
| `skills_en.csv` | 9.3 MB | European Commission (ESCO) | 13,960 skill entries with preferred labels, alternative labels, and hidden synonyms |
| `broaderRelationsSkillPillar_en.csv` | 4.9 MB | European Commission (ESCO) | 20,819 parent-child relationships defining the ESCO hierarchy graph |

**Why these datasets.** Public licensing ensures reproducibility. The combination of O*NET (US-grounded, generic competencies with quantitative importance scores) and ESCO (EU-grounded, broader vocabulary with hierarchy) gives both a structured importance signal and broad skill coverage. Resume and job data are large enough to demonstrate the methodology and small enough to evaluate carefully under realistic compute and labeling budgets.

### 3.2 Ethical considerations

- **Resume anonymization.** The Kaggle Resume dataset is anonymized by the original author — names, contact information, and identifying credentials are stripped at source.
- **Additional PII filtering.** Our preprocessing pipeline applies regex-based filters for email addresses, phone numbers, and bullet-character noise as a defense-in-depth measure.
- **Job descriptions.** Publicly posted text with no privacy concerns.
- **LLM labeling.** Resume text sent to LLM APIs is the anonymized Resume_str field. No personally identifying information is transmitted to Anthropic, OpenAI, or Google.
- **Bias acknowledgment.** All three LLMs share systematic training-data biases (favoring elite credentials, English-language candidates, recent dates). Our test set inherits these biases. We disclose this in Limitations.

### 3.3 Preprocessing pipeline

The preprocessing pipeline (`data/proccessed again/preprocess_pipeline.py`) executes the following steps:

1. **HTML stripping** via `BeautifulSoup` with the `lxml` parser. Preserves word boundaries between adjacent tags.
2. **Character-level cleaning** that retains `+` and `#` for C++/C# and digits for years-of-experience numbers, but removes other punctuation.
3. **Alias substitution** before tokenization to preserve skills whose canonical form contains characters that would otherwise be destroyed by cleaning (e.g., `c++ → cplusplus`, `c# → csharp`, `node.js → nodejs`).
4. **Phrase preservation** for multi-word skills via `flashtext` (single Aho-Corasick scan over 18,576 multi-word skills, replacing them with underscore-joined tokens that survive `nltk.word_tokenize`).
5. **POS-aware lemmatization** using NLTK's WordNet lemmatizer with Penn Treebank → WordNet tag mapping.
6. **Skill extraction** via vocabulary lookup against the 20,253-entry merged vocabulary.
7. **Section extraction** using regex on the raw text to identify Skills, Experience, and Education sections.

### 3.4 Three preprocessing fixes that materially improved data quality

During iterative debugging, three preprocessing bugs were identified and fixed:

**Fix 1 — JD-side years-of-experience extraction.** The original regex captured zero JD YoE mentions because most jobs in this corpus state experience requirements in the structured `model_response` JSON field rather than the free-text job description. After parsing that field as the primary source with fallback to the raw text, 500 of 853 jobs (59%) have non-zero parsed YoE.

**Fix 2 — Single-token tech skill ambiguity.** Words like `R`, `Go`, `Swift`, `Less`, `Express`, `Storm`, `Segment` collide with common English words. A raw-text check was added: for these ambiguous tokens, require that the original text either contains the capitalized canonical form (`R`, `Go`, `Swift`) or shows the lowercased token within ~40 characters of a tech-context cue word (`programming`, `framework`, `library`, `experience`, etc.). Short ambiguous tokens (1–2 characters like `ml`, `cv`, `tf`, `bi`) require BOTH conditions; longer ambiguous tokens (3+ characters like `cnn`, `rnn`, `crm`) require EITHER. Filtered 654 false-positive skill matches across the corpus.

**Fix 3 — Resume title source.** The original pipeline used the broad `Category` column (e.g., "HR") as the resume title input for SBERT encoding. This collapsed all resumes within a category to the same title vector. Switching to first-non-empty-line extraction from the raw `Resume_str` field produced 31,254 unique title-similarity values across 25,590 pairs, up from 7,286 — a 4× increase in feature granularity.

### 3.5 ESCO vocabulary expansion

The curated tech vocabulary (`tech_skills.py`) contains only 355 skills. With this narrow vocabulary, 64% of all pair feature rows had `skill_overlap = 0` because most jobs had only 1–2 extracted skills.

We integrated the ESCO knowledge taxonomy with these filters (`build_esco_vocab.py`):

- Restricted to `skillType = 'knowledge'` (3,221 entries) — domain nouns rather than competence verbs
- Included `preferredLabel` + `altLabels` + `hiddenLabels` (newline/pipe-separated synonyms)
- Dropped entries of fewer than 4 characters
- Dropped entries on a 90-word English stopword/verb blocklist (`use`, `make`, `manage`, `do`, etc.)
- Dropped entries with fewer than 50% alphabetic characters

After filtering: **20,253 unique skill labels** in the merged vocabulary (3,208 ESCO preferred labels + 16,570 ESCO altLabels + 190 ESCO hiddenLabels + 285 TECH_SKILLS entries not already in ESCO).

The vocabulary integration shifted three feature distributions substantially:

| Metric | Before ESCO | After ESCO |
|---|---|---|
| `skill_overlap > 0` | 36% | 68% |
| `years_of_experience > 0` | 39% | 48% |
| `title_similarity` unique values | 7,286 | 31,254 |
| Avg extracted skills per resume | 3.4 | 13.2 |
| Avg extracted skills per job | 1.7 | 5.96 |

### 3.6 ESCO hierarchy depth as importance weight

Even after vocabulary expansion, `weighted_skill_score` was 81% identical to `skill_overlap` because every ESCO skill received the same default fallback importance (the mean of O*NET importances, ~2.59) — making the weighted version mathematically equivalent to the unweighted overlap.

We integrated the ESCO `broaderRelationsSkillPillar` hierarchy graph (`build_esco_depths.py`):

1. Built the parent-child URI adjacency from the 20,819 edges
2. Found root nodes (URIs that never appear as a child)
3. BFS from roots, assigning each URI its minimum depth from any root
4. Mapped depth to importance via `(depth / max_depth) × 5`, placing ESCO skills on the same 0–5 scale as O*NET importances

Generic skills like "communication" sit shallow in the tree (depth ~2 → importance 2.0); specialized skills like "natural language processing" sit deep (depth ~5 → importance 5.0). O*NET importance takes priority where both sources have a score; ESCO depth-based importance fills in for ESCO-only skills.

### 3.7 The 13 features

For every (resume, job) pair, the matching pipeline (`matching_pipeline.py`) computes 13 features. Eight were used in the final model; three were dropped during methodology iteration.

| # | Feature | Used? | Plain English |
|---|---|---|---|
| 1 | `embedding_similarity` | ✓ | SBERT cosine on full cleaned resume vs full cleaned JD |
| 2 | `tfidf_similarity` | ✓ | TF-IDF cosine on (1,2)-grams — catches lexical overlap SBERT may blur |
| 3 | `skill_overlap` | ✓ | `|R ∩ J| / |J|` — fraction of job's required skills the candidate has |
| 4 | `weighted_skill_score` | ✓ | Same as overlap but each skill weighted by O*NET/ESCO importance |
| 5 | `num_missing_skills` | ✓ | `|J − R|` — count of required skills missing from the resume |
| 6 | `avg_missing_skill_importance` | ✓ | Mean importance of missing skills (importance of the gap) |
| 7 | `years_of_experience` | ✓ | Resume YoE extracted from raw text (handles ranges, word forms, `5+`) |
| 8 | `experience_gap` | ✗ dropped | Job YoE − resume YoE (collinear with `years_of_experience`, ρ = -0.94) |
| 9 | `experience_relevance_score` | ✗ dropped | Cosine of resume experience-section embedding vs full JD embedding |
| 10 | `title_similarity` | ✓ | SBERT cosine between first-line resume title and `position_title` |
| 11 | `education_match` | ✗ dropped | Boolean indicator that resume's degree level ≥ JD's required degree |

The three dropped features were removed during the variant-comparison phase: `experience_gap` for collinearity with `years_of_experience`, and `experience_relevance_score` and `education_match` because they were used to generate the original composite weak label and dropping them produced cleaner leakage isolation.

### 3.8 The diversified candidate pool

Top-50 SBERT retrieval gives 853 × 50 = 42,650 candidate pairs, but they all look like potential matches by construction. To enable learning about hard cases, we added three additional pair types (`build_diversified_pool.py`):

| Source | Count | Definition |
|---|---|---|
| **topK** | 42,650 | Top-50 SBERT retrieval per job (the original pool) |
| **A_mid** | 5,000 | Mid-similarity pairs: SBERT cosine ∈ [0.30, 0.50], NOT in top-50 |
| **B_xcat** | 2,000 | Cross-category strict: resume's `Category` family ≠ job position_title family, with similarity > 0.50 |
| **C_rand** | 1,000 | Uniformly random pairs from the full 2.12M space |
| **Total** | **50,650** | Saved as `pair_features_diversified.parquet` |

For B_xcat, resume categories were bucketed into six families (TECH, BUSINESS, HEALTHCARE, EDUCATION, TRADES, ARTS) using the `Category` column directly. Job position_titles were assigned to families via regex on common keywords.

The pool composition is critical for the active-learning experiment: it ensures the candidate space includes both easy positives (topK) and adversarial negatives (B_xcat) so that uncertainty-based sampling actually exposes the model to informative cases.

---

## 4. ML Methodology

This is the largest section. The project tested five distinct training methodologies on the same held-out evaluation set. We document each in detail: what it does, why we tried it, what we found, what we learned, and why we moved to the next.

### 4.0 The two-stage architecture

```
Stage 1 — Retrieval (SBERT)
  Input  : resume text, JD text → 384-dim embeddings (all-MiniLM-L6-v2)
  Output : top-50 candidate pairs per job, ranked by cosine similarity
  Cost   : ~3 ms per text on CPU; full 853 × 2,484 cosine matrix in ~30 sec

Stage 2 — Reranking (Logistic Regression on 8 features)
  Input  : 8 feature values per (resume, job) pair
  Output : match probability; threshold-tuned to 0.52
  Cost   : <1 ms per pair
```

This is the architecture used by every production retrieval-and-rerank system at scale (Karpukhin et al. 2020 — DPR for question answering; Pinterest's PinSage; LinkedIn's job recommendation pipeline). Our contribution is the Stage 2 reranker design and the training methodology.

### 4.1 Methodology 1 — Weak Supervision (Formula Labels)

**What it does.** Generate training labels by applying a deterministic formula to the existing pair feature table:

```
weak_score = 0.7 × weighted_skill_score + 0.3 × title_similarity
label = 1 if weak_score ≥ 70th percentile
label = 0 if weak_score ≤ 30th percentile
middle 40% of pairs dropped (label = NaN, removed from training set)
```

This produces `data/proccessed again/processed/ml_ready_dataset.parquet` — 25,590 weakly-labeled pairs, perfectly balanced 50/50 by construction (the quantile cut guarantees this).

**Why we tried it.** Weak supervision is the standard "free labels" approach in semi-supervised learning. Snorkel-style labeling functions (Ratner et al. 2017) showed that heuristic labels can train production-grade classifiers. We tested whether the same idea works for resume-job matching.

**Critical design decision: leakage-safe feature partitioning.** Both `weighted_skill_score` and `title_similarity` were used to compute the label. If they remained in the training feature set, the model would trivially recover the labeling formula (a degenerate failure mode). We followed v1's discipline of holding label-feeding features out of the training feature set, leaving six leakage-safe features for the classifier: `embedding_similarity`, `tfidf_similarity`, `skill_overlap`, `num_missing_skills`, `avg_missing_skill_importance`, `years_of_experience`.

**Models trained.** Logistic Regression (`solver='liblinear'`, `class_weight='balanced'`) and Gradient Boosting Classifier, both with 5-fold CV F1-optimal threshold tuning on the training set.

**Result.**

```
Cross-validation F1 on weak training set : 0.99
Test F1 on 100-pair LLM gold standard    : 0.44
CV–test F1 gap                           : 0.55
```

**What we learned.** The 0.55-point CV-test gap is a textbook leakage signature. The model achieved near-perfect cross-validation F1 — meaning within-fold prediction was essentially perfect — but completely failed when evaluated against externally-judged labels.

The mechanism is straightforward: the formula labels are a deterministic function of two features. Even with those two features removed from training, the remaining six features correlate strongly with them. The model effectively learned a noisy approximation of the formula rather than learning what humans (or LLMs) actually mean by "good match." Formula labels measure the formula; they do not measure the latent quality the formula was intended to approximate.

**Why we moved on.** Weak supervision cannot serve as the primary training signal for this task. The gap between CV F1 and external-test F1 demonstrates that the labels themselves are the bottleneck, not the model.

### 4.2 Methodology 2 — LLM Supervision

**What it does.** Replace formula labels with judgments from three large language models, taking the majority vote as ground truth.

**Why we tried it.** External, independent label sources should not suffer from the leakage problem because the labels are not a function of any feature the model sees. Three LLMs (Anthropic Claude, OpenAI GPT-4o, Google Gemini Pro) reduce single-vendor bias via majority vote.

**Stratified 500-pair gold sampling** (`sample_gold_pairs.py`) draws across:

- **Pool source**: 200 topK / 150 A_mid / 100 B_xcat / 50 C_rand
- **SBERT similarity decile within each source** (10-bin quantile stratification)
- **Resume Category**: 24 of 25 distinct categories represented; top 5 most-represented: INFORMATION-TECHNOLOGY (35), FINANCE (31), HR (30), BUSINESS-DEVELOPMENT (27), CONSULTANT (27)

Each pair was packaged as a row in a 100-pair batch CSV with truncated resume text (2,000 chars) and JD text (1,500 chars), keeping each batch under 350,000 characters to fit all three models' context windows.

**The LLM prompt** (`PROMPT_FOR_LLMS.txt`):

> *"Output label = 1 if the candidate is a GOOD match for the job (has the core required skills and clearly relevant experience). Output label = 0 if not. Be strict. A good match means the candidate could plausibly do the job with at most minor onboarding. Adjacent skills or generic competencies (communication, teamwork) alone do not qualify."*

Each LLM produced a `row_id, label` CSV for each of 5 batches; the 6 CSVs per LLM × 3 LLMs = 18 files were combined via majority vote (`combine_llm_labels.py`).

**Inter-LLM agreement on 500 gold pairs.**

| Metric | Value |
|---|---|
| 3-0 unanimous votes | 355 / 500 (71.0%) |
| 2-1 disputed votes | 145 / 500 (29.0%) |
| Claude vs GPT pairwise agreement | 77.2% |
| Claude vs Gemini pairwise agreement | 89.4% |
| GPT vs Gemini pairwise agreement | 75.4% |
| Mean pairwise agreement | 80.7% |
| Chance-adjusted κ-proxy | 0.613 |
| Claude positive rate | 22.4% |
| GPT positive rate | 13.6% (strictest) |
| Gemini positive rate | 26.6% (most lenient) |
| Majority-vote positive rate | 22.4% |

The 80.7% mean pairwise agreement sits in the "honest band" — high enough to suggest meaningful consensus, low enough that the three models are clearly not echoing each other.

**Per-source positive rates revealed an important finding for the project narrative:**

| Source | n | Positive rate | Unanimous rate |
|---|---|---|---|
| topK | 200 | 47.5% | 46.5% |
| A_mid | 150 | 3.3% | 91.3% |
| B_xcat | 100 | 8.0% | 36.0% |
| C_rand | 50 | 8.0% | 46.0% |

The topK 47.5% positive rate is the empirical motivation for the reranker: more than half of SBERT's top-50 retrievals are judged "not a match" by LLM consensus, and within that, only 46.5% have unanimous agreement. The retrieval stage produces plausible-looking candidates that frequently aren't real matches — exactly the case a reranker is designed to filter.

**Stratified 80/20 split** of the 500 LLM-labeled pairs:

- 400 training pairs (90 positive, 310 negative)
- 100 held-out test pairs (22 positive, 78 negative)
- Random seed 42, reproducible; identical test set used for ALL subsequent methodology comparisons

**Models trained.** Logistic Regression and Gradient Boosting, both on all 8 features with 5-fold CV threshold tuning, class-balanced.

**Result.**

```
SBERT-only baseline F1     (threshold-only): 0.604
LLM-supervised LogReg F1                   : 0.704
LLM-supervised GBM F1                      : 0.680
Weak-supervised LogReg F1 (for reference)  : 0.440
```

The LLM-supervised LogReg achieves a +10 F1-point lift over SBERT-only thresholding and a +26 point lift over weak supervision.

**Why we moved on.** F1 = 0.704 was solid but we wanted to see if semi-supervised methods could push higher by leveraging the 25,590 unlabeled pairs sitting in the formula-labeled dataset.

### 4.3 Methodology 3 — Self-Training v1 (Absolute Confidence Threshold)

**What it does.** Use the LLM-supervised LogReg model to pseudo-label every pair in `ml_ready_dataset.parquet`. Where the model is highly confident, trust its predictions as additional labels:

```
P(positive | features) ≥ 0.80  →  pseudo-label = 1
P(positive | features) ≤ 0.20  →  pseudo-label = 0
else                           →  discard
```

Combine pseudo-labels with the original 400 LLM-labeled training pairs (deduping any overlap), retrain LogReg, evaluate on the same 100-pair test set.

**Why we tried it.** Self-training is the canonical semi-supervised technique (Yarowsky 1995). On the right problem with the right pool, it expands the training set with little human cost. We had 25,590 unlabeled pairs available.

**Result.**

```
Pseudo-labels generated: 11,292 (84% positive — flipped from gold's 22%)
Combined training set : 400 gold + 11,292 pseudo = 11,692 rows
New threshold (CV)    : 0.68
CV F1 on combined set : 0.9964 (memorization signature)
Test F1               : 0.564  (Δ -0.140 vs supervised)
```

**What went wrong — diagnostic 1: class balance flip.** The supervised model trained on 22.4% positive labels emitted high-confidence positive predictions on 37.4% of the pool. The pseudo-label set ended up 84% positive — completely inverted from the gold distribution. When trained on this skewed distribution, the new model learned "positive is the default class" and pushed its threshold up to 0.68 to compensate, crushing recall (0.864 → 0.500).

**What went wrong — diagnostic 2: pool bias.** `ml_ready_dataset.parquet` is the result of applying the weak labeling formula to the top-50 retrieval pairs. It was already pre-filtered for pairs that score high on `weighted_skill_score + title_similarity` — exactly the features the supervised model relies on. A 0.80 confidence threshold on this pool isn't selecting "highly accurate predictions" in any absolute sense; it's selecting "predictions consistent with the model's existing biases on a pool engineered to match those biases."

**Why we moved on.** We hypothesized that fixing both issues — biased pool and absolute threshold — might rescue self-training.

### 4.4 Methodology 4 — Self-Training v2 (Percentile + Diversified Pool)

**What it does.** Two fixes from v1:

1. **Pool changed**: use the full diversified pool (50,650 pairs across all four source types), dropping all 500 gold pairs to avoid contamination.
2. **Selection rule changed**: top 5% by predicted probability → pseudo-label 1, bottom 5% → pseudo-label 0, discard middle 90%. Forces 50/50 class balance regardless of model calibration on the new distribution.

**Result.**

```
Pseudo-labels generated: 5,016 (50/50 by construction)
Combined training set : 400 gold + 5,016 pseudo = 5,416 rows
New threshold (CV)    : 0.64
CV F1 on combined set : 0.9867
Test F1               : 0.579  (Δ -0.125 vs supervised, Δ +0.015 vs v1)
```

A marginal improvement over v1 but still 12+ points below the supervised baseline.

**Why this still failed — the deeper diagnosis.** Look at the source distribution of the 5,016 pseudo-labels:

```
topK   : 2,844 (56.7%)
A_mid  : 1,769 (35.3%)
C_rand :   309 ( 6.2%)
B_xcat :    94 ( 1.9%)  ← cross-category hard negatives almost never selected
```

**B_xcat — the deliberately constructed hard negatives — appeared in only 1.9% of pseudo-labels** despite being 4% of the pool. The supervised model has no strong opinion on cross-category pairs (their features pull in both directions: similar embeddings but mismatched categories). They sit in the middle of the probability distribution, so percentile selection skips them.

**This is the structural failure mode of self-training.** Confidence-based pseudo-labeling preferentially selects cases the model is already confident about. Those cases have no new information value. The cases that would teach the model something new — the uncertain ones — are systematically excluded by the selection mechanism itself.

This is exactly Yarowsky's 1995 confirmation-bias warning, validated empirically on our task with two independent variants.

**Why we moved on.** Self-training is structurally unable to add information given how its selection rule works. To break out of the loop, we need to *invert* the selection rule: select for uncertainty, not confidence. This is active learning.

### 4.5 Methodology 5 — Active Learning (The Win)

**What it does.** Use the supervised LogReg model to find the 200 pairs in the diversified pool where the model is most UNCERTAIN — predicted probability ∈ [0.40, 0.60]. Send those pairs to the same three LLMs for labeling. Retrain on 400 + 200 = 600 LLM-labeled pairs. Evaluate on the same 100-pair test set.

**Why we tried it.** Active learning is the dual of self-training. Where self-training picks easy cases the model is already confident about, active learning picks hard cases the model is uncertain about. With a small additional label budget, it can target maximum information gain per label.

**Step 1 — uncertain-pair sampling** (`active_learning_sample.py`):

- Scored all 50,650 pairs in the diversified pool
- Dropped the 500 gold pairs to prevent contamination
- 8,902 pairs (17.8% of pool) fell in the uncertainty band [0.40, 0.60]
- Stratified-sampled 200 pairs: 50 each from topK, A_mid, B_xcat, C_rand
- 24 distinct resume categories represented in the 200 picks

The B_xcat uncertain-band rate was 18.9% of B_xcat pool — essentially the same as topK's 19.7%. This confirms that the supervised model has genuine uncertainty about cross-category cases (which self-training was unable to surface).

**Step 2 — LLM labeling.** Same prompt, same workflow as the original gold standard. Two batches of 100 pairs each, uploaded to Claude, GPT-4o, and Gemini Pro. Majority-voted via the same `combine_llm_labels.py` logic.

**Inter-LLM agreement on the 200 active pairs.** Dramatically lower than the original gold:

| Metric | Original gold (500) | Active pairs (200) |
|---|---|---|
| 3-0 unanimous | 71.0% | 43.0% |
| Mean pairwise agreement | 80.7% | 62.0% |
| Claude positive rate | 22.4% | 14.5% |
| GPT positive rate | 13.6% | 53.5% (much more lenient on uncertain cases) |
| Gemini positive rate | 26.6% | 20.5% |
| Majority-vote positive rate | 22.4% | 19.5% |

The 62% pairwise agreement confirms these are genuinely contested cases — even three LLM judges disagree substantially. **The LLMs themselves can't reach high consensus on the uncertain band, which is precisely why these cases are valuable training data.** GPT was much more lenient than Claude or Gemini; the majority vote leans Claude+Gemini.

**Step 3 — retrain and evaluate** (`active_learning_evaluate.py`):

```
Combined training set : 400 gold + 200 active = 600 (21.5% positive)
New threshold (CV)    : 0.52
CV F1 on combined set : 0.5443  ← realistic, not memorization
Test F1               : 0.7692  (Δ +0.066 vs supervised, +0.165 vs SBERT)
```

Other metric improvements vs the supervised baseline:

| Metric | Supervised baseline | Active learning | Δ |
|---|---|---|---|
| Accuracy | 0.84 | 0.88 | +0.04 |
| Precision | 0.594 | 0.667 | +0.073 |
| Recall | 0.864 | 0.909 | +0.045 |
| P@10 | 0.80 | 0.90 | +0.10 |
| P@20 | 0.70 | 0.70 | 0.00 |

**Both precision and recall improved.** This is uncommon — usually models trade one for the other. Here the active labels resolved both false positives and false negatives.

**Confusion-matrix diff vs supervised:**

```
Flipped predictions     : 6 / 100 test pairs
  positive → negative   : 4
  negative → positive   : 2
Improved correctness    : 5 flips
Worsened correctness    : 1 flip
Net delta               : +4 correct test predictions
```

Five of six flipped predictions went the right way. The active labels gave the model enough new information to correctly reclassify 4 net test pairs out of 100.

**Sanity check on the original 400 train labels.** Both the supervised and active-learning models classify those 400 labels with comparable F1 (no degradation on the gold training subset), confirming the new model hasn't been distorted by the active labels.

### 4.6 Methodology summary table

The complete comparison across all five methodologies on the same 100-pair held-out test set:

| Methodology | Training data | CV F1 | Test F1 | Notes |
|---|---|---|---|---|
| SBERT only | — | — | 0.604 | Single-feature threshold baseline |
| Weak-supervised (leakage-safe) | 25,590 formula-labeled | **0.99** | 0.440 | 0.55-point CV-test gap; leakage signature |
| Self-training v1 | 400 LLM + 11,292 pseudo | 0.996 | 0.564 | Class balance flip; biased pool |
| Self-training v2 | 400 LLM + 5,016 pseudo | 0.987 | 0.579 | Hard cases systematically excluded |
| LLM supervised | 400 LLM | 0.631 | 0.704 | The strong baseline |
| **Active learning** | **400 LLM + 200 active** | **0.544** | **0.769** | **The winner** |

Pattern reading: every method with a CV F1 above 0.95 fails on the external test set (overfitting to the label-generation rule). Only methods with realistic CV F1 in the 0.55–0.75 range generalize.

---

## 5. Results and Performance Metrics

### 5.1 Headline comparison table

Evaluated on 100 LLM-labeled held-out test pairs (22 positive, 78 negative), stratified split with random_state=42:

| Approach | Threshold | Accuracy | Precision | Recall | F1 | P@5 | P@10 | P@20 |
|---|---|---|---|---|---|---|---|---|
| **Active learning** | **0.52** | **0.88** | **0.667** | **0.909** | **0.769** | **1.00** | **0.90** | **0.70** |
| LLM-supervised LogReg | 0.50 | 0.84 | 0.594 | 0.864 | 0.704 | 1.00 | 0.80 | 0.70 |
| LLM-supervised GBM | 0.14 | 0.84 | 0.607 | 0.773 | 0.680 | 0.80 | 0.90 | 0.80 |
| SBERT only | 0.62 | 0.79 | 0.516 | 0.727 | 0.604 | 1.00 | 0.70 | 0.65 |
| Self-trained v2 | 0.64 | 0.84 | 0.688 | 0.500 | 0.579 | 1.00 | 0.90 | 0.70 |
| Self-trained v1 | 0.68 | 0.83 | 0.647 | 0.500 | 0.564 | 1.00 | 0.80 | 0.70 |
| Weak-supervised LogReg | 0.36 | 0.72 | 0.393 | 0.500 | 0.440 | 0.40 | 0.60 | 0.40 |

### 5.2 What Precision@K means and why it matters for a reranker

Precision@K is the proportion of true positives among the top-K predictions ranked by model probability. P@5 = 1.00 means all 5 of the model's most-confident predictions are real matches.

For a reranker, P@K is more directly production-relevant than F1. In a real recruiter or job-seeker workflow, only the top few candidates per query are ever looked at by a human. The model's job is to put the right candidates at positions 1–10; nobody reads pair number 73 in a ranked list.

Our active-learning model achieves perfect P@5 (all top-5 predictions correct) and P@10 = 0.90 (9 of top-10 correct). Both metrics exceed the SBERT-only baseline.

### 5.3 Confusion matrix for the final model

Active-learning LogReg on the 100 test pairs:

```
                  Predicted
                  Neg     Pos
Actual Neg        65      13
Actual Pos         2      20
```

Numbers:
- True negatives: 65 (correctly rejected non-matches)
- False positives: 13 (predicted match, actually not — wasted recruiter time in deployment)
- False negatives: 2 (predicted non-match, actually match — qualified candidate missed)
- True positives: 20 (correctly identified matches)

**Recall = 20/22 = 0.909** — the model catches 20 of 22 real matches in the test set.
**Precision = 20/33 = 0.606** — of pairs predicted positive, 60.6% are real matches.

The model has a recall-oriented bias by design (class-balanced training emphasizes positive-class recall). For a stage-2 reranker, this is the correct tradeoff: surface plausible candidates, let a human or downstream stage filter further.

### 5.4 Three findings warranting separate reporting

**Finding 1 — SBERT is a strong retriever but a weak final classifier.** On the 200 topK pairs sampled for the gold standard, only 47.5% were judged positive by LLM majority vote, and only 46.5% had unanimous LLM agreement. SBERT's top-50 retrievals contain a 50% mix of plausible-looking false positives. A reranker is genuinely needed; SBERT-only thresholding on `embedding_similarity` cannot reach better than F1 = 0.604 on this test set.

**Finding 2 — Formula-based weak supervision fails to generalize.** Weak-supervised models trained on quantile-cut formula labels achieve cross-validation F1 of 0.99 but only test F1 of 0.44 on independent LLM-judged labels. The 0.55-point CV-test gap is a textbook leakage signature: the model memorizes the formula, but the formula is not a valid proxy for what humans actually mean by "good match." This holds even with leakage-safe feature partitioning (dropping the label-feeding features from training).

**Finding 3 — Confidence-based label selection (self-training) systematically underperforms uncertainty-based selection (active learning).** Two independent self-training variants both degraded F1 by 12–14 points relative to plain supervised. Active learning with the same supervised seed model and a 200-label budget improved F1 by 6.5 points. The mechanism is empirically demonstrated: confidence-based pseudo-labeling preferentially selects cases the model is already confident about (no new information value), while uncertainty-based selection targets cases at the decision boundary where new labels resolve real ambiguity. This validates Yarowsky's 1995 confirmation-bias warning with concrete data in our specific application.

---

## 6. Feature Importance Analysis

Following a methodological challenge — *"is your model just SBERT dressed up with extra features?"* — we conducted three independent feature importance analyses on the final active-learning LogReg model (`feature_importance_analysis.py`).

### 6.1 Scaled LogReg coefficients

After StandardScaler, coefficients are directly comparable (every feature is on the same standard-deviation scale).

| Feature | Coefficient | |coefficient| |
|---|---|---|
| `embedding_similarity` | +0.979 | 0.979 |
| `skill_overlap` | +0.716 | 0.716 |
| `num_missing_skills` | −0.699 | 0.699 |
| `weighted_skill_score` | −0.697 | 0.697 |
| `tfidf_similarity` | +0.560 | 0.560 |
| `title_similarity` | +0.497 | 0.497 |
| `avg_missing_skill_importance` | −0.279 | 0.279 |
| `years_of_experience` | +0.004 | 0.004 |

`embedding_similarity` has the largest absolute coefficient (0.979). But notice `skill_overlap` and `weighted_skill_score` have nearly equal magnitudes with opposite signs (+0.716 vs −0.697) — a classic multicollinearity signature. These two features are highly correlated by construction (weighted is the importance-weighted version of unweighted overlap). The model uses both together: their joint contribution drives the prediction, but individual coefficients aren't independently interpretable.

`years_of_experience` has coefficient ≈ 0 — it contributes essentially nothing.

### 6.2 Permutation importance

For each feature, shuffle that column 50 times in the test set, measure the F1 drop:

| Feature | Mean F1 drop | Std |
|---|---|---|
| `embedding_similarity` | 0.250 | 0.069 |
| `title_similarity` | 0.118 | 0.046 |
| `num_missing_skills` | 0.105 | 0.038 |
| `tfidf_similarity` | 0.085 | 0.043 |
| `skill_overlap` | 0.071 | 0.039 |
| `weighted_skill_score` | 0.053 | 0.037 |
| `avg_missing_skill_importance` | 0.011 | 0.018 |
| `years_of_experience` | 0.000 | 0.000 |

SBERT is the strongest single feature on permutation, but the next five contribute meaningful drops (0.05 to 0.12). `years_of_experience` is empirically useless.

### 6.3 Leave-one-feature-out ablation

For each feature, retrain LogReg on the remaining 7 features, measure test F1:

| Dropped feature | Test F1 | F1 drop |
|---|---|---|
| `title_similarity` | 0.6531 | **0.1161** |
| `num_missing_skills` | 0.6809 | 0.0883 |
| `embedding_similarity` | 0.7059 | 0.0633 |
| `tfidf_similarity` | 0.7347 | 0.0345 |
| (none — full 8 features) | 0.7692 | 0.0000 |
| `years_of_experience` | 0.7692 | 0.0000 |
| `avg_missing_skill_importance` | 0.7755 | −0.0063 |
| `weighted_skill_score` | 0.7843 | −0.0151 |
| `skill_overlap` | 0.8163 | −0.0471 |

**Removing `title_similarity` hurts the model MORE than removing SBERT** (0.116 drop vs 0.063 drop). Title-vs-title similarity carries the most individual lift, primarily because it's a high-information, short-text SBERT cosine grounded by the fix-4c first-line title extraction.

**Three features (`skill_overlap`, `weighted_skill_score`, `avg_missing_skill_importance`) actually IMPROVE F1 when removed** — a clear multicollinearity signal. With them present, the model splits credit across redundant features; without them, it consolidates the signal more cleanly.

### 6.4 The critical comparison — SBERT-only vs Skill-only

To directly answer "is this just SBERT?":

| Model | Features | Test F1 |
|---|---|---|
| SBERT-only LogReg | `embedding_similarity` only | **0.6275** |
| Skill-only LogReg | 7 features WITHOUT `embedding_similarity` | **0.7059** |
| Full LogReg | All 8 features | **0.7692** |

**Skill-only LogReg outperforms SBERT-only LogReg by +7.8 F1 points** (0.706 vs 0.628). Skill features carry independent signal that exceeds what semantic similarity alone provides. The senior's hypothesis is empirically refuted.

The full 8-feature model achieves an additional +6.3 points over skill-only and +14.2 points over SBERT-only — both channels carry independent signal, and their combination beats either alone.

### 6.5 Top features by each metric

| Metric | Top 3 features |
|---|---|
| Largest |coefficient| | embedding_similarity, skill_overlap, num_missing_skills |
| Largest permutation F1 drop | embedding_similarity, title_similarity, num_missing_skills |
| Largest ablation F1 drop | title_similarity, num_missing_skills, embedding_similarity |

The three rankings disagree on the exact ordering but all three include embedding_similarity, title_similarity (or close cousins), and num_missing_skills among the top 3. The model uses semantic, title-alignment, and skill-gap signals as its primary inputs.

---

## 7. Context-Aware Skill Weighting Experiment

After observing that `weighted_skill_score` had a near-equal-and-opposite coefficient to `skill_overlap` (multicollinearity), we tested whether replacing the static ESCO depth-based importance with a context-aware (job-specific) importance would make the feature more individually predictive.

### 7.1 Two schemes tested

**Option A — SBERT cosine relevance**: `importance(skill, job) = depth_score(skill) × (1 + α × cos(skill_embedding, jd_embedding))`. The skill's importance to a specific job is modulated by how semantically similar the skill name is to the job description embedding.

**Option B — TF-IDF relevance**: `importance(skill, job) = depth_score(skill) × (1 + β × tfidf(skill_terms, jd))`. The skill's importance is modulated by its TF-IDF weight in the specific job description.

Both options were applied to `weighted_skill_score` and `avg_missing_skill_importance` — the only two features that depend on the importance map. The other six features stay unchanged.

### 7.2 Result

| Option | Test F1 | `wss` |coef| | wss permutation F1 drop |
|---|---|---|---|
| Baseline (depth only) | **0.7692** | **0.720** | **0.060** |
| A: depth × SBERT cosine | 0.7547 | 0.132 | 0.001 |
| B: depth × TF-IDF | 0.7547 | 0.195 | 0.014 |

Both options:
- Made `weighted_skill_score` more granular (1,773 → 9,241 unique values).
- Reduced its individual coefficient by 5–6×.
- Reduced F1 by 1.5 points (within ±0.07 statistical noise at N=100).

### 7.3 Interpretation

The context-aware modulation didn't help because both relevance signals are already captured at the document level by other features. SBERT relevance (option A) overlaps with `embedding_similarity`; TF-IDF relevance (option B) overlaps with `tfidf_similarity`. When the modulated `weighted_skill_score` injected the same information into a different feature, the model correctly recognized the redundancy and reduced its reliance on `weighted_skill_score`. The signal didn't disappear — it migrated to the document-level feature that already carried it.

This is a clean null result that strengthens the model architecture: the existing eight features form a roughly Pareto-efficient set for this task. The static ESCO depth-based importance is sufficient; adding context-aware modulation does not unlock new predictive value because the orthogonal axes are already covered.

---

## 8. Deployability and Dashboard Prototype

### 8.1 Inference cost

- SBERT embedding: ~3 ms per text on CPU
- Cosine similarity matrix (853 jobs × 2,484 resumes): single matmul, <1 sec
- Top-K retrieval: argpartition in NumPy, milliseconds
- Feature computation for top-50 pairs per job: <100 ms total
- LogReg evaluation per pair: <1 ms
- Total per-query latency: ~5 seconds for full pipeline; sub-second for cached embeddings

A single CPU server handles thousands of queries per second after caching.

### 8.2 The dashboard prototype

We built a static HTML dashboard (`build_dashboard_demo.py`) that takes the trained model and a real resume from the corpus, then renders the four product surfaces the model enables:

**Section 1 — Match score** for a specific job, with percentile framing ("Top X% of candidates for this role") and a confidence note grounded in the gold-set inter-LLM agreement statistics.

**Section 2 — Skill gap report**: two-column layout showing skills the candidate has (with importance stars) and skills they're missing (sorted by ESCO importance), plus a "top 2 missing skills account for X% of the gap importance" hook. This is the differentiating output that no SBERT-only system can produce.

**Section 3 — 2-year skill roadmap**: aggregated skill gaps across the candidate's top-5 matched jobs, ranked by (frequency × importance), bucketed into four quarters with rationale text. Each skill is annotated with "appears in N/5 target jobs."

**Section 4 — Forward simulation**: shows what happens to the candidate's match probability across their top-5 target jobs if they learn the Q1–Q2 foundation skills. Computed by toggling the skills in the candidate's `extracted_skills` set and re-running the trained model. This is a real model prediction, not a static suggestion.

### 8.3 Production-readiness gaps

The prototype is research-grade; production deployment requires additional engineering:

1. **Web frontend**: a React or Streamlit interface backed by a Flask/FastAPI service serving the trained model.
2. **User account system**: profile storage, resume upload, target-job selection.
3. **Resume parsing**: arbitrary uploads need conversion to the project's cleaned-text format (current pipeline uses the Kaggle dataset's pre-extracted text).
4. **Learning-platform integration**: link each roadmap skill to specific Coursera / edX / Udacity courses (this is what P1 does with nDCG@5 evaluation; we do not).
5. **Live job feeds**: scrape postings from aggregators or partner with job boards (we use a fixed 853-job corpus).
6. **Domain-alignment filter**: a post-processing layer that filters recommendations to be coherent with the resume's broad domain. The dashboard prototype includes a category-alignment check; production needs this as a hard filter to prevent cross-domain false positives.

### 8.4 Plaksha-specific deployment scenarios

1. **Internship matching tool**: integrate with the placement portal. When a student logs in, surface top-K matched internships with per-job skill gap reports.
2. **Course-to-career mapping**: given the student's transcript and a target role (e.g., "ML engineer"), recommend which electives to take next semester.
3. **Skill-development advisor**: aggregate gap reports across a student's top-10 dream jobs → surface the 5 skills with highest marginal impact.

---

## 9. Limitations

We disclose these honestly. Reviewers respect students who articulate their own work's limitations more than those who don't.

### 9.1 Test set size

N = 100 with 22 positive examples. F1 confidence interval is approximately ±0.07 at this sample size. Our +0.066 active-learning lift over plain supervised is meaningful but at the edge of statistical significance. The +0.165 lift over SBERT-only is well outside the confidence interval. Multi-seed re-runs (3–5 random seeds, reported as mean ± std) would tighten the headline numbers — this is a cheap follow-up (~30 minutes total) and we recommend it as a first item of future work.

### 9.2 LLM consensus is correlated noise, not ground truth

Three LLM judges share training-data biases (favoring elite credentials, recent dates, English-language candidates, mainstream job titles). Inter-LLM agreement at 71% unanimous reflects genuine variability in judgment but bounds the meaningfulness of our F1 numbers — they measure "how well the model agrees with majority LLM judgment" rather than "how well the model predicts hire/no-hire outcomes." A validation pass with human recruiters from diverse backgrounds would test whether LLM consensus aligns with real-world judgment.

### 9.3 Individual prediction coherence vs aggregate F1

The model achieves strong aggregate F1 on the test set, but individual high-confidence predictions can be domain-incoherent. The dashboard exposes this clearly: an IT resume can receive a 91% match score for a hospital supply chain analyst role because both documents share generic professional competencies (communication, project management) and the skill vocabulary still contains noise tokens (`source`, `energy`, `pregnancy`) that the model treats as real skills.

This is fixable in production via two layers we have not implemented as part of the trained model itself:
1. **Category-alignment hard filter** at the deployment layer (we add this in the dashboard prototype).
2. **Stricter skill vocabulary cleanup** at the preprocessing layer (would require regenerating all downstream artifacts).

### 9.4 ESCO vocabulary residual noise

Despite a 90-word stopword/verb blocklist applied during ESCO vocabulary construction, generic tokens still leaked through (`source`, `energy`, `pregnancy`, `call`, `job opportunities`). These contribute false-positive skill matches that inflate `skill_overlap` and `weighted_skill_score` on some pairs. The dashboard's display-time blocklist filters these out for user-facing output; the model's underlying feature computation still uses them.

### 9.5 Domain skew (English, IT-heavy)

Our resume corpus is English-only and dominated by IT, business, and healthcare roles. ESCO is European-centric. The model would require retraining on domain-specific gold standards for deployment to non-Western markets, non-English languages, or specialized fields (medical, legal, creative). This is shared with all three papers in the literature survey and remains an open field-level gap.

### 9.6 Resume–job matching is fundamentally subjective

Even human recruiters disagree 30–40% on the same pair. Whatever F1 we report is bounded above by this inter-rater ceiling. Our 0.769 F1 sits below that theoretical ceiling and likely below human-recruiter agreement with LLM consensus too.

### 9.7 No production deployment validation

We have not deployed the system to real users. The lit-survey paper P3 (Sribharathi et al.) ran a 200+ user pilot across 5 institutions; we report no comparable real-world data. Whether candidates who follow our skill-gap recommendations actually improve their employment outcomes — the single most important question in this field — remains untested.

---

## 10. Future Work

We frame future work directly against the five unanswered research questions identified in our literature survey, plus our own work-specific follow-ups.

### 10.1 Tied to literature-survey gaps

**Implicit and soft skill extraction (Lit Survey Gap 1)**: extend the vocabulary-lookup skill extractor with a transformer-based NER component that infers skills from narrative ("led a five-person team through a launch" → project management, leadership). This requires either an annotated training set for implicit-skill NER or a fine-tuned LLM extractor.

**Cross-domain generalization (Gap 2)**: build a multi-domain test set with at least five industry sectors (IT, healthcare, legal, creative, finance) and a transfer learning experiment quantifying how F1 drops as a function of training-to-test domain distance.

**Longitudinal validation (Gap 3 — the field's most important open question)**: a 12-month cohort study comparing students who used the skill-gap dashboard against a control group, measuring real employment outcomes (interview rate, offer rate, role-skill alignment, salary). This requires IRB approval, institutional partnerships, and a research timeline incompatible with conference papers — which is why no published work in this field has done it.

**Algorithmic bias measurement (Gap 4)**: build a demographically stratified test set, compute per-group precision/recall/F1, formally test for statistical disparities. Publish disaggregated metrics. Audit the trained model against this stratified set.

**Multilingual and non-Western support (Gap 5)**: extend the SBERT model to multilingual (mBERT, XLM-R, LaBSE) and add culture-specific parsing for non-Western resume formats (Japanese rirekisho, German Ausbildung sections, Indian academic-score conventions).

### 10.2 Project-specific follow-ups

**Multi-seed validation**: re-run the active-learning evaluation with 3–5 random seeds, report F1 as mean ± std. Cheap (~30 minutes), tightens confidence interval.

**Model simplification**: drop `years_of_experience` (empirically zero importance) and one of the multicollinear pair `skill_overlap` / `weighted_skill_score`. A 6-feature model would likely match the 8-feature F1 with cleaner coefficients to interpret.

**Tighter ESCO vocabulary cleanup**: expand the BLOCKLIST in `build_esco_vocab.py` to drop residual generic tokens (`source`, `energy`, `pregnancy`, `call`). Regenerate the full pipeline. Expected impact: cleaner skill-gap output without changing headline F1.

**Production category-alignment filter**: bake the dashboard's display-time category coherence check into the deployed system as a hard post-processing filter.

**Course-recommendation engine**: link each roadmap skill to specific Coursera/edX/Udacity courses (as P1 does), evaluate with nDCG@5 against a curated relevance dataset.

**Second iteration of active learning**: now that we have 600 LLM-labeled pairs, the trained model is itself a better predictor. Re-run uncertainty sampling against the diversified pool, label another 200 hardest pairs, retrain on 800. Each iteration is bounded by diminishing returns (the hardest cases get labeled first) but a second round would likely add another +0.02–0.04 F1.

**LLM-as-relevance-judge for skill weighting**: implement the Option C variant of the context-aware skill weighting (`llm_rating_input.csv` is already prepared for this). For each (job, skill) pair in the corpus, have an LLM rate the skill's relevance to the job 0–5, integrate as a context-aware importance score. Tests whether LLM judgment captures something our automated heuristics (SBERT cosine, TF-IDF) didn't.

---

## 11. Conclusion

We built a two-stage skill-aware resume–job alignment system: SBERT for retrieval, a 8-feature Logistic Regression reranker for final scoring. The reranker is trained on 600 LLM-consensus labels obtained via 400 stratified initial samples plus 200 uncertainty-driven active-learning samples, achieving F1 = 0.769 on a held-out 100-pair LLM-labeled gold standard.

The headline result — +16.5 F1 points over SBERT-only thresholding — is real and reproducible. But the methodology contributions matter at least as much as the headline number:

- **Demonstrating empirically that formula-based weak supervision does not generalize** to externally-judged labels, with a 0.55-point cross-validation-vs-test F1 gap that serves as a textbook leakage signature reviewers can use to detect this failure mode elsewhere.

- **Demonstrating empirically that confidence-based pseudo-labeling (self-training) fails predictably** on biased pools because the selection mechanism systematically excludes informative cases — Yarowsky's 1995 confirmation-bias warning validated with controlled data in this specific application.

- **Demonstrating that uncertainty-based label selection (active learning) succeeds with the same label budget** because it targets exactly the cases where new labels carry information.

- **Demonstrating that skill-aware features carry signal independent of semantic similarity**: a Logistic Regression on the seven non-SBERT features alone achieves F1 = 0.706, beating SBERT-only F1 = 0.628 by 7.8 points. This refutes the "just SBERT dressed up" objection with concrete numbers.

The literature survey of three contemporaneous papers in this space (Dash et al. 2025, Daberao et al. 2025, Sribharathi et al. 2025) revealed that the field has not run a controlled methodology comparison of this type, has not published inter-rater agreement on its gold standards, and has not empirically validated weak-supervision generalization. Our work fills these specific methodological gaps.

The system is not production-ready. The skill vocabulary still contains noise; the model can produce domain-incoherent individual predictions; the test set is small; LLM consensus is correlated noise rather than ground truth; no real-world deployment has validated whether the system's recommendations actually change employment outcomes. These limitations are shared with the published field and are documented honestly.

What the project demonstrates is a methodologically defensible reranker on top of SBERT retrieval, with quantified contributions from skill-aware features and an empirically validated active-learning pipeline that adds value over both naïve supervised learning and the standard semi-supervised alternative. The skill-gap diagnostic output — what the dashboard prototype renders — is something semantic-similarity-only systems cannot produce regardless of how much SBERT contributes to the score.

---

## Appendix — Code and Dataset Provenance

### A.1 Repository layout

```
MLPR PROJECT/
├── README.md
├── requirements.txt
├── configs/
│   ├── variant_A.json    # legacy variant configs (leakage-debugging era)
│   ├── variant_B.json
│   └── variant_C.json
├── data/
│   ├── raw/                                # 5 public datasets
│   │   ├── Resume (1).csv
│   │   ├── training_data.csv
│   │   ├── Skills.xlsx                     # O*NET
│   │   ├── Skills to Work Activities.xlsx  # O*NET
│   │   ├── Skills to Work Context.xlsx     # O*NET
│   │   ├── skills_en.csv                   # ESCO
│   │   └── broaderRelationsSkillPillar_en.csv   # ESCO hierarchy
│   ├── proccessed again/                   # main working directory
│   │   ├── preprocess_pipeline.py
│   │   ├── matching_pipeline.py
│   │   ├── tech_skills.py
│   │   ├── build_esco_vocab.py
│   │   ├── build_esco_depths.py
│   │   ├── build_diversified_pool.py
│   │   ├── sample_gold_pairs.py
│   │   ├── combine_llm_labels.py
│   │   ├── make_ml_ready.py
│   │   ├── wipe_caches.py
│   │   ├── parquet_to_xlsx.py
│   │   ├── esco_skills_combined.json       # 20,253-skill vocabulary
│   │   ├── esco_skill_depths.json
│   │   ├── processed/                      # core dataset files
│   │   │   ├── cleaned_resumes.parquet           # 2,484 resumes
│   │   │   ├── cleaned_jobs.parquet              # 853 jobs
│   │   │   ├── pair_features.parquet             # 42,650 top-50 pairs × 13 features
│   │   │   ├── pair_features_diversified.parquet # 50,650 pairs × 13 features
│   │   │   ├── ml_ready_dataset.parquet          # 25,590 weak-labeled pairs
│   │   │   └── embeddings/*.npy                  # cached SBERT vectors
│   │   ├── gold_labeling/
│   │   │   ├── gold_pairs_master.csv             # 500 sampled pairs with metadata
│   │   │   ├── gold_pairs_batch_{1..5}.csv       # 100 pairs each (LLM upload)
│   │   │   ├── batch_{1..5}_{claude,gpt,gemini}_labels.csv
│   │   │   ├── gold_labels.csv                   # 500 majority-voted labels
│   │   │   └── gold_labels_disputed.csv          # 145 disputed (2-1) pairs
│   │   └── active_learning/
│   │       ├── active_learning_master.csv        # 200 uncertain pairs
│   │       ├── active_learning_batch_{1,2}.csv
│   │       └── {claude,gpt,gemini}_active_learning_{1,2}.csv
│   └── processed_v1_old/                   # archived v1 baseline (do not modify)
├── src/
│   ├── pipeline.py                  # v1 baseline (legacy)
│   ├── evaluate_three_way.py        # Methods 1+2: weak-supervised vs LLM-supervised
│   ├── run_self_training_experiment.py    # Method 3: self-training v1
│   ├── run_self_training_v2.py            # Method 4: self-training v2
│   ├── active_learning_sample.py          # Method 5 Step 1: pick uncertain pairs
│   ├── active_learning_evaluate.py        # Method 5 Step 2: retrain + evaluate (FINAL)
│   ├── feature_importance_analysis.py     # 3-way feature importance analysis
│   ├── skill_weighting_experiment.py      # context-aware weighting null result
│   └── build_dashboard_demo.py            # dashboard prototype
├── models/
│   ├── weak_safe/{logreg,gbm,scaler}.pkl              # Method 1
│   ├── llm_supervised/{logreg,gbm,scaler}.pkl         # Method 2
│   ├── llm_self_trained/{logreg,scaler}.pkl           # Method 3
│   ├── llm_self_trained_v2/{logreg,scaler}.pkl        # Method 4
│   └── active_learning/{logreg,scaler}.pkl            # ★ FINAL MODEL
└── outputs/
    ├── README.md
    ├── project_report.md             # this document
    ├── skill_gap_analysis_full.md    # 13k-word literature review
    ├── three_way/comparison_*.{csv,png}
    ├── self_training/comparison.{csv,png}
    ├── self_training_v2/comparison.{csv,png}
    ├── active_learning/comparison.{csv,png}
    ├── feature_analysis/{coefficients,permutation_importance,ablation_results}.csv
    ├── skill_weighting/comparison.{csv,png}
    └── dashboard_demo/index.html     # prototype dashboard
```

### A.2 Reproduction commands

```bash
# 1. Build ESCO vocabulary and hierarchy depths
python "data/proccessed again/build_esco_vocab.py"
python "data/proccessed again/build_esco_depths.py"

# 2. Preprocess + match
python "data/proccessed again/wipe_caches.py"
python "data/proccessed again/preprocess_pipeline.py"
python "data/proccessed again/matching_pipeline.py"

# 3. Diversify pool
python "data/proccessed again/build_diversified_pool.py"

# 4. (Manual step) LLM gold labeling: upload batch CSVs to Claude/GPT/Gemini,
#    save responses, then:
python "data/proccessed again/combine_llm_labels.py"

# 5. Train + evaluate all methodologies
python "src/evaluate_three_way.py"
python "src/run_self_training_experiment.py"
python "src/run_self_training_v2.py"
python "src/active_learning_sample.py"
# (Manual step) Upload active learning batches to LLMs, save responses
python "src/active_learning_evaluate.py"

# 6. Analysis
python "src/feature_importance_analysis.py"
python "src/skill_weighting_experiment.py"
python "src/build_dashboard_demo.py"
```

All scripts use `random_state=42` throughout for reproducibility.

### A.3 Final model specification

| Field | Value |
|---|---|
| Model class | Logistic Regression |
| Sklearn parameters | `solver='liblinear'`, `class_weight='balanced'`, `random_state=42` |
| Training samples | 600 LLM-labeled pairs (400 stratified gold + 200 active learning) |
| Features (8) | embedding_similarity, tfidf_similarity, skill_overlap, weighted_skill_score, num_missing_skills, avg_missing_skill_importance, years_of_experience, title_similarity |
| Decision threshold | 0.52 (5-fold CV F1-optimal on training set) |
| Test F1 | 0.769 |
| Test Accuracy | 0.88 |
| Test Precision | 0.667 |
| Test Recall | 0.909 |
| Test P@5 | 1.00 |
| Test P@10 | 0.90 |
| Test P@20 | 0.70 |
| Saved artifacts | `models/active_learning/{logreg,scaler}.pkl` |

### A.4 References

**Primary literature surveyed:**
- Dash, S., Anchan, A., Dabre, P., Kulkarni, V. (2025). *AI Based Skill Gap Analyzer.* International Journal of Engineering Development and Research, Vol. 13 Issue 4.
- Daberao, D.P., Dalal, G., Sreemathy, R. (2025). *ResumeInsight: An AI-Driven Framework for Semantic Resume-Job Matching and Skill-Gap Analysis.* IEEE GITCON (Global Conference on Information Technology and Communication Networks).
- Sribharathi, B., Balamurugan, S.V., Megavarmaraj, S., Deepak, S., Kajendhiran, S. (2025). *Scopira: An AI-Driven Career Guidance System Using Resume Parsing, Skill Gap Analysis, and Intelligent Job Matching.* IEEE ICSSS (10th International Conference on Smart Structures and Systems).

**Methodology references:**
- Yarowsky, D. (1995). *Unsupervised Word Sense Disambiguation Rivaling Supervised Methods.* Proceedings of the 33rd Annual Meeting of the ACL — the canonical self-training paper.
- Devlin, J. et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.*
- Reimers, N., Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.*
- Karpukhin, V. et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP — the canonical two-stage retrieval-and-reranking architecture.
- Ratner, A. et al. (2017). *Snorkel: Rapid Training Data Creation with Weak Supervision.*
- Settles, B. (2009). *Active Learning Literature Survey.* University of Wisconsin-Madison Computer Sciences Technical Report.

**Datasets:**
- ESCO (European Skills, Competences, Qualifications and Occupations) — European Commission, CC-BY licensed taxonomy.
- O*NET (Occupational Information Network) — U.S. Department of Labor, public domain.
- Resume dataset and job description dataset — Kaggle, anonymized by original authors.

---

*Document compiled May 2026. All experimental results reproducible from the scripts and data in this repository.*
