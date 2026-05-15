# AI-Based Skill Gap Analyzer Systems — Complete Analysis
### A detailed explanation of three research papers and the full analytical synthesis produced from them

---

## Table of Contents

1. [What This Document Is](#what-this-document-is)
2. [The Three Research Papers — Detailed Explanations](#the-three-research-papers)
   - [Paper 1 — AI Based Skill Gap Analyzer (Dash et al., 2025)](#paper-1--ai-based-skill-gap-analyzer)
   - [Paper 2 — ResumeInsight (Daberao et al., 2025)](#paper-2--resumeinsight)
   - [Paper 3 — Scopira AI (Sribharathi et al., 2025)](#paper-3--scopira-ai)
3. [Landscape Map — How the Papers Relate](#landscape-map)
4. [Direct Contradictions Between Papers](#direct-contradictions-between-papers)
5. [Intellectual Lineage of Key Concepts](#intellectual-lineage-of-key-concepts)
6. [Five Unanswered Research Questions](#five-unanswered-research-questions)
7. [Research Methodology Comparison](#research-methodology-comparison)
8. [Field Synthesis](#field-synthesis)
9. [Untested Assumptions](#untested-assumptions)
10. [Closing Observation](#closing-observation)

---

## What This Document Is

Three peer-reviewed research papers on AI-based skill gap analysis and resume–job matching were analysed across seven dimensions: paper content, shared assumptions, contradictions, intellectual lineage, research gaps, methodology quality, and critical assumptions. This document explains all of that in full — what each paper actually says, how they agree and disagree, what the field has proven, and what it has never honestly examined.

---

## The Three Research Papers

---

### Paper 1 — AI Based Skill Gap Analyzer

**Full title:** AI Based Skill Gap Analyzer  
**Authors:** Sachinandan Dash, Anush Anchan, Prajot Dabre, Mrs. Veena Kulkarni  
**Affiliation:** Thakur College of Engineering and Technology, Mumbai, India  
**Published in:** International Journal of Engineering Development and Research (IJEDR), Volume 13, Issue 4, November 2025  
**Short name used in analysis:** P1

---

#### What Problem It Tries to Solve

The paper starts from a real and well-documented problem: there is a persistent gap between what job seekers can do and what employers expect. Students and early-career professionals often miss opportunities not because they are unqualified in a general sense, but because they lack specific skills that a particular job requires — and they have no structured way to identify what those skills are or how to acquire them. Meanwhile, recruiters spend enormous time manually reading resumes and comparing them to job descriptions.

The paper's goal is to automate both sides of this process: tell job seekers exactly which skills they are missing and point them toward resources to fill those gaps, while also helping recruiters evaluate candidates faster and more consistently.

---

#### What They Built

The system is a full-stack web application with three distinct layers that communicate with each other:

**Frontend (React + Vite)**
The user interface is built in React using Vite as the build tool for fast performance. Users can:
- Create an account and log in
- Upload their resume (PDF or DOCX)
- Browse and select job postings
- View an interactive skill gap report showing matched skills, missing skills, and learning recommendations
- Export their report

State management uses React Context for most of the app, with Redux brought in for more complex state scenarios. HTTP requests to the backend are made through Axios.

**Backend (Node.js + Express)**
The backend is a REST API built with Node.js and Express.js. Its responsibilities are:
- User authentication using JWT (JSON Web Tokens)
- Accepting and validating uploaded resume files through Multer
- Scanning uploaded files for malware and validating file sizes
- Converting uploaded resumes to plain text using libraries like pdfminer, PyPDF2, or python-docx
- Communicating with the Python microservice for analysis
- Storing and retrieving data from MongoDB
- Rate limiting on public endpoints to prevent abuse

Security features include JWT-based authentication, bcrypt password hashing, and input validation through Joi.

**Python NLP Microservice (Flask or FastAPI)**
This is where the actual intelligence of the system lives. When a resume is uploaded, the backend sends the extracted text to this microservice, which runs it through a multi-step pipeline:

1. **Text preprocessing** — tokenisation, stop-word removal, sentence segmentation using spaCy and NLTK
2. **Skill extraction** — transformer-based NER (Named Entity Recognition) using BERT and RoBERTa fine-tuned on resume-specific datasets
3. **Skill normalisation** — extracted skills are mapped to canonical entries in taxonomies like ESCO (European Skills, Competences, Qualifications and Occupations) or O*NET using rule-based heuristics, embedding-based nearest-neighbour checks, and ontology mappings
4. **Semantic matching** — the normalised candidate skills are compared against job requirements using sentence embeddings from Sentence-BERT, computing cosine similarity scores
5. **Gap identification** — skills present in the job description but absent or insufficient in the candidate's profile are flagged as gaps
6. **Recommendation generation** — the system links each identified gap to relevant learning resources from platforms like Coursera, Udemy, and edX, ranked by semantic relevance, recency, and credibility

**Database (MongoDB)**
MongoDB stores users, resume metadata, job descriptions, and generated reports. An auxiliary index using FAISS, Milvus, Pinecone, or Elasticsearch stores vector embeddings for semantic search queries.

---

#### The NLP Models Used

**BERT (Bidirectional Encoder Representations from Transformers)**
BERT reads text by considering every word in the context of all other words around it simultaneously — bidirectionally. This allows it to identify skills that appear in complex sentence structures or are implied rather than explicitly stated. For example, a sentence like "led the migration of our monolith to microservices" implies skills in Docker, Kubernetes, and system architecture without explicitly naming them.

**RoBERTa (Robustly Optimised BERT Approach)**
RoBERTa is an improved version of BERT trained with dynamic masking (the masking pattern changes each time the data is seen) and on larger datasets. In the paper's evaluation, RoBERTa consistently outperforms BERT across all metrics, which aligns with the broader transformer research literature.

**Sentence-BERT (SBERT)**
Used for generating dense vector representations of skill phrases and job requirement statements, enabling semantic similarity computation between them.

---

#### Results and Performance

The paper evaluates the system across three dimensions:

**Skill extraction performance (comparing BERT vs RoBERTa):**

| Model | Accuracy | Precision | Recall | F1-Score | R-Precision@5 |
|-------|----------|-----------|--------|----------|---------------|
| BERT | 0.86 | 0.87 | 0.83 | 0.85 | 0.80 |
| RoBERTa | 0.90 | 0.89 | 0.85 | 0.87 | 0.83 |

RoBERTa wins on every metric. The authors attribute this to its stronger contextual representation capabilities.

**System performance:**
- Average processing time per resume: **1.8 seconds**
- Concurrent user stress test: **500 simultaneous users** with error rates below 2%

**Recommendation engine:**
- nDCG@5 (how well recommendations are ranked by relevance): **0.81**
- Precision@5 (how many of the top 5 recommendations are relevant): **0.84**
- Coverage (proportion of skill gaps for which relevant resources were found): **78%**

**User satisfaction:** 92% in usability testing

---

#### Limitations the Paper Acknowledges

- Difficulty interpreting context-sensitive skills that vary across industries or appear subtly in narrative text
- No multilingual support — only English resumes
- Further work needed to reduce algorithmic bias and ensure fairness across demographic groups
- Recommendation personalisation could be deeper

---

#### Key Technical Infrastructure

The system is containerised with Docker, orchestrated in production with Kubernetes for automatic scaling and rolling updates. CI/CD pipelines run through GitHub Actions or GitLab CI. Logging uses structured JSON with ELK stack; monitoring uses Prometheus + Grafana. Model training uses PyTorch and Hugging Face on NVIDIA T4 or A100 GPUs. MLflow tracks experiments; DVC or Git LFS handles dataset versioning.

---

### Paper 2 — ResumeInsight

**Full title:** ResumeInsight: An AI-Driven Framework for Semantic Resume–Job Matching and Skill-Gap Analysis  
**Authors:** Dhruv Prakash Daberao, Gitesh Dalal, R. Sreemathy  
**Affiliation:** Pune Institute of Computer Technology, Pune, India  
**Published in:** IEEE GITCON — Global Conference on Information Technology and Communication Networks, Belagavi, India, August 2025  
**Short name used in analysis:** P2

---

#### What Problem It Tries to Solve

The paper opens with a vivid framing of the recruiter's problem: corporate job postings receive over 250 applications on average, recruiters spend up to 75% of their time sorting resumes, and traditional Applicant Tracking Systems (ATS) rely on keyword matching that misses skill equivalencies (a "data scientist" and a "machine learning engineer" have largely overlapping skill sets, but keyword-only systems treat them as different). The system also frequently rejects resumes with non-standard formats, excluding qualified candidates based on presentation rather than competence.

The target audience is explicitly fresh graduates and early-career professionals in India who struggle to align their profiles with job requirements and often fail automated resume screening not because of lack of skill but because of poor resume formatting or the wrong vocabulary.

---

#### What They Built

ResumeInsight is a four-tier architecture:

**Tier 1 — User**
The user uploads a resume.

**Tier 2 — React.js Frontend**
A web interface where the user uploads their resume, selects a target company from a database, and views their match score and recommendations.

**Tier 3 — Express.js Backend API**
Serves as the intermediary between the frontend and the ML model. Receives resume data from the frontend, forwards it to the Flask API, receives the suitability score back, and returns it to the UI.

**Tier 4 — Flask API with ML Model**
Implements the resume parsing pipeline and the prediction model. The model is built with scikit-learn and persisted using joblib.

---

#### The Methodology in Detail

**Step 1 — Data Collection**
The authors built a dataset of 1,500 anonymised resumes and 50 job descriptions, focused on IT and data science roles. They created this dataset themselves and compared it with open datasets on Kaggle.

**Step 2 — Resume Parsing Pipeline**
- PyMuPDF extracts raw text from PDF files, handling complex layouts
- spaCy processes the text: tokenisation, punctuation removal, lemmatisation (turning "programming" and "programmed" into "program")
- Rule-based heuristics and regex patterns identify key sections: Skills, Education, Projects
- The parser extracts specific values: CGPA, 10th grade percentage, 12th grade percentage, and skill keywords

**Step 3 — Skill Similarity Computation**
Rather than using transformer embeddings, P2 uses **Levenshtein distance** — a measure of how many character insertions, deletions, or substitutions are needed to transform one string into another. A lower Levenshtein distance between a candidate's skill list and a company's required skills indicates higher similarity. This is computationally cheap and works well when skill names are spelled consistently, but fails when the same skill is described with different vocabulary ("machine learning" vs "ML" vs "predictive modelling").

**Step 4 — Feature Vector Construction**
For each resume–company pair, a 7-dimensional feature vector is constructed:
- x1: Company's CGPA cutoff
- x2: Company's 12th grade marks cutoff
- x3: Company's 10th grade marks cutoff
- x4: Candidate's CGPA
- x5: Candidate's 12th grade marks
- x6: Candidate's 10th grade marks
- x7: Computed skill-similarity score (from Levenshtein distance)

This design reflects the Indian campus recruitment context, where companies often set hard minimum thresholds for academic scores before even looking at skills.

**Step 5 — Clustering**
K-Means clustering is applied to group similar resume–company pairs, producing cluster labels that are used as targets for the supervised learning models. This means the model is not learning from human-assigned labels of "good fit" or "bad fit" — it is learning to reproduce a clustering artefact, which is a significant methodological limitation.

**Step 6 — Model Training and Selection**
Three models are trained and compared:
- **Random Forest** — an ensemble of decision trees, robust to overfitting, handles high-dimensional noisy data
- **XGBoost** — gradient boosting, computationally efficient for structured data
- **ANN (Artificial Neural Network)** — can capture non-linear patterns but struggled with the dataset's complexity and imbalance

The trained model predicts a suitability score in the range [0, 1], which is scaled to 0–10 for display.

---

#### Results

**Fit prediction model comparison:**

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 72% | 62% | 60% | 61% |
| XGBoost | 63% | 61% | 60% | 61% |
| ANN | 50% | 47% | 45% | 46% |

Random Forest is selected as the primary model.

**Skill extraction performance:**

| Metric | Value |
|--------|-------|
| Precision | 90.1% |
| Recall | 88.5% |
| F1-score | 89.3% |

**Processing speed:** Under 2.5 seconds per resume.

The ROC curve analysis shows ResumeInsight outperforms a baseline keyword-based ATS at most operating thresholds.

---

#### What the System Does NOT Do

P2 explicitly does not include:
- A recommendation engine for upskilling
- A skill gap report explaining which specific skills are missing
- A concurrency or load test
- A user satisfaction study
- Any multilingual support

The system's output is a match percentage and a ranking of the top 5 companies most aligned with the candidate's profile. It is a screening and ranking tool, not a career development tool.

---

#### Limitations

- Dataset limited to IT and data science roles among Indian graduates
- Levenshtein distance fails on semantically equivalent but lexically different skill descriptions
- No OCR support for scanned resumes (planned as future work)
- Academic metrics (CGPA, board scores) are India-specific and would not generalise to other markets
- The 72% accuracy figure is measured against K-Means cluster labels, not against human-judged fitness

---

### Paper 3 — Scopira AI

**Full title:** Scopira: An AI-Driven Career Guidance System Using Resume Parsing, Skill Gap Analysis, and Intelligent Job Matching  
**Authors:** Mrs. B. Sribharathi, Balamurugan SV, Megavarmaraj S, Deepak S, Kajendhiran S  
**Affiliation:** M Kumarasamy College of Engineering, Karur, India  
**Published in:** IEEE ICSSS — 10th International Conference on Smart Structures and Systems, 2025  
**Short name used in analysis:** P3

---

#### What Problem It Tries to Solve

Scopira AI targets the widest problem scope of the three papers. It is not just a skill gap tool or a matching tool — it is framed as a full career guidance platform. The paper explicitly criticises traditional career counselling approaches (face-to-face sessions, static assessment tools, psychological tests like Holland's RIASEC model and Myers-Briggs) for being non-scalable, slow, and generic. They cannot process thousands of resumes, cannot adapt to live job market conditions, and cannot give personalised feedback.

The paper also positions Scopira AI against existing commercial tools: it benchmarks against LinkedIn Skills Match (83.5% accuracy) and IBM Watson Career Explorer (86.8% accuracy), claiming to outperform both at 91.8% accuracy.

---

#### System Architecture — Four Layers

**Input Layer**
Accepts user resumes (PDF or DOCX), personal information, education credentials, and desired career fields.

**Processing Layer (NLP)**
Conducts resume parsing, skill extraction, and job description analysis using TF-IDF and BERT embeddings. Specific operations:
- Tokenisation, lemmatisation, stop-word removal
- Named Entity Recognition (NER) to identify skills, tools, job roles, certifications, education
- TF-IDF vectorisation to convert text into numerical feature vectors

**Machine Learning Layer**
Runs supervised learning algorithms (Logistic Regression, Random Forest, SVM) for skill gap detection and job recommendation. Also uses BERT-based semantic matching for the job recommendation engine and decision tree algorithms for the career path generator.

**Output Layer**
Displays personalised career suggestions, skill gap reports, job matches ranked by suitability score, and professional branding advice (resume quality, LinkedIn profile optimisation, keyword density).

---

#### The Four Core Modules

**1. Resume Parsing Module**
Extracts structured data from unstructured resume text: name, education, experience, certifications, skills. Uses tokenisation, POS (part-of-speech) tagging, and regex-based extraction.

**2. Skill Gap Analysis Module**
Cross-matches extracted user competencies with normalised job skill databases. A supervised ML classifier (Logistic Regression and Random Forest, trained on labeled datasets of professional profiles) identifies missing or weak skills relative to target job roles. The output is a personalised skill gap report with targeted training recommendations.

**3. Job Recommendation Engine**
Uses BERT embeddings to generate dense representations of both candidate profiles and job postings. Cosine similarity scores are computed to rank job matches. The formula:

```
Cosine Similarity = (A · B) / (||A|| × ||B||)
```

A higher cosine similarity score indicates better alignment between candidate qualifications and job requirements.

**4. Career Path Generator**
Uses decision tree algorithms trained on historical career trajectory data from successful professionals. Given a candidate's current profile, it recommends a personalised career path — not just the next job, but a sequence of roles and skill acquisitions over time.

**5. Professional Branding Advisor** *(additional module)*
Analyses the candidate's resume, LinkedIn profile, and online portfolio against keyword density patterns and recruiter interaction trends. Recommends improvements to increase visibility in job search algorithms and professional networks.

---

#### Technical Stack

- **Backend:** Python (Flask framework)
- **Frontend:** React.js
- **Database:** MongoDB (MongoDB Atlas for distributed cloud storage)
- **NLP libraries:** NLTK, spaCy
- **ML libraries:** scikit-learn, TensorFlow
- **Cloud infrastructure:** AWS EC2 for backend processing
- **Optimisations:** ONNX Runtime conversion (27% faster inference), DistilBERT fine-tuning (40% smaller model), batch inference of 20 (reduces API overhead), Redis caching (reduces database load by ~35%)

---

#### Training Data

5,000 anonymised resumes and 3,000 job postings sourced from open repositories and LinkedIn datasets.

---

#### Results

**Model performance (BERT-based matching):**

| Metric | Value |
|--------|-------|
| Accuracy | 89.5% |
| Recall | 92.2% |
| F1-score | 0.905 |
| MRR (Mean Reciprocal Rank) | 0.89 |
| nDCG@5 | 0.86 |
| Top-5 Accuracy | 89% |

The BERT-based model outperforms baseline TF-IDF models by approximately 13–15%.

**Comparative benchmarking:**

| System | Accuracy | Processing Time | User Satisfaction |
|--------|----------|-----------------|-------------------|
| Baseline (aptitude tools) | 74.2% | 3.8s | 65% |
| LinkedIn Skills Match | 83.5% | 2.9s | 78% |
| IBM Watson Career Explorer | 86.8% | 2.6s | 81% |
| **Scopira AI (P3)** | **91.8%** | **1.8s** | **89%** |

**System performance:**
- Average API response time: 1.8 seconds
- Maximum throughput: 8,200 requests/minute
- Database query latency: <250ms
- System uptime: 99.4%
- Capacity: handles over 10,000 resumes per hour

**Usability study (70 participants: 50 students, 20 professionals, 1 week):**

| Dimension | Score |
|-----------|-------|
| Relevance of recommendations | 4.5 / 5 |
| Ease of understanding results | 4.6 / 5 |
| Interface design and navigation | 4.4 / 5 |
| Overall satisfaction | 4.5 / 5 |

**Real-world pilot (200+ users across 3 universities and 2 corporate training centres, 1 month):**
- 65% of users improved their resume scores after one AI-assisted revision
- 70% enrolled in online courses recommended by the system
- 58% reported securing interviews aligned with Scopira AI's recommendations
- 45% reduction in average time taken to decide on a career direction
- 92% positive sentiment in feedback analysis

**Error analysis (200 test resumes):**
- 12% of errors from ambiguous or incomplete resumes
- 8% from overlapping job role descriptions in training data
- 5% from domain-specific technical vocabulary not in generic BERT models

---

#### Ethical Considerations

- Resumes anonymised through SHA-256 encryption before analysis
- Balanced datasets used to mitigate demographic and gender bias
- Every recommendation comes with a confidence score and reasoning trace for interpretability

---

#### Limitations

- Multilingual support not yet implemented
- Domain-adaptive fine-tuning needed for industry-specific vocabulary (IT, healthcare, finance)
- Mobile app with offline capability planned but not built
- No real-time labour market feed integration yet

---

## Landscape Map

With the three papers understood individually, here is how they map against each other as a field.

### Core claims in one sentence

| Paper | Core claim |
|-------|-----------|
| P1 — Dash et al. | A full-stack BERT/RoBERTa + ESCO ontology platform can automatically identify skill gaps and deliver ranked upskilling recommendations with 90% accuracy and 92% satisfaction. |
| P2 — Daberao et al. | A 7-dimensional feature vector using Levenshtein skill similarity and Indian academic metrics, fed into Random Forest, achieves 72% fit-prediction accuracy without needing transformers. |
| P3 — Sribharathi et al. | An end-to-end career guidance system combining TF-IDF, BERT, cosine similarity, and supervised classifiers achieves 91.8% accuracy, outperforming LinkedIn Skills Match and IBM Watson. |

### What unites them

All three papers share the same fundamental pipeline: extract text from a resume → identify skills → compare against job requirements → surface a gap. All three use Python for NLP, React for the frontend, and MongoDB for the database. All three were built primarily for software industry use cases. All three treat ML experimentation as their primary validation method.

### What divides them

The deepest division is about the role of the candidate versus the recruiter. P2 is fundamentally a recruiter tool — it screens and ranks candidates. P1 and P3 are fundamentally candidate tools — they tell individuals how to improve. These are not the same objective, and many of the methodological and architectural differences between the papers flow directly from this unstated difference in whose interests the system serves.

---

## Direct Contradictions Between Papers

Nine points where papers take positions that cannot both be true as general claims:

### Contradiction 1 — Do you need transformer models?

**P1 and P3 say:** BERT and RoBERTa are essential for capturing the contextual and implicit nature of skills. Without transformers, you miss skills that are expressed indirectly or paraphrased.

**P2 says:** spaCy + Levenshtein distance achieves 89.3% F1 without any transformer infrastructure at all.

**Why they disagree:** P2 uses a structured dataset of Indian fresher resumes where skills are listed explicitly in a dedicated skills section with consistent vocabulary. P1 and P3 target diverse, unstructured resumes where skills must be inferred from narrative descriptions. The task definitions are different. P2's dataset rewards string matching; P1 and P3's datasets require semantic understanding.

---

### Contradiction 2 — What predicts job fit?

**P1 and P3 say:** Semantic similarity between resume text and job description text is the primary signal.

**P2 says:** A combination of academic metrics (CGPA, 10th and 12th grade percentages) and skill similarity together form a more complete and reliable prediction vector.

**Why they disagree:** This reflects completely different labour market assumptions. Indian campus recruitment uses hard academic cutoffs as first-pass filters — a candidate below a CGPA threshold will not be considered regardless of skills. The general job market addressed by P1 and P3 rarely uses academic grades as a primary filter for experienced roles.

---

### Contradiction 3 — Are formal ontologies necessary?

**P1 says:** ESCO/O*NET ontologies are essential infrastructure for canonicalising skill names and understanding hierarchical relationships between skills.

**P3 says:** Cosine similarity on BERT vectors is sufficient for matching; formal ontologies are unnecessary overhead.

**Why they disagree:** P1 is research-oriented and heavily grounded in academic literature on knowledge representation. P3 is deployment-oriented and optimises for speed and engineering simplicity. BERT's pretraining implicitly encodes some ontological structure, which is why P3 can skip formal ontologies without catastrophic accuracy loss — but it also means P3 inherits whatever biases and gaps exist in BERT's training data.

---

### Contradiction 4 — What does "accuracy" mean?

**P1** reports 90% model accuracy on skill extraction.  
**P2** reports 72% accuracy on fit prediction.  
**P3** reports 91.8% accuracy on career matching.

These three numbers are measuring entirely different things on entirely different datasets. Comparing them as though they represent performance on the same task is one of the most common and damaging errors in this literature. P2's 72% figure being "lower" does not mean P2's system performs worse at the same task — it means the task P2 chose to measure is harder or differently defined.

---

### Contradictions 5–9 (summarised)

| # | Topic | P1 / P3 position | P2 position |
|---|-------|-----------------|-------------|
| 5 | Recommendations as core output | Upskilling recommendations are a primary deliverable evaluated with nDCG and Precision@K | Fit score is the terminal output; no recommendation engine |
| 6 | String vs semantic matching | Keyword/ATS approaches fail on paraphrased skills | Levenshtein is sufficient for structured, consistent skill labels |
| 7 | Scalability testing | 500–10,000 concurrent users tested | Only single-resume processing speed tested |
| 8 | BERT vs RoBERTa | RoBERTa outperforms BERT on all metrics (P1) | BERT adopted without testing RoBERTa (P3) |
| 9 | User satisfaction | 89–92% satisfaction validates system | No user study; only ML metrics used for validation |

---

## Intellectual Lineage of Key Concepts

This section traces where the field's core ideas came from, who built on them, who challenged them, and where consensus currently stands.

---

### Concept 1 — Transformer-based Skill Extraction

**Origin:** External to these papers. Devlin et al. introduced BERT in 2018 (Google). Liu et al. introduced RoBERTa in 2019 (Facebook). Both were general-purpose language models, not designed for resume analysis.

**Who adopted it:**
- P1 adopted BERT and RoBERTa and ran the only head-to-head comparison in this set of papers, finding RoBERTa consistently superior
- P3 adopted BERT without testing whether alternatives might perform better

**Who challenged it:**
- P2 demonstrated competitive results (89.3% F1) using only classical NLP tools, implicitly arguing that the computational expense and complexity of transformers is not always justified

**Current consensus:** No settled consensus. The appropriate choice depends entirely on how structured and consistent your input data is. For messy, narrative-heavy professional text, transformers are clearly superior. For clean, structured data with explicit skill labels, classical approaches remain viable.

---

### Concept 2 — Semantic Skill Matching

**Origin:** The information retrieval literature (pre-2020) established cosine similarity on TF-IDF vectors as the baseline method for text similarity.

**Who refined it:**
- P3 upgraded the baseline by computing cosine similarity on BERT vectors instead of TF-IDF vectors, demonstrating ~13% improvement
- P1 went further by combining Sentence-BERT embeddings with ontology-based normalisation, ensuring that "Python developer" and "Python programming" are recognised as the same skill before similarity is computed

**Who challenged it:**
- P2 replaced semantic similarity entirely with Levenshtein string distance, which works when skills are named consistently but fails when the same skill is expressed with different vocabulary

**Current consensus:** Weak consensus toward BERT-based semantic embeddings. Cosine + BERT is the dominant approach in the field. String-distance methods survive only as practical shortcuts for narrow, structured use cases.

---

### Concept 3 — Skill Gap Analysis with Upskilling Recommendations

**Origin:** Nguyen et al. (2024), cited in P1, proposed an ontology-driven framework for skill gap analysis that mapped workforce competencies to job requirements using structured domain ontologies.

**Who extended it:**
- P1 added a full recommendation engine that links identified gaps to specific courses on Coursera, Udemy, and edX, ranked by semantic relevance and evaluated with nDCG@5 and Precision@5
- P3 went further by building a career path generator that recommends not just a course but a multi-step career trajectory, and added a professional branding module for resume and LinkedIn optimisation

**Who ignored it:**
- P2 treats skill gap analysis as outside scope. Its output is a match score and company ranking. It does not tell the candidate what they are missing or how to improve.

**Current consensus:** Strong consensus between P1 and P3 that a system that identifies a gap without providing a remedy is incomplete. The recommendation engine is considered a first-class feature, not an optional add-on.

---

## Five Unanswered Research Questions

These are gaps that all three papers acknowledge or imply but none resolves.

---

### Gap 1 — Implicit and Soft Skill Extraction from Narrative Text

**What the gap is:** All three systems can extract explicitly stated skills — if "Python" appears on a resume, it gets tagged. None has a credible solution for extracting skills expressed implicitly in narrative text, such as inferring "project management" from "coordinated a five-person team through a deadline-critical product launch," or inferring "resilience" from a career history that includes multiple pivots.

**Why it exists:** There is no ground-truth corpus for soft skills. Annotators systematically disagree about what counts as a soft skill, what counts as a personality trait, and what threshold of evidence in a resume is sufficient to credit someone with a skill. Without a reliable labelling standard, you cannot train a model and you cannot evaluate one.

**Which paper came closest:** P1 explicitly flags this as a limitation. P3 mentions soft skills in its module descriptions but never demonstrates the system actually extracting them. P2 doesn't address the issue at all.

**What it would take to close it:** A purpose-built annotation project involving occupational psychologists and NLP researchers jointly designing an annotation guide, applied to a diverse corpus of professional narrative text. The evaluation metric would need to allow partial credit rather than binary correct/incorrect, since "this sentence provides moderate evidence of leadership" is a more honest judgement than "this sentence either is or is not about leadership."

---

### Gap 2 — Cross-Domain Model Generalisation

**What the gap is:** Every model in these papers was trained and tested within a narrow domain — primarily IT and data science. None tests whether a model trained on tech resumes can correctly parse a healthcare CV, a legal resume, or a creative portfolio. Domain-specific vocabulary, credential structures, and job title conventions vary so dramatically that cross-domain transfer cannot be assumed.

**Why it exists:** Collecting and annotating diverse multi-domain resume data is expensive. It is much easier and faster to build a working system within one domain and publish. The incentive structure of conference publishing rewards novelty and accuracy within scope, not breadth of applicability.

**Which paper came closest:** P1 uses the ESCO taxonomy, which formally spans all industries, but never tests whether its models perform on non-IT job descriptions.

**What it would take to close it:** A multi-domain benchmark dataset covering at least five industry sectors with different credential structures (IT, healthcare, law, creative industries, finance), paired with a transfer learning experiment that systematically measures how F1 drops as a function of the distance between the training domain and the test domain.

---

### Gap 3 — Longitudinal Validation of Recommendation Outcomes

**What the gap is:** All three papers evaluate their systems using short-term or proxy measures — model accuracy, satisfaction surveys, or one-month deployment observations. None answers the question: did following this system's recommendations actually lead to better employment outcomes?

**Why it exists:** Conference paper timelines are 3–6 months. Longitudinal studies require at minimum 12 months of follow-up and ideally much longer. These two rhythms are structurally incompatible. Additionally, measuring real employment outcomes requires partnerships with employers and institutions, ethical approval, and follow-up infrastructure that most academic research teams cannot sustain.

**Which paper came closest:** P3's one-month pilot with 200+ users is the closest thing in the literature. It found that 58% of users reported securing relevant interviews, but this is self-reported over a very short window and cannot establish causality.

**What it would take to close it:** A cohort study with at minimum 12-month follow-up, comparing a group that used AI-assisted skill gap guidance against a control group that used traditional career counselling. Primary outcomes would be employment rate, role-skill alignment, and salary. The study would need IRB (ethics board) approval and institutional partnerships with universities and employers.

---

### Gap 4 — Algorithmic Bias Measurement and Mitigation

**What the gap is:** P1 mentions "fairness checks" as a design principle. P3 says it uses "balanced datasets." Neither paper publishes disaggregated performance metrics broken down by gender, ethnicity, age, educational background, or socioeconomic status. It is therefore impossible to know whether any of these systems performs differently for different demographic groups — and given what is known about bias in ML systems trained on historical hiring data, the prior probability of demographic disparities is high.

**Why it exists:** It is legally sensitive. Publishing results that show your system performs worse for women or for candidates from certain ethnic groups invites regulatory scrutiny. Obtaining demographically labelled resume datasets raises privacy concerns. And the review culture of engineering conferences does not typically require bias audits.

**Which paper came closest:** P1 is the only paper to even mention the issue in substantive terms — it discusses decision logs for transparency and fairness checks for demographic biases, though without publishing any results.

**What it would take to close it:** A structured audit protocol with a demographically stratified held-out test set. Precision and recall should be computed separately for each protected group, and a formal statistical test should compare these figures against the null hypothesis of no demographic effect. Any disparities above a pre-defined threshold should trigger investigation of training data provenance.

---

### Gap 5 — Multilingual and Non-Western Resume Parsing

**What the gap is:** All three systems operate only on English-language resumes using English-trained models. The global job market is not English-only. Beyond language, resume conventions vary dramatically by culture — Japanese resumes (rirekisho) follow a rigid printed form; French CVs include photographs; Indian resumes emphasise board exam scores; German resumes include a separate section for Ausbildung (vocational training). English-trained parsers cannot handle these structural differences.

**Why it exists:** English-language annotated resume corpora exist in sufficient quantity to train models. Comparable corpora in other languages, particularly with skill annotations, do not exist at scale. Creating them requires native language expertise, domain knowledge, and annotation resources that most research teams cannot access.

**Which paper came closest:** P3 lists multilingual support as a future direction. P1 acknowledges the limitation. P2 never raises it.

**What it would take to close it:** First, assembling a multilingual annotated corpus across at least five languages and three distinct cultural resume formats. Second, benchmarking multilingual transformer models (mBERT, XLM-R, LaBSE) against monolingual baselines to quantify the accuracy cost of language transfer. Third, developing culture-specific parsing heuristics for non-standard resume structures.

---

## Research Methodology Comparison

### Methodology presence by paper

| Methodology type | P1 — Dash et al. | P2 — Daberao et al. | P3 — Sribharathi et al. | Field coverage |
|-----------------|-----------------|--------------------|-----------------------|----------------|
| Survey / usability | Partial — 92% satisfaction; no sample size or instrument published | Absent — no user study of any kind | Strong — 70 participants, Likert scale, 1-week structured interaction | 2 of 3 papers |
| ML experiment | Strong — BERT vs RoBERTa head-to-head; 5 evaluation metrics | Strong — RF vs XGBoost vs ANN; full confusion matrices | Strong — BERT vs TF-IDF baseline; MRR, nDCG@5 | 3 of 3 papers |
| Simulation / load test | Partial — 500 concurrent users, <2% error rate | Absent — only single-resume speed reported | Partial — 10,000 users simulated, 1.8s avg response | 2 of 3 papers |
| Meta-analysis | Absent | Absent | Absent | 0 of 3 papers |
| Case study / pilot | Partial — real UI shown; no structured deployment data | Absent — no evidence of real-world use | Strong — 1-month pilot, 5 institutions, 200+ users | 1 of 3 papers |

---

### Why ML Experimentation Dominates

All three papers were submitted to engineering and computer science venues where the primary review criterion is technical novelty demonstrated through measurable accuracy improvement. A 2% F1 gain over a BERT baseline is publishable. A six-month cohort study measuring employment outcomes belongs in a labour economics journal with a completely different review culture and timeline. The methodology that dominates is the one that gets papers accepted, not the one that answers the most important questions.

---

### Why Meta-Analysis Is Impossible in This Field

Meta-analysis requires synthesising results across multiple studies that measured the same thing in comparable ways. The skill extraction field has no standardised benchmark — Kumar et al. (2022), cited in P1, explicitly documents this. Every paper uses a different dataset, different skill taxonomy, different model, and different evaluation metrics. You cannot aggregate results across studies that measured different things. The prerequisite for meta-analysis — methodological standardisation — does not yet exist.

---

### Why P2 Has the Weakest Methodology

P2's weakness is not its choice of simple methods. Levenshtein distance and Random Forest are defensible choices for a constrained problem. The weakness is threefold:

1. **Single methodology type:** Only ML experimentation. No user study, no load testing, no real-world deployment, no literature synthesis.
2. **Dataset too thin:** 1,500 resumes and only 50 job descriptions. The ratio is so compressed that K-Means clustering (used to generate training labels) will produce unstable cluster boundaries. The model is learning to reproduce a clustering artefact, not to judge actual fit.
3. **No external validation:** There is no baseline other than internal model comparison. There is no evidence that a real recruiter would agree with even one of the system's recommendations. The 72% accuracy figure is presented as a meaningful performance measure when it is actually a measure of self-consistency within a flawed labelling process.

---

## Field Synthesis

### What the field collectively believes

The field has converged on a core pipeline treated as self-evident: extract skills from unstructured text using NLP, compare against job requirements using vector similarity, surface the gap, attach learning recommendations. This architecture is reproduced independently across geographies and institutions as though it were the only reasonable design — not because competing architectures were tested and rejected, but because the pipeline maps cleanly onto available tools and the structure of publishable ML papers.

There is also collective belief, never fully examined, that improving benchmark metrics is equivalent to improving outcomes for real people. The field trusts that a system achieving 90% F1 delivers more value than one achieving 85%, without any evidence that marginal precision improvements translate into better interviews, offers, or career decisions. The proxy is mistaken for the thing itself.

> **The field has achieved consensus on the shape of the solution before achieving consensus on what the problem actually is.**

---

### What remains contested

**Architectural:** Whether transformer-based semantic embeddings are necessary for competent skill matching, or whether simpler string-distance methods are sufficient when resumes are well-structured. This is not a settled empirical question — it is an artefact of researchers using different datasets that reward different methods.

**Feature scope:** Whether fit prediction should incorporate structured academic signals alongside semantic similarity. This reflects incompatible labour market assumptions more than technical evidence.

**Purpose:** One implicit theory holds that the system's job is to serve the recruiter — surface the most technically qualified candidate efficiently. The competing theory holds that the system's job is to serve the candidate — tell them honestly where they stand and how to improve. These are not the same objective, and optimising for one can directly harm the other. The field has not named this tension, let alone resolved it.

---

### What has been proven beyond reasonable doubt

1. **Contextual transformer models — particularly RoBERTa — extract explicitly stated skills from English-language professional text with high precision and recall, comfortably exceeding keyword-based matching.** This holds across independently conducted studies using different datasets and is not seriously disputed.

2. **Users who interact with AI-assisted skill gap tools report higher satisfaction and faster decision-making than users relying on traditional career guidance methods.** This is the most ecologically valid finding in the literature, appearing consistently across different system designs.

3. **Modular microservice architectures decoupling the NLP engine from the application backend can handle realistic concurrent loads without degrading in accuracy.** The engineering pattern is proven.

> **What is proven is narrow: the machines can read skills off a page reliably, and people feel helped when they use these systems. Everything beyond that is inference.**

---

### The single most important unanswered question

> **Does acting on an AI skill gap system's recommendations actually change a person's employment trajectory — and if so, for whom, by how much, and at whose expense?**

Every paper in this field stops at the boundary of the system. They measure what the model outputs; none measures what happens to the human who receives that output and acts on it. A person who follows the system's upskilling recommendation and lands a better role would constitute evidence that these systems work. A person who follows the same recommendation, invests six months and significant money, and remains unemployed because the real barrier was network access or credential discrimination would constitute evidence of harm.

This question is unanswered not because it is technically hard — a longitudinal cohort study is methodologically straightforward — but because answering it requires institutional commitment incompatible with conference paper timelines. The field has optimised for research it can complete in a semester rather than research the problem demands.

---

## Untested Assumptions

Eight load-bearing assumptions shared by the majority of papers that are never explicitly stated, argued for, or empirically tested.

---

### Assumption 01 — Resumes are truthful

**Statement:** Skills listed on a resume accurately represent skills a person actually possesses.

**Papers relying on it most:** P1, P3

**If wrong:** The entire extraction pipeline is measuring self-reported intent, not verified capability. A candidate who lists "machine learning" after watching two YouTube videos and a candidate who built production ML systems are indistinguishable to every model in this literature. If resume skill claims are systematically inflated — and there is substantial evidence from organisational psychology that they are — then skill gap analysis is measuring the distance between two fictions. The field would need to pivot toward verified capability evidence: portfolio assessments, coding challenges, credentialed outcomes.

---

### Assumption 02 — Job descriptions are accurate

**Statement:** Job descriptions accurately represent what the role actually requires day-to-day.

**Papers relying on it most:** P1, P2, P3

**If wrong:** Job descriptions are written by HR generalists, copied from templates, and inflated with aspirational requirements that are rarely validated against what the role's previous occupant actually did. The "credential creep" phenomenon is well-documented — roles routinely require degrees and certifications that perform no predictive function. If job descriptions are unreliable as ground truth, then every skill gap identified is the gap between a candidate's honest self-report and a recruiter's wish list — not an operationally meaningful deficit. The field's entire notion of "gap" collapses at the source.

---

### Assumption 03 — Closing a skill gap improves employment outcomes

**Statement:** Completing the learning recommendations produced by the system will improve a person's probability of employment.

**Papers relying on it most:** P1, P3

**If wrong:** This is the foundational assumption that gives the entire recommendation engine its moral justification. If a person follows the system's advice, invests time and money in certification, and remains unemployed because the real barriers are network access, name-based discrimination, or structural demand deficits, then these systems are not career tools. They are an elaborate redirection of blame onto the candidate. "You are unemployed because of a skill gap" is a comforting story for an economic system that produces structural unemployment. If this assumption fails, the field is not solving a technical problem — it is laundering a social one.

---

### Assumption 04 — Text similarity is a valid proxy for fit

**Statement:** Semantic similarity between a resume and a job description is a valid proxy for how well a candidate would actually perform in that role.

**Papers relying on it most:** P1, P3

**If wrong:** Two documents can be semantically similar while describing people who would perform wildly differently — because fit is determined by working style, team dynamics, motivation, and domain familiarity, none of which appear in text. Conversely, a candidate from an adjacent domain may have a semantically distant resume but be the highest-performing hire. If the proxy is unreliable, every match score in this literature is measuring vocabulary overlap, not suitability.

---

### Assumption 05 — Skill ontologies are neutral and complete

**Statement:** The ESCO and O*NET taxonomies are complete, current, and neutral representations of the skill landscape.

**Papers relying on it most:** P1

**If wrong:** Skill ontologies are built by committees, updated on multi-year cycles, and reflect the labour markets of the countries that fund them — overwhelmingly Western, English-speaking, and formal-sector oriented. Emerging skills appear in job postings for 12–18 months before ontology bodies acknowledge them. Skills common in informal economies, creative industries, and non-Western markets are structurally underrepresented. Any skill not yet named in the ontology is invisible to the system. Any ontological category that encodes historical bias will propagate that bias through every normalisation the system performs.

---

### Assumption 06 — Satisfaction equals usefulness

**Statement:** High user satisfaction scores indicate the system is genuinely useful for career development, not merely pleasant to interact with.

**Papers relying on it most:** P1, P3

**If wrong:** User satisfaction is well-documented as a measure of interface comfort, response speed, and confirmation bias — not accuracy, utility, or downstream impact. A system that produces fluent, beautifully visualised recommendations that are entirely wrong will score highly on satisfaction surveys. P1's 92% and P3's 89% satisfaction figures may be measures of good UX design, not evidence that anyone's employment outcomes improved.

---

### Assumption 07 — The skill gap is the primary obstacle to employment

**Statement:** The main reason qualified candidates fail to obtain suitable employment is a mismatch between what they know and what employers need — and this mismatch is located in the candidate.

**Papers relying on it most:** P1, P2, P3

**If wrong:** This is the deepest ideological assumption in the field, and it is never examined. A competing model — with substantial empirical support — holds that hiring outcomes are primarily determined by network access, referral pathways, name-based discrimination, credential signalling unrelated to competence, and structural demand deficits. If the obstacle is the gate rather than the candidate's preparation, building a better gate-measurement tool does not help the candidate. It is an active misdirection that consumes the candidate's time, money, and energy while leaving the actual barrier untouched.

---

### Assumption 08 — Historical hiring data encodes quality, not bias

**Statement:** Training data collected from existing hiring practices encodes what good hiring looks like, not what biased hiring looked like.

**Papers relying on it most:** P2, P3

**If wrong:** If training data reflects a hiring history in which qualified candidates from underrepresented groups were systematically rejected, the model will learn that those candidates are poor fits — and reproduce that judgement at scale, with the additional authority of algorithmic objectivity. This is not merely a theoretical risk. Multiple independent audits of commercial ATS systems have documented exactly this phenomenon. Based on how these systems are built and what is known about the historical composition of tech-sector hiring, demographic disparities in model performance are close to a certainty, not a possibility. The difference between P2 and P3 is only that P3 at least mentions the problem.

---

## Closing Observation

The pattern formed by these eight assumptions, read together, reveals something structural about the field.

Assumptions 01 and 02 are mirror images of the same problem: both inputs to the system — the resume and the job description — are treated as reliable ground truth when they are in fact strategic documents written to manage impressions. The entire analytical chain built on top of these inputs is *precise measurement of noisy signals*, not noisy measurement of precise ones. The distinction matters because the field's response to the first kind of problem is better algorithms, while the correct response to the second kind is better inputs — a completely different research agenda.

Assumptions 03 and 07 constitute the field's core political assumption: that unemployment is a technical problem located in the individual, solvable by better information. This is a specific theory of labour markets that shifts moral responsibility from structural conditions to personal deficits. The field has chosen this framing not because it is obviously correct but because it is the framing that produces fundable, publishable, commercialisable research. A field that concluded "the primary obstacle is discriminatory hiring, and no algorithm can fix that" would cease to attract engineering investment.

Assumption 08 is the most urgent. Assumptions 01–07 being wrong would mean these systems are ineffective. Assumption 08 being wrong means they are actively making things worse for the most vulnerable users — encoding historical discrimination into a tool that presents itself as objective — at exactly the moment when its recommendations carry the most weight.

> The field has built a precise instrument for measuring the wrong thing, pointed at unreliable inputs, on behalf of a theory of the problem that has never been tested. That is not a criticism of the individual papers reviewed here — each is a competent piece of engineering research within its own scope. It is a structural observation about what a field looks like when it optimises for publishability over explanatory power.

---

*Document compiled from analysis of P1 (Dash et al., IJEDR 2025), P2 (Daberao et al., IEEE GITCON 2025), and P3 (Sribharathi et al., IEEE ICSSS 2025). Analysis conducted May 2026.*
