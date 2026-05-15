"""
Resume–Job Alignment Pipeline
==============================
MLPR Course Project

Steps:
  1. Data Cleaning
  2. Skill Extraction
  3. Skill Normalization
  4. Pair Construction (domain-aware)
  5. Feature Engineering (SBERT + skill metrics)
  6. Weak Supervision Labels
  7. Visualizations
  8. Save final dataset
"""

import re
import json
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend --works without a display
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# PATHS  (edit if your files are elsewhere)
# ─────────────────────────────────────────────
RESUME_PATH = "../data/raw/Resume (1).csv"
JOB_PATH    = "../data/raw/training_data.csv"
OUT_CSV     = "../data/processed/ml_ready_dataset.csv"
PLOT_DIR    = "../outputs"          # where to save plots


# ══════════════════════════════════════════════
# STEP 1 --DATA CLEANING
# ══════════════════════════════════════════════

# --- 1a. Text utilities ---

_BULLET_CHARS = re.compile(r"[●■◆►▪•◦▸✓✔✗✘→★☆]")
_MULTI_WS     = re.compile(r"\s+")
_PII_EMAIL    = re.compile(r"\S+@\S+\.\S+")
_PII_PHONE    = re.compile(
    r"(\+?\d[\d\s\-().]{7,}\d)"   # international & local formats
)
_SECTION_HEADERS = re.compile(
    r"\b(minimum qualifications?|preferred qualifications?|"
    r"responsibilities|requirements|about (the )?role|"
    r"what you('ll| will) do|who you are|equal opportunity)\b",
    re.IGNORECASE,
)

def clean_text(text: str, remove_pii: bool = True) -> str:
    """Lowercase, strip bullets, PII, and extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _BULLET_CHARS.sub(" ", text)
    # remove HTML tags if any leaked through
    text = re.sub(r"<[^>]+>", " ", text)
    if remove_pii:
        text = _PII_EMAIL.sub(" ", text)
        text = _PII_PHONE.sub(" ", text)
    # remove lone special chars (keep alphanumerics, +, #, spaces)
    text = re.sub(r"[^a-z0-9\s+#./,;:()\-]", " ", text)
    text = _MULTI_WS.sub(" ", text).strip()
    return text


def clean_job_text(text: str) -> str:
    """Clean job description and strip section headers."""
    text = clean_text(text, remove_pii=False)
    text = _SECTION_HEADERS.sub(" ", text)
    # fix obvious merged words: lowercase letter directly followed by uppercase
    # (already lowercased so we won't catch camelCase, but fix digit-glued words)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)   # pre-lower --no-op now
    text = _MULTI_WS.sub(" ", text).strip()
    return text


# --- 1b. JSON parsing ---

def parse_model_response(raw: str) -> dict:
    """
    Safely parse model_response JSON.
    Returns a dict with keys 'required_skills' and 'responsibilities'.
    Falls back to empty strings on any error.
    """
    default = {"required_skills": "", "responsibilities": ""}
    if not isinstance(raw, str):
        return default
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # attempt to extract with regex if JSON is malformed
        skills_match = re.search(
            r'"Required Skills"\s*:\s*"([^"]*)"', raw, re.IGNORECASE
        )
        resp_match = re.search(
            r'"Core Responsibilities"\s*:\s*"([^"]*)"', raw, re.IGNORECASE
        )
        return {
            "required_skills": skills_match.group(1) if skills_match else "",
            "responsibilities": resp_match.group(1) if resp_match else "",
        }

    return {
        "required_skills": data.get("Required Skills", ""),
        "responsibilities": data.get("Core Responsibilities", ""),
    }


# --- 1c. Load & clean datasets ---

def load_resumes(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    df = df.dropna(subset=["Resume_str", "Category"]).reset_index(drop=True)
    df["clean_text"] = df["Resume_str"].apply(clean_text)
    print(f"[resumes] loaded {len(df)} rows | {df['Category'].nunique()} categories")
    return df[["ID", "Category", "clean_text"]]


def load_jobs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    df = df.dropna(subset=["job_description"]).reset_index(drop=True)
    df["job_id"] = df.index
    df["clean_jd"] = df["job_description"].apply(clean_job_text)

    parsed = df["model_response"].apply(parse_model_response)
    df["required_skills_raw"] = parsed.apply(lambda x: x["required_skills"])
    df["responsibilities_raw"] = parsed.apply(lambda x: x["responsibilities"])

    print(f"[jobs]    loaded {len(df)} rows | {df['position_title'].nunique()} unique titles")
    return df[["job_id", "position_title", "clean_jd", "required_skills_raw", "responsibilities_raw"]]


# ══════════════════════════════════════════════
# STEP 2 --SKILL EXTRACTION
# ══════════════════════════════════════════════

# Common skill keywords for seed matching (lightweight, no heavy ontology)
_SKILL_SEEDS = {
    # languages & frameworks
    "python", "java", "javascript", "typescript", "c++", "c#", "r", "scala",
    "sql", "nosql", "html", "css", "php", "ruby", "go", "swift", "kotlin",
    "react", "angular", "vue", "node", "django", "flask", "spring", "tensorflow",
    "pytorch", "keras", "scikit-learn", "pandas", "numpy", "matplotlib",
    "fastapi", "spring boot", "node.js",
    # data & ml
    "machine learning", "deep learning", "nlp", "computer vision", "data analysis",
    "data science", "data analytics", "data visualization", "data engineering",
    "statistics", "mathematics", "regression", "classification", "clustering",
    "neural network", "random forest", "gradient boosting", "xgboost", "llm",
    "tableau", "power bi", "excel", "spark", "hadoop", "kafka", "airflow",
    "dashboards", "etl", "data warehouse", "big data",
    # cloud & devops
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "github", "jenkins",
    "terraform", "linux", "unix", "bash", "rest api", "api", "graphql",
    "microservices", "ci/cd", "devops",
    # databases
    "mongodb", "postgresql", "redis", "elasticsearch", "mysql",
    # business & soft skills
    "project management", "agile", "scrum", "jira", "communication", "leadership",
    "teamwork", "problem solving", "critical thinking", "time management",
    "customer service", "sales", "marketing", "accounting", "finance",
    "ui", "ux", "research", "writing",
    # domain-specific
    "photoshop", "illustrator", "figma", "sketch", "autocad", "solidworks",
    "sap", "oracle", "salesforce", "crm", "erp", "quickbooks",
    "nursing", "patient care", "medical coding", "clinical research",
    "teaching", "curriculum", "lesson planning",
    "microsoft office", "word", "powerpoint",
    "database", "networking", "security", "cloud",
}

# pattern: skill-like noun phrases (2-4 word phrases, no verbs at start)
_SKILL_PATTERN = re.compile(
    r"\b([a-z][a-z0-9+#]*(?:[\s\-][a-z][a-z0-9+#]*){0,3})\b"
)

def extract_skills_from_text(text) -> list:
    """
    Hybrid skill extractor:
      1. Seed matching  --checks text against known skill keywords
      2. Comma-list extraction --captures items in comma-separated skill lists
    Returns a deduplicated list of skill strings.
    """
    if not text:
        return []

    # Handle list inputs (e.g. from JSON parsed fields)
    if isinstance(text, list):
        text = ", ".join(str(t) for t in text)
    text = str(text).lower()

    found = set()

    # Pass 1: seed keyword matching (lookarounds for c++, c#, etc.)
    for seed in _SKILL_SEEDS:
        if re.search(r'(?<!\w)' + re.escape(seed) + r'(?!\w)', text):
            found.add(seed)

    # Pass 2: extract comma-separated lists near skill trigger words
    # e.g. "skills: python, java, sql" or "proficient in react, angular"
    trigger = re.compile(
        r"(?:skills?|tools?|technologies?|proficient in|experience with|"
        r"knowledge of|familiar with|expertise in)[:\s]+([^\n.;]{5,120})",
        re.IGNORECASE,
    )
    for match in trigger.finditer(text):
        segment = match.group(1)
        items = re.split(r"[,/|•&]", segment)
        for item in items:
            item = item.strip().strip("()")
            if 2 <= len(item.split()) <= 5 and len(item) > 2:
                # only keep if mostly alphanumeric
                if re.match(r"^[a-z0-9][a-z0-9\s#+.\-/]*$", item):
                    found.add(item)

    return sorted(found)


# ── UPGRADE 2: Expanded dictionary for enriched job skill extraction ──
SKILL_DICT = [
    "python", "sql", "aws", "azure", "gcp", "react", "java", "c++", "c#",
    "javascript", "typescript", "node.js", "flask", "django", "fastapi",
    "machine learning", "ml", "deep learning", "dl", "tensorflow", "pytorch",
    "keras", "scikit-learn", "nlp", "computer vision", "llm",
    "pandas", "numpy", "statistics", "mathematics",
    "excel", "tableau", "power bi", "powerbi", "data analysis",
    "data analytics", "data visualization", "dashboards",
    "git", "github", "docker", "kubernetes", "linux", "unix", "bash",
    "spark", "hadoop", "big data", "kafka", "airflow",
    "rest api", "api", "microservices", "spring boot",
    "html", "css", "ui", "ux", "agile", "scrum", "jira", "ci/cd",
    # additional coverage
    "r", "scala", "go", "swift", "kotlin", "php", "ruby",
    "angular", "vue", "node", "spring", "matplotlib",
    "regression", "classification", "clustering", "neural network",
    "random forest", "gradient boosting", "xgboost",
    "graphql", "jenkins", "terraform",
    "project management", "communication", "leadership",
    "problem solving", "critical thinking", "teamwork",
    "customer service", "sales", "marketing", "accounting", "finance",
    "photoshop", "illustrator", "figma", "autocad",
    "sap", "oracle", "salesforce", "quickbooks", "crm", "erp",
    "nosql", "mongodb", "postgresql", "redis", "elasticsearch",
    "data science", "data engineering", "etl", "data warehouse",
]

# FIX 2: Safe regex using lookarounds (handles c++, c#, node.js, ci/cd)
_SKILL_DICT_PATTERNS = [
    (s, re.compile(r'(?<!\w)' + re.escape(s) + r'(?!\w)'))
    for s in SKILL_DICT
]

# FIX 4: Pattern-based extraction phrases
_PHRASE_PATTERNS = [
    re.compile(r"experience (?:in|with) ([a-zA-Z0-9 ,+#/.\-]+)", re.IGNORECASE),
    re.compile(r"proficien(?:t|cy) (?:in|with) ([a-zA-Z0-9 ,+#/.\-]+)", re.IGNORECASE),
    re.compile(r"knowledge of ([a-zA-Z0-9 ,+#/.\-]+)", re.IGNORECASE),
    re.compile(r"working with ([a-zA-Z0-9 ,+#/.\-]+)", re.IGNORECASE),
    re.compile(r"familiar(?:ity)? with ([a-zA-Z0-9 ,+#/.\-]+)", re.IGNORECASE),
    re.compile(r"expertise in ([a-zA-Z0-9 ,+#/.\-]+)", re.IGNORECASE),
    re.compile(r"skilled in ([a-zA-Z0-9 ,+#/.\-]+)", re.IGNORECASE),
]

# Set version of SKILL_DICT for fast membership checks in pattern extraction
_SKILL_DICT_SET = set(SKILL_DICT)


def _extract_phrases_from_patterns(text: str) -> set:
    """FIX 4: Extract skill-like phrases from contextual patterns."""
    found = set()
    for pat in _PHRASE_PATTERNS:
        for m in pat.finditer(text):
            segment = m.group(1)
            # split on comma, slash, semicolon, and/or
            items = re.split(r"[,;/]|\band\b|\bor\b", segment)
            for item in items:
                clean_phrase = item.strip().lower().strip(".()")
                if not clean_phrase:
                    continue
                # FIX 4 filter: length check + dictionary or short phrase
                if 2 <= len(clean_phrase) <= 30 and (
                    clean_phrase in _SKILL_DICT_SET or len(clean_phrase.split()) <= 3
                ):
                    found.add(clean_phrase)
    return found


def extract_skills_from_job(required_skills_raw: str, clean_jd: str) -> list:
    """
    For jobs: extract from BOTH structured 'Required Skills' AND the
    full job description, merge with dictionary + pattern matches, and deduplicate.
    """
    skills = set()

    # Source 1: structured field from model_response
    if required_skills_raw:
        skills.update(extract_skills_from_text(required_skills_raw))

    # Source 2: ALWAYS also scan the full job description text
    skills.update(extract_skills_from_text(clean_jd))

    # FIX 3: Normalize text before dictionary matching
    jd_lower = str(clean_jd).lower() if clean_jd else ""
    jd_lower = jd_lower.replace("powerbi", "power bi")

    # Source 3: dictionary matching with safe regex (FIX 1 + FIX 2)
    for skill, pattern in _SKILL_DICT_PATTERNS:
        if pattern.search(jd_lower):
            skills.add(skill)

    # Source 4: pattern-based extraction (FIX 4)
    skills.update(_extract_phrases_from_patterns(jd_lower))

    return sorted(skills)


# ══════════════════════════════════════════════
# STEP 3 --SKILL NORMALIZATION
# ══════════════════════════════════════════════

_NORM_MAP = {
    # abbreviations & synonyms
    "ml":               "machine learning",
    "dl":               "deep learning",
    "data analytics":   "data analysis",
    "ai":               "artificial intelligence",
    "nlp":              "natural language processing",
    "cv":               "computer vision",
    "oop":              "object oriented programming",
    "ds":               "data science",
    "da":               "data analysis",
    "bi":               "business intelligence",
    "rpa":              "robotic process automation",
    # tools/brands
    "powerbi":          "power bi",
    "power-bi":         "power bi",
    "ms excel":         "excel",
    "microsoft excel":  "excel",
    "ms word":          "word",
    "ms office":        "microsoft office",
    "vscode":           "visual studio code",
    "vs code":          "visual studio code",
    "sklearn":          "scikit-learn",
    "scikit learn":     "scikit-learn",
    "pytorch":          "pytorch",
    "tf":               "tensorflow",
    "node.js":          "node",
    "nodejs":           "node",
    "reactjs":          "react",
    "react.js":         "react",
    "vuejs":            "vue",
    "vue.js":           "vue",
    "angularjs":        "angular",
    "postgres":         "postgresql",
    "mongo":            "mongodb",
    "k8s":              "kubernetes",
    # soft skills normalisation
    "comm":             "communication",
    "mgmt":             "management",
}

def normalize_skills(skills: list) -> list:
    """Lowercase, deduplicate, and apply normalisation map."""
    normalized = []
    seen = set()
    for s in skills:
        s = s.lower().strip()
        s = _NORM_MAP.get(s, s)          # direct lookup
        # also try without hyphens/dots
        s_clean = re.sub(r"[-.]", " ", s).strip()
        s = _NORM_MAP.get(s_clean, s)
        if s and s not in seen and len(s) > 1:
            seen.add(s)
            normalized.append(s)
    return sorted(normalized)


# ══════════════════════════════════════════════
# STEP 4 --TOP-K SBERT PAIR CONSTRUCTION
# ══════════════════════════════════════════════

TOP_K = 6   # number of best-matching jobs per resume

def build_pairs_topk(resumes_df: pd.DataFrame, jobs_df: pd.DataFrame,
                     model_name: str = "all-MiniLM-L6-v2",
                     batch_size: int = 128) -> pd.DataFrame:
    """
    UPGRADE 1: For each resume, select the TOP_K most semantically similar
    jobs using SBERT embeddings + cosine similarity.  Returns pairs with
    pre-computed similarity scores (avoids re-encoding later).
    """
    MAX_CHARS = 4000

    # ── Step 4a: encode all unique texts ONCE ─────────────────
    print(f"[sbert]   loading model '{model_name}' ...")
    model = SentenceTransformer(model_name)

    res_texts = resumes_df["clean_text"].str[:MAX_CHARS].tolist()
    job_texts = jobs_df["clean_jd"].str[:MAX_CHARS].tolist()

    print(f"[sbert]   encoding {len(res_texts)} resumes ...")
    res_emb = model.encode(res_texts, batch_size=batch_size, show_progress_bar=True,
                           convert_to_numpy=True, normalize_embeddings=True)
    print(f"[sbert]   encoding {len(job_texts)} jobs ...")
    job_emb = model.encode(job_texts, batch_size=batch_size, show_progress_bar=True,
                           convert_to_numpy=True, normalize_embeddings=True)

    # ── Step 4b: full cosine similarity matrix (vectorised) ───
    print("[sbert]   computing similarity matrix ...")
    sim_matrix = cosine_similarity(res_emb, job_emb)   # (n_res, n_jobs)

    # ── Step 4c: top-K selection (no loops) ───────────────────
    top_k_idx = np.argsort(-sim_matrix, axis=1)[:, :TOP_K]   # (n_res, K)

    n_res = len(resumes_df)
    resume_ix = np.repeat(np.arange(n_res), TOP_K)           # [0,0,0,0,0,0, 1,1,...]
    job_ix    = top_k_idx.flatten()                           # flattened job indices
    sim_scores = sim_matrix[resume_ix, job_ix]                # corresponding scores

    # ── Step 4d: construct pairs dataframe ────────────────────
    pairs = pd.DataFrame({
        "resume_id":           resumes_df["ID"].values[resume_ix],
        "job_id":              jobs_df["job_id"].values[job_ix],
        "category":            resumes_df["Category"].values[resume_ix],
        "position_title":      jobs_df["position_title"].values[job_ix],
        "resume_text":         resumes_df["clean_text"].values[resume_ix],
        "job_text":            jobs_df["clean_jd"].values[job_ix],
        "resume_skills":       resumes_df["resume_skills"].values[resume_ix],
        "job_skills":          jobs_df["job_skills"].values[job_ix],
        "semantic_similarity": sim_scores.astype(float),
    })

    avg_sim = pairs["semantic_similarity"].mean()
    print(f"[pairs]   constructed {len(pairs)} pairs | avg similarity: {avg_sim:.4f}")
    return pairs


# ══════════════════════════════════════════════
# STEP 5 --FEATURE ENGINEERING
# ══════════════════════════════════════════════

def compute_skill_features(row) -> dict:
    r_skills = set(row["resume_skills"])
    j_skills = set(row["job_skills"])

    if not j_skills:
        coverage = 0.0
        gap      = 0
    else:
        overlap  = r_skills & j_skills
        coverage = len(overlap) / len(j_skills)
        gap      = len(j_skills - r_skills)

    return {
        "skill_coverage":    round(coverage, 4),
        "skill_gap":         gap,
        "num_resume_skills": len(r_skills),
        "num_job_skills":    len(j_skills),
    }


# ══════════════════════════════════════════════
# STEP 6 --WEAK SUPERVISION LABELS
# ══════════════════════════════════════════════

def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quantile-based weak supervision labels.
      skill_coverage >= 70th percentile  ->  label 1  (good match)
      skill_coverage <= 30th percentile  ->  label 0  (poor match)
      middle 40%                         ->  dropped
    """
    high_thresh = df["skill_coverage"].quantile(0.70)
    low_thresh  = df["skill_coverage"].quantile(0.30)

    print(f"[labels]  skill_coverage thresholds -- high (>={high_thresh:.4f}) = 1 | low (<={low_thresh:.4f}) = 0")

    df = df.copy()
    df["label"] = None
    df.loc[df["skill_coverage"] >= high_thresh, "label"] = 1
    df.loc[df["skill_coverage"] <= low_thresh, "label"] = 0

    labeled = df.dropna(subset=["label"]).copy()
    labeled["label"] = labeled["label"].astype(int)

    counts = labeled["label"].value_counts()
    print(f"[labels]  retained {len(labeled)} pairs | label=1: {counts.get(1,0)} | label=0: {counts.get(0,0)}")
    return labeled


# ══════════════════════════════════════════════
# STEP 7 --VISUALIZATIONS
# ══════════════════════════════════════════════

def plot_distributions(df: pd.DataFrame, save_dir: str = "../outputs") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # --- Skill Coverage ---
    axes[0].hist(df["skill_coverage"], bins=30, color="#4C72B0", edgecolor="white")
    axes[0].set_title("Skill Coverage Distribution", fontsize=12)
    axes[0].set_xlabel("skill_coverage")
    axes[0].set_ylabel("Count")

    # --- Semantic Similarity ---
    axes[1].hist(df["semantic_similarity"], bins=30, color="#55A868", edgecolor="white")
    axes[1].set_title("Semantic Similarity Distribution", fontsize=12)
    axes[1].set_xlabel("cosine similarity (SBERT)")
    axes[1].set_ylabel("Count")

    # --- Label Distribution ---
    label_counts = df["label"].value_counts().sort_index()
    axes[2].bar(label_counts.index.astype(str), label_counts.values,
                color=["#C44E52", "#4C72B0"])
    axes[2].set_title("Label Distribution", fontsize=12)
    axes[2].set_xlabel("label")
    axes[2].set_ylabel("Count")
    for i, v in enumerate(label_counts.values):
        axes[2].text(i, v + 2, str(v), ha="center", fontsize=10)

    plt.tight_layout()
    out = f"{save_dir}/feature_distributions.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plots]   saved -> {out}")


# ══════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════

def run_pipeline():
    print("=" * 60)
    print(" RESUME–JOB ALIGNMENT PIPELINE")
    print("=" * 60)

    # ── Step 1: Load & Clean ──────────────────────────────────
    print("\n[1/7] Loading and cleaning data ...")
    resumes = load_resumes(RESUME_PATH)
    jobs    = load_jobs(JOB_PATH)

    # ── Step 2: Extract Skills ────────────────────────────────
    print("\n[2/7] Extracting skills ...")
    resumes["resume_skills_raw"] = resumes["clean_text"].apply(extract_skills_from_text)
    jobs["job_skills_raw"] = jobs.apply(
        lambda r: extract_skills_from_job(r["required_skills_raw"], r["clean_jd"]),
        axis=1,
    )

    # ── Step 3: Normalize Skills ──────────────────────────────
    print("\n[3/7] Normalizing skills ...")
    resumes["resume_skills"] = resumes["resume_skills_raw"].apply(normalize_skills)
    jobs["job_skills"]       = jobs["job_skills_raw"].apply(normalize_skills)

    # Quick sanity check + extraction metrics
    avg_r = resumes["resume_skills"].apply(len).mean()
    avg_j = jobs["job_skills"].apply(len).mean()
    max_j = jobs["job_skills"].apply(len).max()
    zero_j = (jobs["job_skills"].apply(len) == 0).sum()
    print(f"  avg resume skills: {avg_r:.1f} | avg job skills: {avg_j:.1f}")
    print(f"  max job skills: {max_j} | jobs with 0 skills: {zero_j}/{len(jobs)}")

    # ── Step 4: Top-K SBERT Pairing (embeddings computed once) ─
    print("\n[4/7] Building Top-K SBERT pairs (encodes once) ...")
    pairs = build_pairs_topk(resumes, jobs)
    # semantic_similarity is already in pairs from the similarity matrix

    # ── Step 5: Feature Engineering ──────────────────────────
    print("\n[5/7] Computing skill features ...")
    skill_feats = pairs.apply(compute_skill_features, axis=1, result_type="expand")
    pairs = pd.concat([pairs, skill_feats], axis=1)

    # ── Step 6: Labels ────────────────────────────────────────
    print("\n[6/7] Assigning weak supervision labels ...")
    labeled = assign_labels(pairs)

    # ── Final Dataset ─────────────────────────────────────────
    feature_cols = [
        "resume_id", "job_id", "category", "position_title",
        "resume_text", "job_text",
        "semantic_similarity", "skill_coverage", "skill_gap",
        "num_resume_skills", "num_job_skills",
        "resume_skills", "job_skills",
        "label",
    ]
    final = labeled[feature_cols].copy()
    final.to_csv(OUT_CSV, index=False)
    print(f"\n[output]  saved -> {OUT_CSV}  ({len(final)} rows)")

    # ── Step 7: Visualizations ────────────────────────────────
    print("\n[7/7] Generating visualizations ...")
    plot_distributions(final, PLOT_DIR)

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETE -- DATASET PREVIEW")
    print("=" * 60)

    # Training features (X) -- excludes skill_coverage and skill_gap
    # to prevent data leakage (labels are derived from skill_coverage)
    X_cols = [
        "semantic_similarity",
        "num_resume_skills",
        "num_job_skills",
    ]
    # Analysis columns kept in CSV but NOT used as model features
    analysis_cols = ["skill_coverage", "skill_gap"]
    all_metric_cols = X_cols + analysis_cols

    print("\n--- TRAINING FEATURES (X) ---")
    print("  ", X_cols)
    print("\n--- ANALYSIS-ONLY columns (NOT in X, kept for reference) ---")
    print("  ", analysis_cols)

    print("\nAll feature statistics:")
    print(final[all_metric_cols].describe().round(4).to_string())

    total = len(final)
    counts = final["label"].value_counts()
    print(f"\nDataset summary: {total} total rows")
    print(f"  label=0: {counts.get(0,0)}  ({100*counts.get(0,0)/total:.1f}%)")
    print(f"  label=1: {counts.get(1,0)}  ({100*counts.get(1,0)/total:.1f}%)")

    print("\nSample rows:")
    print(final[all_metric_cols + ["label"]].head(8).to_string(index=False))

    # ── Skill Extraction Metrics ──────────────────────────────
    print("\n" + "-" * 60)
    print(" SKILL EXTRACTION METRICS")
    print("-" * 60)
    print(f"  Avg job skills:           {avg_j:.1f}  (previous: ~3.8)")
    print(f"  Max job skills:           {max_j}")
    print(f"  Jobs with 0 skills:       {zero_j}/{len(jobs)}")
    cov_mean = final["skill_coverage"].mean()
    cov_zero_pct = 100 * (final["skill_coverage"] == 0).sum() / len(final)
    print(f"  skill_coverage mean:      {cov_mean:.4f}  (previous: ~0.4157)")
    print(f"  skill_coverage == 0:      {cov_zero_pct:.1f}%")

    return final


if __name__ == "__main__":
    df = run_pipeline()
