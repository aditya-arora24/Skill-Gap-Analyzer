"""
upgrade_features.py
===================
Upgrades ml_ready_dataset.csv and gold_standard_final.csv with a robust,
leakage-safe 5-feature setup:

  [semantic_similarity, tfidf_similarity, skill_imbalance,
   num_resume_skills, num_job_skills]

STRICT RULES ENFORCED:
  - Vectorizer fitted on TRAINING DATA only (no gold leakage)
  - Labels are NEVER modified
  - Row order is NEVER shuffled
  - No existing columns are dropped
  - No models are trained
"""

import ast
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

# Import skill utilities from the existing pipeline (no heavy re-execution)
from pipeline import (
    extract_skills_from_text,
    normalize_skills,
    clean_text,
    clean_job_text,
)

import sys
sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "semantic_similarity",
    "tfidf_similarity",
    "skill_imbalance",
    "num_resume_skills",
    "num_job_skills",
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def rowwise_cosine_sim(mat_a, mat_b) -> np.ndarray:
    """Vectorized row-wise cosine similarity between two sparse matrices."""
    norm_a = normalize(mat_a, norm="l2")
    norm_b = normalize(mat_b, norm="l2")
    sim = np.asarray(norm_a.multiply(norm_b).sum(axis=1)).flatten()
    return sim.astype(float)


def count_skills_in_text(text: str) -> int:
    """Extract, normalize, and count skills from raw text."""
    raw = extract_skills_from_text(text)
    return len(normalize_skills(raw))


def safe_parse_list(val) -> list:
    """Parse a stringified Python list; returns empty list on failure."""
    if isinstance(val, list):
        return val
    try:
        parsed = ast.literal_eval(str(val))
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATASETS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 1: Loading datasets")
print("=" * 65)

train = pd.read_csv("../data/processed/ml_ready_dataset.csv")
gold  = pd.read_csv("../data/processed/gold_standard_final.csv")

print(f"\nml_ready_dataset.csv  — shape: {train.shape}")
print(f"  Columns: {list(train.columns)}")
print(f"\ngold_standard_final.csv — shape: {gold.shape}")
print(f"  Columns: {list(gold.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — ENSURE TEXT COLUMNS EXIST
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2: Ensuring text columns exist")
print("=" * 65)

# ── 2a. Training data: resume_text ───────────────────────────────────────────
if "resume_text" not in train.columns:
    print("[train] resume_text missing — merging from Resume (1).csv ...")
    resumes_raw = pd.read_csv("../data/raw/Resume (1).csv", encoding="utf-8", on_bad_lines="skip")
    resumes_raw = resumes_raw[["ID", "Resume_str"]].dropna().copy()
    resumes_raw["resume_text"] = resumes_raw["Resume_str"].apply(clean_text)
    train = train.merge(
        resumes_raw[["ID", "resume_text"]].rename(columns={"ID": "resume_id"}),
        on="resume_id",
        how="left",
    )
    missing = train["resume_text"].isna().sum()
    print(f"  Merged. Rows with no resume match: {missing}")
else:
    print("[train] resume_text already present")

# ── 2b. Training data: job_description ───────────────────────────────────────
if "job_description" not in train.columns:
    print("[train] job_description missing — merging from training_data.csv ...")
    jobs_raw = pd.read_csv("../data/raw/training_data.csv", encoding="utf-8", on_bad_lines="skip")
    jobs_raw = jobs_raw[["job_description"]].dropna().copy()
    jobs_raw["job_description"] = jobs_raw["job_description"].apply(clean_job_text)
    jobs_raw["job_id"] = jobs_raw.index
    train = train.merge(
        jobs_raw[["job_id", "job_description"]],
        on="job_id",
        how="left",
    )
    missing = train["job_description"].isna().sum()
    print(f"  Merged. Rows with no job match: {missing}")
else:
    print("[train] job_description already present")

# ── 2c. Gold standard already carries both text columns ──────────────────────
print("[gold]  resume_text and job_description present: "
      f"{'resume_text' in gold.columns and 'job_description' in gold.columns}")

# Fill NaN text with empty string (safe fallback; does not remove rows)
train["resume_text"]     = train["resume_text"].fillna("")
train["job_description"] = train["job_description"].fillna("")
gold["resume_text"]      = gold["resume_text"].fillna("")
gold["job_description"]  = gold["job_description"].fillna("")

# ── Assert both datasets have the required text columns ──────────────────────
assert "resume_text"     in train.columns, "Training data still missing resume_text"
assert "job_description" in train.columns, "Training data still missing job_description"
assert "resume_text"     in gold.columns,  "Gold standard missing resume_text"
assert "job_description" in gold.columns,  "Gold standard missing job_description"
print("\nAssertion PASSED: both datasets contain ['resume_text', 'job_description'] ✓")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — CREATE FEATURE: skill_imbalance
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3: Creating feature — skill_imbalance")
print("=" * 65)

# ── 3a. Training data: num_resume_skills / num_job_skills already exist ──────
#         (they were computed in pipeline.py from the stringified skill lists)
print("[train] Using existing num_resume_skills and num_job_skills ...")
train["skill_imbalance"] = train["num_job_skills"] - train["num_resume_skills"]
si = train["skill_imbalance"]
print(f"  skill_imbalance — min: {si.min()}, max: {si.max()}, mean: {si.mean():.4f}")
assert si.isna().sum() == 0, "NaN found in train skill_imbalance!"
print("  No NaNs ✓")

# ── 3b. Gold standard: compute num_resume_skills, num_job_skills ─────────────
print("[gold]  Computing num_resume_skills, num_job_skills via skill extractor ...")
gold["num_resume_skills"] = gold["resume_text"].apply(count_skills_in_text)
gold["num_job_skills"]    = gold["job_description"].apply(count_skills_in_text)
gold["skill_imbalance"]   = gold["num_job_skills"] - gold["num_resume_skills"]

for col in ["num_resume_skills", "num_job_skills", "skill_imbalance"]:
    s = gold[col]
    print(f"  {col:25s} — min: {s.min()}, max: {s.max()}, mean: {s.mean():.4f}")
    assert s.isna().sum() == 0, f"NaN found in gold {col}!"
print("  No NaNs ✓")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — CREATE FEATURE: tfidf_similarity
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4: Creating feature — tfidf_similarity")
print("=" * 65)

# ── 4a. Fit vectorizer ONLY on training texts ─────────────────────────────────
print("[tfidf] Fitting TfidfVectorizer on TRAINING DATA only ...")
train_corpus = list(train["resume_text"]) + list(train["job_description"])
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
vectorizer.fit(train_corpus)
print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")

# ── 4b. Transform training data, then compute row-wise cosine similarity ──────
print("[tfidf] Transforming training data ...")
tfidf_tr_resume = vectorizer.transform(train["resume_text"])
tfidf_tr_job    = vectorizer.transform(train["job_description"])
train["tfidf_similarity"] = rowwise_cosine_sim(tfidf_tr_resume, tfidf_tr_job)
ts = train["tfidf_similarity"]
print(f"  tfidf_similarity — min: {ts.min():.4f}, max: {ts.max():.4f}, mean: {ts.mean():.4f}")
assert ts.isna().sum() == 0, "NaN found in train tfidf_similarity!"
print("  No NaNs ✓")

# ── 4c. Apply SAME fitted vectorizer to gold standard (DO NOT refit) ──────────
print("[tfidf] Transforming gold standard with pre-fitted vectorizer (no refit) ...")
tfidf_gs_resume = vectorizer.transform(gold["resume_text"])
tfidf_gs_job    = vectorizer.transform(gold["job_description"])
gold["tfidf_similarity"] = rowwise_cosine_sim(tfidf_gs_resume, tfidf_gs_job)
gs = gold["tfidf_similarity"]
print(f"  tfidf_similarity — min: {gs.min():.4f}, max: {gs.max():.4f}, mean: {gs.mean():.4f}")
assert gs.isna().sum() == 0, "NaN found in gold tfidf_similarity!"
print("  No NaNs ✓")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4b — COMPUTE semantic_similarity FOR GOLD STANDARD (SBERT)
# ─────────────────────────────────────────────────────────────────────────────
# Training data already has semantic_similarity from the original pipeline.
# Gold standard does not — compute it now using the same SBERT model.
print("\n" + "=" * 65)
print("STEP 4b: Computing semantic_similarity for gold standard (SBERT)")
print("=" * 65)

if "semantic_similarity" not in gold.columns:
    print("[sbert] Loading model 'all-MiniLM-L6-v2' ...")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"[sbert] Encoding {len(gold)} gold resumes ...")
    gold_res_emb = sbert.encode(
        gold["resume_text"].tolist(),
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    print(f"[sbert] Encoding {len(gold)} gold job descriptions ...")
    gold_job_emb = sbert.encode(
        gold["job_description"].tolist(),
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    # L2-normalized embeddings: dot product == cosine similarity
    gold["semantic_similarity"] = (gold_res_emb * gold_job_emb).sum(axis=1).astype(float)
    ss = gold["semantic_similarity"]
    print(f"  semantic_similarity — min: {ss.min():.4f}, max: {ss.max():.4f}, mean: {ss.mean():.4f}")
else:
    print("[sbert] semantic_similarity already present in gold standard")

assert gold["semantic_similarity"].isna().sum() == 0, "NaN in gold semantic_similarity!"
print("  No NaNs ✓")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — VERIFY ALL 5 FEATURE COLUMNS ARE PRESENT IN BOTH DATASETS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5: Verifying all 5 feature columns in both datasets")
print("=" * 65)

for col in FEATURE_COLS:
    assert col in train.columns, f"FAIL: training data missing '{col}'"
    assert col in gold.columns,  f"FAIL: gold standard missing '{col}'"
    print(f"  {col:30s} ✓  (train) ✓  (gold)")

print("\nAll 5 features present in BOTH datasets ✓")

# Sanity: label columns untouched
assert "label"      in train.columns, "label column dropped from training data!"
assert "gold_label" in gold.columns,  "gold_label column dropped from gold standard!"
print("Labels untouched (label / gold_label still present) ✓")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — VALIDATE (STRICT CHECKS)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6: Validation — strict checks")
print("=" * 65)

for ds_name, df in [("ml_ready_dataset", train), ("gold_standard_final", gold)]:
    print(f"\n{'─'*45}")
    print(f"  Dataset : {ds_name}")
    print(f"  Shape   : {df.shape}")

    nan_report = df[FEATURE_COLS].isna().sum()
    total_nan  = nan_report.sum()
    if total_nan > 0:
        print(f"  WARNING — NaN values detected:")
        print(nan_report[nan_report > 0].to_string())
    else:
        print("  Missing values in features : NONE ✓")

    print("\n  Feature summary (mean ± std):")
    for col in FEATURE_COLS:
        m = df[col].mean()
        s = df[col].std()
        print(f"    {col:30s}  mean={m:8.4f}  std={s:7.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — SAVE UPDATED DATASETS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 7: Saving updated datasets")
print("=" * 65)

import os
base_dir = os.path.dirname(os.path.abspath(__file__))

def save_csv(df: pd.DataFrame, preferred_name: str, fallback_name: str) -> str:
    """
    Save df to preferred_name; if that file is locked (e.g. open in Excel),
    fall back to fallback_name so no work is lost.
    Returns the actual path used.
    """
    path = os.path.join(base_dir, preferred_name)
    try:
        df.to_csv(path, index=False, encoding="utf-8")
        return path
    except PermissionError:
        fallback = os.path.join(base_dir, fallback_name)
        df.to_csv(fallback, index=False, encoding="utf-8")
        print(f"  WARNING: '{preferred_name}' is locked (open in another app?).")
        print(f"           Saved to '{fallback_name}' instead.")
        print(f"           Close the file in Excel/Notepad and rename manually, or rerun the script.")
        return fallback

train_saved = save_csv(train, "../data/processed/ml_ready_dataset.csv",    "../data/processed/ml_ready_dataset_updated.csv")
gold_saved  = save_csv(gold,  "../data/processed/gold_standard_final.csv", "../data/processed/gold_standard_final_updated.csv")

print(f"  ml_ready_dataset    -> {os.path.basename(train_saved)} ({train.shape[0]} rows, {train.shape[1]} cols)")
print(f"  gold_standard_final -> {os.path.basename(gold_saved)}  ({gold.shape[0]} rows, {gold.shape[1]} cols)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — MANDATORY OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Feature upgrade successful ✅")
print("=" * 65)

print("\n[ml_ready_dataset] — head(3) of feature columns:")
print(train[FEATURE_COLS].head(3).to_string(index=True))

print("\n[gold_standard_final] — head(3) of feature columns:")
print(gold[FEATURE_COLS].head(3).to_string(index=True))

print("\n" + "=" * 65)
print(f"Final feature set: {FEATURE_COLS}")
print(f"  ml_ready_dataset.csv   : {train.shape[0]} rows, {train.shape[1]} cols")
print(f"  gold_standard_final.csv: {gold.shape[0]} rows, {gold.shape[1]} cols")
print("=" * 65)
