"""
build_final_dataset.py
======================
Two-phase dataset builder for the resume-job alignment model (v2).

Phase 1 (variant-agnostic) -- produces the shared pair table:
    data/processed/pair_features_final.parquet

Phase 2 (variant-specific) -- reads a JSON config and produces:
    data/processed/variant_<X>/ml_ready_dataset.parquet
    data/processed/variant_<X>/gold_standard_final.parquet

Usage
-----
    # Build the shared feature table (run once after v2 preprocessing)
    python build_final_dataset.py --phase pairs

    # Build a variant from a config file
    python build_final_dataset.py --phase variant --config ../configs/variant_A.json
    python build_final_dataset.py --phase variant --config ../configs/variant_B.json
    python build_final_dataset.py --phase variant --config ../configs/variant_C.json

    # Convenience: do both phases in sequence
    python build_final_dataset.py --phase all --config ../configs/variant_A.json

Three preprocessing fixes (4a, 4b, 4c) are applied during phase 1 -- so
preprocess_pipeline.py never needs to re-run.

SBERT handling for fix 4c (resume titles)
-----------------------------------------
fix 4c changes resume titles from `Category` (the broad dataset label) to
the actual first-line role text (e.g. `HR DIRECTOR`). The cached
title_resume_emb.npy in the v2 embeddings directory was generated against
the OLD title source (`Category`), so it is STALE for fix 4c.

  - If sentence-transformers is available, the new titles are re-encoded
    with SBERT and the cache is overwritten. This is the canonical path.
  - If sentence-transformers is NOT available (e.g. in the Cowork sandbox),
    we fall back to the stale cached embeddings -- the script still runs,
    but `title_similarity` will reflect the OLD title source. The output
    log makes this explicit; the run is a smoke test, not canonical.

Other cached embeddings (resume_emb, job_emb, resume_exp_emb, job_title_emb)
are derived from inputs that DID NOT change, so they are reused as-is.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

# SBERT is preferred for fix-4c title re-encoding but optional.
try:
    from sentence_transformers import SentenceTransformer  # noqa: F401
    _HAVE_SBERT = True
except Exception:
    _HAVE_SBERT = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# v2 preprocessing outputs. The user's preprocess_pipeline.py writes to
# `processed/` under its working dir, but at one point that subfolder was
# named `again proccesed data/`. Try both so the script is robust to either
# layout, prefer `processed/`.
_V2_CANDIDATES = [
    PROJECT_ROOT / "data" / "proccessed again" / "processed",
    PROJECT_ROOT / "data" / "proccessed again" / "again proccesed data",
]
V2_DIR = next((p for p in _V2_CANDIDATES if p.exists()), _V2_CANDIDATES[0])

V1_OLD_DIR   = PROJECT_ROOT / "data" / "processed_v1_old"
PROC_DIR     = PROJECT_ROOT / "data" / "processed"

RESUME_PARQUET   = V2_DIR / "cleaned_resumes.parquet"
JOB_PARQUET      = V2_DIR / "cleaned_jobs.parquet"
SKILL_DICT_JSON  = V2_DIR / "skill_dictionary_merged.json"
SKILL_CTX_JSON   = V2_DIR / "skill_to_context_map.json"
EMB_DIR          = V2_DIR / "embeddings"

PROC_DIR.mkdir(parents=True, exist_ok=True)
PAIRS_PATH = PROC_DIR / "pair_features_final.parquet"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SBERT_MODEL = "all-MiniLM-L6-v2"
TOP_K       = 50
TFIDF_MAX_FEATURES = 20_000
TFIDF_NGRAM = (1, 2)
TFIDF_MIN_DF = 2

ALL_FEATURES = [
    "embedding_similarity",
    "tfidf_similarity",
    "skill_overlap",
    "weighted_skill_score",
    "num_missing_skills",
    "avg_missing_skill_importance",
    "years_of_experience",
    "experience_gap",
    "experience_relevance_score",
    "title_similarity",
    "education_match",
]

# ---------------------------------------------------------------------------
# Fix 4b: ambiguous single-token tech skills -- raw-case match required
# ---------------------------------------------------------------------------
AMBIGUOUS_TOKENS = {"r", "go", "less", "express", "swift", "storm", "segment", "sketch"}
TECH_CONTEXT_CUES = {
    "language", "programming", "develop", "framework", "library", "package",
    "stack", "code", "coding", "scripting", "lang", "skills", "tools",
    "experience", "proficient", "knowledge", "using", "with",
}


def _ambiguous_in_raw(skill: str, raw_text: str) -> bool:
    if not isinstance(raw_text, str) or not raw_text:
        return False
    cap_token = skill.upper() if len(skill) <= 2 else skill[0].upper() + skill[1:]
    if re.search(rf"(?<![A-Za-z0-9_]){re.escape(cap_token)}(?![A-Za-z0-9_])", raw_text):
        post = re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(cap_token)}(?![A-Za-z0-9_])\s+(\w+)",
            raw_text,
        )
        if post:
            nxt = post.group(1).lower()
            if nxt in {"is", "are", "was", "were", "than", "in", "on", "the"} and skill in {"less", "go"}:
                pass
            else:
                return True
        else:
            return True
    s = re.escape(skill)
    lower = raw_text.lower()
    for m in re.finditer(rf"(?<![A-Za-z0-9_]){s}(?![A-Za-z0-9_])", lower):
        start, end = m.start(), m.end()
        window = lower[max(0, start - 40): min(len(lower), end + 40)]
        if any(cue in window for cue in TECH_CONTEXT_CUES):
            return True
    return False


def filter_ambiguous_skills(skills_value, raw_text: str) -> list[dict]:
    if skills_value is None:
        return []
    if isinstance(skills_value, np.ndarray):
        items = list(skills_value)
    elif isinstance(skills_value, list):
        items = skills_value
    elif isinstance(skills_value, str):
        try:
            items = ast.literal_eval(skills_value)
        except Exception:
            items = []
    else:
        items = []
    cleaned: list[dict] = []
    for it in items:
        if isinstance(it, dict):
            name = str(it.get("skill", "")).strip().lower()
            src  = it.get("source", "tech")
        else:
            name = str(it).strip().lower()
            src  = "tech"
        if not name:
            continue
        if name in AMBIGUOUS_TOKENS:
            if _ambiguous_in_raw(name, raw_text):
                cleaned.append({"skill": name, "source": src})
        else:
            cleaned.append({"skill": name, "source": src})
    return cleaned


# ---------------------------------------------------------------------------
# Fix 4c: derive resume title from first line of Resume_str
# ---------------------------------------------------------------------------
_TITLE_BAD_PATTERNS = [
    re.compile(r"\d{3,}"),
    re.compile(r"@"),
    re.compile(r"https?://", re.IGNORECASE),
]
_NAME_LIKE = re.compile(r"^[A-Z][a-z]+ [A-Z][a-z]+$")


def derive_resume_title(resume_str: str, category_fallback: str) -> str:
    if not isinstance(resume_str, str) or not resume_str.strip():
        return category_fallback
    for line in resume_str.split("\n"):
        line = line.strip()
        if not line:
            continue
        first = re.split(r"\s{3,}", line, maxsplit=1)[0].strip()
        if not first or len(first) < 3 or len(first) > 90:
            continue
        if any(p.search(first) for p in _TITLE_BAD_PATTERNS):
            continue
        if _NAME_LIKE.match(first):
            continue
        return first.lower()
    return category_fallback


# ---------------------------------------------------------------------------
# Fix 4a: YoE extraction with model_response fallback
# ---------------------------------------------------------------------------
_YOE_PATTERNS = [
    re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:to\s*\d+\s*)?(?:years?|yrs?)\b", re.IGNORECASE),
    re.compile(r"(\d+(?:\.\d+)?)\s*-\s*\d+\s*(?:years?|yrs?)\b", re.IGNORECASE),
]
_YOE_CAP = 50.0


def _scan_yoe(text: str) -> float:
    if not isinstance(text, str) or not text:
        return 0.0
    found = []
    for p in _YOE_PATTERNS:
        for m in p.finditer(text):
            try:
                v = float(m.group(1))
            except (ValueError, IndexError):
                continue
            if 0 < v <= _YOE_CAP:
                found.append(v)
    return max(found) if found else 0.0


def extract_resume_yoe(resume_str: str) -> float:
    return _scan_yoe(resume_str)


def extract_job_yoe(raw_jd: str, model_response: str) -> float:
    if isinstance(model_response, str) and model_response.strip():
        try:
            obj = json.loads(model_response.strip())
            if isinstance(obj, dict):
                exp_level = obj.get("Experience Level") or obj.get("experience_level")
                if isinstance(exp_level, str):
                    v = _scan_yoe(exp_level)
                    if v > 0:
                        return v
                v = _scan_yoe(json.dumps(obj))
                if v > 0:
                    return v
        except (json.JSONDecodeError, ValueError):
            v = _scan_yoe(model_response)
            if v > 0:
                return v
    return _scan_yoe(raw_jd)


# ---------------------------------------------------------------------------
# Education extraction
# ---------------------------------------------------------------------------
_DEGREE_PATTERNS = [
    (re.compile(r"\b(ph\.?\s?d|doctorate|doctoral)\b", re.IGNORECASE), 3),
    (re.compile(r"\b(m\.?s\.?c?|m\.?a\.?|master'?s?|m\.?eng|mba|m\.?tech)\b", re.IGNORECASE), 2),
    (re.compile(r"\b(b\.?s\.?c?|b\.?a\.?|bachelor'?s?|b\.?eng|b\.?tech|undergrad)\b", re.IGNORECASE), 1),
]


def extract_degree_level(text: str) -> int:
    if not isinstance(text, str) or not text:
        return 0
    best = 0
    for patt, lvl in _DEGREE_PATTERNS:
        if patt.search(text):
            best = max(best, lvl)
    return best


# ---------------------------------------------------------------------------
# Skill set helpers
# ---------------------------------------------------------------------------
def skills_to_set(skills_value) -> set[str]:
    if skills_value is None:
        return set()
    if isinstance(skills_value, np.ndarray):
        items = list(skills_value)
    elif isinstance(skills_value, list):
        items = skills_value
    elif isinstance(skills_value, str):
        try:
            items = ast.literal_eval(skills_value)
        except Exception:
            items = []
    else:
        items = []
    out = set()
    for it in items:
        if isinstance(it, dict):
            s = str(it.get("skill", "")).strip().lower()
        else:
            s = str(it).strip().lower()
        if s:
            out.add(s)
    return out


# ---------------------------------------------------------------------------
# Top-K retrieval
# ---------------------------------------------------------------------------
def topk_retrieval(job_emb, resume_emb, k):
    sims = job_emb @ resume_emb.T
    n_resumes = sims.shape[1]
    k = min(k, n_resumes)
    if k == n_resumes:
        order = np.argsort(-sims, axis=1)
        return order, np.take_along_axis(sims, order, axis=1)
    part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    rows = np.arange(sims.shape[0])[:, None]
    part_scores = sims[rows, part]
    order = np.argsort(-part_scores, axis=1)
    idx = part[rows, order]
    scores = part_scores[rows, order]
    return idx, scores


# ---------------------------------------------------------------------------
# TF-IDF pair similarities
# ---------------------------------------------------------------------------
def tfidf_pair_similarities(resume_text, job_text, pairs_job, pairs_res, batch=4096):
    print("[tfidf] vectorising")
    vec = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM,
        min_df=TFIDF_MIN_DF,
        sublinear_tf=True,
    )
    vec.fit(resume_text + job_text)
    R = sk_normalize(vec.transform(resume_text))
    J = sk_normalize(vec.transform(job_text))
    sims = np.empty(len(pairs_job), dtype=np.float32)
    for i in range(0, len(pairs_job), batch):
        j_idx = pairs_job[i:i + batch]
        r_idx = pairs_res[i:i + batch]
        prod = J[j_idx].multiply(R[r_idx]).sum(axis=1)
        sims[i:i + batch] = np.asarray(prod).ravel()
    return sims


# ---------------------------------------------------------------------------
# Skill features
# ---------------------------------------------------------------------------
def build_importance_map(skill_dict_merged, skill_ctx):
    onet_scores = {}
    for skill in skill_dict_merged:
        ctx = skill_ctx.get(skill)
        if ctx and ctx.get("importance_mean", 0.0) > 0:
            onet_scores[skill] = float(ctx["importance_mean"])
    default_importance = (
        float(np.mean(list(onet_scores.values()))) if onet_scores else 2.5
    )
    return onet_scores, default_importance


def skill_features(job_skills, resume_skills, pairs_job, pairs_res,
                   importance_map, default_importance):
    n = len(pairs_job)
    overlap      = np.zeros(n, dtype=np.float32)
    weighted     = np.zeros(n, dtype=np.float32)
    n_missing    = np.zeros(n, dtype=np.int32)
    avg_miss_imp = np.zeros(n, dtype=np.float32)
    for k in range(n):
        js = job_skills[pairs_job[k]]
        rs = resume_skills[pairs_res[k]]
        if not js:
            continue
        inter = js & rs
        miss  = js - rs
        overlap[k] = len(inter) / len(js)
        wj = sum(importance_map.get(s, default_importance) for s in js)
        wi = sum(importance_map.get(s, default_importance) for s in inter)
        weighted[k] = (wi / wj) if wj > 0 else 0.0
        n_missing[k] = len(miss)
        if miss:
            avg_miss_imp[k] = float(
                np.mean([importance_map.get(s, default_importance) for s in miss])
            )
    return {
        "skill_overlap": overlap,
        "weighted_skill_score": weighted,
        "num_missing_skills": n_missing,
        "avg_missing_skill_importance": avg_miss_imp,
    }


# ---------------------------------------------------------------------------
# Title encoding (fix 4c) -- SBERT preferred, stale cache fallback documented
# ---------------------------------------------------------------------------
def encode_titles(resume_titles, job_titles, n_resumes, n_jobs):
    """
    Returns (title_resume_emb, title_job_emb, mode_str).

    Priority:
      1. If SBERT is available -> re-encode with new resume titles (fix 4c).
      2. Else if cached title_resume_emb.npy exists with matching shape ->
         use it AS-IS, but log clearly that this is the OLD Category-based
         encoding (fix 4c partially reverted).
      3. Else fall back to TF-IDF char-n-gram (last resort).
    """
    if _HAVE_SBERT:
        print("[titles] re-encoding with SBERT (fix 4c applied)")
        model = SentenceTransformer(SBERT_MODEL)
        title_r = model.encode(
            resume_titles, batch_size=64, convert_to_numpy=True,
            show_progress_bar=False, normalize_embeddings=True,
        ).astype(np.float32)
        title_j = model.encode(
            job_titles, batch_size=64, convert_to_numpy=True,
            show_progress_bar=False, normalize_embeddings=True,
        ).astype(np.float32)
        # Persist back to cache so subsequent runs skip encoding
        try:
            np.save(EMB_DIR / "resume_title_emb.npy", title_r)
            np.save(EMB_DIR / "job_title_emb.npy",    title_j)
        except Exception:
            pass
        return title_r, title_j, "sbert"

    cached_r = EMB_DIR / "resume_title_emb.npy"
    cached_j = EMB_DIR / "job_title_emb.npy"
    if cached_r.exists() and cached_j.exists():
        title_r = np.load(cached_r)
        title_j = np.load(cached_j)
        if title_r.shape[0] == n_resumes and title_j.shape[0] == n_jobs:
            print("[titles] !!! sentence-transformers NOT installed -- "
                  "using STALE cached SBERT title embeddings (Category-based).")
            print("[titles]     fix 4c is PARTIALLY REVERTED for title_similarity.")
            print("[titles]     re-run on a machine with SBERT for canonical numbers.")
            return title_r, title_j, "stale-sbert-cache"

    # Last-resort fallback (should never be needed in normal use)
    print("[titles] no SBERT, no cache -- TF-IDF char-n-gram fallback")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, analyzer="char_wb")
    vec.fit(resume_titles + job_titles)
    return sk_normalize(vec.transform(resume_titles)), sk_normalize(vec.transform(job_titles)), "tfidf-char_wb"


# ---------------------------------------------------------------------------
# PHASE 1 -- build the variant-agnostic pair feature table
# ---------------------------------------------------------------------------
def build_pair_features() -> dict:
    print("=" * 70)
    print(" PHASE 1: Building shared pair_features table")
    print("=" * 70)

    print("\n[load] cleaned_resumes / cleaned_jobs")
    print(f"       v2 dir: {V2_DIR}")
    if not RESUME_PARQUET.exists():
        raise FileNotFoundError(
            f"{RESUME_PARQUET} not found. Tried these v2 dirs: {_V2_CANDIDATES}. "
            f"Update _V2_CANDIDATES in build_final_dataset.py if your preprocessing "
            f"pipeline wrote to a different location."
        )
    resumes = pd.read_parquet(RESUME_PARQUET).reset_index(drop=True)
    jobs    = pd.read_parquet(JOB_PARQUET).reset_index(drop=True)
    print(f"       resumes={len(resumes)}  jobs={len(jobs)}")

    skill_dict = json.load(open(SKILL_DICT_JSON, encoding="utf-8"))
    skill_ctx  = json.load(open(SKILL_CTX_JSON,  encoding="utf-8"))
    importance_map, default_importance = build_importance_map(skill_dict, skill_ctx)

    # Fix 4b
    print("\n[fix 4b] filtering ambiguous single-token skills")
    before_r = sum(len(skills_to_set(v)) for v in resumes["extracted_skills"])
    before_j = sum(len(skills_to_set(v)) for v in jobs["extracted_skills"])
    resumes["extracted_skills_v2"] = [
        filter_ambiguous_skills(s, raw)
        for s, raw in zip(resumes["extracted_skills"], resumes["Resume_str"])
    ]
    jobs["extracted_skills_v2"] = [
        filter_ambiguous_skills(s, raw)
        for s, raw in zip(jobs["extracted_skills"], jobs["job_description"])
    ]
    after_r = sum(len(skills_to_set(v)) for v in resumes["extracted_skills_v2"])
    after_j = sum(len(skills_to_set(v)) for v in jobs["extracted_skills_v2"])
    print(f"         resume skills: {before_r} -> {after_r}  (dropped {before_r - after_r})")
    print(f"         job    skills: {before_j} -> {after_j}  (dropped {before_j - after_j})")

    # Fix 4c
    print("\n[fix 4c] deriving resume titles from first line of Resume_str")
    resumes["resume_title"] = [
        derive_resume_title(rs, cat)
        for rs, cat in zip(resumes["Resume_str"], resumes["Category"])
    ]
    n_kept_cat = (resumes["resume_title"].str.lower() == resumes["Category"].str.lower()).sum()
    print(f"         {n_kept_cat}/{len(resumes)} fell back to Category")

    # Fix 4a
    print("\n[fix 4a] extracting JD YoE with model_response fallback")
    job_yoe = np.array(
        [extract_job_yoe(jd, mr)
         for jd, mr in zip(jobs["job_description"].fillna(""),
                            jobs["model_response"].fillna(""))],
        dtype=np.float32,
    )
    n_jd_pos = int((job_yoe > 0).sum())
    print(f"         {n_jd_pos}/{len(jobs)} jobs now have YoE > 0")

    resume_yoe = np.array(
        [extract_resume_yoe(rs) for rs in resumes["Resume_str"].fillna("")],
        dtype=np.float32,
    )
    print(f"         resume YoE: mean={resume_yoe.mean():.2f} max={resume_yoe.max():.0f}")

    resume_edu = np.array(
        [extract_degree_level(t) for t in resumes["section_education"].fillna("")],
        dtype=np.int8,
    )
    job_edu = np.array(
        [extract_degree_level(jd) for jd in jobs["job_description"].fillna("")],
        dtype=np.int8,
    )

    # Embeddings: cached resume_emb / job_emb / resume_exp_emb are reused as-is
    # (their inputs didn't change).
    print("\n[emb] reusing cached SBERT embeddings (cleaned text unchanged)")
    resume_emb     = np.load(EMB_DIR / "resume_emb.npy")
    job_emb        = np.load(EMB_DIR / "job_emb.npy")
    resume_exp_emb = np.load(EMB_DIR / "resume_exp_emb.npy")
    print(f"      resume_emb: {resume_emb.shape}")
    print(f"      job_emb:    {job_emb.shape}")
    print(f"      resume_exp_emb: {resume_exp_emb.shape}")

    # Title encoding (only place fix 4c affects the embedding cache)
    title_resume_emb, title_job_emb, title_mode = encode_titles(
        resumes["resume_title"].astype(str).tolist(),
        jobs["position_title"].fillna("").astype(str).tolist(),
        len(resumes), len(jobs),
    )

    # Top-K retrieval
    k = min(TOP_K, len(resumes))
    print(f"\n[topk] top-{k} retrieval")
    topk_idx, topk_scores = topk_retrieval(job_emb, resume_emb, k=k)
    pairs_job   = np.repeat(np.arange(len(jobs)), k)
    pairs_res   = topk_idx.reshape(-1)
    pairs_sbert = topk_scores.reshape(-1).astype(np.float32)
    print(f"       top-K pairs: {len(pairs_job)}")

    # Force-include gold pairs that fell outside top-K
    old_gold_csv = V1_OLD_DIR / "gold_standard_final.csv"
    gold_raw = pd.read_csv(old_gold_csv)
    res_lookup = {str(resumes.iloc[i]["Resume_str"]): i for i in range(len(resumes))}
    job_lookup = {str(jobs.iloc[i]["job_description"]):  i for i in range(len(jobs))}
    gold_raw["resume_id"] = gold_raw["resume_text"].astype(str).map(res_lookup)
    gold_raw["job_id"]    = gold_raw["job_description"].astype(str).map(job_lookup)
    gold_pairs = gold_raw.dropna(subset=["resume_id", "job_id"]).copy()
    gold_pairs["resume_id"] = gold_pairs["resume_id"].astype(int)
    gold_pairs["job_id"]    = gold_pairs["job_id"].astype(int)

    seen = set(zip(pairs_job.tolist(), pairs_res.tolist()))
    extra_pairs = [
        (int(j), int(r))
        for j, r in zip(gold_pairs["job_id"], gold_pairs["resume_id"])
        if (int(j), int(r)) not in seen
    ]
    if extra_pairs:
        ej, er = zip(*extra_pairs)
        ej_arr = np.array(ej, dtype=np.int32)
        er_arr = np.array(er, dtype=np.int32)
        extra_sbert = (job_emb[ej_arr] * resume_emb[er_arr]).sum(axis=1).astype(np.float32)
        pairs_job   = np.concatenate([pairs_job, ej_arr])
        pairs_res   = np.concatenate([pairs_res, er_arr])
        pairs_sbert = np.concatenate([pairs_sbert, extra_sbert])
        print(f"       force-included {len(extra_pairs)} gold pairs outside top-K")
    print(f"       total pairs: {len(pairs_job)}")

    # TF-IDF
    resume_text = resumes["cleaned_resume"].fillna("").tolist()
    job_text    = jobs["cleaned_job_description"].fillna("").tolist()
    tfidf_sims = tfidf_pair_similarities(resume_text, job_text, pairs_job, pairs_res)

    # Skill features
    print("\n[skills] building per-pair skill features (post-fix-4b)")
    resume_skill_sets = [skills_to_set(v) for v in resumes["extracted_skills_v2"]]
    job_skill_sets    = [skills_to_set(v) for v in jobs["extracted_skills_v2"]]
    skill_feats = skill_features(
        job_skill_sets, resume_skill_sets, pairs_job, pairs_res,
        importance_map, default_importance,
    )

    # Experience / title / education
    print("\n[features] experience + title + education")
    pair_resume_yoe = resume_yoe[pairs_res]
    pair_job_yoe    = job_yoe[pairs_job]
    exp_gap = (pair_job_yoe - pair_resume_yoe).astype(np.float32)
    exp_relevance = (
        resume_exp_emb[pairs_res] * job_emb[pairs_job]
    ).sum(axis=1).astype(np.float32)

    # Title similarity -- dense if SBERT, sparse if TF-IDF fallback
    if title_mode == "tfidf-char_wb":
        n = len(pairs_job)
        title_sim = np.empty(n, dtype=np.float32)
        BATCH = 4096
        for i in range(0, n, BATCH):
            j_idx = pairs_job[i:i + BATCH]
            r_idx = pairs_res[i:i + BATCH]
            prod = title_job_emb[j_idx].multiply(title_resume_emb[r_idx]).sum(axis=1)
            title_sim[i:i + BATCH] = np.asarray(prod).ravel()
    else:
        title_sim = (
            title_resume_emb[pairs_res] * title_job_emb[pairs_job]
        ).sum(axis=1).astype(np.float32)

    pair_resume_edu = resume_edu[pairs_res]
    pair_job_edu    = job_edu[pairs_job]
    edu_match = ((pair_job_edu == 0) | (pair_resume_edu >= pair_job_edu)).astype(np.int8)

    pair_df = pd.DataFrame({
        "job_id":    pairs_job.astype(np.int32),
        "resume_id": pairs_res.astype(np.int32),
        "embedding_similarity":         pairs_sbert,
        "tfidf_similarity":             tfidf_sims,
        "skill_overlap":                skill_feats["skill_overlap"],
        "weighted_skill_score":         skill_feats["weighted_skill_score"],
        "num_missing_skills":           skill_feats["num_missing_skills"],
        "avg_missing_skill_importance": skill_feats["avg_missing_skill_importance"],
        "years_of_experience":          pair_resume_yoe,
        "experience_gap":               exp_gap,
        "experience_relevance_score":   exp_relevance,
        "title_similarity":             title_sim,
        "education_match":              edu_match,
    })
    pair_df.to_parquet(PAIRS_PATH, index=False)
    print(f"\n[write] {PAIRS_PATH}  ({len(pair_df)} rows)")
    return {"title_mode": title_mode, "n_pairs": len(pair_df)}


# ---------------------------------------------------------------------------
# PHASE 2 -- apply a variant config to the shared pair table
# ---------------------------------------------------------------------------
def assign_labels(pair_df: pd.DataFrame, weights: dict, high_pct: float, low_pct: float):
    score = np.zeros(len(pair_df), dtype=np.float32)
    for col, w in weights.items():
        v = pair_df[col].astype(np.float32).clip(lower=0).values
        score += w * v
    out = pair_df.copy()
    out["composite_score"] = score
    high = float(np.quantile(score, high_pct))
    low  = float(np.quantile(score, low_pct))
    print(f"[label] composite quantiles: low({low_pct})={low:.4f}  high({high_pct})={high:.4f}")
    label = np.full(len(out), np.nan, dtype=np.float32)
    label[score >= high] = 1.0
    label[score <= low]  = 0.0
    out["label"] = label
    return out


def remap_gold(pair_df: pd.DataFrame, resumes: pd.DataFrame, jobs: pd.DataFrame,
               old_gold_csv: Path) -> pd.DataFrame:
    print("[gold] loading old gold standard")
    old_gold = pd.read_csv(old_gold_csv)
    res_lookup = {str(resumes.iloc[i]["Resume_str"]): i for i in range(len(resumes))}
    job_lookup = {str(jobs.iloc[i]["job_description"]):  i for i in range(len(jobs))}
    old_gold["resume_id"] = old_gold["resume_text"].astype(str).map(res_lookup)
    old_gold["job_id"]    = old_gold["job_description"].astype(str).map(job_lookup)
    found = old_gold.dropna(subset=["resume_id", "job_id"]).copy()
    found["resume_id"] = found["resume_id"].astype(int)
    found["job_id"]    = found["job_id"].astype(int)
    merged = pair_df.merge(
        found[["resume_id", "job_id", "gold_label"]],
        on=["resume_id", "job_id"], how="inner",
    )
    print(f"[gold] eval set: {len(merged)} pairs")
    return merged


def build_variant(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    name        = config["variant_name"]
    weights     = config["label_weights"]
    high_pct    = config["label_quantiles"]["high_pct"]
    low_pct     = config["label_quantiles"]["low_pct"]
    holdout     = set(config.get("holdout_features", []))
    train_feats = config["training_features"]

    print("=" * 70)
    print(f" PHASE 2: Building variant {name}")
    print("=" * 70)
    print(f"  weights       : {weights}")
    print(f"  quantile cut  : {low_pct} / {high_pct}")
    print(f"  holdout       : {sorted(holdout)}")
    print(f"  training cols : {train_feats}")

    # Sanity check: training_features and holdout_features must be disjoint
    overlap_check = set(train_feats) & holdout
    assert not overlap_check, (
        f"training_features and holdout_features overlap: {overlap_check}"
    )
    # Sanity check: every name is a real feature
    unknown = (set(train_feats) | holdout) - set(ALL_FEATURES)
    assert not unknown, f"unknown features in config: {unknown}"

    # Load shared pair table
    if not PAIRS_PATH.exists():
        raise FileNotFoundError(
            f"{PAIRS_PATH} not found. Run with --phase pairs first."
        )
    pair_df = pd.read_parquet(PAIRS_PATH)
    print(f"\n[load] {PAIRS_PATH}  ({len(pair_df)} rows)")

    # Apply labels
    labeled = assign_labels(pair_df, weights, high_pct, low_pct)
    train_df = labeled.dropna(subset=["label"]).copy()
    train_df["label"] = train_df["label"].astype(int)
    n_pos = int((train_df["label"] == 1).sum())
    n_neg = int((train_df["label"] == 0).sum())
    print(f"[label] kept {len(train_df)}/{len(labeled)} pairs  | pos={n_pos} neg={n_neg}")

    # Sanity check: balance after quantile cut
    bal = max(n_pos, n_neg) / max(1, n_pos + n_neg)
    if bal > 0.60:
        print(f"[WARN] label distribution is {bal:.1%} -- expected ~50/50 from quantile cut")

    # Re-map gold standard
    print("\n[gold] re-mapping old gold standard onto pair universe")
    resumes = pd.read_parquet(RESUME_PARQUET).reset_index(drop=True)
    jobs    = pd.read_parquet(JOB_PARQUET).reset_index(drop=True)
    gold_df = remap_gold(pair_df, resumes, jobs, V1_OLD_DIR / "gold_standard_final.csv")

    # Write variant outputs
    out_dir = PROC_DIR / f"variant_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "ml_ready_dataset.parquet"
    gold_path  = out_dir / "gold_standard_final.parquet"
    train_df.to_parquet(train_path, index=False)
    gold_df.to_parquet(gold_path, index=False)
    print(f"\n[write] {train_path}")
    print(f"[write] {gold_path}")

    # Persist a summary in the variant dir for later report generation
    summary = {
        "variant_name": name,
        "n_pairs_total": int(len(pair_df)),
        "n_train": int(len(train_df)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_gold": int(len(gold_df)),
        "training_features": train_feats,
        "holdout_features": sorted(holdout),
        "label_weights": weights,
        "label_quantiles": config["label_quantiles"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="v2 dataset builder")
    ap.add_argument("--phase", choices=["pairs", "variant", "all"], required=True,
                    help="pairs = build shared feature table; variant = apply config; all = both")
    ap.add_argument("--config", type=Path, default=None,
                    help="Path to variant config JSON (required for phase variant/all)")
    args = ap.parse_args()

    if args.phase in ("pairs", "all"):
        info = build_pair_features()
        print(f"\n[done] pair_features built (title_mode={info['title_mode']})")

    if args.phase in ("variant", "all"):
        if args.config is None:
            ap.error("--config is required for phase variant/all")
        build_variant(args.config)


if __name__ == "__main__":
    main()
