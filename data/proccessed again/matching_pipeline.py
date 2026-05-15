"""
Two-stage resume <-> job matching pipeline.

  Stage 1  SBERT embeddings + top-K retrieval per job
  Stage 2  Rich feature extraction over the (job, resume) shortlist

Inputs (produced by preprocess_pipeline.py):
    processed/cleaned_resumes.parquet
    processed/cleaned_jobs.parquet
    processed/skill_dictionary_merged.json
    processed/skill_to_context_map.json

Output:
    processed/pair_features.parquet   (~ n_jobs x TOP_K rows)
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SBERT_MODEL = "all-MiniLM-L6-v2"
TOP_K = 50
ENCODE_BATCH = 64
TFIDF_MAX_FEATURES = 20_000
TFIDF_NGRAM = (1, 2)
TFIDF_MIN_DF = 2

# Resolve paths relative to this script's location so the script works when
# launched from anywhere (project root, this dir, etc.).
_SCRIPT_DIR = Path(__file__).resolve().parent
PROC = _SCRIPT_DIR / "processed"
RESUME_PARQUET = PROC / "cleaned_resumes.parquet"
JOB_PARQUET = PROC / "cleaned_jobs.parquet"
SKILL_DICT_JSON = PROC / "skill_dictionary_merged.json"
SKILL_CTX_JSON = PROC / "skill_to_context_map.json"
OUT_PAIRS = PROC / "pair_features.parquet"
OUT_EMB_DIR = PROC / "embeddings"


# ---------------------------------------------------------------------------
# Skill column parsing
# ---------------------------------------------------------------------------
def parse_skill_field(value) -> list[str]:
    """The extracted_skills column may be list[dict], np.ndarray, or string repr."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        items = list(value)
    elif isinstance(value, list):
        items = value
    elif isinstance(value, str):
        try:
            items = ast.literal_eval(value)
        except Exception:
            items = []
    else:
        items = []
    out: list[str] = []
    for it in items:
        if isinstance(it, dict):
            s = it.get("skill", "")
        else:
            s = str(it)
        s = s.strip().lower()
        if s:
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Years-of-experience extraction (expanded)
# ---------------------------------------------------------------------------
# Cap to reject parsed garbage (e.g. "1995 years" from a date)
_YOE_CAP = 50.0

# Word forms one through ten (covers 90%+ of word-form mentions in resumes/JDs)
_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
}

# Numeric range patterns ("3-5 years", "3 to 5 years"). Take midpoint.
# Note hyphen and en-dash both, and trailing optional "+".
_YOE_RANGE_PATTERNS = [
    re.compile(
        r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(\d+(?:\.\d+)?)\s*to\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b",
        re.IGNORECASE,
    ),
]

# Single numeric forms: "5 years", "5+ years", "5 yrs", "5 years of experience"
_YOE_SINGLE_PATTERNS = [
    re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b", re.IGNORECASE),
    re.compile(r"(\d+(?:\.\d+)?)\s*yrs?\b", re.IGNORECASE),
]

# Word-form: "five years (of experience)"
_YOE_WORD_PATTERN = re.compile(
    r"\b(" + "|".join(_WORD_TO_NUM.keys()) + r")\s*\+?\s*(?:years?|yrs?)\b",
    re.IGNORECASE,
)


def extract_years(text: str) -> float:
    """
    Maximum plausible years-of-experience mentioned in `text`. Handles:
      - "5 years", "5+ years", "5 yrs"
      - "3-5 years" / "3 to 5 years"  (returns midpoint = 4)
      - "five years"                  (word forms one through twenty)
    Range matches are processed first and substituted out so a "3-5 years"
    range doesn't double-count as "5 years" via the single-number pattern.
    """
    if not isinstance(text, str) or not text:
        return 0.0
    found: list[float] = []
    remaining = text

    # 1. Range patterns (midpoint), substituted out of `remaining` after.
    for p in _YOE_RANGE_PATTERNS:
        for m in p.finditer(remaining):
            try:
                lo = float(m.group(1))
                hi = float(m.group(2))
            except (ValueError, IndexError):
                continue
            if 0 < lo <= _YOE_CAP and 0 < hi <= _YOE_CAP and lo <= hi:
                found.append((lo + hi) / 2.0)
        remaining = p.sub(" ", remaining)

    # 2. Single numeric forms on the cleaned remainder
    for p in _YOE_SINGLE_PATTERNS:
        for m in p.finditer(remaining):
            try:
                v = float(m.group(1))
            except (ValueError, IndexError):
                continue
            if 0 < v <= _YOE_CAP:
                found.append(v)

    # 3. Word forms ("five years")
    for m in _YOE_WORD_PATTERN.finditer(remaining):
        word = m.group(1).lower()
        if word in _WORD_TO_NUM:
            v = float(_WORD_TO_NUM[word])
            if 0 < v <= _YOE_CAP:
                found.append(v)

    return max(found) if found else 0.0


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
# Resume title from first line of Resume_str (replaces Category as title source)
# ---------------------------------------------------------------------------
_TITLE_BAD_PATTERNS = [
    re.compile(r"\d{3,}"),                   # phone-like digits
    re.compile(r"@"),                         # email
    re.compile(r"https?://", re.IGNORECASE),  # url
]
_NAME_LIKE = re.compile(r"^[A-Z][a-z]+ [A-Z][a-z]+$")


def derive_resume_title(resume_str: str, category_fallback: str) -> str:
    """
    First non-empty line of `Resume_str`, cut at the first run of 3+ spaces
    (Resume_str uses 3+ spaces as a pseudo-tab between the role line and
    "Summary" / "Experience" headers). Skips obvious contact lines and pure
    name lines. Falls back to `category_fallback` if nothing usable is found.
    """
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
# 1. SBERT embeddings (batched, normalized)
# ---------------------------------------------------------------------------
def encode_texts(model: SentenceTransformer, texts: list[str], desc: str) -> np.ndarray:
    print(f"      encoding {desc}: {len(texts)} texts")
    return model.encode(
        texts,
        batch_size=ENCODE_BATCH,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine == dot product
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Top-K retrieval
# ---------------------------------------------------------------------------
def topk_retrieval(
    job_emb: np.ndarray, resume_emb: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        idx     (n_jobs, k) resume indices, sorted by descending similarity
        scores  (n_jobs, k) cosine similarities aligned to idx
    """
    sims = job_emb @ resume_emb.T  # both row-normalized -> cosine
    n_resumes = sims.shape[1]
    k = min(k, n_resumes)

    if k == n_resumes:
        order = np.argsort(-sims, axis=1)
        return order, np.take_along_axis(sims, order, axis=1)

    # argpartition pulls top-k unsorted, then we sort just those k
    part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    rows = np.arange(sims.shape[0])[:, None]
    part_scores = sims[rows, part]
    order = np.argsort(-part_scores, axis=1)
    idx = part[rows, order]
    scores = part_scores[rows, order]
    return idx, scores


# ---------------------------------------------------------------------------
# 4.1 TF-IDF pair similarities (sparse, batched)
# ---------------------------------------------------------------------------
def tfidf_pair_similarities(
    resume_text: list[str],
    job_text: list[str],
    pairs_job: np.ndarray,
    pairs_res: np.ndarray,
    batch: int = 4096,
) -> np.ndarray:
    print("[4.1] TF-IDF vectorization")
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
        j_idx = pairs_job[i : i + batch]
        r_idx = pairs_res[i : i + batch]
        # Element-wise product of paired sparse rows, then row-sum = cosine (rows are unit-norm)
        prod = J[j_idx].multiply(R[r_idx]).sum(axis=1)
        sims[i : i + batch] = np.asarray(prod).ravel()
    return sims


# ---------------------------------------------------------------------------
# 4.2 Skill-overlap features
# ---------------------------------------------------------------------------
def skill_features(
    job_skill_sets: list[set],
    resume_skill_sets: list[set],
    pairs_job: np.ndarray,
    pairs_res: np.ndarray,
    importance_map: dict[str, float],
    default_importance: float,
) -> dict[str, np.ndarray]:
    n = len(pairs_job)
    overlap = np.zeros(n, dtype=np.float32)
    weighted = np.zeros(n, dtype=np.float32)
    n_missing = np.zeros(n, dtype=np.int32)
    avg_miss_imp = np.zeros(n, dtype=np.float32)

    for k in range(n):
        js = job_skill_sets[pairs_job[k]]
        rs = resume_skill_sets[pairs_res[k]]
        if not js:
            continue
        inter = js & rs
        miss = js - rs
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
# Importance map.
#   O*NET skills:  use the curated mean importance (0..5 from skill_to_context_map).
#   ESCO skills:   use ESCO hierarchy depth as a proxy for specificity and map
#                  to a 0..5 score via importance = (depth / max_depth) * 5.
#                  Deeper in the tree = more specialised = higher importance.
#   Otherwise:     fall back to default_importance (mean of O*NET importances).
#
# O*NET takes priority over ESCO when both have a score for the same label.
# ---------------------------------------------------------------------------
ESCO_DEPTH_JSON = _SCRIPT_DIR / "esco_skill_depths.json"


def build_importance_map(skill_dict_merged: dict, skill_ctx: dict) -> tuple[dict[str, float], float]:
    onet_scores: dict[str, float] = {}
    for skill in skill_dict_merged:
        ctx = skill_ctx.get(skill)
        if ctx and ctx.get("importance_mean", 0.0) > 0:
            onet_scores[skill] = float(ctx["importance_mean"])

    default_importance = (
        float(np.mean(list(onet_scores.values()))) if onet_scores else 2.5
    )

    # Layer ESCO depth-based importance on top, but never overwrite O*NET.
    importance_map: dict[str, float] = dict(onet_scores)
    n_esco_added = 0
    if ESCO_DEPTH_JSON.exists():
        try:
            data = json.load(open(ESCO_DEPTH_JSON, encoding="utf-8"))
            depth_by_label = data.get("depth_by_label", {})
            max_depth = int(data.get("max_depth") or 1)
            if max_depth <= 0:
                max_depth = 1
            for label, d in depth_by_label.items():
                if not isinstance(d, (int, float)) or d <= 0:
                    continue
                if label in importance_map:
                    continue   # O*NET wins
                imp = (float(d) / float(max_depth)) * 5.0
                # Clamp into the same 0..5 range as O*NET for downstream comparability
                importance_map[label] = max(0.0, min(5.0, imp))
                n_esco_added += 1
            print(f"[importance] ESCO depth scores added: {n_esco_added:,} "
                  f"(max_depth={max_depth}, formula=(depth/max_depth)*5)")
        except Exception as e:
            print(f"[warn] could not load {ESCO_DEPTH_JSON.name}: {e}")
    else:
        print(f"[warn] {ESCO_DEPTH_JSON.name} not found -> ESCO skills fall "
              f"back to default_importance ({default_importance:.3f}). "
              f"Run build_esco_depths.py to enable depth-based weighting.")

    return importance_map, default_importance


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(top_k: int = TOP_K, sample: int | None = None) -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    OUT_EMB_DIR.mkdir(parents=True, exist_ok=True)

    # --- Snapshot existing pair_features.parquet for before/after comparison ---
    # This is purely for the importance-update report; the new pair_features
    # will overwrite this file at the end of main().
    old_weighted_score = None
    old_avg_miss_imp   = None
    old_skill_overlap  = None
    if OUT_PAIRS.exists():
        try:
            _old = pd.read_parquet(
                OUT_PAIRS,
                columns=["weighted_skill_score", "avg_missing_skill_importance",
                         "skill_overlap"],
            )
            old_weighted_score = _old["weighted_skill_score"].to_numpy()
            old_avg_miss_imp   = _old["avg_missing_skill_importance"].to_numpy()
            old_skill_overlap  = _old["skill_overlap"].to_numpy()
            print(f"[snapshot] captured prior weighted_skill_score / "
                  f"avg_missing_skill_importance for diff "
                  f"({len(old_weighted_score):,} rows)")
        except Exception as e:
            print(f"[snapshot] could not snapshot prior pair_features: {e}")

    # --- Load ---
    print("[load] resumes / jobs / skill artefacts")
    resumes = pd.read_parquet(RESUME_PARQUET).reset_index(drop=True)
    jobs = pd.read_parquet(JOB_PARQUET).reset_index(drop=True)
    if sample:
        resumes = resumes.head(sample).reset_index(drop=True)
        jobs = jobs.head(sample).reset_index(drop=True)
    skill_dict = json.load(open(SKILL_DICT_JSON, encoding="utf-8"))
    skill_ctx = json.load(open(SKILL_CTX_JSON, encoding="utf-8"))
    importance_map, default_importance = build_importance_map(skill_dict, skill_ctx)
    print(f"      resumes={len(resumes)}  jobs={len(jobs)}  "
          f"importance default={default_importance:.3f}")

    # --- Resume side ---
    resume_text = resumes["cleaned_resume"].fillna("").tolist()
    resume_exp_text = resumes["section_experience"].fillna("").tolist()
    # Title source: first line of Resume_str (granular role like "HR DIRECTOR")
    # rather than the broad Category column (e.g. "HR"). This is fix 4c —
    # without re-encoding here the title cache stays stuck on Category and
    # title_similarity collapses to ~340 unique values across 25k pairs.
    resume_titles = [
        derive_resume_title(rs, cat)
        for rs, cat in zip(
            resumes["Resume_str"].fillna("").astype(str).tolist(),
            resumes["Category"].fillna("").astype(str).tolist(),
        )
    ]
    resume_yoe = np.array(
        [extract_years(t) for t in resumes["Resume_str"].fillna("").tolist()],
        dtype=np.float32,
    )
    resume_edu = np.array(
        [extract_degree_level(t) for t in resumes["section_education"].fillna("").tolist()],
        dtype=np.int8,
    )
    resume_skills = [set(parse_skill_field(v)) for v in resumes["extracted_skills"].tolist()]

    # --- Job side ---
    job_text = jobs["cleaned_job_description"].fillna("").tolist()
    job_titles = jobs["position_title"].fillna("").astype(str).tolist()
    raw_jd = jobs["job_description"].fillna("").tolist()  # YoE/degree extraction off raw, not cleaned
    job_yoe = np.array([extract_years(t) for t in raw_jd], dtype=np.float32)
    job_edu = np.array([extract_degree_level(t) for t in raw_jd], dtype=np.int8)
    job_skills = [set(parse_skill_field(v)) for v in jobs["extracted_skills"].tolist()]

    # ----------------------------------------------------------------- #
    # Step 1: SBERT embeddings (cached)                                  #
    # ----------------------------------------------------------------- #
    print(f"[1] loading SBERT model: {SBERT_MODEL}")
    model = SentenceTransformer(SBERT_MODEL)

    def cached_encode(texts: list[str], name: str) -> np.ndarray:
        cache = OUT_EMB_DIR / f"{name}.npy"
        if cache.exists() and not sample:
            arr = np.load(cache)
            if arr.shape[0] == len(texts):
                print(f"      [cache hit] {name}: {arr.shape}")
                return arr
        emb = encode_texts(model, texts, name)
        if not sample:
            np.save(cache, emb)
        return emb

    resume_emb = cached_encode(resume_text, "resume_emb")
    job_emb = cached_encode(job_text, "job_emb")
    resume_exp_emb = cached_encode(resume_exp_text, "resume_exp_emb")
    title_resume_emb = cached_encode(resume_titles, "resume_title_emb")
    title_job_emb = cached_encode(job_titles, "job_title_emb")

    # ----------------------------------------------------------------- #
    # Step 2: Top-K retrieval                                            #
    # ----------------------------------------------------------------- #
    k = min(top_k, len(resumes))
    print(f"[2] top-{k} retrieval over {len(jobs)} jobs x {len(resumes)} resumes")
    topk_idx, topk_scores = topk_retrieval(job_emb, resume_emb, k=k)

    # ----------------------------------------------------------------- #
    # Step 3: Build pair set                                             #
    # ----------------------------------------------------------------- #
    pairs_job = np.repeat(np.arange(len(jobs)), k)
    pairs_res = topk_idx.reshape(-1)
    pairs_sbert = topk_scores.reshape(-1).astype(np.float32)
    print(f"[3] pair set: {len(pairs_job)} rows  "
          f"({len(jobs)} jobs x top-{k})")

    # ----------------------------------------------------------------- #
    # Step 4.1: TF-IDF cosine similarity                                 #
    # ----------------------------------------------------------------- #
    tfidf_sims = tfidf_pair_similarities(resume_text, job_text, pairs_job, pairs_res)

    # ----------------------------------------------------------------- #
    # Step 4.2: Skill features                                           #
    # ----------------------------------------------------------------- #
    print("[4.2] skill overlap + weighted score + missing-skill features")
    skill_feats = skill_features(
        job_skills, resume_skills, pairs_job, pairs_res,
        importance_map, default_importance,
    )

    # ----------------------------------------------------------------- #
    # Step 4.3: Experience features                                      #
    # ----------------------------------------------------------------- #
    print("[4.3] experience features (yoe, gap, relevance)")
    pair_resume_yoe = resume_yoe[pairs_res]
    pair_job_yoe = job_yoe[pairs_job]
    exp_gap = (pair_job_yoe - pair_resume_yoe).astype(np.float32)
    # exp relevance: cosine(resume.experience_section, job.full_jd)
    exp_relevance = (resume_exp_emb[pairs_res] * job_emb[pairs_job]).sum(axis=1).astype(np.float32)

    # ----------------------------------------------------------------- #
    # Step 4.4: Title similarity                                         #
    # ----------------------------------------------------------------- #
    print("[4.4] title similarity")
    title_sim = (title_resume_emb[pairs_res] * title_job_emb[pairs_job]).sum(axis=1).astype(np.float32)

    # ----------------------------------------------------------------- #
    # Step 4.5: Education match                                          #
    # ----------------------------------------------------------------- #
    print("[4.5] education match")
    pair_resume_edu = resume_edu[pairs_res]
    pair_job_edu = job_edu[pairs_job]
    # 1 if candidate degree level >= required (or no requirement parseable)
    edu_match = ((pair_job_edu == 0) | (pair_resume_edu >= pair_job_edu)).astype(np.int8)

    # ----------------------------------------------------------------- #
    # Step 5: Assemble output                                            #
    # ----------------------------------------------------------------- #
    print("[5] assembling pair feature table")
    out = pd.DataFrame({
        "job_id": pairs_job.astype(np.int32),
        "resume_id": pairs_res.astype(np.int32),
        "embedding_similarity": pairs_sbert,
        "tfidf_similarity": tfidf_sims,
        "skill_overlap": skill_feats["skill_overlap"],
        "weighted_skill_score": skill_feats["weighted_skill_score"],
        "num_missing_skills": skill_feats["num_missing_skills"],
        "avg_missing_skill_importance": skill_feats["avg_missing_skill_importance"],
        "years_of_experience": pair_resume_yoe,
        "experience_gap": exp_gap,
        "experience_relevance_score": exp_relevance,
        "title_similarity": title_sim,
        "education_match": edu_match,
    })
    out.to_parquet(OUT_PAIRS, index=False)
    print(f"[done] wrote {len(out)} rows -> {OUT_PAIRS}")

    # ----------------------------------------------------------------- #
    # Explicit success metrics                                           #
    # ----------------------------------------------------------------- #
    print()
    print("=== success metrics (compare to pre-ESCO baseline) ===")
    pct_overlap_pos = float((out["skill_overlap"] > 0).mean()) * 100
    pct_yoe_pos     = float((out["years_of_experience"] > 0).mean()) * 100
    n_unique_title  = int(out["title_similarity"].round(6).nunique())
    print(f"  skill_overlap > 0:           {pct_overlap_pos:5.1f}%   "
          f"(was 36.4%; target >= 50%)")
    print(f"  years_of_experience > 0:     {pct_yoe_pos:5.1f}%   "
          f"(was 39.1%)")
    print(f"  title_similarity unique:     {n_unique_title:>6,}   "
          f"(was 7,286; target >= 20,000)")

    if pct_overlap_pos < 40:
        print()
        print("[STOP] skill_overlap > 0 is below 40%. Vocabulary integration "
              "may have failed. Inspect cleaned_*_parquet.extracted_skills "
              "before regenerating ml_ready_dataset.")

    # ----------------------------------------------------------------- #
    # Importance-update before/after comparison                          #
    # ----------------------------------------------------------------- #
    # Only the two importance-weighted columns can change when the
    # importance map changes. skill_overlap and num_missing_skills are
    # unweighted counts and must be identical to the previous run; we use
    # that as a sanity check.
    if old_weighted_score is not None and len(old_weighted_score) == len(out):
        print()
        print("=== importance update: before / after ===")

        new_w = out["weighted_skill_score"].to_numpy()
        new_a = out["avg_missing_skill_importance"].to_numpy()
        new_overlap = out["skill_overlap"].to_numpy()

        # Sanity check: skill_overlap should be unchanged
        overlap_unchanged = bool(np.allclose(new_overlap, old_skill_overlap, atol=1e-6))
        print(f"  skill_overlap unchanged from prior run: {overlap_unchanged}")
        if not overlap_unchanged:
            print("  [warn] skill_overlap changed. Either the vocabulary or the "
                  "extracted_skills set changed too -- not just the importance map.")

        # weighted_skill_score == skill_overlap fraction (success metric)
        old_match = float(np.isclose(old_weighted_score, old_skill_overlap, atol=1e-6).mean()) * 100
        new_match = float(np.isclose(new_w, new_overlap, atol=1e-6).mean()) * 100
        print(f"  weighted_skill_score == skill_overlap: "
              f"was {old_match:5.1f}%  ->  now {new_match:5.1f}%   "
              f"(success metric: target < 50%)")

        # avg_missing_skill_importance default-value share
        old_default_share = float(np.isclose(old_avg_miss_imp, 2.59328, atol=1e-3).mean()) * 100
        # Re-pick the new default so the share is computed against the right value
        # default_importance is recomputed in build_importance_map; emit it so the
        # user can spot the new fallback rate. We approximate it from the data.
        new_default_share_2_59 = float(np.isclose(new_a, 2.59328, atol=1e-3).mean()) * 100
        print(f"  avg_missing_skill_importance == 2.593: "
              f"was {old_default_share:5.1f}%  ->  now {new_default_share_2_59:5.1f}%")

        # Unique value counts
        n_w_old = int(np.unique(np.round(old_weighted_score, 6)).size)
        n_w_new = int(np.unique(np.round(new_w,            6)).size)
        n_a_old = int(np.unique(np.round(old_avg_miss_imp, 6)).size)
        n_a_new = int(np.unique(np.round(new_a,            6)).size)
        print(f"  weighted_skill_score unique values:    "
              f"was {n_w_old:>6,}  ->  now {n_w_new:>6,}")
        print(f"  avg_missing_skill_importance unique:   "
              f"was {n_a_old:>6,}  ->  now {n_a_new:>6,}")

        if new_match >= 70:
            print()
            print("[STOP] weighted_skill_score == skill_overlap is still >= 70%. "
                  "Hierarchy mapping may not be picking up the right skill labels. "
                  "Inspect esco_skill_depths.json -- if depth_by_label is small or "
                  "max_depth is 1, the depth computation likely failed.")

    # ----------------------------------------------------------------- #
    # Precision spot-check: 5 random pairs with their matched skills     #
    # ----------------------------------------------------------------- #
    print()
    print("=== precision spot-check (5 random pairs) ===")
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(out), size=min(5, len(out)), replace=False)
    for k, idx in enumerate(sample_idx, start=1):
        row = out.iloc[idx]
        j_id = int(row["job_id"])
        r_id = int(row["resume_id"])
        j_title = jobs.iloc[j_id]["position_title"] if j_id < len(jobs) else "?"
        r_title = resume_titles[r_id] if r_id < len(resume_titles) else "?"
        j_sk = job_skills[j_id] if j_id < len(job_skills) else set()
        r_sk = resume_skills[r_id] if r_id < len(resume_skills) else set()
        inter = sorted(j_sk & r_sk)[:8]
        print(f"  [{k}] job={j_id} ({j_title[:35]!r:40s}) "
              f"resume={r_id} (title={r_title[:35]!r:40s})")
        print(f"      embedding_sim={row['embedding_similarity']:.3f}  "
              f"skill_overlap={row['skill_overlap']:.3f}  "
              f"weighted_skill={row['weighted_skill_score']:.3f}  "
              f"title_sim={row['title_similarity']:.3f}")
        print(f"      n_job_skills={len(j_sk)}  n_resume_skills={len(r_sk)}  "
              f"matched={inter}")

    print()
    print("Summary statistics:")
    print(out.describe(include="all").T[["mean", "std", "min", "max"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--sample", type=int, default=None,
                    help="Smoke-test on first N resumes and N jobs.")
    args = ap.parse_args()
    main(top_k=args.top_k, sample=args.sample)
