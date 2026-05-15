"""
build_diversified_pool.py
=========================
Phase 1 of the new methodology. Add three classes of pairs to the existing
top-50 retrieval pool so the LLM gold standard (and any model trained on it)
can learn to discriminate hard cases:

  Type A — mid-similarity (SBERT cosine in [0.3, 0.5], NOT in current top-50).
           Borderline candidates the retriever passed over. Target: 5,000.
  Type B — cross-category strict. Resume Category vs job position_title family
           are deliberately mismatched (e.g., HR resume against Software Eng
           job). Target: 2,000.
  Type C — random. Uniform sample over (job_id, resume_id) excluding pairs
           already in the pool. Genuine negatives for calibration. Target: 1,000.

Computes the same 13 features used by matching_pipeline.py for the new pairs,
reusing cached SBERT embeddings (no re-encoding). Output:
    data/proccessed again/processed/pair_features_diversified.parquet

Run:
    python "data/proccessed again/build_diversified_pool.py"
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Reuse helpers from matching_pipeline so feature definitions stay in sync.
from matching_pipeline import (    # noqa: E402
    extract_years,
    extract_degree_level,
    parse_skill_field,
    build_importance_map,
    derive_resume_title,
    skill_features as compute_skill_features,
    tfidf_pair_similarities,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROC = SCRIPT_DIR / "processed"
EMB_DIR = PROC / "embeddings"

EXISTING_PAIRS  = PROC / "pair_features.parquet"
RESUMES_PARQ    = PROC / "cleaned_resumes.parquet"
JOBS_PARQ       = PROC / "cleaned_jobs.parquet"
SKILL_DICT_JSON = PROC / "skill_dictionary_merged.json"
SKILL_CTX_JSON  = PROC / "skill_to_context_map.json"

OUT_PAIRS = PROC / "pair_features_diversified.parquet"

# ---------------------------------------------------------------------------
# Sampling targets and config
# ---------------------------------------------------------------------------
N_TYPE_A = 5000
N_TYPE_B = 2000
N_TYPE_C = 1000
RANDOM_SEED = 42

# Type A bounds
SIM_LO = 0.30
SIM_HI = 0.50

# ---------------------------------------------------------------------------
# Category families (strict cross-category definition for Type B)
# Source: Resume.csv.Category column standard values; we bucket them into
# coarse families so a "cross" pair is one whose families differ.
# ---------------------------------------------------------------------------
CATEGORY_FAMILIES: dict[str, str] = {
    # TECH family
    "INFORMATION-TECHNOLOGY": "TECH",
    "ENGINEERING":            "TECH",
    "DESIGNER":               "TECH",
    "DIGITAL-MEDIA":          "TECH",
    # BUSINESS family (HR, finance, sales, advisory)
    "HR":                     "BUSINESS",
    "FINANCE":                "BUSINESS",
    "ACCOUNTANT":             "BUSINESS",
    "SALES":                  "BUSINESS",
    "BANKING":                "BUSINESS",
    "BUSINESS-DEVELOPMENT":   "BUSINESS",
    "ADVOCATE":               "BUSINESS",
    "CONSULTANT":             "BUSINESS",
    "PUBLIC-RELATIONS":       "BUSINESS",
    # HEALTHCARE family
    "HEALTHCARE":             "HEALTHCARE",
    "FITNESS":                "HEALTHCARE",
    # TRADES family
    "AUTOMOBILE":             "TRADES",
    "AVIATION":               "TRADES",
    "CONSTRUCTION":           "TRADES",
    "AGRICULTURE":            "TRADES",
    "CHEF":                   "TRADES",
    # ARTS / EDUCATION
    "TEACHER":                "EDUCATION",
    "ARTS":                   "ARTS",
    "APPAREL":                "ARTS",
}

# Job position_title -> family via simple keyword heuristics. Anything that
# matches multiple families resolves to the first match in this priority
# order (TECH highest because it's the most-likely-to-be-confused-with-any).
JOB_FAMILY_RULES: list[tuple[str, re.Pattern]] = [
    ("TECH",       re.compile(r"\b(software|developer|engineer|programmer|data scientist|"
                              r"data analyst|devops|cloud|architect|qa\b|qa engineer|sre|"
                              r"machine learning|ml engineer|full stack|backend|frontend|"
                              r"web developer|mobile developer|systems administrator|"
                              r"network|security engineer|database administrator)\b",
                              re.IGNORECASE)),
    ("BUSINESS",   re.compile(r"\b(hr\b|human resources|recruiter|talent acquisition|"
                              r"finance|financial|accountant|accounting|controller|"
                              r"sales|account executive|business development|"
                              r"marketing|brand manager|product manager|"
                              r"banker|banking|loan|financial advisor|legal counsel|"
                              r"paralegal|attorney|consultant)\b", re.IGNORECASE)),
    ("HEALTHCARE", re.compile(r"\b(nurse|physician|doctor|medical|healthcare|clinical|"
                              r"therapist|pharmacist|dental|surgeon|technician)\b",
                              re.IGNORECASE)),
    ("EDUCATION",  re.compile(r"\b(teacher|professor|instructor|tutor|education|"
                              r"academic)\b", re.IGNORECASE)),
    ("TRADES",     re.compile(r"\b(mechanic|technician|electrician|plumber|construction|"
                              r"chef|cook|driver|operator|maintenance)\b", re.IGNORECASE)),
    ("ARTS",       re.compile(r"\b(designer|artist|writer|editor|content|graphic|"
                              r"creative|copywriter|illustrator)\b", re.IGNORECASE)),
]


def job_family(position_title: str) -> str:
    if not isinstance(position_title, str) or not position_title:
        return "UNKNOWN"
    for fam, patt in JOB_FAMILY_RULES:
        if patt.search(position_title):
            return fam
    return "UNKNOWN"


def resume_family(category: str) -> str:
    if not isinstance(category, str):
        return "UNKNOWN"
    return CATEGORY_FAMILIES.get(category.strip().upper(), "UNKNOWN")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 70)
    print(" Phase 1 — building diversified candidate pool")
    print("=" * 70)

    # ---- Load existing pool ----
    print(f"\n[load] {EXISTING_PAIRS}")
    existing = pd.read_parquet(EXISTING_PAIRS)
    print(f"       {len(existing):,} existing pairs (top-50 retrieval)")
    existing_set: set[tuple[int, int]] = set(zip(
        existing["job_id"].astype(int).tolist(),
        existing["resume_id"].astype(int).tolist(),
    ))

    # ---- Load resumes / jobs ----
    resumes = pd.read_parquet(RESUMES_PARQ).reset_index(drop=True)
    jobs    = pd.read_parquet(JOBS_PARQ).reset_index(drop=True)
    print(f"[load] resumes={len(resumes)}  jobs={len(jobs)}")

    # ---- Load cached embeddings (no SBERT call) ----
    resume_emb     = np.load(EMB_DIR / "resume_emb.npy")
    job_emb        = np.load(EMB_DIR / "job_emb.npy")
    resume_exp_emb = np.load(EMB_DIR / "resume_exp_emb.npy")
    title_resume_emb = np.load(EMB_DIR / "resume_title_emb.npy")
    title_job_emb    = np.load(EMB_DIR / "job_title_emb.npy")
    print(f"[emb]  resume={resume_emb.shape}  job={job_emb.shape}  "
          f"exp={resume_exp_emb.shape}  title_r={title_resume_emb.shape}  "
          f"title_j={title_job_emb.shape}")

    # Sanity: existing pair_features has 42,650 rows, top-50 over 853 jobs.
    assert resume_emb.shape[0] == len(resumes), "resume_emb / resumes mismatch"
    assert job_emb.shape[0] == len(jobs), "job_emb / jobs mismatch"

    # ---- Skill artefacts and importance map ----
    skill_dict = json.load(open(SKILL_DICT_JSON, encoding="utf-8"))
    skill_ctx  = json.load(open(SKILL_CTX_JSON,  encoding="utf-8"))
    importance_map, default_importance = build_importance_map(skill_dict, skill_ctx)
    print(f"[imp]  importance_map size={len(importance_map):,}  "
          f"default_importance={default_importance:.3f}")

    # ---- Per-resume / per-job static features ----
    print("[feats] computing per-row static features (yoe / edu / skill sets)")
    resume_yoe = np.array(
        [extract_years(t) for t in resumes["Resume_str"].fillna("").tolist()],
        dtype=np.float32,
    )
    resume_edu = np.array(
        [extract_degree_level(t) for t in resumes["section_education"].fillna("").tolist()],
        dtype=np.int8,
    )
    resume_skill_sets = [set(parse_skill_field(v)) for v in resumes["extracted_skills"].tolist()]

    raw_jd = jobs["job_description"].fillna("").tolist()
    job_yoe = np.array([extract_years(t) for t in raw_jd], dtype=np.float32)
    job_edu = np.array([extract_degree_level(t) for t in raw_jd], dtype=np.int8)
    job_skill_sets = [set(parse_skill_field(v)) for v in jobs["extracted_skills"].tolist()]

    # Resume + job category families for cross-category sampling
    resume_fam = np.array([resume_family(c) for c in resumes["Category"].astype(str).tolist()])
    job_fam    = np.array([job_family(t) for t in jobs["position_title"].astype(str).tolist()])

    print(f"[fam]  resume family counts: "
          f"{dict(zip(*np.unique(resume_fam, return_counts=True)))}")
    print(f"[fam]  job    family counts: "
          f"{dict(zip(*np.unique(job_fam, return_counts=True)))}")

    # ---- Full SBERT similarity matrix (job × resume) ----
    # Both embeddings are row-normalised; dot product == cosine.
    print("[sim]  computing full SBERT cosine matrix (job × resume)")
    sim_mat = job_emb @ resume_emb.T   # (n_jobs, n_resumes), float32
    print(f"       sim_mat shape={sim_mat.shape}  "
          f"min={sim_mat.min():.3f}  max={sim_mat.max():.3f}  "
          f"mean={sim_mat.mean():.3f}")

    # ---- Type A: mid-similarity, NOT in top-50 ----
    print(f"\n[Type A] mid-similarity pairs in [{SIM_LO}, {SIM_HI}], "
          f"excluding top-50 (target {N_TYPE_A:,})")
    # Mask within target range
    mask_mid = (sim_mat >= SIM_LO) & (sim_mat <= SIM_HI)
    j_idx_a, r_idx_a = np.where(mask_mid)
    print(f"         candidates in band: {len(j_idx_a):,}")
    # Filter out existing pairs
    candidates_a = [
        (int(j), int(r))
        for j, r in zip(j_idx_a, r_idx_a)
        if (int(j), int(r)) not in existing_set
    ]
    print(f"         after excluding top-50: {len(candidates_a):,}")
    if len(candidates_a) < N_TYPE_A:
        print(f"[warn]   wanted {N_TYPE_A:,} but only {len(candidates_a):,} available; using all")
        sample_a = candidates_a
    else:
        idx = rng.choice(len(candidates_a), size=N_TYPE_A, replace=False)
        sample_a = [candidates_a[i] for i in idx]
    print(f"         sampled: {len(sample_a):,}")

    # ---- Type B: cross-category strict, embedding_similarity > 0.50 ----
    # We only want HARD cross-category pairs (not random low-sim mismatches),
    # so require similarity > 0.50 and family mismatch.
    print(f"\n[Type B] cross-category strict, sim > 0.50 (target {N_TYPE_B:,})")
    # Iterate per job, for each job find resumes whose family != job's family
    # AND sim > 0.50, then sample.
    candidates_b: list[tuple[int, int]] = []
    seen_in_a = set(sample_a) | existing_set
    for j in range(len(jobs)):
        jf = job_fam[j]
        if jf == "UNKNOWN":
            continue
        # similarity row, mask family mismatch
        sims = sim_mat[j]
        ok = (sims > 0.50) & (resume_fam != jf) & (resume_fam != "UNKNOWN")
        for r in np.where(ok)[0]:
            pair = (j, int(r))
            if pair not in seen_in_a:
                candidates_b.append(pair)
    print(f"         cross-family candidates with sim > 0.50: {len(candidates_b):,}")
    if len(candidates_b) < N_TYPE_B:
        print(f"[warn]   wanted {N_TYPE_B:,} but only {len(candidates_b):,} available; using all")
        sample_b = candidates_b
    else:
        idx = rng.choice(len(candidates_b), size=N_TYPE_B, replace=False)
        sample_b = [candidates_b[i] for i in idx]
    print(f"         sampled: {len(sample_b):,}")

    # ---- Type C: random uniform over (job, resume) ----
    print(f"\n[Type C] random pairs, any similarity (target {N_TYPE_C:,})")
    seen_so_far = set(sample_a) | set(sample_b) | existing_set
    sample_c: list[tuple[int, int]] = []
    n_jobs = len(jobs)
    n_res  = len(resumes)
    attempts = 0
    while len(sample_c) < N_TYPE_C and attempts < N_TYPE_C * 10:
        j = int(rng.integers(0, n_jobs))
        r = int(rng.integers(0, n_res))
        pair = (j, r)
        if pair in seen_so_far:
            attempts += 1
            continue
        sample_c.append(pair)
        seen_so_far.add(pair)
        attempts += 1
    print(f"         sampled: {len(sample_c):,}")

    # ---- Build the new pair set + provenance tag ----
    new_pairs_job: list[int] = []
    new_pairs_res: list[int] = []
    new_pair_source: list[str] = []
    for (j, r) in sample_a:
        new_pairs_job.append(j); new_pairs_res.append(r); new_pair_source.append("A_mid")
    for (j, r) in sample_b:
        new_pairs_job.append(j); new_pairs_res.append(r); new_pair_source.append("B_xcat")
    for (j, r) in sample_c:
        new_pairs_job.append(j); new_pairs_res.append(r); new_pair_source.append("C_rand")

    n_new = len(new_pairs_job)
    print(f"\n[total] new pairs: {n_new:,}")

    pj = np.array(new_pairs_job, dtype=np.int32)
    pr = np.array(new_pairs_res, dtype=np.int32)

    # ---- Compute features for the new pairs ----
    print(f"\n[feats] computing 13 features for {n_new:,} new pairs")

    # SBERT cosine (already in matrix)
    new_emb_sim = sim_mat[pj, pr].astype(np.float32)

    # TF-IDF: re-fit on full corpus, transform, batch-multiply.
    resume_text = resumes["cleaned_resume"].fillna("").tolist()
    job_text    = jobs["cleaned_job_description"].fillna("").tolist()
    new_tfidf_sim = tfidf_pair_similarities(resume_text, job_text, pj, pr)

    # Skill features (overlap, weighted, missing, avg miss imp)
    skill_feats = compute_skill_features(
        job_skill_sets, resume_skill_sets, pj, pr,
        importance_map, default_importance,
    )

    # Experience features
    pair_resume_yoe = resume_yoe[pr]
    pair_job_yoe    = job_yoe[pj]
    exp_gap = (pair_job_yoe - pair_resume_yoe).astype(np.float32)
    exp_relevance = (resume_exp_emb[pr] * job_emb[pj]).sum(axis=1).astype(np.float32)

    # Title similarity from cached title embeddings
    title_sim = (title_resume_emb[pr] * title_job_emb[pj]).sum(axis=1).astype(np.float32)

    # Education match
    pair_resume_edu = resume_edu[pr]
    pair_job_edu    = job_edu[pj]
    edu_match = ((pair_job_edu == 0) | (pair_resume_edu >= pair_job_edu)).astype(np.int8)

    # ---- Assemble new-pair DataFrame matching pair_features.parquet schema ----
    new_df = pd.DataFrame({
        "job_id":                       pj,
        "resume_id":                    pr,
        "embedding_similarity":         new_emb_sim,
        "tfidf_similarity":             new_tfidf_sim,
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
    # provenance column for downstream stratification (kept separate so the
    # parquet schema still aligns with the existing pair_features table)
    new_df["pool_source"] = new_pair_source

    # Tag the existing pool with its own source label
    existing_tagged = existing.copy()
    existing_tagged["pool_source"] = "topK"

    diversified = pd.concat([existing_tagged, new_df], ignore_index=True)
    diversified.to_parquet(OUT_PAIRS, index=False)
    print(f"\n[write] {OUT_PAIRS}")
    print(f"        total rows: {len(diversified):,} "
          f"(existing {len(existing):,} + new {n_new:,})")

    # ---- Distribution sanity check ----
    print("\n=== Pool composition ===")
    by_src = diversified["pool_source"].value_counts()
    for src, n in by_src.items():
        print(f"  {src:>8s}: {n:>7,}  ({100 * n / len(diversified):5.1f}%)")

    print("\n=== embedding_similarity distribution by source ===")
    for src in ["topK", "A_mid", "B_xcat", "C_rand"]:
        sub = diversified[diversified["pool_source"] == src]["embedding_similarity"]
        if len(sub) == 0:
            continue
        print(f"  {src:>8s}: n={len(sub):>6,}  "
              f"min={sub.min():.3f}  q25={sub.quantile(0.25):.3f}  "
              f"med={sub.median():.3f}  q75={sub.quantile(0.75):.3f}  "
              f"max={sub.max():.3f}  mean={sub.mean():.3f}")

    print("\n=== skill_overlap distribution by source ===")
    for src in ["topK", "A_mid", "B_xcat", "C_rand"]:
        sub = diversified[diversified["pool_source"] == src]["skill_overlap"]
        if len(sub) == 0:
            continue
        nonzero = (sub > 0).mean() * 100
        print(f"  {src:>8s}: n={len(sub):>6,}  "
              f"mean={sub.mean():.3f}  nonzero={nonzero:5.1f}%")

    print("\n[done] Phase 1 complete. Inspect distributions above; if they look "
          "reasonable, proceed to Phase 2 (sample_gold_pairs.py).")


if __name__ == "__main__":
    main()
