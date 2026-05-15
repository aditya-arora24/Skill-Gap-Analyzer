"""
active_learning_sample.py
==========================
Step 1 of the active-learning experiment. Selects 200 pairs from the
diversified pool that the supervised LogReg is MOST UNCERTAIN about
(predicted probability in [0.40, 0.60]) — the cases self-training would
NEVER pick because they're not confident-positive or confident-negative,
but exactly the cases that would teach the model something new.

Steps:
  1. Score every row of pair_features_diversified.parquet with the
     supervised LogReg.
  2. Filter to the uncertainty band, prob ∈ [0.40, 0.60].
  3. Drop ANY pair already in gold_labels.csv (to avoid relabeling).
  4. Stratified sample 200 pairs across pool_source
     (topK / A_mid / B_xcat / C_rand), targeting equal allocation but
     respecting per-source availability.
  5. Build the upload CSVs in the same format as the original
     gold_pairs_batch_*.csv files, and a master CSV with metadata.

Inputs (read-only):
  models/llm_supervised/{scaler,logreg}.pkl
  data/proccessed again/processed/pair_features_diversified.parquet
  data/proccessed again/processed/cleaned_resumes.parquet
  data/proccessed again/processed/cleaned_jobs.parquet
  data/proccessed again/gold_labeling/gold_labels.csv
  data/proccessed again/gold_labeling/PROMPT_FOR_LLMS.txt

Outputs (NEW directory only):
  data/proccessed again/active_learning/
    active_learning_master.csv
    active_learning_batch_1.csv          (100 pairs)
    active_learning_batch_2.csv          (100 pairs)
    PROMPT_FOR_LLMS.txt                  (copy, for convenience)
    sample_metadata.json

NO models are retrained. NO existing artifacts are modified.

Run:
    python "src/active_learning_sample.py"
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUPERVISED_MODEL_DIR = PROJECT_ROOT / "models" / "llm_supervised"
POOL_PARQUET   = PROJECT_ROOT / "data" / "proccessed again" / "processed" / "pair_features_diversified.parquet"
RESUMES_PARQ   = PROJECT_ROOT / "data" / "proccessed again" / "processed" / "cleaned_resumes.parquet"
JOBS_PARQ      = PROJECT_ROOT / "data" / "proccessed again" / "processed" / "cleaned_jobs.parquet"
GOLD_CSV       = PROJECT_ROOT / "data" / "proccessed again" / "gold_labeling" / "gold_labels.csv"
PROMPT_TXT     = PROJECT_ROOT / "data" / "proccessed again" / "gold_labeling" / "PROMPT_FOR_LLMS.txt"

OUT_DIR = PROJECT_ROOT / "data" / "proccessed again" / "active_learning"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# Uncertainty band
PROB_LO = 0.40
PROB_HI = 0.60

# Sampling
N_TARGET = 200
BATCH_SIZE = 100
N_BATCHES = N_TARGET // BATCH_SIZE        # 2

# Truncation (match Phase 2 sampling for consistency)
RESUME_MAX_CHARS = 2000
JD_MAX_CHARS     = 1500

ALL_FEATURES = [
    "embedding_similarity",
    "tfidf_similarity",
    "skill_overlap",
    "weighted_skill_score",
    "num_missing_skills",
    "avg_missing_skill_importance",
    "years_of_experience",
    "title_similarity",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def truncate(text, max_chars: int) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " [...]"


def stratified_sample_with_caps(df: pd.DataFrame, source_col: str,
                                 sources: list[str], n_target: int,
                                 rng: np.random.Generator) -> pd.DataFrame:
    """
    Aim for n_target / len(sources) per source. If a source has fewer than
    its share, take all of it; redistribute the deficit proportionally to
    remaining sources.
    """
    per_source_target = n_target // len(sources)
    remainder = n_target - per_source_target * len(sources)

    picks: list[pd.DataFrame] = []
    short_sources = []
    long_sources = []
    for s in sources:
        sub = df[df[source_col] == s]
        if len(sub) <= per_source_target:
            picks.append(sub.copy())
            short_sources.append((s, per_source_target - len(sub)))
        else:
            long_sources.append(s)
    # First pass deficit
    deficit = sum(d for _, d in short_sources)

    # Allocate remaining target across long sources
    if long_sources:
        # Take per_source_target from each long source
        for s in long_sources:
            sub = df[df[source_col] == s]
            picked = sub.sample(n=per_source_target,
                                random_state=int(rng.integers(0, 1_000_000)))
            picks.append(picked)
        # Now redistribute deficit + remainder
        extra_needed = deficit + remainder
        if extra_needed > 0:
            already = pd.concat(picks, ignore_index=True)
            already_keys = set(zip(already["job_id"], already["resume_id"]))
            available = df[df[source_col].isin(long_sources)].copy()
            available_mask = ~available.apply(
                lambda r: (r["job_id"], r["resume_id"]) in already_keys, axis=1
            )
            available = available[available_mask]
            if len(available) > 0:
                extra = available.sample(n=min(extra_needed, len(available)),
                                          random_state=int(rng.integers(0, 1_000_000)))
                picks.append(extra)

    out = pd.concat(picks, ignore_index=True)
    if len(out) > n_target:
        out = out.sample(n=n_target,
                         random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    print("=" * 70)
    print(" Active learning — Step 1: select uncertain pairs for labeling")
    print("=" * 70)

    # 1. Load supervised model + pool + gold + raw text
    print("\n[1] loading inputs (read-only)")
    scaler = joblib.load(SUPERVISED_MODEL_DIR / "scaler.pkl")
    model  = joblib.load(SUPERVISED_MODEL_DIR / "logreg.pkl")

    pool    = pd.read_parquet(POOL_PARQUET)
    resumes = pd.read_parquet(RESUMES_PARQ).reset_index(drop=True)
    jobs    = pd.read_parquet(JOBS_PARQ).reset_index(drop=True)
    gold    = pd.read_csv(GOLD_CSV)

    print(f"    pool         : {len(pool):,} rows  (sources: "
          f"{pool['pool_source'].value_counts().to_dict()})")
    print(f"    gold-labeled : {len(gold):,} pairs (will be excluded)")

    # 2. Score the pool
    print("\n[2] scoring pool with supervised LogReg")
    X = pool[ALL_FEATURES].values
    Xs = scaler.transform(X)
    probs = model.predict_proba(Xs)[:, 1]
    pool = pool.copy()
    pool["pred_prob"] = probs

    # 3. Drop gold pairs to avoid relabeling
    gold_keys = set(zip(gold["job_id"].astype(int), gold["resume_id"].astype(int)))
    pool_keys = list(zip(pool["job_id"].astype(int), pool["resume_id"].astype(int)))
    pool["_in_gold"] = [k in gold_keys for k in pool_keys]
    pool_no_gold = pool[~pool["_in_gold"]].drop(columns=["_in_gold"]).reset_index(drop=True)
    print(f"    after dropping gold: {len(pool_no_gold):,} pairs")

    # 4. Filter to uncertainty band
    band = pool_no_gold[
        (pool_no_gold["pred_prob"] >= PROB_LO) &
        (pool_no_gold["pred_prob"] <= PROB_HI)
    ].reset_index(drop=True)

    print(f"\n[3] uncertainty band: prob ∈ [{PROB_LO}, {PROB_HI}]")
    print(f"    pairs in band: {len(band):,}  "
          f"({100 * len(band) / len(pool_no_gold):5.1f}% of pool)")
    print(f"    distribution by pool_source:")
    src_counts = band["pool_source"].value_counts()
    for src, n in src_counts.items():
        # Compare to source's full size
        src_total = (pool_no_gold["pool_source"] == src).sum()
        print(f"      {src:>8s}: {n:>5,}  "
              f"({100*n/src_total:5.1f}% of {src_total:,})")

    # Probability sub-distribution within the band
    print(f"    probability quartiles within band:")
    q = band["pred_prob"].quantile([0.25, 0.5, 0.75])
    print(f"      q25={q.iloc[0]:.4f}  med={q.iloc[1]:.4f}  q75={q.iloc[2]:.4f}")

    if len(band) < N_TARGET:
        print(f"[warn] only {len(band)} uncertain pairs available (target {N_TARGET}). "
              f"Will take all and stop.")

    # 5. Stratified sampling across sources
    print(f"\n[4] sampling {N_TARGET} pairs stratified across pool_source")
    sources = sorted(band["pool_source"].unique().tolist())
    sampled = stratified_sample_with_caps(
        band, "pool_source", sources, n_target=min(N_TARGET, len(band)), rng=rng,
    )
    print(f"    sampled: {len(sampled):,}")
    print(f"    actual source distribution:")
    for src, n in sampled["pool_source"].value_counts().items():
        print(f"      {src:>8s}: {n:>3,}")

    # Stable random shuffle so batches are mixed
    sampled = sampled.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    # Use row_id space disjoint from the original gold (1..500). Start at 1001.
    sampled["row_id"] = range(1001, 1001 + len(sampled))

    # 6. Attach resume + JD text (truncated)
    print("\n[5] joining truncated resume + JD text")
    resumes_lookup_str = {i: resumes.iloc[i]["Resume_str"]      for i in range(len(resumes))}
    resumes_lookup_cat = {i: resumes.iloc[i]["Category"]        for i in range(len(resumes))}
    jobs_lookup_jd     = {i: jobs.iloc[i]["job_description"]    for i in range(len(jobs))}
    jobs_lookup_pos    = {i: jobs.iloc[i]["position_title"]     for i in range(len(jobs))}

    sampled["resume_category"]    = sampled["resume_id"].map(resumes_lookup_cat)
    sampled["job_position_title"] = sampled["job_id"].map(jobs_lookup_pos)
    sampled["resume_text"]        = sampled["resume_id"].map(
        lambda i: truncate(resumes_lookup_str.get(i, ""), RESUME_MAX_CHARS)
    )
    sampled["job_description"]    = sampled["job_id"].map(
        lambda i: truncate(jobs_lookup_jd.get(i, ""), JD_MAX_CHARS)
    )

    # 7. Write master file (you keep this; it's not for LLMs)
    master_cols = [
        "row_id", "job_id", "resume_id", "pool_source", "pred_prob",
        "embedding_similarity", "skill_overlap", "weighted_skill_score",
        "title_similarity",
        "resume_category", "job_position_title",
        "resume_text", "job_description",
    ]
    master_path = OUT_DIR / "active_learning_master.csv"
    sampled[master_cols].to_csv(master_path, index=False)
    print(f"\n[write] {master_path}")

    # 8. Write batches in the original gold_pairs format
    upload_cols = ["row_id", "job_position_title", "resume_text", "job_description"]
    n = len(sampled)
    n_batches = max(1, (n + BATCH_SIZE - 1) // BATCH_SIZE)
    print(f"\n[batches] writing {n_batches} batches of <= {BATCH_SIZE} pairs each")
    for b in range(n_batches):
        chunk = sampled.iloc[b * BATCH_SIZE: (b + 1) * BATCH_SIZE][upload_cols]
        if len(chunk) == 0:
            continue
        path = OUT_DIR / f"active_learning_batch_{b + 1}.csv"
        chunk.to_csv(path, index=False)
        n_chars = chunk["resume_text"].str.len().sum() + chunk["job_description"].str.len().sum()
        print(f"  batch {b + 1}: {len(chunk):>3d} pairs  "
              f"~{n_chars:,} chars  ({path.name})")

    # 9. Copy the prompt file for convenience
    if PROMPT_TXT.exists():
        shutil.copy(PROMPT_TXT, OUT_DIR / "PROMPT_FOR_LLMS.txt")
        print(f"[copy ] {OUT_DIR / 'PROMPT_FOR_LLMS.txt'}")
    else:
        print(f"[warn] {PROMPT_TXT} not found; create it manually before LLM upload")

    # 10. Run metadata
    meta = {
        "random_seed":        RANDOM_SEED,
        "uncertainty_band":   [PROB_LO, PROB_HI],
        "n_pool":             int(len(pool)),
        "n_pool_after_drop_gold": int(len(pool_no_gold)),
        "n_in_band":          int(len(band)),
        "n_sampled":          int(len(sampled)),
        "row_id_range":       [1001, 1001 + len(sampled) - 1],
        "n_batches":          int(n_batches),
        "batch_size":         BATCH_SIZE,
        "source_distribution": {
            str(k): int(v) for k, v in sampled["pool_source"].value_counts().items()
        },
        "category_distribution_top10": {
            str(k): int(v) for k, v in sampled["resume_category"].value_counts().head(10).items()
        },
        "n_categories":       int(sampled["resume_category"].nunique()),
        "prob_quartiles":     {
            "q25": float(sampled["pred_prob"].quantile(0.25)),
            "q50": float(sampled["pred_prob"].quantile(0.50)),
            "q75": float(sampled["pred_prob"].quantile(0.75)),
        },
    }
    (OUT_DIR / "sample_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[write] {OUT_DIR / 'sample_metadata.json'}")

    # 11. Final distribution report
    print("\n" + "=" * 70)
    print(" Sample distribution (review before sending to LLMs)")
    print("=" * 70)
    print(f"\n  Source mix:")
    for s, n in sampled["pool_source"].value_counts().items():
        print(f"    {s:>8s}: {n:>3,}")

    print(f"\n  Probability spread (pred_prob, lower = harder):")
    print(f"    min: {sampled['pred_prob'].min():.4f}")
    print(f"    q25: {sampled['pred_prob'].quantile(0.25):.4f}")
    print(f"    med: {sampled['pred_prob'].quantile(0.50):.4f}")
    print(f"    q75: {sampled['pred_prob'].quantile(0.75):.4f}")
    print(f"    max: {sampled['pred_prob'].max():.4f}")

    print(f"\n  Top resume Categories represented "
          f"({sampled['resume_category'].nunique()} total):")
    for cat, n in sampled["resume_category"].value_counts().head(10).items():
        print(f"    {cat:<25s}: {n}")

    print(f"\n  Embedding similarity in sample:")
    print(f"    min: {sampled['embedding_similarity'].min():.3f}")
    print(f"    med: {sampled['embedding_similarity'].median():.3f}")
    print(f"    max: {sampled['embedding_similarity'].max():.3f}")

    print(f"\n  Skill overlap in sample:")
    nonzero = (sampled['skill_overlap'] > 0).mean() * 100
    print(f"    mean: {sampled['skill_overlap'].mean():.3f}  "
          f"nonzero: {nonzero:.1f}%")

    print()
    print("=" * 70)
    print(" Done. Next steps:")
    print("=" * 70)
    print(f"  1. Inspect the distribution above. If it looks balanced, proceed.")
    print(f"  2. For each LLM (Claude / GPT / Gemini), upload the 2 batch CSVs")
    print(f"     in {OUT_DIR}/ with the same prompt as the gold labeling round.")
    print(f"  3. Save responses as:")
    print(f"     {OUT_DIR}/labels_claude.csv     (or batch_*_claude_labels.csv)")
    print(f"     {OUT_DIR}/labels_gpt.csv        (or batch_*_gpt_labels.csv)")
    print(f"     {OUT_DIR}/labels_gemini.csv     (or batch_*_gemini_labels.csv)")
    print(f"  4. Then we'll run Step 2 (combine LLM votes + retrain on 600).")


if __name__ == "__main__":
    main()
