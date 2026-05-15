"""
sample_gold_pairs.py
====================
Phase 2 of the new methodology. Samples 500 pairs from the diversified pool
for LLM labeling, stratified across pool source and similarity bands.

Allocation across pool sources:
    topK    : 200 pairs  (the easy-positive distribution; stratify by sim decile)
    A_mid   : 150 pairs  (boundary zone; stratify by sim band)
    B_xcat  : 100 pairs  (cross-category hard negatives)
    C_rand  :  50 pairs  (uniform random calibration)
                  ----
    total   : 500

Output: 5 CSVs, 100 pairs each, ready to upload to Claude / GPT / Gemini.
Each CSV has the truncated resume + JD text plus a stable row_id so we can
join the 3 LLMs' returned labels back together in Phase 3.

Truncation: resume to 2,000 raw chars, JD to 1,500 raw chars.
At 100 pairs / batch this is ~350k chars (~87k tokens) — fits GPT-4o's
128k window with room for prompt + output.

A master file `gold_pairs_master.csv` keeps the full metadata
(job_id, resume_id, pool_source, embedding_similarity) so Phase 3 can
join LLM votes back to the pair table.

Run:
    python "data/proccessed again/sample_gold_pairs.py"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROC = SCRIPT_DIR / "processed"

POOL_PATH    = PROC / "pair_features_diversified.parquet"
RESUMES_PATH = PROC / "cleaned_resumes.parquet"
JOBS_PATH    = PROC / "cleaned_jobs.parquet"

OUT_DIR     = SCRIPT_DIR / "gold_labeling"
MASTER_CSV  = OUT_DIR / "gold_pairs_master.csv"

# ---------------------------------------------------------------------------
# Sampling allocation
# ---------------------------------------------------------------------------
TARGETS: dict[str, int] = {
    "topK":   200,
    "A_mid":  150,
    "B_xcat": 100,
    "C_rand":  50,
}
TOTAL_TARGET = sum(TARGETS.values())   # 500
RANDOM_SEED  = 42

# Per-batch size for upload to LLMs
BATCH_SIZE = 100
N_BATCHES  = TOTAL_TARGET // BATCH_SIZE  # 5

# Truncation lengths (raw chars)
RESUME_MAX_CHARS = 2000
JD_MAX_CHARS     = 1500


# ---------------------------------------------------------------------------
# Stratified sampling helpers
# ---------------------------------------------------------------------------
def stratified_decile_sample(df: pd.DataFrame, n: int, rng: np.random.Generator,
                             sim_col: str = "embedding_similarity") -> pd.DataFrame:
    """Stratify by similarity decile within the input DataFrame; sample
    proportionally so the output spans the full similarity range."""
    if len(df) <= n:
        return df.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
    df = df.copy()
    # 10 quantile bins (or fewer if df is small)
    n_bins = min(10, len(df) // 5)
    df["_bin"] = pd.qcut(df[sim_col], q=n_bins, labels=False, duplicates="drop")
    per_bin = max(1, n // df["_bin"].nunique())
    samples = []
    for b, group in df.groupby("_bin"):
        take = min(per_bin, len(group))
        samples.append(group.sample(n=take, random_state=int(rng.integers(0, 1_000_000))))
    out = pd.concat(samples, ignore_index=True)
    # Trim or top-up to exactly n
    if len(out) > n:
        out = out.sample(n=n, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
    elif len(out) < n:
        # top-up from un-sampled rows
        used = set(zip(out["job_id"], out["resume_id"]))
        rest = df[~df.apply(lambda r: (r["job_id"], r["resume_id"]) in used, axis=1)]
        more = rest.sample(n=min(n - len(out), len(rest)),
                           random_state=int(rng.integers(0, 1_000_000)))
        out = pd.concat([out, more], ignore_index=True)
    return out.drop(columns=["_bin"], errors="ignore").reset_index(drop=True)


def truncate(text, max_chars: int) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " [...]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" Phase 2 — sampling 500 stratified pairs for LLM labeling")
    print("=" * 70)

    # --- Load pool + raw text ---
    print(f"\n[load] {POOL_PATH}")
    pool = pd.read_parquet(POOL_PATH)
    print(f"       {len(pool):,} pairs  | source mix: "
          f"{pool['pool_source'].value_counts().to_dict()}")

    resumes = pd.read_parquet(RESUMES_PATH).reset_index(drop=True)
    jobs    = pd.read_parquet(JOBS_PATH).reset_index(drop=True)

    # --- Stratified sample per source ---
    sampled_chunks: list[pd.DataFrame] = []
    for source, n in TARGETS.items():
        sub = pool[pool["pool_source"] == source]
        if len(sub) == 0:
            print(f"[warn] no pairs for source={source!r}")
            continue
        if source in ("topK", "A_mid"):
            picked = stratified_decile_sample(sub, n, rng)
        else:
            # B_xcat and C_rand: random within source
            picked = sub.sample(n=min(n, len(sub)),
                                random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
        picked["source_for_sample"] = source
        print(f"[sample] {source:>7s}: requested {n:>3d}  got {len(picked):>3d}  "
              f"sim mean={picked['embedding_similarity'].mean():.3f}")
        sampled_chunks.append(picked)

    sampled = pd.concat(sampled_chunks, ignore_index=True)
    print(f"\n[total] sampled: {len(sampled):,}  "
          f"(target {TOTAL_TARGET:,})")

    # Sanity: drop accidental duplicates (shouldn't happen but be safe)
    before = len(sampled)
    sampled = sampled.drop_duplicates(subset=["job_id", "resume_id"]).reset_index(drop=True)
    if len(sampled) != before:
        print(f"[dedupe] removed {before - len(sampled)} duplicate (job_id, resume_id) rows")

    # --- Stable shuffle so batches are mixed across sources ---
    sampled = sampled.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    sampled["row_id"] = range(1, len(sampled) + 1)

    # --- Attach raw text for the LLMs ---
    print("\n[text] joining truncated resume + JD text")
    res_lookup_str = {i: resumes.iloc[i]["Resume_str"] for i in range(len(resumes))}
    res_lookup_cat = {i: resumes.iloc[i]["Category"] for i in range(len(resumes))}
    job_lookup_jd  = {i: jobs.iloc[i]["job_description"] for i in range(len(jobs))}
    job_lookup_pos = {i: jobs.iloc[i]["position_title"] for i in range(len(jobs))}

    sampled["resume_category"] = sampled["resume_id"].map(res_lookup_cat)
    sampled["job_position_title"] = sampled["job_id"].map(job_lookup_pos)
    sampled["resume_text"] = sampled["resume_id"].map(
        lambda i: truncate(res_lookup_str.get(i, ""), RESUME_MAX_CHARS)
    )
    sampled["job_description"] = sampled["job_id"].map(
        lambda i: truncate(job_lookup_jd.get(i, ""), JD_MAX_CHARS)
    )

    # --- Master file (full metadata, never sent to LLMs) ---
    master_cols = [
        "row_id", "job_id", "resume_id", "source_for_sample",
        "embedding_similarity", "skill_overlap", "weighted_skill_score",
        "resume_category", "job_position_title",
        "resume_text", "job_description",
    ]
    sampled[master_cols].to_csv(MASTER_CSV, index=False)
    print(f"[write] {MASTER_CSV}")

    # --- Per-batch upload CSVs (only the columns the LLM needs) ---
    upload_cols = ["row_id", "job_position_title", "resume_text", "job_description"]
    print(f"\n[batches] writing {N_BATCHES} batches of {BATCH_SIZE} pairs each "
          f"to {OUT_DIR}")
    for b in range(N_BATCHES):
        chunk = sampled.iloc[b * BATCH_SIZE: (b + 1) * BATCH_SIZE][upload_cols]
        path = OUT_DIR / f"gold_pairs_batch_{b + 1}.csv"
        chunk.to_csv(path, index=False)
        n_chars = chunk["resume_text"].str.len().sum() + chunk["job_description"].str.len().sum()
        print(f"  batch {b + 1}: {len(chunk):>3d} pairs  "
              f"~{n_chars:,} chars  ({path.name})")

    # --- Distribution sanity check ---
    print("\n=== Sample distribution ===")
    by_src = sampled["source_for_sample"].value_counts()
    for s, n in by_src.items():
        sub = sampled[sampled["source_for_sample"] == s]
        print(f"  {s:>7s}: n={n:>3d}  "
              f"sim med={sub['embedding_similarity'].median():.3f}  "
              f"q25={sub['embedding_similarity'].quantile(0.25):.3f}  "
              f"q75={sub['embedding_similarity'].quantile(0.75):.3f}")

    print("\n=== Resume Category coverage ===")
    cat_counts = sampled["resume_category"].value_counts().head(15)
    for c, n in cat_counts.items():
        print(f"  {c:<25s}: {n}")
    n_categories = sampled["resume_category"].nunique()
    print(f"  ({n_categories} distinct resume categories represented)")

    # --- Prompt template the user will paste alongside each upload ---
    prompt_path = OUT_DIR / "PROMPT_FOR_LLMS.txt"
    prompt = """\
You are evaluating whether a candidate's resume is a good match for a job description.

For each row in the CSV I have uploaded:
- Read the resume_text and job_description.
- Output label = 1 if the candidate is a GOOD match for the job (has the
  core required skills and clearly relevant experience).
- Output label = 0 if the candidate is NOT a good match.
- Be strict. A good match means the candidate could plausibly do the job
  with at most minor onboarding. Adjacent skills or generic competencies
  ("communication", "teamwork") alone do not qualify as a match.
- Do not consider formatting, writing quality, or resume length.
- Only judge based on skills, experience, and role fit.

Return ONLY a CSV with two columns and a header row:

    row_id,label

One row per input pair. Do not include any other text, explanation, or commentary.
If you cannot judge a pair (text is empty or unreadable), output label = 0 for that row.
"""
    prompt_path.write_text(prompt)
    print(f"\n[write] {prompt_path}")
    print()
    print("=" * 70)
    print(" Phase 2 complete. Next steps:")
    print("=" * 70)
    print(f"  1. Open {prompt_path}")
    print(f"  2. For each LLM (Claude, GPT, Gemini), upload the 5 batch CSVs from")
    print(f"     {OUT_DIR}/ together with the prompt.")
    print(f"  3. Save each LLM's output as:")
    print(f"     {OUT_DIR}/labels_claude.csv")
    print(f"     {OUT_DIR}/labels_gpt.csv")
    print(f"     {OUT_DIR}/labels_gemini.csv")
    print(f"     Each must have columns: row_id,label")
    print(f"  4. Then run combine_llm_labels.py for Phase 3.")


if __name__ == "__main__":
    main()
