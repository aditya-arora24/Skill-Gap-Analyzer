"""
make_ml_ready.py
================
Single-purpose builder. Reads pair_features.parquet, drops 3 noisy columns,
generates weak labels via a 70/30 quantile cut on a composite score, and
writes the ML-ready dataset.

Source : data/proccessed again/processed/pair_features.parquet  (42,650 rows)
Output : data/proccessed again/processed/ml_ready_dataset.parquet

Run from anywhere:
    python "data/proccessed again/make_ml_ready.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths (resolved relative to this script's location)
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PROC_DIR     = SCRIPT_DIR / "processed"

SRC_PATH  = PROC_DIR / "pair_features.parquet"
OUT_PATH  = PROC_DIR / "ml_ready_dataset.parquet"
GOLD_PATH = PROJECT_ROOT / "data" / "processed_v1_old" / "gold_standard_final.csv"

# Helper inputs only used to recover gold (resume_id, job_id) — gold CSV
# stores the raw text, not the IDs, so we have to look them up.
RESUME_PARQUET = PROC_DIR / "cleaned_resumes.parquet"
JOB_PARQUET    = PROC_DIR / "cleaned_jobs.parquet"


# ---------------------------------------------------------------------------
# Column contracts
# ---------------------------------------------------------------------------
DROP_COLS = ["experience_gap", "experience_relevance_score", "education_match"]

FEATURE_COLS = [
    "embedding_similarity",
    "tfidf_similarity",
    "skill_overlap",
    "weighted_skill_score",
    "num_missing_skills",
    "avg_missing_skill_importance",
    "years_of_experience",
    "title_similarity",
]
KEY_COLS = ["job_id", "resume_id"]

# Final output column order
OUTPUT_COLS = [
    "job_id", "resume_id", "label",
    "embedding_similarity", "tfidf_similarity", "skill_overlap",
    "weighted_skill_score", "num_missing_skills",
    "avg_missing_skill_importance", "years_of_experience",
    "title_similarity",
]

EXPECTED_INPUT_COLS = KEY_COLS + FEATURE_COLS + DROP_COLS  # 13 columns total


def main() -> None:
    # -------------------------------------------------------------------
    # Load and validate
    # -------------------------------------------------------------------
    if not SRC_PATH.exists():
        print(f"ERROR: source file not found: {SRC_PATH}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(SRC_PATH)
    src_rows = len(df)

    missing = [c for c in EXPECTED_INPUT_COLS if c not in df.columns]
    if missing:
        print("ERROR: source file is missing expected columns. STOP.", file=sys.stderr)
        for c in missing:
            print(f"  missing: {c}", file=sys.stderr)
        sys.exit(2)

    # -------------------------------------------------------------------
    # Step 1 — drop 3 columns
    # -------------------------------------------------------------------
    df = df.drop(columns=DROP_COLS)

    # -------------------------------------------------------------------
    # Step 2 — weak labels via 30/70 quantile cut on composite score
    # -------------------------------------------------------------------
    weak_score = (
        0.7 * df["weighted_skill_score"].astype(np.float32)
        + 0.3 * df["title_similarity"].astype(np.float32)
    )
    p30 = float(np.quantile(weak_score, 0.30))
    p70 = float(np.quantile(weak_score, 0.70))

    label = pd.Series(np.full(len(df), -1, dtype=np.int8), index=df.index)
    label[weak_score <= p30] = 0
    label[weak_score >= p70] = 1

    keep_mask = label != -1
    rows_kept    = int(keep_mask.sum())
    rows_dropped = src_rows - rows_kept

    df_kept = df[keep_mask].copy()
    df_kept["label"] = label[keep_mask].astype(np.int8).values

    # -------------------------------------------------------------------
    # Step 3 — assemble output with exact dtypes and column order
    # -------------------------------------------------------------------
    for c in FEATURE_COLS:
        df_kept[c] = df_kept[c].astype(np.float32)
    df_kept["label"] = df_kept["label"].astype(np.int8)

    out_df = df_kept[OUTPUT_COLS].reset_index(drop=True)
    out_df.to_parquet(OUT_PATH, index=False)

    # -------------------------------------------------------------------
    # Step 4 — gold standard check (read-only, count only)
    # -------------------------------------------------------------------
    gold_found = 0
    if GOLD_PATH.exists() and RESUME_PARQUET.exists() and JOB_PARQUET.exists():
        gold_df = pd.read_csv(GOLD_PATH)
        resumes = pd.read_parquet(RESUME_PARQUET, columns=["Resume_str"]).reset_index(drop=True)
        jobs    = pd.read_parquet(JOB_PARQUET,    columns=["job_description"]).reset_index(drop=True)

        # Gold's `resume_text` is the raw Resume_str string; same for jobs.
        # Map each gold row's text back to a row index in the v2 parquets.
        res_lookup = {str(s): i for i, s in enumerate(resumes["Resume_str"].tolist())}
        job_lookup = {str(s): i for i, s in enumerate(jobs["job_description"].tolist())}
        gold_df["resume_id"] = gold_df["resume_text"].astype(str).map(res_lookup)
        gold_df["job_id"]    = gold_df["job_description"].astype(str).map(job_lookup)

        keyed = gold_df.dropna(subset=["resume_id", "job_id"]).copy()
        keyed["resume_id"] = keyed["resume_id"].astype(int)
        keyed["job_id"]    = keyed["job_id"].astype(int)

        out_keys = set(zip(out_df["job_id"].tolist(), out_df["resume_id"].tolist()))
        gold_found = int(sum(
            (int(j), int(r)) in out_keys
            for j, r in zip(keyed["job_id"], keyed["resume_id"])
        ))

    # -------------------------------------------------------------------
    # Step 5 — validation report (exact format)
    # -------------------------------------------------------------------
    n_pos = int((out_df["label"] == 1).sum())
    n_neg = int((out_df["label"] == 0).sum())
    ratio = "n/a" if n_neg == 0 else f"{n_pos / n_neg:.4f}"

    null_counts = out_df[OUTPUT_COLS].isnull().sum()

    print("=== ML-READY DATASET VALIDATION ===")
    print(f"Source rows:          {src_rows:,}")
    print(f"Rows kept after cut:  {rows_kept:,}")
    print(f"Rows dropped (middle):{rows_dropped:,}")
    print(f"Label=1 (positive):   {n_pos:,}")
    print(f"Label=0 (negative):   {n_neg:,}")
    print(f"Ratio:                {ratio}")
    print()
    print("Null values per column:")
    for c in OUTPUT_COLS:
        print(f"  {c}: {int(null_counts[c])}")
    print()
    print("Feature statistics:")
    print(f"  {'feature':<32} | {'min':>10} | {'max':>10} | {'mean':>10}")
    for c in FEATURE_COLS:
        col = out_df[c].astype(np.float64)
        print(f"  {c:<32} | {col.min():>10.4f} | {col.max():>10.4f} | {col.mean():>10.4f}")
    print()
    print("Weak score thresholds:")
    print(f"  30th percentile: {p30:.6f}")
    print(f"  70th percentile: {p70:.6f}")
    print()
    print(f"Columns in output ({len(OUTPUT_COLS)}): {OUTPUT_COLS}")
    print()
    print(f"Gold standard pairs found in ml_ready_dataset: {gold_found} / 100")
    print("=== DONE ===")
    print(f"Saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
