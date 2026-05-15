"""
Create gold standard evaluation dataset for human labeling.
Stratified sample: 50 label=1 + 50 label=0 from ml_ready_dataset.csv
Output: gold_standard_to_label.csv (resume_text, job_description only — no label)
"""

import pandas as pd

ML_READY   = "../data/processed/ml_ready_dataset.csv"
RESUME_SRC = "../data/raw/Resume (1).csv"
JOB_SRC    = "../data/raw/training_data.csv"
OUT_CSV    = "../data/processed/gold_standard_to_label.csv"

# ── Load ML dataset ───────────────────────────────────────────────────────────
df = pd.read_csv(ML_READY)

# ── Stratified sample: 50 per label ──────────────────────────────────────────
pos = df[df["label"] == 1].sample(n=50, random_state=42)
neg = df[df["label"] == 0].sample(n=50, random_state=42)
sample = pd.concat([pos, neg]).reset_index(drop=True)
label_counts = sample["label"].value_counts().sort_index()

# ── Attach text columns ───────────────────────────────────────────────────────
if "resume_text" in df.columns and "job_text" in df.columns:
    # Pipeline already saved text columns — use directly
    gold = sample[["resume_text", "job_text"]].rename(
        columns={"job_text": "job_description"}
    )
else:
    # Reconstruct from source files (pipeline run before text columns were added)
    resumes = pd.read_csv(RESUME_SRC, encoding="utf-8", on_bad_lines="skip")[
        ["ID", "Resume_str"]
    ]
    jobs = pd.read_csv(JOB_SRC, encoding="utf-8", on_bad_lines="skip")[
        ["job_description"]
    ]
    jobs["job_id"] = jobs.index

    sample = sample.merge(resumes, left_on="resume_id", right_on="ID", how="left")
    sample = sample.merge(jobs, on="job_id", how="left")
    gold = sample[["Resume_str", "job_description"]].rename(
        columns={"Resume_str": "resume_text"}
    )

# ── Save (no label column) ────────────────────────────────────────────────────
gold.to_csv(OUT_CSV, index=False)

# ── Print summary ─────────────────────────────────────────────────────────────
print(f"Total rows:         {len(gold)}")
print(f"Label distribution: label=1: {label_counts.get(1, 0)} | label=0: {label_counts.get(0, 0)}")
print(f"Saved -> {OUT_CSV}")
