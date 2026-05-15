"""
Merge human labels into the gold standard evaluation dataset.
Uses row_index to ensure correct alignment — does NOT rely on row order.
Output: gold_standard_final.csv (resume_text, job_description, gold_label)
"""

import pandas as pd

TO_LABEL = "../data/processed/gold_standard_to_label.csv"
LABELS   = "../data/processed/gold_standard_labels.csv"
OUT_CSV  = "../data/processed/gold_standard_final.csv"

# ── Load files ────────────────────────────────────────────────────────────────
texts  = pd.read_csv(TO_LABEL)
labels = pd.read_csv(LABELS)

# ── Attach row_index to texts (preserves original order, no shuffling) ────────
texts["row_index"] = texts.index

# ── Merge on row_index ────────────────────────────────────────────────────────
merged = texts.merge(labels[["row_index", "gold_label"]], on="row_index", how="inner")

# ── Final columns only ────────────────────────────────────────────────────────
final = merged[["resume_text", "job_description", "gold_label"]].reset_index(drop=True)

# ── Validate ──────────────────────────────────────────────────────────────────
missing = final.isnull().sum()
duplicates = final.duplicated().sum()

assert missing.sum() == 0,    f"Missing values found: {missing[missing > 0].to_dict()}"
assert duplicates == 0,       f"Duplicate rows found: {duplicates}"
assert len(final) == 100,     f"Expected 100 rows, got {len(final)}"

# ── Save ──────────────────────────────────────────────────────────────────────
final.to_csv(OUT_CSV, index=False)

# ── Print summary ─────────────────────────────────────────────────────────────
print(f"Total rows: {len(final)}")
print(f"\nFirst 5 rows:")
preview = final.head(5).copy()
preview["resume_text"] = preview["resume_text"].str[:60] + "..."
preview["job_description"] = preview["job_description"].str[:50] + "..."
print(preview.to_string())
print(f"\ngold_label distribution:")
print(final["gold_label"].value_counts().sort_index().to_string())
print(f"\nMissing values: {missing.to_dict()}")
print(f"Duplicate rows: {duplicates}")
print(f"\nSaved -> {OUT_CSV}")
