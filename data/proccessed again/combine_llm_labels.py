"""
combine_llm_labels.py
=====================
Phase 3 of the new methodology. Take the three LLM-labeled CSVs (one per
LLM: Claude, GPT, Gemini) and produce a single majority-vote gold standard.

Inputs (all under data/proccessed again/gold_labeling/):
    labels_claude.csv   (columns: row_id, label)
    labels_gpt.csv      (columns: row_id, label)
    labels_gemini.csv   (columns: row_id, label)
    gold_pairs_master.csv  (from Phase 2; provides job_id, resume_id, etc.)

Output:
    gold_labels.csv  -- one row per pair, with:
        row_id, job_id, resume_id, source_for_sample,
        claude_label, gpt_label, gemini_label,
        majority_label, agreement, is_unanimous,
        embedding_similarity, skill_overlap, weighted_skill_score,
        resume_category, job_position_title

Run:
    python "data/proccessed again/combine_llm_labels.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
GOLD_DIR   = SCRIPT_DIR / "gold_labeling"

MASTER_CSV     = GOLD_DIR / "gold_pairs_master.csv"
OUT_CSV        = GOLD_DIR / "gold_labels.csv"
DISPUTED_CSV   = GOLD_DIR / "gold_labels_disputed.csv"

# Two supported file layouts:
#   (a) Single concatenated file per LLM:   labels_<llm>.csv
#   (b) One file per (batch, LLM):          batch_<N>_<llm>_labels.csv
# We try (a) first; if not found, fall back to (b) and concatenate.
PER_LLM_NAMES = ("claude", "gpt", "gemini")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_one_label_csv(path: Path, llm_name: str) -> pd.DataFrame:
    """Read a single labels CSV and return df with columns ['row_id', 'label'].
    Tolerates LLMs returning 'Row_ID' / 'Label' / extra columns."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "row_id" not in df.columns or "label" not in df.columns:
        print(f"ERROR: {path} must have columns 'row_id' and 'label'. "
              f"Got: {list(df.columns)}", file=sys.stderr)
        sys.exit(2)
    return df[["row_id", "label"]].copy()


def load_llm_labels_combined(llm_name: str) -> pd.DataFrame:
    """
    Resolve labels for one LLM by either:
      (a) reading the single concatenated file labels_<llm>.csv, or
      (b) reading and concatenating batch_<N>_<llm>_labels.csv (any N), or
      (c) reading and concatenating batch_<N>_labels_<llm>.csv (legacy alt).
    """
    single = GOLD_DIR / f"labels_{llm_name}.csv"
    if single.exists():
        df = _read_one_label_csv(single, llm_name)
        source_desc = f"{single.name}"
    else:
        # Find per-batch files, accept either naming convention
        batch_files = sorted(GOLD_DIR.glob(f"batch_*_{llm_name}_labels.csv"))
        if not batch_files:
            batch_files = sorted(GOLD_DIR.glob(f"batch_*_labels_{llm_name}.csv"))
        if not batch_files:
            print(f"ERROR: no labels found for '{llm_name}'. Looked for:",
                  file=sys.stderr)
            print(f"  - {single}", file=sys.stderr)
            print(f"  - {GOLD_DIR}/batch_*_{llm_name}_labels.csv",
                  file=sys.stderr)
            print(f"  - {GOLD_DIR}/batch_*_labels_{llm_name}.csv",
                  file=sys.stderr)
            sys.exit(1)
        parts = [_read_one_label_csv(p, llm_name) for p in batch_files]
        df = pd.concat(parts, ignore_index=True)
        source_desc = f"{len(batch_files)} batch file(s): {', '.join(p.name for p in batch_files)}"
    print(f"[load] {llm_name:>6s}: {len(df):>3d} rows from {source_desc}")
    return df


def load_llm_labels(name: str) -> pd.DataFrame:
    df = load_llm_labels_combined(name)

    # Coerce types
    df["row_id"] = pd.to_numeric(df["row_id"], errors="coerce").astype("Int64")
    df["label"]  = pd.to_numeric(df["label"],  errors="coerce")

    # Drop rows with bad row_id
    bad_id = df["row_id"].isna().sum()
    if bad_id:
        print(f"[warn] {name}: {bad_id} rows had non-numeric row_id and were dropped")
        df = df.dropna(subset=["row_id"])

    # Sanity-check labels (must be 0 or 1)
    bad_lab = (~df["label"].isin([0, 1])).sum()
    if bad_lab:
        print(f"[warn] {name}: {bad_lab} rows had label values other than 0/1; "
              f"coercing via threshold 0.5")
        df["label"] = (df["label"].fillna(0) >= 0.5).astype(int)
    else:
        df["label"] = df["label"].astype(int)

    # Drop duplicates on row_id (keep first)
    before = len(df)
    df = df.drop_duplicates(subset=["row_id"], keep="first")
    if len(df) != before:
        print(f"[warn] {name}: removed {before - len(df)} duplicate row_ids")

    df["row_id"] = df["row_id"].astype(int)
    df = df.rename(columns={"label": f"{name}_label"})
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print(" Phase 3 — combining 3 LLM label sets via majority vote")
    print("=" * 70)

    if not MASTER_CSV.exists():
        print(f"ERROR: missing {MASTER_CSV}. Run sample_gold_pairs.py first.",
              file=sys.stderr)
        sys.exit(1)

    print(f"\n[load] {MASTER_CSV}")
    master = pd.read_csv(MASTER_CSV)
    print(f"       {len(master):,} sampled pairs")

    claude = load_llm_labels("claude")
    gpt    = load_llm_labels("gpt")
    gemini = load_llm_labels("gemini")

    # ---- Inner-join the three LLM tables on row_id ----
    merged = master.merge(claude, on="row_id", how="left") \
                   .merge(gpt,    on="row_id", how="left") \
                   .merge(gemini, on="row_id", how="left")

    # Coverage report
    n_total = len(merged)
    n_claude_missing = merged["claude_label"].isna().sum()
    n_gpt_missing    = merged["gpt_label"].isna().sum()
    n_gemini_missing = merged["gemini_label"].isna().sum()
    print(f"\n[merge] master rows: {n_total}")
    print(f"        claude missing: {n_claude_missing}")
    print(f"        gpt    missing: {n_gpt_missing}")
    print(f"        gemini missing: {n_gemini_missing}")

    # Drop rows where any LLM didn't label (can't majority-vote)
    full = merged.dropna(subset=["claude_label", "gpt_label", "gemini_label"]).copy()
    n_dropped = n_total - len(full)
    if n_dropped:
        print(f"[drop ] dropped {n_dropped} rows missing at least one LLM label")

    full["claude_label"] = full["claude_label"].astype(int)
    full["gpt_label"]    = full["gpt_label"].astype(int)
    full["gemini_label"] = full["gemini_label"].astype(int)

    # ---- Majority vote ----
    s = full[["claude_label", "gpt_label", "gemini_label"]].sum(axis=1)
    full["majority_label"] = (s >= 2).astype(int)
    full["agreement"]      = full[["claude_label", "gpt_label", "gemini_label"]].apply(
        lambda r: "3-0" if r.iloc[0] == r.iloc[1] == r.iloc[2] else "2-1",
        axis=1,
    )
    full["is_unanimous"] = (full["agreement"] == "3-0")

    # ---- Reorder + write ----
    out_cols = [
        "row_id", "job_id", "resume_id", "source_for_sample",
        "claude_label", "gpt_label", "gemini_label",
        "majority_label", "agreement", "is_unanimous",
        "embedding_similarity", "skill_overlap", "weighted_skill_score",
        "resume_category", "job_position_title",
    ]
    full = full[out_cols].sort_values("row_id").reset_index(drop=True)
    full.to_csv(OUT_CSV, index=False)
    print(f"\n[write] {OUT_CSV}")

    # ---- Disputed-only side file (2-1 votes) ----
    disputed = full[~full["is_unanimous"]].copy()
    disputed.to_csv(DISPUTED_CSV, index=False)
    print(f"[write] {DISPUTED_CSV}  ({len(disputed)} disputed rows)")

    # ---- Reporting ----
    print("\n=== Inter-LLM agreement ===")
    n_unanimous = int(full["is_unanimous"].sum())
    n_disputed  = len(full) - n_unanimous
    print(f"  3-0 unanimous : {n_unanimous:>3d}  ({100*n_unanimous/len(full):5.1f}%)")
    print(f"  2-1 disputed  : {n_disputed:>3d}  ({100*n_disputed/len(full):5.1f}%)")

    # Pairwise agreement
    pair_pairs = [
        ("claude_label", "gpt_label",    "Claude vs GPT   "),
        ("claude_label", "gemini_label", "Claude vs Gemini"),
        ("gpt_label",    "gemini_label", "GPT vs Gemini   "),
    ]
    print("\n  Pairwise agreement rate:")
    for a, b, name in pair_pairs:
        agree = (full[a] == full[b]).mean() * 100
        print(f"    {name}: {agree:5.1f}%")

    # Cohen's kappa-style: but with 3 raters; report Fleiss-style proxy
    # (probability of any two raters agreeing on a random pair, vs chance)
    p_a = (full["claude_label"] == full["gpt_label"]).mean()
    p_b = (full["claude_label"] == full["gemini_label"]).mean()
    p_c = (full["gpt_label"] == full["gemini_label"]).mean()
    p_obs = (p_a + p_b + p_c) / 3
    p_chance = 0.5  # binary task; if labels are uniform, random match probability is 0.5
    kappa_proxy = (p_obs - p_chance) / (1 - p_chance) if p_obs > p_chance else 0.0
    print(f"\n  Mean pairwise agreement: {p_obs * 100:5.1f}%")
    print(f"  Chance-adjusted (kappa-proxy): {kappa_proxy:.3f}")

    # ---- Per-LLM label rates ----
    print("\n=== Per-LLM positive-class rate ===")
    for col in ["claude_label", "gpt_label", "gemini_label"]:
        rate = full[col].mean() * 100
        print(f"  {col:<14s}: {rate:5.1f}% positive")
    print(f"  majority      : {full['majority_label'].mean() * 100:5.1f}% positive")

    # ---- Label distribution by source band ----
    print("\n=== Majority-label distribution by sample source ===")
    for src in ["topK", "A_mid", "B_xcat", "C_rand"]:
        sub = full[full["source_for_sample"] == src]
        if len(sub) == 0:
            continue
        pos = sub["majority_label"].sum()
        unan = sub["is_unanimous"].sum()
        print(f"  {src:>7s}: n={len(sub):>3d}  "
              f"pos={pos:>3d} ({100*pos/len(sub):4.1f}%)  "
              f"unanimous={unan:>3d} ({100*unan/len(sub):4.1f}%)")

    print()
    print("=" * 70)
    print(" Phase 3 complete. Sanity checks to look at above:")
    print("=" * 70)
    print("  - Pairwise agreement should be 70-85% for honest LLM judgment.")
    print("    Below 60%: LLMs are answering different questions (prompt issue).")
    print("    Above 95%: LLMs are echoing each other (likely they're all defaulting")
    print("    to the same heuristic). Check disputed CSV to see what's contested.")
    print("  - B_xcat positive rate should be LOW (these are deliberate cross-")
    print("    category mismatches). If it's > 30%, the LLMs aren't being strict")
    print("    enough or the family rules in build_diversified_pool need tightening.")
    print("  - topK positive rate should be HIGH (these are top-50 retrieval pairs).")
    print("    If it's < 50%, top-50 retrieval is producing many bad matches.")


if __name__ == "__main__":
    main()
