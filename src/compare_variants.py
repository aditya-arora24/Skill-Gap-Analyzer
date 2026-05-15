"""
compare_variants.py
===================
Reads outputs/variant_<X>/metrics_<X>.csv for X in {A,B,C} and produces:

  outputs/variant_comparison.csv   -- unified table, all variants stacked
  outputs/variant_comparison.png   -- grouped bar chart, F1 by model x variant

Run AFTER `evaluate_models.py --config ../configs/variant_<X>.json` has
executed for every variant you want to compare.

Usage
-----
    python compare_variants.py                 # all available variants
    python compare_variants.py --variants A B  # subset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=None,
                    help="Subset of variant names (A B C). Default: all available.")
    args = ap.parse_args()

    variants = args.variants or [d.name.replace("variant_", "")
                                  for d in OUTPUTS_DIR.glob("variant_*")
                                  if (d / f"metrics_{d.name.replace('variant_', '')}.csv").exists()]
    variants = sorted(variants)
    if not variants:
        print("No variant metrics CSVs found in outputs/variant_*/. "
              "Run evaluate_models.py for each variant first.")
        return

    print(f"Found variants: {variants}")
    frames = []
    for v in variants:
        path = OUTPUTS_DIR / f"variant_{v}" / f"metrics_{v}.csv"
        if not path.exists():
            print(f"  [skip] {path} missing")
            continue
        df = pd.read_csv(path)
        if "variant" not in df.columns:
            df["variant"] = v
        frames.append(df)
    if not frames:
        print("Nothing to compare.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[["variant", "Model", "Threshold", "Accuracy", "Precision", "Recall", "F1"]]
    combined = combined.sort_values(["variant", "F1"], ascending=[True, False]).reset_index(drop=True)

    out_csv = OUTPUTS_DIR / "variant_comparison.csv"
    combined.to_csv(out_csv, index=False)
    print(f"\n[saved] {out_csv}")
    print("\n=== Unified comparison ===")
    print(combined.to_string(index=False))

    # ---- Grouped bar chart of F1 ----
    pivot = combined.pivot_table(index="Model", columns="variant", values="F1", aggfunc="first")
    # sort models so trained ones come first
    model_order = [m for m in ["LogReg (tuned)", "GBM (tuned)",
                               "SBERT Only", "TF-IDF Only",
                               "Title Similarity", "Weighted Skills",
                               "Composite Label"] if m in pivot.index]
    pivot = pivot.reindex(model_order)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    n_models = len(pivot.index)
    n_var = len(pivot.columns)
    x = np.arange(n_models)
    width = 0.8 / max(n_var, 1)
    palette = plt.cm.tab10.colors

    for i, v in enumerate(pivot.columns):
        vals = pivot[v].fillna(0).values
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                      label=f"Variant {v}", color=palette[i % len(palette)],
                      edgecolor="white")
        for b, val in zip(bars, vals):
            if val > 0:
                ax.text(b.get_x() + b.get_width() / 2, val + 0.01, f"{val:.2f}",
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.set_ylabel("F1 (gold standard)")
    ax.set_ylim(0, max(0.85, pivot.max().max() * 1.12))
    ax.set_title("Variant comparison — F1 on the held-out gold set", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_png = OUTPUTS_DIR / "variant_comparison.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[saved] {out_png}")


if __name__ == "__main__":
    main()
