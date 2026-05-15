"""
Ablation Study: Contribution of Feature Groups in Hybrid ML System
Standalone script — read-only on data, does not modify any existing files.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "ml_ready_dataset.csv")
GOLD_PATH  = os.path.join(BASE_DIR, "..", "data", "processed", "gold_standard_final.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")
PLOT_PATH  = os.path.join(OUTPUT_DIR, "ablation_study_results.png")

# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Data")
print("=" * 60)

train_df = pd.read_csv(TRAIN_PATH)
gold_df  = pd.read_csv(GOLD_PATH)

print(f"\n[Train]  Shape : {train_df.shape}")
print(f"[Train]  Columns: {list(train_df.columns)}")
print(f"\n[Gold]   Shape : {gold_df.shape}")
print(f"[Gold]   Columns: {list(gold_df.columns)}")

REQUIRED_TRAIN = {"semantic_similarity", "tfidf_similarity", "skill_imbalance",
                  "num_resume_skills", "num_job_skills", "label"}
REQUIRED_GOLD  = {"semantic_similarity", "tfidf_similarity", "skill_imbalance",
                  "num_resume_skills", "num_job_skills", "gold_label"}

missing_train = REQUIRED_TRAIN - set(train_df.columns)
missing_gold  = REQUIRED_GOLD  - set(gold_df.columns)

if missing_train:
    sys.exit(f"[ERROR] Training data missing columns: {missing_train}")
if missing_gold:
    sys.exit(f"[ERROR] Gold data missing columns: {missing_gold}")

print("\n[OK] All required columns present in both datasets.")

# ─────────────────────────────────────────────
# STEP 2: DEFINE FEATURE SETS
# ─────────────────────────────────────────────
STAGES = [
    {
        "name": "Stage 1 - Semantic Only",
        "label": "Semantic",
        "features": ["semantic_similarity"],
    },
    {
        "name": "Stage 2 - Semantic + Lexical",
        "label": "Semantic\n+Lexical",
        "features": ["semantic_similarity", "tfidf_similarity"],
    },
    {
        "name": "Stage 3 - Full Hybrid Model",
        "label": "Full Hybrid",
        "features": [
            "semantic_similarity",
            "tfidf_similarity",
            "skill_imbalance",
            "num_resume_skills",
            "num_job_skills",
        ],
    },
]

THRESHOLDS = np.arange(0.10, 0.91, 0.05)

# ─────────────────────────────────────────────
# STEP 3: CORE LOOP
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Running Ablation Stages")
print("=" * 60)

results = []

for stage in STAGES:
    print(f"\n{'-' * 50}")
    print(f"  {stage['name']}")
    print(f"{'-' * 50}")

    feats = stage["features"]

    # ── STEP 5 sanity: NaN check ──
    nan_train = train_df[feats].isna().sum().sum()
    nan_gold  = gold_df[feats].isna().sum().sum()
    print(f"  Feature count : {len(feats)}")
    print(f"  NaNs in train : {nan_train}")
    print(f"  NaNs in gold  : {nan_gold}")
    if nan_train > 0 or nan_gold > 0:
        sys.exit(f"[ERROR] NaN values detected in stage '{stage['name']}'")
    print(f"  [OK] No NaNs detected.")

    # 3.1 Prepare Data
    X_train = train_df[feats].copy()
    y_train = train_df["label"].copy()
    X_gold  = gold_df[feats].copy()
    y_gold  = gold_df["gold_label"].copy()

    # 3.2 Leakage-safe scaling: fit only on train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_gold_scaled  = scaler.transform(X_gold)

    # 3.3 Threshold tuning via cross-validated OOF probabilities (training only)
    model_cv = GradientBoostingClassifier(random_state=42)
    oof_probs = cross_val_predict(
        model_cv, X_train_scaled, y_train, cv=5, method="predict_proba"
    )[:, 1]

    best_f1   = -1
    best_thresh = 0.5
    for t in THRESHOLDS:
        preds = (oof_probs >= t).astype(int)
        f1 = f1_score(y_train, preds, zero_division=0)
        if f1 > best_f1:
            best_f1   = f1
            best_thresh = t

    best_thresh = round(best_thresh, 2)
    print(f"  Best threshold (CV): {best_thresh:.2f}  (OOF F1 = {best_f1:.4f})")

    # 3.4 Train final model on full training data
    model_final = GradientBoostingClassifier(random_state=42)
    model_final.fit(X_train_scaled, y_train)

    # 3.5 Evaluate on gold standard
    gold_probs = model_final.predict_proba(X_gold_scaled)[:, 1]
    gold_preds = (gold_probs >= best_thresh).astype(int)

    # 3.6 Compute metrics
    acc  = accuracy_score(y_gold, gold_preds)
    prec = precision_score(y_gold, gold_preds, zero_division=0)
    rec  = recall_score(y_gold, gold_preds, zero_division=0)
    f1   = f1_score(y_gold, gold_preds, zero_division=0)

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")

    results.append({
        "stage_num":  len(results) + 1,
        "stage_name": stage["name"],
        "label":      stage["label"],
        "features":   feats,
        "threshold":  best_thresh,
        "accuracy":   acc,
        "precision":  prec,
        "recall":     rec,
        "f1":         f1,
    })

# ─────────────────────────────────────────────
# STEP 4.1: MARKDOWN TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Results")
print("=" * 60)

header = f"{'Stage':<8} {'Features Used':<52} {'Thresh':>6} {'Acc':>7} {'Prec':>8} {'Rec':>7} {'F1':>7}"
sep    = "-" * len(header)
print(f"\n{sep}")
print(header)
print(sep)

for r in results:
    feat_str = ", ".join(r["features"])
    print(
        f"{r['stage_num']:<8} {feat_str:<52} {r['threshold']:>6.2f} "
        f"{r['accuracy']:>7.4f} {r['precision']:>8.4f} {r['recall']:>7.4f} {r['f1']:>7.4f}"
    )
print(sep)

# Also print as Markdown
print("\n### Markdown Table\n")
print("| Stage | Features Used | Threshold | Accuracy | Precision | Recall | F1 |")
print("|-------|--------------|-----------|----------|-----------|--------|-----|")
for r in results:
    feat_str = ", ".join(r["features"])
    print(
        f"| {r['stage_num']} | {feat_str} | {r['threshold']:.2f} "
        f"| {r['accuracy']:.4f} | {r['precision']:.4f} "
        f"| {r['recall']:.4f} | {r['f1']:.4f} |"
    )

# ─────────────────────────────────────────────
# STEP 4.2: VISUALIZATION
# ─────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

labels    = [r["label"] for r in results]
f1_scores = [r["f1"]        for r in results]
prec_scores = [r["precision"] for r in results]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars_f1   = ax.bar(x - width / 2, f1_scores,   width, label="F1 Score")
bars_prec = ax.bar(x + width / 2, prec_scores,  width, label="Precision")

ax.set_xlabel("Feature Stage")
ax.set_ylabel("Score")
ax.set_title("Ablation Study — Feature Group Contributions")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.05)
ax.legend()

# Annotate bars
for bar in bars_f1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
            ha="center", va="bottom", fontsize=8)
for bar in bars_prec:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
            ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150)
plt.close()
print(f"\n[OK] Plot saved -> {PLOT_PATH}")

# ─────────────────────────────────────────────
# STEP 5: SANITY CHECK SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Sanity Check Summary")
print("=" * 60)
for r in results:
    print(f"  {r['stage_name']}")
    print(f"    Feature count : {len(r['features'])}")
    print(f"    Best threshold: {r['threshold']:.2f}")
    print(f"    No NaNs       : confirmed")

# ─────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Ablation study complete [DONE]")
print("=" * 60)
print("\nSummary:")
for r in results:
    print(f"  Stage {r['stage_num']} | threshold={r['threshold']:.2f} | "
          f"F1={r['f1']:.4f} | Prec={r['precision']:.4f} | Rec={r['recall']:.4f}")
print(f"\nPlot saved: {PLOT_PATH}")
