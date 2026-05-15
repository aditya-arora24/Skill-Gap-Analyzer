"""
run_self_training_v2.py
========================
Fixed self-training variant. Two changes from run_self_training_experiment.py:

  1. Pool = pair_features_diversified.parquet (50,650 pairs across all four
     sources: topK / A_mid / B_xcat / C_rand). NOT ml_ready_dataset.parquet.
     This restores the full distribution including hard cases that the
     formula filter previously removed.

  2. Pseudo-label selection: top 5% by P(positive) → pseudo-label = 1,
     bottom 5% by P(positive) → pseudo-label = 0, discard middle 90%.
     Controlled 50/50 class balance regardless of the model's calibration
     on the new pool. Replaces the absolute 0.80 / 0.20 thresholds that
     caused the v1 class-balance flip.

All 500 gold pairs (both train and test) are dropped from the pool before
scoring so no LLM-labeled pair ever appears as a pseudo-label.

Inputs (read-only):
  models/llm_supervised/{scaler,logreg}.pkl
  data/proccessed again/gold_labeling/gold_labels.csv
  data/proccessed again/processed/pair_features_diversified.parquet
  outputs/three_way/run_metadata.json
  outputs/self_training/comparison.csv  (v1 numbers, for side-by-side print)

Outputs (NEW paths only):
  models/llm_self_trained_v2/{scaler,logreg}.pkl
  outputs/self_training_v2/comparison.csv         3-way comparison
  outputs/self_training_v2/comparison.png         grouped bars
  outputs/self_training_v2/pseudo_label_distribution.png
  outputs/self_training_v2/test_set_diff.csv
  outputs/self_training_v2/run_metadata.json

Run:
    python "src/run_self_training_v2.py"
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUPERVISED_MODEL_DIR = PROJECT_ROOT / "models" / "llm_supervised"
GOLD_CSV     = PROJECT_ROOT / "data" / "proccessed again" / "gold_labeling" / "gold_labels.csv"
POOL_PARQUET = PROJECT_ROOT / "data" / "proccessed again" / "processed" / "pair_features_diversified.parquet"
PHASE4_META  = PROJECT_ROOT / "outputs" / "three_way" / "run_metadata.json"
V1_RESULTS   = PROJECT_ROOT / "outputs" / "self_training" / "comparison.csv"

NEW_MODEL_DIR = PROJECT_ROOT / "models" / "llm_self_trained_v2"
OUT_DIR       = PROJECT_ROOT / "outputs" / "self_training_v2"
NEW_MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RANDOM_SEED   = 42
TEST_FRACTION = 0.20
TOP_PCT       = 0.05      # top 5% → pseudo-positive
BOT_PCT       = 0.05      # bottom 5% → pseudo-negative
THRESHOLD_GRID = np.arange(0.10, 0.91, 0.02)
PRECISION_AT_K = (5, 10, 20)

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
def precision_at_k(scores, y_true, k):
    if len(scores) == 0:
        return 0.0
    k = min(k, len(scores))
    top_idx = np.argsort(-scores)[:k]
    return float(np.sum(y_true[top_idx] == 1)) / k


def metrics_row(name, threshold, y_true, y_pred, scores):
    row = {
        "Approach":  name,
        "Threshold": round(float(threshold), 2),
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    for k in PRECISION_AT_K:
        row[f"P@{k}"] = round(precision_at_k(scores, y_true, k), 4)
    return row


def find_best_threshold_cv(model_kwargs, X, y, thresholds, cv=5,
                           seed=RANDOM_SEED):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(X, y):
        m = LogisticRegression(**model_kwargs)
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        f = f1_score(y, (oof >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return round(float(best_t), 2), round(float(best_f1), 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(" Self-training v2 (fixed pool + percentile threshold)")
    print("=" * 70)

    # 1. Load supervised baseline
    print("\n[1] loading supervised LogReg baseline (read-only)")
    old_scaler = joblib.load(SUPERVISED_MODEL_DIR / "scaler.pkl")
    old_model  = joblib.load(SUPERVISED_MODEL_DIR / "logreg.pkl")

    # 2. Reproduce Phase 4 train/test split
    print("\n[2] reproducing Phase 4 train/test split (random_state=42)")
    gold = pd.read_csv(GOLD_CSV)
    pool_full = pd.read_parquet(POOL_PARQUET)

    gold = gold[[c for c in gold.columns if c not in ALL_FEATURES]]
    gold_full = gold.merge(
        pool_full[["job_id", "resume_id"] + ALL_FEATURES],
        on=["job_id", "resume_id"], how="left",
    )
    gold_full = gold_full.dropna(subset=ALL_FEATURES).reset_index(drop=True)
    gold_full["majority_label"] = gold_full["majority_label"].astype(int)

    train_df, test_df = train_test_split(
        gold_full,
        test_size=TEST_FRACTION,
        stratify=gold_full["majority_label"],
        random_state=RANDOM_SEED,
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
    print(f"    train: {len(train_df)} (pos={int(train_df['majority_label'].sum())}, "
          f"neg={int((1 - train_df['majority_label']).sum())})")
    print(f"    test : {len(test_df)} (pos={int(test_df['majority_label'].sum())}, "
          f"neg={int((1 - test_df['majority_label']).sum())})")

    if PHASE4_META.exists():
        meta4 = json.loads(PHASE4_META.read_text())
        expected_ids = set(meta4.get("test_row_ids", []))
        actual_ids   = set(test_df["row_id"].astype(int).tolist())
        if expected_ids and expected_ids != actual_ids:
            raise SystemExit("test split differs from Phase 4")
        elif expected_ids:
            print(f"    [OK] test row_ids match Phase 4 ({len(actual_ids)} pairs)")

    y_train = train_df["majority_label"].values
    y_test  = test_df["majority_label"].values
    X_train_full = train_df[ALL_FEATURES].values
    X_test_full  = test_df[ALL_FEATURES].values

    # 3. Build the unlabeled pool: pair_features_diversified MINUS all 500 gold pairs
    print("\n[3] preparing unlabeled pool (pair_features_diversified \\ 500 gold)")
    print(f"    starting pool: {len(pool_full):,} rows  | "
          f"source mix: {pool_full['pool_source'].value_counts().to_dict()}")

    # Drop ALL gold pairs (train + test) so no LLM-labeled pair becomes a pseudo-label
    gold_keys = set(zip(
        gold_full["job_id"].astype(int), gold_full["resume_id"].astype(int)
    ))
    pool_keys = list(zip(
        pool_full["job_id"].astype(int), pool_full["resume_id"].astype(int)
    ))
    pool_mask = np.array([k not in gold_keys for k in pool_keys])
    pool = pool_full[pool_mask].reset_index(drop=True)
    n_dropped = int((~pool_mask).sum())
    print(f"    dropped {n_dropped} gold-overlap rows  -> pool size: {len(pool):,}")
    print(f"    source mix: {pool['pool_source'].value_counts().to_dict()}")

    # 4. Score the pool
    print("\n[4] scoring pool with supervised model")
    X_pool = pool[ALL_FEATURES].values
    X_pool_s = old_scaler.transform(X_pool)
    probs_pool = old_model.predict_proba(X_pool_s)[:, 1]
    print(f"    mean P(positive) = {probs_pool.mean():.4f}  "
          f"std = {probs_pool.std():.4f}")

    # ---- Diagnostic: probability histogram (10 bins) ----
    print("\n    P(positive) distribution (10 bins):")
    hist_counts, hist_edges = np.histogram(probs_pool, bins=10, range=(0.0, 1.0))
    for i, c in enumerate(hist_counts):
        bar = "#" * int(40 * c / max(hist_counts))
        print(f"      {hist_edges[i]:.1f}–{hist_edges[i+1]:.1f}: "
              f"{c:>6,}  {bar}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(probs_pool, bins=20, color="steelblue", edgecolor="white")
    ax.axvline(np.quantile(probs_pool, 1 - TOP_PCT), color="seagreen",
               linestyle="--", label=f"top {int(TOP_PCT*100)}% cutoff")
    ax.axvline(np.quantile(probs_pool, BOT_PCT), color="crimson",
               linestyle="--", label=f"bottom {int(BOT_PCT*100)}% cutoff")
    ax.set_xlabel("P(positive | features)")
    ax.set_ylabel("Count")
    ax.set_title("Probability distribution on diversified pool "
                 f"(N={len(probs_pool):,})", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pseudo_label_distribution.png", dpi=150)
    plt.close()

    # 5. Pseudo-label selection: top 5% / bottom 5% by probability
    print(f"\n[5] selecting pseudo-labels by percentile")
    n_pool = len(probs_pool)
    n_top  = int(np.ceil(n_pool * TOP_PCT))
    n_bot  = int(np.ceil(n_pool * BOT_PCT))
    print(f"    pool size: {n_pool:,}")
    print(f"    top {int(TOP_PCT*100)}%   = {n_top:>5,}  -> pseudo-label = 1")
    print(f"    bottom {int(BOT_PCT*100)}% = {n_bot:>5,}  -> pseudo-label = 0")

    # argsort by probability ascending; bottom indices are the lowest, top indices the highest
    order = np.argsort(probs_pool)
    bot_idx = order[:n_bot]
    top_idx = order[-n_top:]

    pseudo_idx   = np.concatenate([top_idx, bot_idx])
    pseudo_labels = np.concatenate([np.ones(n_top, dtype=int), np.zeros(n_bot, dtype=int)])
    pseudo_probs  = probs_pool[pseudo_idx]
    pseudo_X      = X_pool[pseudo_idx]
    pseudo_pool_source = pool.iloc[pseudo_idx]["pool_source"].values

    print(f"    total pseudo-labels: {len(pseudo_labels):,}")
    print(f"    class balance: 50/50 by construction")
    print(f"    pseudo-positive probability range: "
          f"[{pseudo_probs[pseudo_labels==1].min():.4f}, "
          f"{pseudo_probs[pseudo_labels==1].max():.4f}]")
    print(f"    pseudo-negative probability range: "
          f"[{pseudo_probs[pseudo_labels==0].min():.4f}, "
          f"{pseudo_probs[pseudo_labels==0].max():.4f}]")
    print(f"    pseudo-label source distribution:")
    for src, cnt in pd.Series(pseudo_pool_source).value_counts().items():
        print(f"      {src:>8s}: {cnt:>5,}  "
              f"({100*cnt/len(pseudo_labels):4.1f}%)")

    # 6. Combine with 400 LLM gold labels
    print("\n[6] combining with 400 LLM-labeled training pairs")
    combined_X = np.vstack([X_train_full, pseudo_X])
    combined_y = np.concatenate([y_train, pseudo_labels])
    print(f"    combined size: {len(combined_y):,} "
          f"(400 gold + {len(pseudo_labels):,} pseudo)")
    print(f"    overall class balance: "
          f"{100 * combined_y.mean():.1f}% positive")

    # 7. Re-train fresh LogReg on combined
    print("\n[7] training new LogReg on combined set")
    new_scaler = StandardScaler().fit(combined_X)
    combined_Xs = new_scaler.transform(combined_X)

    model_kwargs = dict(
        solver="liblinear",
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    new_t, new_cv_f1 = find_best_threshold_cv(
        model_kwargs, combined_Xs, combined_y, THRESHOLD_GRID,
    )
    new_model = LogisticRegression(**model_kwargs)
    new_model.fit(combined_Xs, combined_y)
    print(f"    new threshold (CV): {new_t}  CV F1 = {new_cv_f1:.4f}")

    joblib.dump(new_scaler, NEW_MODEL_DIR / "scaler.pkl")
    joblib.dump(new_model,  NEW_MODEL_DIR / "logreg.pkl")
    print(f"    saved -> {NEW_MODEL_DIR / 'scaler.pkl'}")
    print(f"    saved -> {NEW_MODEL_DIR / 'logreg.pkl'}")

    # 8. Evaluate both on the same 100-pair test set
    print("\n[8] evaluating on the same 100-pair held-out test set")
    X_train_old_s = old_scaler.transform(X_train_full)
    old_t, old_cv_f1 = find_best_threshold_cv(
        model_kwargs, X_train_old_s, y_train, THRESHOLD_GRID,
    )

    X_test_old_s = old_scaler.transform(X_test_full)
    X_test_new_s = new_scaler.transform(X_test_full)
    probs_old = old_model.predict_proba(X_test_old_s)[:, 1]
    probs_new = new_model.predict_proba(X_test_new_s)[:, 1]
    y_pred_old = (probs_old >= old_t).astype(int)
    y_pred_new = (probs_new >= new_t).astype(int)

    sup_metrics = metrics_row("Supervised (baseline)", old_t, y_test, y_pred_old, probs_old)
    v2_metrics  = metrics_row("Self-trained v2 (fixed)", new_t, y_test, y_pred_new, probs_new)

    # ---- Pull v1 metrics from existing CSV (for the 3-way comparison) ----
    v1_row = None
    if V1_RESULTS.exists():
        v1_df = pd.read_csv(V1_RESULTS)
        if (v1_df["Approach"] == "Self-trained (1 iter)").any():
            v1 = v1_df[v1_df["Approach"] == "Self-trained (1 iter)"].iloc[0].to_dict()
            v1["Approach"] = "Self-trained v1 (abs threshold)"
            v1_row = v1
    if v1_row is None:
        # hard-coded from the v1 run, in case the CSV isn't where expected
        v1_row = {
            "Approach": "Self-trained v1 (abs threshold)",
            "Threshold": 0.68, "Accuracy": 0.83, "Precision": 0.6471,
            "Recall": 0.5000, "F1": 0.5641, "P@5": 1.00, "P@10": 0.80, "P@20": 0.70,
        }

    cmp_df = pd.DataFrame([sup_metrics, v1_row, v2_metrics])
    print("\n" + "=" * 70)
    print(" Three-way comparison on the SAME 100-pair test set")
    print("=" * 70)
    print(cmp_df.to_string(index=False))
    cmp_df.to_csv(OUT_DIR / "comparison.csv", index=False)
    print(f"\n[write] {OUT_DIR / 'comparison.csv'}")

    # ---- Test-set diff: which pairs flipped from supervised to v2 ----
    diff = pd.DataFrame({
        "row_id":   test_df["row_id"].astype(int).values if "row_id" in test_df.columns else range(len(test_df)),
        "true":     y_test,
        "old_pred": y_pred_old,
        "v2_pred":  y_pred_new,
        "old_prob": np.round(probs_old, 4),
        "v2_prob":  np.round(probs_new, 4),
    })
    diff["flipped"]     = diff["old_pred"] != diff["v2_pred"]
    diff["old_correct"] = diff["old_pred"] == diff["true"]
    diff["v2_correct"]  = diff["v2_pred"]  == diff["true"]
    diff["delta"]       = diff["v2_correct"].astype(int) - diff["old_correct"].astype(int)
    diff.to_csv(OUT_DIR / "test_set_diff.csv", index=False)

    n_flipped = int(diff["flipped"].sum())
    n_p2n = int(((diff["old_pred"] == 1) & (diff["v2_pred"] == 0)).sum())
    n_n2p = int(((diff["old_pred"] == 0) & (diff["v2_pred"] == 1)).sum())
    n_now_correct = int((diff["delta"] > 0).sum())
    n_now_wrong   = int((diff["delta"] < 0).sum())
    print(f"\n--- Confusion-matrix diff (supervised vs v2) ---")
    print(f"  flipped: {n_flipped} / 100")
    print(f"    pos -> neg: {n_p2n}    neg -> pos: {n_n2p}")
    print(f"  flips that improved prediction: {n_now_correct}")
    print(f"  flips that worsened prediction: {n_now_wrong}")

    # ---- Sanity check on original 400 train labels ----
    X_train_new_s = new_scaler.transform(X_train_full)
    train_pred_old = (old_model.predict_proba(X_train_old_s)[:, 1] >= old_t).astype(int)
    train_pred_new = (new_model.predict_proba(X_train_new_s)[:, 1] >= new_t).astype(int)
    train_acc_old = accuracy_score(y_train, train_pred_old)
    train_acc_new = accuracy_score(y_train, train_pred_new)
    train_f1_old  = f1_score(y_train, train_pred_old, zero_division=0)
    train_f1_new  = f1_score(y_train, train_pred_new, zero_division=0)
    print(f"\n--- Sanity check on original 400 train labels ---")
    print(f"  supervised : accuracy={train_acc_old:.4f}  F1={train_f1_old:.4f}")
    print(f"  self-train v2: accuracy={train_acc_new:.4f}  F1={train_f1_new:.4f}")

    # ---- Plot: 3-way grouped bars ----
    metric_cols = ["F1", "P@5", "P@10", "P@20", "Accuracy"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(cmp_df))
    width = 0.15
    palette = plt.cm.tab10.colors
    for i, m in enumerate(metric_cols):
        if m not in cmp_df.columns:
            continue
        vals = pd.to_numeric(cmp_df[m], errors="coerce").fillna(0).values
        bars = ax.bar(x + i * width - width * (len(metric_cols) - 1) / 2, vals,
                      width, label=m, color=palette[i % len(palette)],
                      edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(cmp_df["Approach"], rotation=8, ha="center")
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Score on held-out test (100 pairs)")
    ax.set_title("Self-training comparison: baseline vs v1 (absolute "
                 "threshold) vs v2 (percentile)", fontweight="bold")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "comparison.png", dpi=150)
    plt.close()
    print(f"[write] {OUT_DIR / 'comparison.png'}")

    # ---- Run metadata ----
    meta = {
        "random_seed":          RANDOM_SEED,
        "test_fraction":        TEST_FRACTION,
        "top_percentile":       TOP_PCT,
        "bottom_percentile":    BOT_PCT,
        "n_pool":               int(len(pool)),
        "n_pool_dropped_gold":  int(n_dropped),
        "n_top":                int(n_top),
        "n_bot":                int(n_bot),
        "n_pseudo":             int(len(pseudo_labels)),
        "n_combined":           int(len(combined_y)),
        "old_threshold":        old_t,
        "old_cv_f1":            old_cv_f1,
        "new_threshold":        new_t,
        "new_cv_f1":            new_cv_f1,
        "old_test_metrics":     sup_metrics,
        "v2_test_metrics":      v2_metrics,
        "n_flipped":            n_flipped,
        "n_now_correct":        n_now_correct,
        "n_now_wrong":          n_now_wrong,
        "train_accuracy_old":   float(train_acc_old),
        "train_accuracy_new":   float(train_acc_new),
        "train_f1_old":         float(train_f1_old),
        "train_f1_new":         float(train_f1_new),
        "pseudo_label_source_distribution": {
            str(k): int(v) for k, v in
            pd.Series(pseudo_pool_source).value_counts().items()
        },
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[write] {OUT_DIR / 'run_metadata.json'}")

    # Bottom line
    print("\n" + "=" * 70)
    print(" Bottom line")
    print("=" * 70)
    delta_v2_baseline = v2_metrics["F1"]  - sup_metrics["F1"]
    delta_v2_v1       = v2_metrics["F1"]  - float(v1_row["F1"])
    print(f"  Supervised baseline      F1 = {sup_metrics['F1']:.4f}")
    print(f"  Self-trained v1 (abs)    F1 = {float(v1_row['F1']):.4f}")
    print(f"  Self-trained v2 (fixed)  F1 = {v2_metrics['F1']:.4f}")
    print(f"  v2 vs supervised  delta F1 = {delta_v2_baseline:+.4f}")
    print(f"  v2 vs v1          delta F1 = {delta_v2_v1:+.4f}")
    if v2_metrics["F1"] > sup_metrics["F1"] + 0.02:
        print("  Verdict: fix worked. Self-training adds value when the pool is "
              "diversified and balance is enforced.")
    elif v2_metrics["F1"] > sup_metrics["F1"] - 0.02:
        print("  Verdict: roughly a wash with supervised baseline. Pool diversity "
              "and balance fixed the v1 collapse, but supervised already saturated.")
    else:
        print("  Verdict: self-training still degrades vs supervised. Even with "
              "the diversified pool and percentile selection, the supervised "
              "baseline is the right ceiling for this dataset size.")


if __name__ == "__main__":
    main()
