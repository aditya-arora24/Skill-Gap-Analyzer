"""
run_self_training_experiment.py
================================
Single-iteration self-training experiment on top of the LLM-supervised
LogReg baseline (F1 = 0.7037 on the 100-pair test set).

Procedure:
  1. Load the supervised LogReg + scaler from models/llm_supervised/.
  2. Score every row of ml_ready_dataset.parquet with that model.
  3. Pseudo-label rows where:
        prob >= 0.80  ->  pseudo-label = 1
        prob <= 0.20  ->  pseudo-label = 0
        else            discard
  4. Combine the pseudo-labeled rows with the original 400 LLM-labeled
     training pairs (deduped on (job_id, resume_id) -- LLM labels win).
  5. Re-train a fresh Logistic Regression on the combined set, with
     5-fold CV threshold tuning.
  6. Evaluate on the SAME 100-pair held-out test set used by Phase 4
     (verified by reproducing the random_state=42 stratified split).

Diagnostics emitted:
  (a) Probability histogram on ml_ready_dataset before pseudo-labeling.
  (b) Confusion-matrix diff: which test pairs flipped prediction.
  (c) Sanity check: both models evaluated on the original 400 LLM labels.

Inputs (read-only, never modified):
  models/llm_supervised/{scaler,logreg}.pkl
  data/proccessed again/gold_labeling/gold_labels.csv
  data/proccessed again/processed/pair_features_diversified.parquet
  data/proccessed again/processed/ml_ready_dataset.parquet
  outputs/three_way/run_metadata.json   (split sanity check)

Outputs (NEW paths only, nothing existing is overwritten):
  models/llm_self_trained/{scaler,logreg}.pkl
  outputs/self_training/comparison.csv
  outputs/self_training/comparison.png
  outputs/self_training/pseudo_label_distribution.png
  outputs/self_training/test_set_diff.csv
  outputs/self_training/run_metadata.json

Run:
    python "src/run_self_training_experiment.py"
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
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUPERVISED_MODEL_DIR = PROJECT_ROOT / "models" / "llm_supervised"
GOLD_CSV             = PROJECT_ROOT / "data" / "proccessed again" / "gold_labeling" / "gold_labels.csv"
POOL_PARQUET         = PROJECT_ROOT / "data" / "proccessed again" / "processed" / "pair_features_diversified.parquet"
ML_READY_PARQUET     = PROJECT_ROOT / "data" / "proccessed again" / "processed" / "ml_ready_dataset.parquet"
PHASE4_META          = PROJECT_ROOT / "outputs" / "three_way" / "run_metadata.json"

NEW_MODEL_DIR = PROJECT_ROOT / "models" / "llm_self_trained"
OUT_DIR       = PROJECT_ROOT / "outputs" / "self_training"
NEW_MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config (matches Phase 4 exactly so the test split is reproducible)
# ---------------------------------------------------------------------------
RANDOM_SEED   = 42
TEST_FRACTION = 0.20
PSEUDO_HIGH   = 0.80
PSEUDO_LOW    = 0.20
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
def precision_at_k(scores: np.ndarray, y_true: np.ndarray, k: int) -> float:
    if len(scores) == 0:
        return 0.0
    k = min(k, len(scores))
    top_idx = np.argsort(-scores)[:k]
    return float(np.sum(y_true[top_idx] == 1)) / k


def metrics_row(name: str, threshold: float, y_true, y_pred, scores) -> dict:
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


def find_best_threshold_cv(model_kwargs: dict, X, y, thresholds, cv=5,
                           seed=RANDOM_SEED) -> tuple[float, float]:
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
def main() -> None:
    print("=" * 70)
    print(" Self-training experiment (1 iteration)")
    print("=" * 70)

    # -------------------------------------------------------------- #
    # 1. Load supervised baseline (read-only)                        #
    # -------------------------------------------------------------- #
    print("\n[1] loading supervised LogReg baseline")
    if not (SUPERVISED_MODEL_DIR / "scaler.pkl").exists():
        raise FileNotFoundError(f"missing {SUPERVISED_MODEL_DIR}/scaler.pkl")
    if not (SUPERVISED_MODEL_DIR / "logreg.pkl").exists():
        raise FileNotFoundError(f"missing {SUPERVISED_MODEL_DIR}/logreg.pkl")
    old_scaler = joblib.load(SUPERVISED_MODEL_DIR / "scaler.pkl")
    old_model  = joblib.load(SUPERVISED_MODEL_DIR / "logreg.pkl")

    # -------------------------------------------------------------- #
    # 2. Reproduce the Phase 4 train/test split                      #
    # -------------------------------------------------------------- #
    print("\n[2] reproducing Phase 4 train/test split")
    gold = pd.read_csv(GOLD_CSV)
    pool = pd.read_parquet(POOL_PARQUET)

    # Strip feature columns from gold (they were carried from the master CSV
    # and would conflict with a clean pool merge).
    gold = gold[[c for c in gold.columns if c not in ALL_FEATURES]]
    gold_full = gold.merge(
        pool[["job_id", "resume_id"] + ALL_FEATURES],
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

    # Sanity: test split must match Phase 4
    if PHASE4_META.exists():
        phase4 = json.loads(PHASE4_META.read_text())
        expected_ids = set(phase4.get("test_row_ids", []))
        actual_ids   = set(test_df["row_id"].astype(int).tolist())
        if expected_ids and expected_ids != actual_ids:
            print(f"ERROR: test split differs from Phase 4. "
                  f"intersection: {len(expected_ids & actual_ids)} / {len(expected_ids)}",
                  flush=True)
            raise SystemExit(1)
        elif expected_ids:
            print(f"    [OK] test row_ids match Phase 4 metadata "
                  f"({len(actual_ids)} pairs)")
    else:
        print(f"    [warn] {PHASE4_META} not found -- can't verify split match")

    y_train = train_df["majority_label"].values
    y_test  = test_df["majority_label"].values
    X_train_full = train_df[ALL_FEATURES].values
    X_test_full  = test_df[ALL_FEATURES].values

    # -------------------------------------------------------------- #
    # 3. Score ml_ready_dataset.parquet with supervised model        #
    # -------------------------------------------------------------- #
    print("\n[3] scoring ml_ready_dataset.parquet with supervised model")
    ml_ready = pd.read_parquet(ML_READY_PARQUET)
    print(f"    {len(ml_ready):,} rows in unlabeled pool")
    X_ml = ml_ready[ALL_FEATURES].values
    X_ml_scaled = old_scaler.transform(X_ml)
    probs_ml = old_model.predict_proba(X_ml_scaled)[:, 1]

    # ---- Diagnostic (a): probability histogram ----
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(probs_ml, bins=40, color="steelblue", edgecolor="white")
    ax.axvline(PSEUDO_LOW,  color="crimson", linestyle="--", label=f"low={PSEUDO_LOW}")
    ax.axvline(PSEUDO_HIGH, color="seagreen", linestyle="--", label=f"high={PSEUDO_HIGH}")
    ax.set_xlabel("P(positive | features) from supervised LogReg")
    ax.set_ylabel("Count")
    ax.set_title("Probability distribution on ml_ready_dataset.parquet "
                 f"(N={len(probs_ml):,})", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    hist_path = OUT_DIR / "pseudo_label_distribution.png"
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"    histogram -> {hist_path}")

    # Bin counts for the report
    n_below = int((probs_ml <= PSEUDO_LOW).sum())
    n_above = int((probs_ml >= PSEUDO_HIGH).sum())
    n_mid   = len(probs_ml) - n_below - n_above
    print(f"    probability bins:")
    print(f"      <= {PSEUDO_LOW}: {n_below:>5,}  ({100*n_below/len(probs_ml):5.1f}%)")
    print(f"      mid     : {n_mid:>5,}  ({100*n_mid/len(probs_ml):5.1f}%)  (discarded)")
    print(f"      >= {PSEUDO_HIGH}: {n_above:>5,}  ({100*n_above/len(probs_ml):5.1f}%)")

    # -------------------------------------------------------------- #
    # 4. Apply confidence thresholds                                  #
    # -------------------------------------------------------------- #
    print("\n[4] generating pseudo-labels")
    pseudo_pos_mask = probs_ml >= PSEUDO_HIGH
    pseudo_neg_mask = probs_ml <= PSEUDO_LOW
    pseudo_keep_mask = pseudo_pos_mask | pseudo_neg_mask

    pseudo_df = ml_ready[pseudo_keep_mask].copy()
    pseudo_df["pseudo_label"] = pseudo_pos_mask[pseudo_keep_mask].astype(int)
    pseudo_df["pseudo_prob"]  = probs_ml[pseudo_keep_mask]
    print(f"    pseudo-labels generated: {len(pseudo_df):,}")
    print(f"      positive: {int(pseudo_df['pseudo_label'].sum()):>5,}")
    print(f"      negative: {int((1 - pseudo_df['pseudo_label']).sum()):>5,}")
    print(f"      class balance: "
          f"{100 * pseudo_df['pseudo_label'].mean():.1f}% positive")

    # -------------------------------------------------------------- #
    # 5. Combine with original 400 LLM-labeled training pairs        #
    # -------------------------------------------------------------- #
    print("\n[5] combining with 400 LLM-labeled training pairs")

    # Dedupe: if a pair is in BOTH the gold train set and the pseudo set,
    # keep the gold version (LLM judgment beats self-prediction).
    train_keys = set(zip(
        train_df["job_id"].astype(int), train_df["resume_id"].astype(int)
    ))
    if len(pseudo_df) > 0:
        pseudo_keys = list(zip(
            pseudo_df["job_id"].astype(int), pseudo_df["resume_id"].astype(int)
        ))
        keep_mask = np.array([k not in train_keys for k in pseudo_keys])
        n_overlap = int((~keep_mask).sum())
        pseudo_df = pseudo_df[keep_mask].reset_index(drop=True)
        if n_overlap:
            print(f"    [dedupe] removed {n_overlap} pseudo-labels that "
                  f"overlapped the gold train set")

    combined_X = np.vstack([
        X_train_full,
        pseudo_df[ALL_FEATURES].values,
    ])
    combined_y = np.concatenate([
        y_train,
        pseudo_df["pseudo_label"].astype(int).values,
    ])
    print(f"    combined training set: {len(combined_y):,} rows "
          f"({len(y_train)} gold + {len(pseudo_df):,} pseudo)")
    print(f"    overall class balance: "
          f"{100 * combined_y.mean():.1f}% positive")

    # -------------------------------------------------------------- #
    # 6. Re-train LogReg on combined set                             #
    # -------------------------------------------------------------- #
    print("\n[6] re-training LogReg on combined set")
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
    print(f"    new threshold (CV): {new_t}  (CV F1 = {new_cv_f1:.4f})")

    joblib.dump(new_scaler, NEW_MODEL_DIR / "scaler.pkl")
    joblib.dump(new_model,  NEW_MODEL_DIR / "logreg.pkl")
    print(f"    saved -> {NEW_MODEL_DIR / 'scaler.pkl'}")
    print(f"    saved -> {NEW_MODEL_DIR / 'logreg.pkl'}")

    # -------------------------------------------------------------- #
    # 7. Evaluate both models on the 100-pair test set               #
    # -------------------------------------------------------------- #
    print("\n[7] evaluating on the 100-pair held-out test set")

    # Old (supervised) model uses its existing scaler and the threshold we
    # re-tune via CV on the original 400 train pairs (same procedure as Phase 4).
    X_train_old_s = old_scaler.transform(X_train_full)
    old_t, old_cv_f1 = find_best_threshold_cv(
        model_kwargs, X_train_old_s, y_train, THRESHOLD_GRID,
    )
    print(f"    old (supervised) threshold (CV on 400): {old_t}  CV F1={old_cv_f1:.4f}")

    X_test_old_s = old_scaler.transform(X_test_full)
    X_test_new_s = new_scaler.transform(X_test_full)

    probs_old = old_model.predict_proba(X_test_old_s)[:, 1]
    probs_new = new_model.predict_proba(X_test_new_s)[:, 1]
    y_pred_old = (probs_old >= old_t).astype(int)
    y_pred_new = (probs_new >= new_t).astype(int)

    old_metrics = metrics_row("Supervised (baseline)",   old_t, y_test, y_pred_old, probs_old)
    new_metrics = metrics_row("Self-trained (1 iter)",   new_t, y_test, y_pred_new, probs_new)

    # Print side-by-side comparison
    print("\n" + "=" * 70)
    print(" Comparison on 100-pair held-out test set")
    print("=" * 70)
    cmp_df = pd.DataFrame([old_metrics, new_metrics])
    print(cmp_df.to_string(index=False))

    cmp_df.to_csv(OUT_DIR / "comparison.csv", index=False)
    print(f"\n[write] {OUT_DIR / 'comparison.csv'}")

    # ---- Diagnostic (b): test-set diff ----
    diff = pd.DataFrame({
        "row_id":     test_df["row_id"].astype(int).values if "row_id" in test_df.columns else range(len(test_df)),
        "true":       y_test,
        "old_pred":   y_pred_old,
        "new_pred":   y_pred_new,
        "old_prob":   np.round(probs_old, 4),
        "new_prob":   np.round(probs_new, 4),
    })
    diff["flipped"]      = diff["old_pred"] != diff["new_pred"]
    diff["old_correct"]  = diff["old_pred"] == diff["true"]
    diff["new_correct"]  = diff["new_pred"] == diff["true"]
    diff["delta"]        = diff["new_correct"].astype(int) - diff["old_correct"].astype(int)

    diff.to_csv(OUT_DIR / "test_set_diff.csv", index=False)

    print("\n--- Confusion-matrix diff ---")
    n_flipped = int(diff["flipped"].sum())
    n_pos_to_neg = int(((diff["old_pred"] == 1) & (diff["new_pred"] == 0)).sum())
    n_neg_to_pos = int(((diff["old_pred"] == 0) & (diff["new_pred"] == 1)).sum())
    n_now_correct = int((diff["delta"] > 0).sum())
    n_now_wrong   = int((diff["delta"] < 0).sum())
    print(f"  flipped predictions: {n_flipped} / 100")
    print(f"    pos -> neg: {n_pos_to_neg}")
    print(f"    neg -> pos: {n_neg_to_pos}")
    print(f"  flips that improved prediction: {n_now_correct}")
    print(f"  flips that worsened prediction: {n_now_wrong}")
    print(f"[write] {OUT_DIR / 'test_set_diff.csv'}")

    # ---- Diagnostic (c): sanity check on original 400 train labels ----
    print("\n--- Sanity check: both models on original 400 train labels ---")
    X_train_new_s = new_scaler.transform(X_train_full)
    train_pred_old = (old_model.predict_proba(X_train_old_s)[:, 1] >= old_t).astype(int)
    train_pred_new = (new_model.predict_proba(X_train_new_s)[:, 1] >= new_t).astype(int)
    train_acc_old = accuracy_score(y_train, train_pred_old)
    train_acc_new = accuracy_score(y_train, train_pred_new)
    train_f1_old  = f1_score(y_train, train_pred_old, zero_division=0)
    train_f1_new  = f1_score(y_train, train_pred_new, zero_division=0)
    print(f"  supervised   : accuracy={train_acc_old:.4f}  F1={train_f1_old:.4f}")
    print(f"  self-trained : accuracy={train_acc_new:.4f}  F1={train_f1_new:.4f}")
    if train_acc_new < train_acc_old - 0.05:
        print(f"  [warn] self-trained accuracy on original labels DROPPED by "
              f">5pp (old={train_acc_old:.3f}, new={train_acc_new:.3f}). "
              f"Self-training may have hurt baseline behavior.")
    else:
        print(f"  [OK] self-trained model still classifies original labels at "
              f"comparable performance.")

    # ---- Plot: grouped bars of F1 / P@5 / P@10 / P@20 ----
    metric_cols = ["F1", "P@5", "P@10", "P@20", "Accuracy", "Precision", "Recall"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(cmp_df))
    width = 0.10
    palette = plt.cm.tab10.colors
    for i, m in enumerate(metric_cols):
        if m not in cmp_df.columns:
            continue
        vals = cmp_df[m].values
        bars = ax.bar(x + i * width - width * (len(metric_cols) - 1) / 2, vals,
                      width, label=m, color=palette[i % len(palette)],
                      edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(cmp_df["Approach"], rotation=0)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Score on held-out test (100 pairs)")
    ax.set_title("Self-training (1 iteration) vs. supervised baseline",
                 fontweight="bold")
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "comparison.png", dpi=150)
    plt.close()
    print(f"[write] {OUT_DIR / 'comparison.png'}")

    # ---- Run metadata for reproducibility ----
    meta = {
        "random_seed":            RANDOM_SEED,
        "test_fraction":          TEST_FRACTION,
        "pseudo_high_threshold":  PSEUDO_HIGH,
        "pseudo_low_threshold":   PSEUDO_LOW,
        "n_train_gold":           int(len(y_train)),
        "n_pseudo_kept":          int(len(pseudo_df)),
        "n_combined":             int(len(combined_y)),
        "combined_pos_rate":      float(combined_y.mean()),
        "old_threshold":          old_t,
        "old_cv_f1":              old_cv_f1,
        "new_threshold":          new_t,
        "new_cv_f1":              new_cv_f1,
        "old_test_metrics":       old_metrics,
        "new_test_metrics":       new_metrics,
        "n_flipped":              n_flipped,
        "n_now_correct":          n_now_correct,
        "n_now_wrong":            n_now_wrong,
        "train_accuracy_old":     float(train_acc_old),
        "train_accuracy_new":     float(train_acc_new),
        "train_f1_old":           float(train_f1_old),
        "train_f1_new":           float(train_f1_new),
        "ml_ready_n_rows":        int(len(ml_ready)),
        "ml_ready_pseudo_high":   int(n_above),
        "ml_ready_pseudo_low":    int(n_below),
        "ml_ready_pseudo_mid":    int(n_mid),
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[write] {OUT_DIR / 'run_metadata.json'}")

    # ---- Bottom-line summary ----
    print("\n" + "=" * 70)
    print(" Bottom line")
    print("=" * 70)
    delta_f1  = new_metrics["F1"]  - old_metrics["F1"]
    delta_acc = new_metrics["Accuracy"] - old_metrics["Accuracy"]
    print(f"  Supervised     F1 = {old_metrics['F1']:.4f}   "
          f"acc = {old_metrics['Accuracy']:.4f}")
    print(f"  Self-trained   F1 = {new_metrics['F1']:.4f}   "
          f"acc = {new_metrics['Accuracy']:.4f}")
    print(f"  Delta          F1 = {delta_f1:+.4f}     "
          f"acc = {delta_acc:+.4f}")

    if delta_f1 > 0.02:
        verdict = ("self-training added measurable value -- "
                   "expand to more iterations or report as final model")
    elif delta_f1 < -0.02:
        verdict = ("self-training degraded test F1 -- "
                   "publishable negative result; supervised baseline stands")
    else:
        verdict = ("self-training was approximately a wash -- "
                   "supervised baseline already saturated; ship it as-is")
    print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
