"""
evaluate_three_way.py
=====================
Phase 4 of the new methodology. Train + evaluate three approaches on the
SAME 100-pair LLM-labeled held-out test set:

  (1) SBERT-only baseline
        Threshold-only predictor on `embedding_similarity`. Threshold tuned
        on training fold by F1; this is the fairest single-feature baseline.

  (2) Weak-supervised, leakage-safe (≈ Variant B partition)
        LogReg + GBM trained on the existing weak-labeled dataset
        (`data/proccessed again/processed/ml_ready_dataset.parquet`) using
        only the 6 leakage-safe features. We exclude `weighted_skill_score`
        and `title_similarity` because the weak label rule
            score = 0.7·wss + 0.3·title_sim
        derives from those two columns; including them would let the model
        recover the rule.

  (3) LLM-supervised reranker (the new approach)
        LogReg + GBM trained on 400 LLM-labeled pairs, all 8 features.
        No leakage concern — the labels come from an external source.

All three evaluated on the same 100-pair held-out test set drawn from the
500 LLM-labeled gold standard. Stratified 80/20 split (preserves the
22.4% positive rate in both halves).

Metrics reported:
  - Accuracy, Precision, Recall, F1
  - Precision@K (top-K of the test set ranked by model score)

Outputs:
  outputs/three_way/comparison_table.csv
  outputs/three_way/comparison_chart.png
  models/llm_supervised/{logreg,gbm,scaler}.pkl
  models/weak_safe/{logreg,gbm,scaler}.pkl

Run:
    python "src/evaluate_three_way.py"
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

POOL_PARQUET = PROJECT_ROOT / "data" / "proccessed again" / "processed" / "pair_features_diversified.parquet"
WEAK_TRAIN   = PROJECT_ROOT / "data" / "proccessed again" / "processed" / "ml_ready_dataset.parquet"
GOLD_CSV     = PROJECT_ROOT / "data" / "proccessed again" / "gold_labeling" / "gold_labels.csv"

OUT_DIR    = PROJECT_ROOT / "outputs" / "three_way"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = PROJECT_ROOT / "models"


# ---------------------------------------------------------------------------
# Feature config
# ---------------------------------------------------------------------------
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

# Leakage-safe subset for the weak-supervised baseline. Drops the two
# features that go into the weak label formula (0.7*wss + 0.3*title_sim).
WEAK_SAFE_FEATURES = [
    "embedding_similarity",
    "tfidf_similarity",
    "skill_overlap",
    "num_missing_skills",
    "avg_missing_skill_importance",
    "years_of_experience",
]

RANDOM_SEED = 42
TEST_FRACTION = 0.20   # 100 / 500
SBERT_THRESHOLD_GRID = np.arange(0.30, 0.86, 0.02)
PRECISION_AT_K_VALUES = (5, 10, 20)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def metrics_row(name: str, threshold, y_true, y_pred, scores=None) -> dict:
    row = {
        "Approach": name,
        "Threshold": threshold,
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if scores is not None:
        for k in PRECISION_AT_K_VALUES:
            row[f"P@{k}"] = round(precision_at_k(scores, y_true, k), 4)
    return row


def precision_at_k(scores: np.ndarray, y_true: np.ndarray, k: int) -> float:
    """Of the top-K scoring items in `scores`, what fraction are positive in y_true?"""
    if len(scores) == 0:
        return 0.0
    k = min(k, len(scores))
    top_idx = np.argsort(-scores)[:k]
    return float(np.sum(y_true[top_idx] == 1)) / k


def find_best_threshold_cv(model, X, y, thresholds, cv=5, seed=RANDOM_SEED) -> tuple[float, float]:
    """5-fold CV out-of-fold predictions; pick threshold that maximises F1."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(X, y):
        m = type(model)(**model.get_params())
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        f = f1_score(y, (oof >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return round(best_t, 2), round(best_f1, 4)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print(" Phase 4 — three-way comparison (SBERT / weak-safe / LLM-supervised)")
    print("=" * 70)

    if not GOLD_CSV.exists():
        raise FileNotFoundError(f"{GOLD_CSV} missing — run combine_llm_labels.py first")
    if not POOL_PARQUET.exists():
        raise FileNotFoundError(f"{POOL_PARQUET} missing — run build_diversified_pool.py first")
    if not WEAK_TRAIN.exists():
        raise FileNotFoundError(f"{WEAK_TRAIN} missing — run make_ml_ready.py first")

    print(f"\n[load] {GOLD_CSV.name}")
    gold = pd.read_csv(GOLD_CSV)
    print(f"       {len(gold):,} gold-labeled pairs")
    print(f"       majority pos: {gold['majority_label'].mean() * 100:.1f}%")

    print(f"[load] {POOL_PARQUET.name}")
    pool = pd.read_parquet(POOL_PARQUET)
    print(f"       {len(pool):,} pairs in feature pool")

    # gold_labels.csv carries 3 feature columns (embedding_similarity,
    # skill_overlap, weighted_skill_score) from gold_pairs_master.csv. Strip
    # those before merging so we get the canonical values from the pool
    # without _x / _y suffix headaches.
    gold = gold[[c for c in gold.columns if c not in ALL_FEATURES]]

    # Join gold pairs to their feature rows
    gold_full = gold.merge(
        pool[["job_id", "resume_id"] + ALL_FEATURES],
        on=["job_id", "resume_id"],
        how="left",
    )
    n_missing = gold_full[ALL_FEATURES[0]].isna().sum()
    if n_missing:
        print(f"[warn] {n_missing} gold pairs had no feature row in pool; dropping")
        gold_full = gold_full.dropna(subset=[ALL_FEATURES[0]])
    gold_full = gold_full.reset_index(drop=True)
    gold_full["majority_label"] = gold_full["majority_label"].astype(int)

    # Stratified 80/20 split (preserves positive rate in both halves)
    train_df, test_df = train_test_split(
        gold_full,
        test_size=TEST_FRACTION,
        stratify=gold_full["majority_label"],
        random_state=RANDOM_SEED,
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
    print(f"\n[split] train: {len(train_df)}  pos={int(train_df['majority_label'].sum())}  "
          f"neg={int((1 - train_df['majority_label']).sum())}")
    print(f"[split] test:  {len(test_df)}   pos={int(test_df['majority_label'].sum())}  "
          f"neg={int((1 - test_df['majority_label']).sum())}")

    y_test = test_df["majority_label"].values
    X_test_full = test_df[ALL_FEATURES].values

    results: list[dict] = []

    # -----------------------------------------------------------------
    # (1) SBERT-only baseline
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" (1) SBERT-only baseline (threshold on embedding_similarity)")
    print("=" * 70)

    train_sbert = train_df["embedding_similarity"].values
    y_train     = train_df["majority_label"].values

    # Tune threshold on train by F1
    best_t, best_f1 = 0.5, 0.0
    for t in SBERT_THRESHOLD_GRID:
        preds = (train_sbert >= t).astype(int)
        f = f1_score(y_train, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    print(f"  best threshold on train: {best_t:.2f}  F1_train={best_f1:.4f}")

    test_scores = test_df["embedding_similarity"].values
    y_pred = (test_scores >= best_t).astype(int)
    row = metrics_row("SBERT only", round(float(best_t), 2),
                      y_test, y_pred, scores=test_scores)
    results.append(row)
    print(f"  test F1={row['F1']}  acc={row['Accuracy']}  "
          f"prec={row['Precision']}  rec={row['Recall']}")

    # -----------------------------------------------------------------
    # (2) Weak-supervised, leakage-safe (≈ Variant B)
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" (2) Weak-supervised, leakage-safe (LogReg + GBM, 6 features)")
    print("=" * 70)

    weak_df = pd.read_parquet(WEAK_TRAIN)
    print(f"  weak training set: {len(weak_df):,} rows  "
          f"pos={int((weak_df['label']==1).sum())}  neg={int((weak_df['label']==0).sum())}")

    X_weak_train = weak_df[WEAK_SAFE_FEATURES].values
    y_weak_train = weak_df["label"].astype(int).values
    scaler_weak = StandardScaler().fit(X_weak_train)
    X_weak_train_s = scaler_weak.transform(X_weak_train)

    # Test features for weak path: same 6 columns from gold test set
    X_test_weak = scaler_weak.transform(test_df[WEAK_SAFE_FEATURES].values)

    weak_models_dir = MODELS_DIR / "weak_safe"
    weak_models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler_weak, weak_models_dir / "scaler.pkl")

    for ModelCls, name in [(LogisticRegression, "LogReg"), (GradientBoostingClassifier, "GBM")]:
        kwargs = {"random_state": RANDOM_SEED}
        if ModelCls is LogisticRegression:
            kwargs.update(solver="liblinear", class_weight="balanced")
        elif ModelCls is GradientBoostingClassifier:
            pass  # no class_weight; we'll rely on threshold tuning

        # Tune threshold via 5-fold CV on the weak training set
        m_for_cv = ModelCls(**kwargs)
        best_t, best_f1 = find_best_threshold_cv(
            m_for_cv, X_weak_train_s, y_weak_train,
            np.arange(0.10, 0.91, 0.02),
        )
        # Final fit on full train, score test
        final = ModelCls(**kwargs)
        final.fit(X_weak_train_s, y_weak_train)
        joblib.dump(final, weak_models_dir / f"{name.lower()}.pkl")

        scores_test = final.predict_proba(X_test_weak)[:, 1]
        y_pred = (scores_test >= best_t).astype(int)
        row = metrics_row(f"Weak-safe {name}", round(float(best_t), 2),
                          y_test, y_pred, scores=scores_test)
        results.append(row)
        print(f"  {name}: best_t={best_t:.2f}  CV F1={best_f1:.4f}  "
              f"test F1={row['F1']}")

    # -----------------------------------------------------------------
    # (3) LLM-supervised reranker
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" (3) LLM-supervised reranker (LogReg + GBM, 8 features)")
    print("=" * 70)

    X_llm_train = train_df[ALL_FEATURES].values
    y_llm_train = train_df["majority_label"].values
    scaler_llm = StandardScaler().fit(X_llm_train)
    X_llm_train_s = scaler_llm.transform(X_llm_train)
    X_llm_test_s  = scaler_llm.transform(test_df[ALL_FEATURES].values)

    llm_models_dir = MODELS_DIR / "llm_supervised"
    llm_models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler_llm, llm_models_dir / "scaler.pkl")

    # With ~90 positives in 400 train, stratified CV needs care; use 5 folds.
    for ModelCls, name in [(LogisticRegression, "LogReg"), (GradientBoostingClassifier, "GBM")]:
        kwargs = {"random_state": RANDOM_SEED}
        if ModelCls is LogisticRegression:
            kwargs.update(solver="liblinear", class_weight="balanced")

        m_for_cv = ModelCls(**kwargs)
        best_t, best_f1 = find_best_threshold_cv(
            m_for_cv, X_llm_train_s, y_llm_train,
            np.arange(0.10, 0.91, 0.02),
        )
        final = ModelCls(**kwargs)
        final.fit(X_llm_train_s, y_llm_train)
        joblib.dump(final, llm_models_dir / f"{name.lower()}.pkl")

        scores_test = final.predict_proba(X_llm_test_s)[:, 1]
        y_pred = (scores_test >= best_t).astype(int)
        row = metrics_row(f"LLM-supervised {name}", round(float(best_t), 2),
                          y_test, y_pred, scores=scores_test)
        results.append(row)
        print(f"  {name}: best_t={best_t:.2f}  CV F1={best_f1:.4f}  "
              f"test F1={row['F1']}")

    # -----------------------------------------------------------------
    # Comparison table
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" Comparison table (sorted by F1)")
    print("=" * 70)

    res_df = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
    out_csv = OUT_DIR / "comparison_table.csv"
    res_df.to_csv(out_csv, index=False)
    print(res_df.to_string(index=False))
    print(f"\n[write] {out_csv}")

    # -----------------------------------------------------------------
    # Plot — grouped bars by approach for F1, P@5, P@10
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.5))
    metric_cols = ["F1", "P@5", "P@10"]
    n_metrics = len(metric_cols)
    x = np.arange(len(res_df))
    width = 0.8 / n_metrics
    palette = ["#3b82f6", "#10b981", "#f59e0b"]

    for i, metric in enumerate(metric_cols):
        if metric not in res_df.columns:
            continue
        vals = res_df[metric].values
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                      label=metric, color=palette[i], edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(res_df["Approach"], rotation=15, ha="right")
    ax.set_ylabel("Score on held-out test set")
    ax.set_ylim(0, 1.05)
    ax.set_title("Three-way comparison on 100 LLM-labeled test pairs",
                 fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_png = OUT_DIR / "comparison_chart.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[write] {out_png}")

    # -----------------------------------------------------------------
    # Best-model confusion matrix
    # -----------------------------------------------------------------
    best_name = res_df.iloc[0]["Approach"]
    print(f"\n[best] {best_name}  (F1 = {res_df.iloc[0]['F1']})")
    print()

    # -----------------------------------------------------------------
    # Save split metadata so the run is reproducible
    # -----------------------------------------------------------------
    split_meta = {
        "random_seed": RANDOM_SEED,
        "test_fraction": TEST_FRACTION,
        "n_train": len(train_df),
        "n_test":  len(test_df),
        "train_pos": int(train_df["majority_label"].sum()),
        "test_pos":  int(test_df["majority_label"].sum()),
        "all_features":       ALL_FEATURES,
        "weak_safe_features": WEAK_SAFE_FEATURES,
        "test_row_ids": test_df["row_id"].tolist() if "row_id" in test_df.columns else [],
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(split_meta, indent=2))
    print(f"[write] {OUT_DIR / 'run_metadata.json'}")


if __name__ == "__main__":
    main()
