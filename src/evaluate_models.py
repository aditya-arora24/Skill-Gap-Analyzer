"""
evaluate_models.py
==================
Per-variant, leakage-aware evaluation pipeline (v2).

Reads a variant config (JSON) and the per-variant data files written by
build_final_dataset.py phase 2:

  data/processed/variant_<X>/ml_ready_dataset.parquet
  data/processed/variant_<X>/gold_standard_final.parquet
  configs/variant_<X>.json   (training_features, holdout_features, label_weights)

Trains LogReg + GBM, tunes thresholds via 5-fold CV on training data,
evaluates on the gold standard. Saves models, scaler, feature-importance
plot, metrics CSV, and confusion matrix all under per-variant directories.

Usage
-----
    python evaluate_models.py --config ../configs/variant_A.json
    python evaluate_models.py --config ../configs/variant_B.json
    python evaluate_models.py --config ../configs/variant_C.json
"""

from __future__ import annotations

import argparse
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

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "processed"
MODELS_DIR   = PROJECT_ROOT / "models"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_best_threshold(model, X, y, thresholds, cv=5):
    oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        f = f1_score(y, (oof >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return round(best_t, 2), round(best_f1, 4), oof


def metrics_row(name, threshold, y_true, y_pred):
    return {
        "Model":     name,
        "Threshold": threshold,
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


# ---------------------------------------------------------------------------
# Main per-variant evaluator
# ---------------------------------------------------------------------------
def evaluate_variant(config_path: Path) -> pd.DataFrame:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    name      = config["variant_name"]
    features  = config["training_features"]
    holdout   = config.get("holdout_features", [])
    weights   = config["label_weights"]
    rng_seed  = config.get("random_state", 42)

    variant_dir = DATA_DIR / f"variant_{name}"
    train_path  = variant_dir / "ml_ready_dataset.parquet"
    gold_path   = variant_dir / "gold_standard_final.parquet"
    out_dir     = OUTPUTS_DIR / f"variant_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir  = MODELS_DIR / f"variant_{name}"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f" Evaluating variant {name}")
    print("=" * 70)
    print(f"  training_features ({len(features)}): {features}")
    print(f"  holdout_features  ({len(holdout)}):  {holdout}")

    # ---- Load ----
    train_df = pd.read_parquet(train_path)
    gold_df  = pd.read_parquet(gold_path)
    print(f"\n[Train] {train_df.shape}  pos/neg = "
          f"{int((train_df['label']==1).sum())}/{int((train_df['label']==0).sum())}")
    print(f"[Gold]  {gold_df.shape}  pos/neg = "
          f"{int((gold_df['gold_label']==1).sum())}/{int((gold_df['gold_label']==0).sum())}")

    for col in features:
        assert col in train_df.columns, f"missing {col} in train"
        assert col in gold_df.columns,  f"missing {col} in gold"
    assert "label"      in train_df.columns
    assert "gold_label" in gold_df.columns
    assert train_df[features + ["label"]].isnull().sum().sum() == 0
    assert gold_df[features + ["gold_label"]].isnull().sum().sum() == 0

    # ---- Prepare ----
    X_train = train_df[features].values
    y_train = train_df["label"].astype(int).values
    X_gold  = gold_df[features].values
    y_gold  = gold_df["gold_label"].astype(int).values

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_gold_s  = scaler.transform(X_gold)

    # ---- Threshold tuning via OOF ----
    thresholds = np.arange(0.10, 0.91, 0.05)
    print("\n[CV] tuning LogReg threshold...")
    logreg = LogisticRegression(solver="liblinear", random_state=rng_seed)
    t_lr, f1_lr, _ = find_best_threshold(logreg, X_train_s, y_train, thresholds)
    print(f"     LogReg best threshold={t_lr}  CV F1={f1_lr}")

    print("[CV] tuning GBM threshold...")
    gbm = GradientBoostingClassifier(random_state=rng_seed)
    t_gbm, f1_gbm, _ = find_best_threshold(gbm, X_train_s, y_train, thresholds)
    print(f"     GBM    best threshold={t_gbm}  CV F1={f1_gbm}")

    # ---- Fit final models ----
    logreg.fit(X_train_s, y_train)
    gbm.fit(X_train_s, y_train)
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(logreg, models_dir / "logreg_model.pkl")
    joblib.dump(gbm,    models_dir / "gbm_model.pkl")

    # ---- Apply tuned thresholds to gold ----
    p_lr  = logreg.predict_proba(X_gold_s)[:, 1]
    p_gbm = gbm.predict_proba(X_gold_s)[:, 1]
    y_lr  = (p_lr  >= t_lr).astype(int)
    y_gb  = (p_gbm >= t_gbm).astype(int)

    # ---- Baselines (held-out features as references; available regardless of variant) ----
    baselines = []
    if "embedding_similarity" in gold_df.columns:
        y = (gold_df["embedding_similarity"].values >= 0.65).astype(int)
        baselines.append(metrics_row("SBERT Only", 0.65, y_gold, y))
    if "tfidf_similarity" in gold_df.columns:
        y = (gold_df["tfidf_similarity"].values >= 0.10).astype(int)
        baselines.append(metrics_row("TF-IDF Only", 0.10, y_gold, y))
    if "weighted_skill_score" in gold_df.columns:
        y = (gold_df["weighted_skill_score"].values >= 0.30).astype(int)
        baselines.append(metrics_row("Weighted Skills", 0.30, y_gold, y))
    if "title_similarity" in gold_df.columns:
        y = (gold_df["title_similarity"].values >= 0.30).astype(int)
        baselines.append(metrics_row("Title Similarity", 0.30, y_gold, y))

    # Composite-label-as-classifier sanity check
    score = np.zeros(len(gold_df), dtype=np.float32)
    for col, w in weights.items():
        score += w * gold_df[col].astype(np.float32).clip(lower=0).values
    # Use the high-quantile threshold on the gold's composite score
    high = float(np.quantile(score, config["label_quantiles"]["high_pct"]))
    y_lab = (score >= high).astype(int)
    label_baseline = metrics_row(f"Composite Label", round(high, 3), y_gold, y_lab)

    # ---- Assemble results ----
    rows = [
        {**metrics_row("LogReg (tuned)", t_lr,  y_gold, y_lr ), "variant": name},
        {**metrics_row("GBM (tuned)",    t_gbm, y_gold, y_gb ), "variant": name},
    ]
    for b in baselines:
        rows.append({**b, "variant": name})
    rows.append({**label_baseline, "variant": name})

    df = pd.DataFrame(rows)
    df = df[["variant", "Model", "Threshold", "Accuracy", "Precision", "Recall", "F1"]]
    df = df.sort_values("F1", ascending=False).reset_index(drop=True)

    # ---- Print ----
    print("\n" + "=" * 70)
    print(f"  Variant {name} -- gold-set results (sorted by F1)")
    print("=" * 70)
    print(df.to_string(index=False))

    # ---- Feature importance plot (GBM) ----
    importances = gbm.feature_importances_
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 0.5 + 0.4 * len(features)))
    bars = ax.barh([features[i] for i in order], importances[order],
                   color="steelblue", edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title(f"GBM Feature Importance — Variant {name}", fontweight="bold")
    ax.set_xlim(0, max(importances) * 1.2 if max(importances) > 0 else 1)
    plt.tight_layout()
    plt.savefig(out_dir / f"feature_importance_{name}.png", dpi=150)
    # Top-level copy too, for the unified comparison report
    plt.savefig(OUTPUTS_DIR / f"feature_importance_{name}.png", dpi=150)
    plt.close()

    # ---- Confusion matrix for best ML model ----
    best = df.iloc[0]
    pred_map = {
        "LogReg (tuned)":  y_lr,
        "GBM (tuned)":     y_gb,
    }
    if best["Model"] in pred_map:
        cm = confusion_matrix(y_gold, pred_map[best["Model"]])
        print(f"\nBest model: {best['Model']}  (F1={best['F1']})")
        print(f"  Confusion matrix:")
        print(f"    Pred   ->     0       1")
        print(f"    Actual 0  | {cm[0,0]:>6}  {cm[0,1]:>6}")
        print(f"    Actual 1  | {cm[1,0]:>6}  {cm[1,1]:>6}")

    # ---- Save metrics CSV ----
    df.to_csv(out_dir / f"metrics_{name}.csv", index=False)
    print(f"\n[saved] {out_dir}/metrics_{name}.csv")
    print(f"[saved] {out_dir}/feature_importance_{name}.png")
    print(f"[saved] {models_dir}/{{scaler,logreg_model,gbm_model}}.pkl")

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True,
                    help="Path to variant config JSON")
    args = ap.parse_args()
    evaluate_variant(args.config)


if __name__ == "__main__":
    main()
