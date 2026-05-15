"""
feature_importance_analysis.py
===============================
Settles the question: does the active-learning reranker actually use the
skill features, or is it just SBERT dressed up with extra columns?

Three analyses, all on the final active-learning LogReg (F1 = 0.769)
evaluated against the same 100-pair LLM-labeled test set:

  1. Scaled coefficients
     LogReg coefficients after StandardScaler are directly comparable
     across features (everything is on the same SD scale). Absolute
     magnitude tells you how much each feature shifts the decision.

  2. Permutation importance
     For each feature, shuffle that column N times in the test set,
     measure the F1 drop. Big drop = the model relies heavily on
     this feature. No drop = the feature is decoration.

  3. Leave-one-feature-out ablation
     For each of 8 features, retrain LogReg on the remaining 7
     (same training procedure, same threshold tuning, same test set).
     Compare F1 against the full 8-feature model. The drop tells you
     what each feature is contributing on its own.

  Plus the critical comparison your senior asked for:
     - SBERT-only F1 (already have from Phase 4)
     - Skill-only F1 (drop embedding_similarity, retrain on remaining 7)

Inputs (read-only):
  models/active_learning/{scaler,logreg}.pkl
  data/proccessed again/gold_labeling/gold_labels.csv
  data/proccessed again/processed/pair_features_diversified.parquet
  data/proccessed again/active_learning/active_labels.csv

Outputs (NEW directory):
  outputs/feature_analysis/coefficients.csv
  outputs/feature_analysis/permutation_importance.csv
  outputs/feature_analysis/ablation_results.csv
  outputs/feature_analysis/feature_importance.png
  outputs/feature_analysis/run_metadata.json

Run:
    python "src/feature_importance_analysis.py"
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
from sklearn.metrics import f1_score, accuracy_score


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

AL_MODEL_DIR    = PROJECT_ROOT / "models" / "active_learning"
GOLD_CSV        = PROJECT_ROOT / "data" / "proccessed again" / "gold_labeling" / "gold_labels.csv"
ACTIVE_LBL_CSV  = PROJECT_ROOT / "outputs" / "active_learning" / "active_labels.csv"
POOL_PARQUET    = PROJECT_ROOT / "data" / "proccessed again" / "processed" / "pair_features_diversified.parquet"

OUT_DIR = PROJECT_ROOT / "outputs" / "feature_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Config — must match active_learning_evaluate.py exactly so the test split
# is identical
# ---------------------------------------------------------------------------
RANDOM_SEED   = 42
TEST_FRACTION = 0.20
THRESHOLD_GRID = np.arange(0.10, 0.91, 0.02)
N_PERMUTATIONS = 50

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
def find_best_threshold_cv(model_kwargs, X, y, thresholds, cv=5, seed=RANDOM_SEED):
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


def train_and_eval(features: list[str], X_train_full, y_train,
                   X_test_full, y_test, name: str = "") -> dict:
    """Train fresh LogReg on the given feature subset, return test metrics."""
    if not features:
        return None
    X_train = X_train_full[:, [ALL_FEATURES.index(f) for f in features]]
    X_test  = X_test_full[:,  [ALL_FEATURES.index(f) for f in features]]

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model_kwargs = dict(
        solver="liblinear",
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    best_t, cv_f1 = find_best_threshold_cv(
        model_kwargs, X_train_s, y_train, THRESHOLD_GRID,
    )
    final = LogisticRegression(**model_kwargs)
    final.fit(X_train_s, y_train)
    probs = final.predict_proba(X_test_s)[:, 1]
    y_pred = (probs >= best_t).astype(int)
    return {
        "name":      name,
        "n_features": len(features),
        "features":  features,
        "threshold": best_t,
        "cv_f1":     cv_f1,
        "test_f1":   round(f1_score(y_test, y_pred, zero_division=0), 4),
        "test_acc":  round(accuracy_score(y_test, y_pred), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print(" Feature importance analysis on active-learning LogReg (F1=0.769)")
    print("=" * 70)

    # 1. Reproduce the active-learning training set (400 gold + 200 active)
    print("\n[1] reproducing the 600-pair training set + 100-pair test set")
    gold = pd.read_csv(GOLD_CSV)
    pool = pd.read_parquet(POOL_PARQUET)

    gold_no_feat = gold[[c for c in gold.columns if c not in ALL_FEATURES]]
    gold_full = gold_no_feat.merge(
        pool[["job_id", "resume_id"] + ALL_FEATURES],
        on=["job_id", "resume_id"], how="left",
    ).dropna(subset=ALL_FEATURES).reset_index(drop=True)
    gold_full["majority_label"] = gold_full["majority_label"].astype(int)

    train_df, test_df = train_test_split(
        gold_full,
        test_size=TEST_FRACTION,
        stratify=gold_full["majority_label"],
        random_state=RANDOM_SEED,
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
    y_train_orig = train_df["majority_label"].values
    y_test       = test_df["majority_label"].values
    X_train_orig = train_df[ALL_FEATURES].values
    X_test       = test_df[ALL_FEATURES].values

    # Add active learning labels
    active = pd.read_csv(ACTIVE_LBL_CSV)
    active_X = active[ALL_FEATURES].values
    active_y = active["majority_label"].astype(int).values
    X_train  = np.vstack([X_train_orig, active_X])
    y_train  = np.concatenate([y_train_orig, active_y])
    print(f"    train: {len(y_train)} pairs (400 gold + 200 active)")
    print(f"    test : {len(y_test)} pairs")

    # 2. Load the trained active-learning model
    print("\n[2] loading trained active-learning LogReg")
    scaler = joblib.load(AL_MODEL_DIR / "scaler.pkl")
    model  = joblib.load(AL_MODEL_DIR / "logreg.pkl")

    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)
    probs = model.predict_proba(X_test_s)[:, 1]
    # Recover the threshold from CV on the same training set
    best_t, cv_f1 = find_best_threshold_cv(
        dict(solver="liblinear", class_weight="balanced", random_state=RANDOM_SEED),
        X_train_s, y_train, THRESHOLD_GRID,
    )
    y_pred = (probs >= best_t).astype(int)
    full_f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"    threshold = {best_t}, CV F1 = {cv_f1}, test F1 = {full_f1:.4f}")
    if abs(full_f1 - 0.7692) > 0.01:
        print(f"[warn] reproduced F1 ({full_f1:.4f}) differs from reported 0.7692")

    # ----------------------------------------------------------------- #
    # Analysis 1: scaled LogReg coefficients                             #
    # ----------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print(" (1) Scaled LogReg coefficients (signed, sorted by |coef|)")
    print("=" * 70)
    coefs = model.coef_[0]
    coef_df = pd.DataFrame({
        "feature":     ALL_FEATURES,
        "coef":        np.round(coefs, 4),
        "abs_coef":    np.round(np.abs(coefs), 4),
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)
    print(coef_df.to_string(index=False))
    coef_df.to_csv(OUT_DIR / "coefficients.csv", index=False)
    print(f"\n[write] {OUT_DIR / 'coefficients.csv'}")

    # ----------------------------------------------------------------- #
    # Analysis 2: permutation importance on test set                     #
    # ----------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print(f" (2) Permutation importance on test set (N={N_PERMUTATIONS} per feature)")
    print("=" * 70)
    rng = np.random.default_rng(RANDOM_SEED)
    perm_results = []
    for fi, fname in enumerate(ALL_FEATURES):
        f1_drops = []
        for _ in range(N_PERMUTATIONS):
            X_perm = X_test_s.copy()
            X_perm[:, fi] = rng.permutation(X_perm[:, fi])
            probs_perm = model.predict_proba(X_perm)[:, 1]
            y_pred_perm = (probs_perm >= best_t).astype(int)
            f1_perm = f1_score(y_test, y_pred_perm, zero_division=0)
            f1_drops.append(full_f1 - f1_perm)
        mean_drop = float(np.mean(f1_drops))
        std_drop  = float(np.std(f1_drops))
        perm_results.append({
            "feature":   fname,
            "mean_f1_drop": round(mean_drop, 4),
            "std_f1_drop":  round(std_drop, 4),
        })
    perm_df = pd.DataFrame(perm_results).sort_values(
        "mean_f1_drop", ascending=False
    ).reset_index(drop=True)
    print(perm_df.to_string(index=False))
    perm_df.to_csv(OUT_DIR / "permutation_importance.csv", index=False)
    print(f"\n[write] {OUT_DIR / 'permutation_importance.csv'}")

    # ----------------------------------------------------------------- #
    # Analysis 3: leave-one-feature-out ablation                         #
    # ----------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print(" (3) Leave-one-feature-out ablation (retrain LogReg on 7 features)")
    print("=" * 70)
    print(f"    full 8-feature reference F1 = {full_f1:.4f}\n")

    ablation_rows = []
    # Full reference
    ablation_rows.append({
        "dropped_feature": "(none — full 8 features)",
        "test_f1":         round(full_f1, 4),
        "f1_drop":         0.0,
    })
    for fname in ALL_FEATURES:
        keep_features = [f for f in ALL_FEATURES if f != fname]
        result = train_and_eval(keep_features, X_train, y_train,
                                 X_test, y_test, name=f"drop {fname}")
        ablation_rows.append({
            "dropped_feature": fname,
            "test_f1":         result["test_f1"],
            "f1_drop":         round(full_f1 - result["test_f1"], 4),
        })
    ablation_df = pd.DataFrame(ablation_rows).sort_values(
        "f1_drop", ascending=False
    ).reset_index(drop=True)
    print(ablation_df.to_string(index=False))
    ablation_df.to_csv(OUT_DIR / "ablation_results.csv", index=False)
    print(f"\n[write] {OUT_DIR / 'ablation_results.csv'}")

    # ----------------------------------------------------------------- #
    # The senior's question, answered directly                           #
    # ----------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print(" THE CRITICAL COMPARISON — does the model do anything beyond SBERT?")
    print("=" * 70)

    # Re-evaluate: SBERT-only model (LogReg on 1 feature)
    sbert_only = train_and_eval(
        ["embedding_similarity"], X_train, y_train, X_test, y_test, "SBERT-only LogReg"
    )
    # Skill-only model (drop embedding_similarity, train on remaining 7)
    skill_only_features = [f for f in ALL_FEATURES if f != "embedding_similarity"]
    skill_only = train_and_eval(
        skill_only_features, X_train, y_train, X_test, y_test, "Skill-only (no SBERT)"
    )
    # Full model (already computed)
    print()
    print(f"  Full model (8 features, embedding + skill + structural):")
    print(f"    test F1 = {full_f1:.4f}")
    print()
    print(f"  SBERT-only (just embedding_similarity, retrained as LogReg):")
    print(f"    test F1 = {sbert_only['test_f1']:.4f}")
    print()
    print(f"  Skill-only (7 features WITHOUT embedding_similarity, retrained):")
    print(f"    test F1 = {skill_only['test_f1']:.4f}")
    print()

    delta_sbert  = full_f1 - sbert_only["test_f1"]
    delta_skill  = full_f1 - skill_only["test_f1"]
    print(f"  Lift over SBERT-only           : +{delta_sbert:.4f} F1")
    print(f"  Lift over Skill-only-no-SBERT  : +{delta_skill:.4f} F1")
    print()
    print(f"  Verdict signal:")
    if skill_only["test_f1"] > sbert_only["test_f1"] + 0.03:
        print(f"    Skill features alone OUTPERFORM SBERT alone "
              f"({skill_only['test_f1']:.3f} vs {sbert_only['test_f1']:.3f}).")
        print(f"    Strong evidence that skill features add independent signal,")
        print(f"    not just SBERT decoration.")
    elif skill_only["test_f1"] > sbert_only["test_f1"] - 0.03:
        print(f"    Skill features alone match SBERT alone within noise. Both")
        print(f"    channels carry comparable signal; their combination beats either.")
    else:
        print(f"    SBERT alone outperforms skill features alone "
              f"({sbert_only['test_f1']:.3f} vs {skill_only['test_f1']:.3f}).")
        print(f"    SBERT is the dominant signal; skill features add a smaller")
        print(f"    incremental lift on top of it.")

    # ----------------------------------------------------------------- #
    # Plot
    # ----------------------------------------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Coefficients (signed, color by sign)
    ax = axes[0]
    sorted_coef = coef_df.sort_values("coef")
    colors = ["#dc2626" if c < 0 else "#16a34a" for c in sorted_coef["coef"]]
    ax.barh(sorted_coef["feature"], sorted_coef["coef"], color=colors, edgecolor="white")
    for i, v in enumerate(sorted_coef["coef"]):
        ax.text(v + (0.02 if v >= 0 else -0.02), i, f"{v:.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("LogReg scaled coefficients\n(positive = pushes prediction toward match)",
                 fontweight="bold", fontsize=11)
    ax.set_xlabel("Coefficient (after StandardScaler)")
    ax.grid(axis="x", alpha=0.3)

    # Permutation importance + ablation drop side-by-side
    ax = axes[1]
    # Order by permutation drop, descending
    perm_sorted = perm_df.set_index("feature").reindex(ALL_FEATURES)
    abl_sorted  = ablation_df[ablation_df["dropped_feature"].isin(ALL_FEATURES)] \
                              .set_index("dropped_feature").reindex(ALL_FEATURES)
    order = perm_sorted["mean_f1_drop"].sort_values().index.tolist()

    y = np.arange(len(order))
    h = 0.4
    perm_vals = perm_sorted.loc[order, "mean_f1_drop"].values
    abl_vals  = abl_sorted.loc[order,  "f1_drop"].values
    ax.barh(y - h/2, perm_vals, h, color="#3b82f6", label="Permutation F1 drop", edgecolor="white")
    ax.barh(y + h/2, abl_vals,  h, color="#f59e0b", label="Ablation F1 drop",    edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Per-feature importance\n(higher = more important)",
                 fontweight="bold", fontsize=11)
    ax.set_xlabel("F1 drop when this feature is removed/permuted")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_importance.png", dpi=150)
    plt.close()
    print(f"\n[write] {OUT_DIR / 'feature_importance.png'}")

    # ----------------------------------------------------------------- #
    # Run metadata
    # ----------------------------------------------------------------- #
    meta = {
        "random_seed":            RANDOM_SEED,
        "n_train":                int(len(y_train)),
        "n_test":                 int(len(y_test)),
        "test_threshold":         best_t,
        "test_f1_full":           round(float(full_f1), 4),
        "n_permutations":         N_PERMUTATIONS,
        "sbert_only_f1":          float(sbert_only["test_f1"]),
        "skill_only_f1":          float(skill_only["test_f1"]),
        "lift_over_sbert":        round(float(delta_sbert), 4),
        "lift_over_skill_only":   round(float(delta_skill), 4),
        "coefficients":           coef_df.to_dict(orient="records"),
        "permutation_importance": perm_df.to_dict(orient="records"),
        "ablation":               ablation_df.to_dict(orient="records"),
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[write] {OUT_DIR / 'run_metadata.json'}")

    # Summary box
    print("\n" + "=" * 70)
    print(" Summary box (paste into project_report.md)")
    print("=" * 70)
    top3_coef = coef_df.head(3)["feature"].tolist()
    top3_perm = perm_df.head(3)["feature"].tolist()
    top3_abl  = ablation_df[ablation_df["dropped_feature"].isin(ALL_FEATURES)] \
                .head(3)["dropped_feature"].tolist()
    print(f"  Top 3 by |coefficient|         : {top3_coef}")
    print(f"  Top 3 by permutation importance: {top3_perm}")
    print(f"  Top 3 by ablation F1 drop      : {top3_abl}")
    print(f"  SBERT-only F1                  : {sbert_only['test_f1']:.4f}")
    print(f"  Skill-only F1 (no SBERT)       : {skill_only['test_f1']:.4f}")
    print(f"  Full 8-feature F1              : {full_f1:.4f}")


if __name__ == "__main__":
    main()
