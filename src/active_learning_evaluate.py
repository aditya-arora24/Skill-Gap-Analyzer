"""
active_learning_evaluate.py
============================
Step 2 of the active-learning experiment. Combines the 6 active-learning
LLM label CSVs into a majority-vote gold extension, retrains the LogReg
on the combined 600-pair training set (400 original gold + 200 active),
and evaluates on the SAME 100-pair held-out test set used by Phase 4.

Inputs (read-only):
  models/llm_supervised/{scaler,logreg}.pkl                 (baseline reference)
  data/proccessed again/active_learning/
      active_learning_master.csv          (row_id -> job_id, resume_id, features)
      claude_active_learning_{1,2}.csv    (row_id, label)
      gpt_active_learning_{1,2}.csv       (row_id, label)
      gemini_active_learning_{1,2}.csv    (row_id, label)
  data/proccessed again/gold_labeling/gold_labels.csv       (original 500)
  data/proccessed again/processed/pair_features_diversified.parquet
  outputs/three_way/run_metadata.json                       (test split sanity)
  outputs/self_training/comparison.csv                      (v1 numbers)
  outputs/self_training_v2/comparison.csv                   (v2 numbers)

Outputs (NEW paths only):
  models/active_learning/{scaler,logreg}.pkl
  outputs/active_learning/active_labels.csv                  the 200 majority-voted active labels
  outputs/active_learning/comparison.csv                     4-way comparison
  outputs/active_learning/comparison.png                     grouped bars
  outputs/active_learning/test_set_diff.csv                  per-test-pair diff vs supervised
  outputs/active_learning/run_metadata.json

Run:
    python "src/active_learning_evaluate.py"
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
AL_DIR        = PROJECT_ROOT / "data" / "proccessed again" / "active_learning"
AL_MASTER     = AL_DIR / "active_learning_master.csv"
GOLD_CSV      = PROJECT_ROOT / "data" / "proccessed again" / "gold_labeling" / "gold_labels.csv"
POOL_PARQUET  = PROJECT_ROOT / "data" / "proccessed again" / "processed" / "pair_features_diversified.parquet"
PHASE4_META   = PROJECT_ROOT / "outputs" / "three_way" / "run_metadata.json"
V1_RESULTS    = PROJECT_ROOT / "outputs" / "self_training" / "comparison.csv"
V2_RESULTS    = PROJECT_ROOT / "outputs" / "self_training_v2" / "comparison.csv"

NEW_MODEL_DIR = PROJECT_ROOT / "models" / "active_learning"
OUT_DIR       = PROJECT_ROOT / "outputs" / "active_learning"
NEW_MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config (matches Phase 4 + earlier self-training scripts exactly)
# ---------------------------------------------------------------------------
RANDOM_SEED   = 42
TEST_FRACTION = 0.20
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

# Per-LLM file naming used in this round
LLM_NAMES = ("claude", "gpt", "gemini")


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


def load_one_llm_labels(llm: str) -> pd.DataFrame:
    """Concatenate that LLM's two batch CSVs (batch_1 + batch_2)."""
    parts = []
    for b in (1, 2):
        path = AL_DIR / f"{llm}_active_learning_{b}.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        if "row_id" not in df.columns or "label" not in df.columns:
            raise ValueError(f"{path} must have columns row_id,label; got {list(df.columns)}")
        df = df[["row_id", "label"]].copy()
        df["row_id"] = pd.to_numeric(df["row_id"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["row_id"])
        df["row_id"] = df["row_id"].astype(int)
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
        df["label"] = (df["label"] >= 1).astype(int)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["row_id"], keep="first")
    return out.rename(columns={"label": f"{llm}_label"})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print(" Active learning — Step 2: combine labels, retrain, evaluate")
    print("=" * 70)

    # 1. Load + combine 3 LLMs' active-learning labels
    print("\n[1] loading + combining 3 LLMs' active-learning labels")
    if not AL_MASTER.exists():
        raise FileNotFoundError(f"{AL_MASTER} missing -- run active_learning_sample.py first")
    master = pd.read_csv(AL_MASTER)
    print(f"    master file: {len(master)} active pairs (row_ids "
          f"{int(master['row_id'].min())}..{int(master['row_id'].max())})")

    tables = {llm: load_one_llm_labels(llm) for llm in LLM_NAMES}
    for llm in LLM_NAMES:
        print(f"    {llm:>6s}: {len(tables[llm])} rows")

    merged = master.merge(tables["claude"], on="row_id", how="left") \
                   .merge(tables["gpt"],    on="row_id", how="left") \
                   .merge(tables["gemini"], on="row_id", how="left")

    miss_c = merged["claude_label"].isna().sum()
    miss_g = merged["gpt_label"].isna().sum()
    miss_m = merged["gemini_label"].isna().sum()
    if miss_c or miss_g or miss_m:
        print(f"[warn] missing labels: claude={miss_c} gpt={miss_g} gemini={miss_m}; "
              f"dropping affected rows")
        merged = merged.dropna(subset=["claude_label", "gpt_label", "gemini_label"])
    merged["claude_label"] = merged["claude_label"].astype(int)
    merged["gpt_label"]    = merged["gpt_label"].astype(int)
    merged["gemini_label"] = merged["gemini_label"].astype(int)

    # 2. Majority vote
    print("\n[2] computing majority vote")
    s = merged[["claude_label", "gpt_label", "gemini_label"]].sum(axis=1)
    merged["majority_label"] = (s >= 2).astype(int)
    merged["agreement"]      = merged.apply(
        lambda r: "3-0" if r["claude_label"] == r["gpt_label"] == r["gemini_label"] else "2-1",
        axis=1,
    )
    merged["is_unanimous"] = merged["agreement"] == "3-0"

    # Inter-LLM agreement statistics
    n_unanim = int(merged["is_unanimous"].sum())
    n_disp   = len(merged) - n_unanim
    print(f"    n active labels: {len(merged)}")
    print(f"    3-0 unanimous : {n_unanim:>3d} ({100*n_unanim/len(merged):5.1f}%)")
    print(f"    2-1 disputed  : {n_disp:>3d} ({100*n_disp/len(merged):5.1f}%)")

    pair_a = (merged["claude_label"] == merged["gpt_label"]).mean()
    pair_b = (merged["claude_label"] == merged["gemini_label"]).mean()
    pair_c = (merged["gpt_label"]    == merged["gemini_label"]).mean()
    print(f"    pairwise agreement:")
    print(f"      Claude vs GPT   : {pair_a*100:5.1f}%")
    print(f"      Claude vs Gemini: {pair_b*100:5.1f}%")
    print(f"      GPT vs Gemini   : {pair_c*100:5.1f}%")
    mean_pair = (pair_a + pair_b + pair_c) / 3
    print(f"      mean            : {mean_pair*100:5.1f}%")

    print(f"    per-LLM positive rates:")
    for llm in LLM_NAMES:
        rate = merged[f"{llm}_label"].mean() * 100
        print(f"      {llm:>6s}: {rate:5.1f}%")
    print(f"    majority    : {merged['majority_label'].mean()*100:5.1f}%")

    # By-source breakdown
    print(f"    majority labels by pool_source:")
    for src in sorted(merged["pool_source"].unique()):
        sub = merged[merged["pool_source"] == src]
        pos = sub["majority_label"].sum()
        unan = sub["is_unanimous"].sum()
        print(f"      {src:>8s}: n={len(sub):>3d}  "
              f"pos={pos:>2d} ({100*pos/len(sub):4.1f}%)  "
              f"unanimous={unan:>2d} ({100*unan/len(sub):4.1f}%)")

    # The master CSV only carries 4 of the 8 features. Pull the full set from
    # pair_features_diversified.parquet so downstream training has everything.
    pool_for_features = pd.read_parquet(POOL_PARQUET)
    feat_cols_in_master = [c for c in ALL_FEATURES if c in merged.columns]
    if feat_cols_in_master:
        merged = merged.drop(columns=feat_cols_in_master)
    merged = merged.merge(
        pool_for_features[["job_id", "resume_id"] + ALL_FEATURES],
        on=["job_id", "resume_id"], how="left",
    )
    n_missing_feats = merged[ALL_FEATURES[0]].isna().sum()
    if n_missing_feats:
        print(f"[warn] {n_missing_feats} active rows missing feature joins; dropping")
        merged = merged.dropna(subset=ALL_FEATURES).reset_index(drop=True)

    # Save the combined labels (now with all 8 features)
    save_cols = [
        "row_id", "job_id", "resume_id", "pool_source", "pred_prob",
        "claude_label", "gpt_label", "gemini_label",
        "majority_label", "agreement", "is_unanimous",
    ] + ALL_FEATURES + ["resume_category", "job_position_title"]
    save_cols = [c for c in save_cols if c in merged.columns]
    al_out = merged[save_cols].copy()
    al_out.to_csv(OUT_DIR / "active_labels.csv", index=False)
    print(f"\n[write] {OUT_DIR / 'active_labels.csv'}")

    # 3. Reproduce Phase 4 train/test split on the original 500 gold
    print("\n[3] reproducing Phase 4 train/test split on the original 500 gold")
    gold = pd.read_csv(GOLD_CSV)
    pool = pd.read_parquet(POOL_PARQUET)

    gold_no_feat = gold[[c for c in gold.columns if c not in ALL_FEATURES]]
    gold_full = gold_no_feat.merge(
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
    print(f"    original gold split  -> train: {len(train_df)}  test: {len(test_df)}")

    if PHASE4_META.exists():
        meta4 = json.loads(PHASE4_META.read_text())
        expected = set(meta4.get("test_row_ids", []))
        actual   = set(test_df["row_id"].astype(int).tolist())
        if expected and expected != actual:
            raise SystemExit("ERROR: test split differs from Phase 4")
        elif expected:
            print(f"    [OK] test row_ids match Phase 4 ({len(actual)} pairs)")

    y_train = train_df["majority_label"].values
    y_test  = test_df["majority_label"].values
    X_train_orig = train_df[ALL_FEATURES].values
    X_test  = test_df[ALL_FEATURES].values

    # 4. Build combined training set: 400 original gold + 200 active
    print("\n[4] combining 400 gold train + 200 active learning labels")
    al_X = al_out[ALL_FEATURES].values
    al_y = al_out["majority_label"].values
    combined_X = np.vstack([X_train_orig, al_X])
    combined_y = np.concatenate([y_train, al_y])
    print(f"    combined: {len(combined_y)} rows  "
          f"(pos {int(combined_y.sum())}, neg {int((1-combined_y).sum())})")
    print(f"    overall positive rate: {100 * combined_y.mean():.1f}%")
    print(f"    breakdown:")
    print(f"      original gold train: {len(y_train)}  pos={int(y_train.sum())}")
    print(f"      active learning:    {len(al_y)}  pos={int(al_y.sum())}")

    # 5. Train new LogReg + scaler on combined set
    print("\n[5] training new LogReg on combined set")
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

    # 6. Load supervised baseline + tune threshold on original 400 (same as Phase 4)
    old_scaler = joblib.load(SUPERVISED_MODEL_DIR / "scaler.pkl")
    old_model  = joblib.load(SUPERVISED_MODEL_DIR / "logreg.pkl")
    X_train_old_s = old_scaler.transform(X_train_orig)
    old_t, old_cv_f1 = find_best_threshold_cv(
        model_kwargs, X_train_old_s, y_train, THRESHOLD_GRID,
    )

    # 7. Evaluate both on the same 100-pair test set
    print("\n[6] evaluating on the same 100-pair held-out test set")
    X_test_old_s = old_scaler.transform(X_test)
    X_test_new_s = new_scaler.transform(X_test)
    probs_old = old_model.predict_proba(X_test_old_s)[:, 1]
    probs_new = new_model.predict_proba(X_test_new_s)[:, 1]
    y_pred_old = (probs_old >= old_t).astype(int)
    y_pred_new = (probs_new >= new_t).astype(int)

    sup_metrics = metrics_row("Supervised (400 gold)", old_t, y_test, y_pred_old, probs_old)
    al_metrics  = metrics_row("Active learning (600)", new_t, y_test, y_pred_new, probs_new)

    # Pull v1 + v2 numbers for context
    rows = [sup_metrics]
    for label, path in [
        ("Self-trained v1 (abs)",   V1_RESULTS),
        ("Self-trained v2 (pct)",   V2_RESULTS),
    ]:
        if path.exists():
            df = pd.read_csv(path)
            # Find the "self-trained" row in either CSV
            cand = df[df["Approach"].str.contains("Self-trained", na=False, case=False)]
            if len(cand):
                r = cand.iloc[-1].to_dict()  # latest row
                r["Approach"] = label
                rows.append(r)
    rows.append(al_metrics)
    cmp_df = pd.DataFrame(rows)
    # Coerce columns
    for c in ["Threshold", "Accuracy", "Precision", "Recall", "F1"] + [f"P@{k}" for k in PRECISION_AT_K]:
        if c in cmp_df.columns:
            cmp_df[c] = pd.to_numeric(cmp_df[c], errors="coerce")
    cmp_df.to_csv(OUT_DIR / "comparison.csv", index=False)

    print("\n" + "=" * 70)
    print(" Comparison on the SAME 100-pair test set")
    print("=" * 70)
    print(cmp_df.round(4).to_string(index=False))
    print(f"\n[write] {OUT_DIR / 'comparison.csv'}")

    # 8. Test-set diff vs supervised
    diff = pd.DataFrame({
        "row_id":   test_df["row_id"].astype(int).values if "row_id" in test_df.columns else range(len(test_df)),
        "true":     y_test,
        "old_pred": y_pred_old,
        "al_pred":  y_pred_new,
        "old_prob": np.round(probs_old, 4),
        "al_prob":  np.round(probs_new, 4),
    })
    diff["flipped"]     = diff["old_pred"] != diff["al_pred"]
    diff["old_correct"] = diff["old_pred"] == diff["true"]
    diff["al_correct"]  = diff["al_pred"]  == diff["true"]
    diff["delta"]       = diff["al_correct"].astype(int) - diff["old_correct"].astype(int)
    diff.to_csv(OUT_DIR / "test_set_diff.csv", index=False)

    n_flipped = int(diff["flipped"].sum())
    n_p2n = int(((diff["old_pred"] == 1) & (diff["al_pred"] == 0)).sum())
    n_n2p = int(((diff["old_pred"] == 0) & (diff["al_pred"] == 1)).sum())
    n_now_correct = int((diff["delta"] > 0).sum())
    n_now_wrong   = int((diff["delta"] < 0).sum())
    print(f"\n--- Confusion-matrix diff (supervised vs active-learning) ---")
    print(f"  flipped: {n_flipped} / 100")
    print(f"    pos -> neg: {n_p2n}    neg -> pos: {n_n2p}")
    print(f"  flips that improved prediction: {n_now_correct}")
    print(f"  flips that worsened prediction: {n_now_wrong}")
    print(f"[write] {OUT_DIR / 'test_set_diff.csv'}")

    # 9. Plot grouped bars
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
    ax.set_xticklabels(cmp_df["Approach"], rotation=10, ha="right")
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Score on held-out test (100 pairs)")
    ax.set_title("Active learning vs. supervised baseline + self-training variants",
                 fontweight="bold")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "comparison.png", dpi=150)
    plt.close()
    print(f"[write] {OUT_DIR / 'comparison.png'}")

    # 10. Run metadata
    meta = {
        "random_seed":         RANDOM_SEED,
        "n_active_labels":     int(len(merged)),
        "active_unanimous":    int(n_unanim),
        "active_disputed":     int(n_disp),
        "active_positive_rate": float(merged["majority_label"].mean()),
        "pairwise_agreement":  {
            "claude_gpt":    float(pair_a),
            "claude_gemini": float(pair_b),
            "gpt_gemini":    float(pair_c),
            "mean":          float(mean_pair),
        },
        "n_combined_train":    int(len(combined_y)),
        "old_threshold":       old_t,
        "old_cv_f1":           old_cv_f1,
        "new_threshold":       new_t,
        "new_cv_f1":           new_cv_f1,
        "old_test_metrics":    sup_metrics,
        "al_test_metrics":     al_metrics,
        "n_flipped":           n_flipped,
        "n_now_correct":       n_now_correct,
        "n_now_wrong":         n_now_wrong,
        "active_label_source_distribution": {
            str(k): int(v) for k, v in al_out["pool_source"].value_counts().items()
        },
        "active_label_positive_by_source": {
            str(k): float(g["majority_label"].mean())
            for k, g in al_out.groupby("pool_source")
        },
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[write] {OUT_DIR / 'run_metadata.json'}")

    # 11. Bottom line
    print("\n" + "=" * 70)
    print(" Bottom line")
    print("=" * 70)
    delta = al_metrics["F1"] - sup_metrics["F1"]
    print(f"  Supervised baseline (400 LLM labels)     F1 = {sup_metrics['F1']:.4f}")
    print(f"  Active learning   (400 + 200 = 600)     F1 = {al_metrics['F1']:.4f}")
    print(f"  Delta                                    F1 = {delta:+.4f}")
    if delta > 0.02:
        verdict = ("Active learning added measurable value. The 200 uncertain-band "
                   "labels carried information the supervised baseline didn't have.")
    elif delta > -0.02:
        verdict = ("Active learning was approximately a wash. Either the supervised "
                   "baseline was already saturated for this feature set, or the "
                   "uncertain-band labels weren't different enough from gold to "
                   "shift the decision boundary.")
    else:
        verdict = ("Active learning degraded F1. Likely the LLMs disagreed too much "
                   "on the uncertain band for the labels to help; check disputed-"
                   "rate and pairwise agreement above.")
    print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
