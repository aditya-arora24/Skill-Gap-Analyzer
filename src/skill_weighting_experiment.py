"""
skill_weighting_experiment.py
==============================
Tests three context-aware weighting schemes for weighted_skill_score, with
the goal of making the importance per skill JOB-SPECIFIC instead of using
the static ESCO depth alone.

Schemes compared:
  Baseline : importance(skill, job)  =  depth_score(skill)
              (current implementation -- universal weight per skill)

  A (SBERT) : importance(skill, job) = depth_score(skill) × (1 + α × cos(skill_emb, jd_emb))
              SBERT cosine between the skill's name embedding and the job
              description embedding -- semantic relevance.

  B (TFIDF) : importance(skill, job) = depth_score(skill) × (1 + β × tfidf(skill, jd))
              TF-IDF score of the skill's term(s) in the JD -- lexical relevance.

  C (LLM)   : importance(skill, job) = depth_score(skill) × (1 + γ × llm_rating)
              LLM-rated 0-5 relevance per (skill, job) pair. NOT run here;
              this script generates the prep CSVs you can upload to an LLM,
              then re-run with --include_C if you want.

For each scheme we:
  1. Compute relevance scores
  2. Recompute weighted_skill_score and avg_missing_skill_importance
     for all 50,650 pairs (the only two features that depend on the
     importance map)
  3. Re-train LogReg with the same procedure as active_learning_evaluate.py
     (same train/test split, same feature set, same threshold tuning)
  4. Evaluate on the same 100-pair held-out test set
  5. Run permutation importance specifically on weighted_skill_score
  6. Report side-by-side comparison

Inputs (read-only):
  models/llm_supervised/{scaler,logreg}.pkl
  data/proccessed again/processed/cleaned_jobs.parquet
  data/proccessed again/processed/cleaned_resumes.parquet
  data/proccessed again/processed/pair_features_diversified.parquet
  data/proccessed again/esco_skill_depths.json
  data/proccessed again/esco_skills_combined.json
  data/proccessed again/processed/embeddings/job_emb.npy
  data/proccessed again/gold_labeling/gold_labels.csv
  outputs/active_learning/active_labels.csv

Outputs (NEW directory):
  outputs/skill_weighting/
    skill_emb.npy                            (cached SBERT skill embeddings)
    relevance_sbert.npy                      (option A relevance matrix)
    relevance_tfidf.npy                      (option B relevance matrix)
    comparison.csv                           (the 4-way comparison table)
    comparison.png                           (chart)
    weighted_score_changes.csv               (per-pair before/after for diagnostics)
    llm_rating_input.csv                     (option C prep, if needed)
    run_metadata.json

Run:
    python "src/skill_weighting_experiment.py"
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROC          = PROJECT_ROOT / "data" / "proccessed again" / "processed"
PROC_AGAIN    = PROJECT_ROOT / "data" / "proccessed again"

CLEANED_JOBS    = PROC / "cleaned_jobs.parquet"
CLEANED_RESUMES = PROC / "cleaned_resumes.parquet"
POOL_PARQUET    = PROC / "pair_features_diversified.parquet"
DEPTH_JSON      = PROC_AGAIN / "esco_skill_depths.json"
VOCAB_JSON      = PROC_AGAIN / "esco_skills_combined.json"
JOB_EMB_NPY     = PROC / "embeddings" / "job_emb.npy"
GOLD_CSV        = PROC_AGAIN / "gold_labeling" / "gold_labels.csv"
ACTIVE_LBL_CSV  = PROJECT_ROOT / "outputs" / "active_learning" / "active_labels.csv"
SKILL_CTX_JSON  = PROC / "skill_to_context_map.json"
SKILL_DICT_JSON = PROC / "skill_dictionary_merged.json"

OUT_DIR = PROJECT_ROOT / "outputs" / "skill_weighting"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RANDOM_SEED   = 42
TEST_FRACTION = 0.20
THRESHOLD_GRID = np.arange(0.10, 0.91, 0.02)
N_PERMUTATIONS = 50

# Relevance scaling factors
ALPHA_SBERT = 1.0
BETA_TFIDF  = 1.0

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

# SBERT model — must match what was used for cached job embeddings
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_skill_field(value):
    """Match preprocessing: extract list of skill strings from extracted_skills column."""
    import ast
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        items = list(value)
    elif isinstance(value, list):
        items = value
    elif isinstance(value, str):
        try:
            items = ast.literal_eval(value)
        except Exception:
            items = []
    else:
        items = []
    out = []
    for it in items:
        if isinstance(it, dict):
            s = str(it.get("skill", "")).strip().lower()
        else:
            s = str(it).strip().lower()
        if s:
            out.append(s)
    return out


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


def build_skill_embeddings(skills_list: list[str]) -> np.ndarray:
    """Encode each skill name with SBERT. One-time, cached."""
    cache = OUT_DIR / "skill_emb.npy"
    skills_path = OUT_DIR / "skill_emb_keys.json"
    if cache.exists() and skills_path.exists():
        cached_skills = json.loads(skills_path.read_text())
        if cached_skills == skills_list:
            print(f"    [cache hit] skill_emb: {cache}")
            return np.load(cache)

    from sentence_transformers import SentenceTransformer
    print(f"    encoding {len(skills_list):,} skill names with SBERT...")
    model = SentenceTransformer(SBERT_MODEL_NAME)
    emb = model.encode(
        skills_list, batch_size=128,
        convert_to_numpy=True, show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    np.save(cache, emb)
    skills_path.write_text(json.dumps(skills_list))
    return emb


def compute_sbert_relevance(skills_list, skill_emb, job_emb) -> np.ndarray:
    """
    relevance[i, j] = cosine similarity between skill i's embedding and job j's
    embedding. Both inputs are row-normalized so cosine == dot product.
    Output is in [-1, 1]; we clip negatives to 0 since negative relevance would
    be weird for "this skill is anti-relevant to this job".
    """
    rel = (skill_emb @ job_emb.T).astype(np.float32)
    rel = np.clip(rel, 0.0, 1.0)
    return rel


def compute_tfidf_relevance(skills_list, jd_corpus) -> np.ndarray:
    """
    For each skill, sum the TF-IDF scores of its component tokens in each JD.
    Multi-word skills are scored by summing their term TF-IDFs.
    """
    vec = TfidfVectorizer(
        max_features=50_000, ngram_range=(1, 1),
        min_df=1, sublinear_tf=True,
    )
    jd_tfidf = vec.fit_transform(jd_corpus)  # (n_jobs, vocab) sparse
    feature_names = vec.get_feature_names_out()
    name_to_col = {n: i for i, n in enumerate(feature_names)}

    n_skills = len(skills_list)
    n_jobs = jd_tfidf.shape[0]
    relevance = np.zeros((n_skills, n_jobs), dtype=np.float32)

    for i, skill in enumerate(skills_list):
        # Get component tokens (skills can be multi-word)
        tokens = skill.lower().replace("-", " ").split()
        cols = [name_to_col[t] for t in tokens if t in name_to_col]
        if cols:
            # Sum TF-IDF across component tokens, for each JD
            sub = jd_tfidf[:, cols].toarray().sum(axis=1)
            relevance[i] = sub.astype(np.float32)

    # Per-skill normalization to [0, 1]: divide by max over all JDs for that skill
    max_per_skill = relevance.max(axis=1, keepdims=True)
    max_per_skill[max_per_skill == 0] = 1.0
    relevance = relevance / max_per_skill
    return relevance


def get_depth_importance(skills_list: list[str], depth_data: dict, default_importance: float) -> dict[str, float]:
    """Build the static depth-based importance lookup."""
    depth_by_label = depth_data.get("depth_by_label", {})
    max_depth = float(depth_data.get("max_depth", 1)) or 1.0
    out = {}
    for s in skills_list:
        d = depth_by_label.get(s)
        if d:
            out[s] = (float(d) / max_depth) * 5.0
        else:
            out[s] = default_importance
    return out


def recompute_weighted_features(
    pairs_job: np.ndarray, pairs_res: np.ndarray,
    job_skill_sets: list[set], resume_skill_sets: list[set],
    skill_to_idx: dict[str, int], depth_imp: dict[str, float],
    relevance: np.ndarray | None,
    default_importance: float,
    alpha: float = 1.0,
):
    """
    Recompute weighted_skill_score, avg_missing_skill_importance, num_missing_skills.

    If relevance is None: importance(skill, job) = depth_imp[skill].
    Otherwise:           importance(skill, job) = depth_imp[skill] × (1 + alpha × relevance[skill, job]).
    """
    n = len(pairs_job)
    weighted = np.zeros(n, dtype=np.float32)
    avg_miss_imp = np.zeros(n, dtype=np.float32)
    n_missing = np.zeros(n, dtype=np.int32)

    for k in range(n):
        j = int(pairs_job[k])
        r = int(pairs_res[k])
        js = job_skill_sets[j]
        rs = resume_skill_sets[r]
        if not js:
            continue
        inter = js & rs
        miss = js - rs
        n_missing[k] = len(miss)

        def imp(skill: str, job_idx: int = j) -> float:
            base = depth_imp.get(skill, default_importance)
            if relevance is not None and skill in skill_to_idx:
                rel = relevance[skill_to_idx[skill], job_idx]
                return float(base * (1.0 + alpha * rel))
            return float(base)

        wj = sum(imp(s) for s in js)
        wi = sum(imp(s) for s in inter)
        weighted[k] = (wi / wj) if wj > 0 else 0.0
        if miss:
            avg_miss_imp[k] = float(np.mean([imp(s) for s in miss]))

    return weighted, avg_miss_imp, n_missing


def evaluate_option(name, pair_df, gold_full, train_df, test_df,
                    al_X_full, al_y, suffix=""):
    """
    Re-merge pair features into gold_full + active_labels, build train/test,
    train LogReg, evaluate on test, run permutation importance specifically
    on weighted_skill_score.
    """
    # Build train: 400 gold + 200 active, with the new feature values for ALL
    train_keys = list(zip(train_df["job_id"].astype(int), train_df["resume_id"].astype(int)))
    test_keys  = list(zip(test_df["job_id"].astype(int),  test_df["resume_id"].astype(int)))
    al_keys    = list(zip(al_X_full["job_id"].astype(int), al_X_full["resume_id"].astype(int)))

    pair_lookup = pair_df.set_index(["job_id", "resume_id"])

    def lookup_features(keys):
        out = []
        for j, r in keys:
            row = pair_lookup.loc[(j, r), ALL_FEATURES].values
            out.append(row)
        return np.array(out, dtype=np.float64)

    X_train = np.vstack([
        lookup_features(train_keys),
        lookup_features(al_keys),
    ])
    y_train = np.concatenate([
        train_df["majority_label"].astype(int).values,
        al_y,
    ])
    X_test = lookup_features(test_keys)
    y_test = test_df["majority_label"].astype(int).values

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model_kwargs = dict(solver="liblinear", class_weight="balanced", random_state=RANDOM_SEED)
    best_t, cv_f1 = find_best_threshold_cv(
        model_kwargs, X_train_s, y_train, THRESHOLD_GRID,
    )
    final = LogisticRegression(**model_kwargs)
    final.fit(X_train_s, y_train)

    probs = final.predict_proba(X_test_s)[:, 1]
    y_pred = (probs >= best_t).astype(int)

    f1     = round(f1_score(y_test, y_pred, zero_division=0), 4)
    acc    = round(accuracy_score(y_test, y_pred), 4)
    prec   = round(precision_score(y_test, y_pred, zero_division=0), 4)
    rec    = round(recall_score(y_test, y_pred, zero_division=0), 4)

    # Coefficient for weighted_skill_score (column index)
    wss_idx = ALL_FEATURES.index("weighted_skill_score")
    avg_idx = ALL_FEATURES.index("avg_missing_skill_importance")
    coefs = final.coef_[0]

    # Permutation importance specifically for weighted_skill_score
    rng = np.random.default_rng(RANDOM_SEED)
    drops = []
    for _ in range(N_PERMUTATIONS):
        X_perm = X_test_s.copy()
        X_perm[:, wss_idx] = rng.permutation(X_perm[:, wss_idx])
        f1_p = f1_score(y_test, (final.predict_proba(X_perm)[:, 1] >= best_t).astype(int),
                        zero_division=0)
        drops.append(f1 - f1_p)
    perm_drop = float(np.mean(drops))

    return {
        "Option":                    name,
        "Threshold":                 best_t,
        "CV F1":                     cv_f1,
        "Test F1":                   f1,
        "Accuracy":                  acc,
        "Precision":                 prec,
        "Recall":                    rec,
        "wss_coef":                  round(float(coefs[wss_idx]), 4),
        "wss_abs_coef":              round(abs(float(coefs[wss_idx])), 4),
        "avg_missing_imp_coef":      round(float(coefs[avg_idx]), 4),
        "wss_permutation_drop":      round(perm_drop, 4),
        "wss_mean":                  round(float(pair_df["weighted_skill_score"].mean()), 4),
        "wss_std":                   round(float(pair_df["weighted_skill_score"].std()), 4),
        "wss_unique":                int(pair_df["weighted_skill_score"].round(6).nunique()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(" Context-aware skill weighting experiment")
    print("=" * 70)

    # ---- Load data ----
    print("\n[1] loading data")
    pool = pd.read_parquet(POOL_PARQUET)
    resumes = pd.read_parquet(CLEANED_RESUMES).reset_index(drop=True)
    jobs = pd.read_parquet(CLEANED_JOBS).reset_index(drop=True)
    print(f"    pool: {len(pool):,} pairs   resumes: {len(resumes)}  jobs: {len(jobs)}")

    job_emb = np.load(JOB_EMB_NPY)
    print(f"    job_emb: {job_emb.shape}")

    depth_data = json.loads(DEPTH_JSON.read_text())
    skill_ctx  = json.loads(SKILL_CTX_JSON.read_text())
    skill_dict = json.loads(SKILL_DICT_JSON.read_text())
    print(f"    depth_by_label: {len(depth_data.get('depth_by_label', {})):,} skills "
          f"(max_depth={depth_data.get('max_depth')})")

    # default importance = mean of O*NET importances
    onet_imps = []
    for s, ctx in skill_ctx.items():
        v = ctx.get("importance_mean", 0.0)
        if v > 0:
            onet_imps.append(float(v))
    default_importance = float(np.mean(onet_imps)) if onet_imps else 2.5
    print(f"    default_importance = {default_importance:.3f}")

    # Build skill universe: union of skills appearing in any extracted_skills
    print("\n[2] building skill universe from extracted_skills")
    job_skill_sets = [set(parse_skill_field(v)) for v in jobs["extracted_skills"].tolist()]
    resume_skill_sets = [set(parse_skill_field(v)) for v in resumes["extracted_skills"].tolist()]
    all_skills_in_use = set()
    for s in job_skill_sets + resume_skill_sets:
        all_skills_in_use.update(s)
    skills_list = sorted(all_skills_in_use)
    skill_to_idx = {s: i for i, s in enumerate(skills_list)}
    print(f"    {len(skills_list):,} unique skills appear in the corpus")

    depth_imp = get_depth_importance(skills_list, depth_data, default_importance)

    # ---- Build relevance matrices for A and B ----
    print("\n[3] computing relevance matrices")

    # A: SBERT skill-vs-JD
    print("    Option A — SBERT cosine")
    skill_emb = build_skill_embeddings(skills_list)
    rel_sbert = compute_sbert_relevance(skills_list, skill_emb, job_emb)
    np.save(OUT_DIR / "relevance_sbert.npy", rel_sbert)
    print(f"       relevance_sbert: {rel_sbert.shape}  "
          f"min={rel_sbert.min():.3f}  mean={rel_sbert.mean():.3f}  max={rel_sbert.max():.3f}")

    # B: TF-IDF
    print("    Option B — TF-IDF")
    jd_corpus = jobs["cleaned_job_description"].fillna("").tolist()
    rel_tfidf = compute_tfidf_relevance(skills_list, jd_corpus)
    np.save(OUT_DIR / "relevance_tfidf.npy", rel_tfidf)
    print(f"       relevance_tfidf: {rel_tfidf.shape}  "
          f"min={rel_tfidf.min():.3f}  mean={rel_tfidf.mean():.3f}  max={rel_tfidf.max():.3f}")

    # ---- Compute pair indices ----
    pairs_job = pool["job_id"].astype(int).values
    pairs_res = pool["resume_id"].astype(int).values

    # ---- For each option, recompute the two affected features ----
    print("\n[4] recomputing weighted_skill_score and avg_missing_skill_importance")

    options = {
        "Baseline (depth only)":    None,
        "A: depth × SBERT":         (rel_sbert, ALPHA_SBERT),
        "B: depth × TFIDF":         (rel_tfidf, BETA_TFIDF),
    }

    pair_dfs = {}
    base_pool = pool.copy()
    base_pool["weighted_skill_score_baseline"] = base_pool["weighted_skill_score"].values
    base_pool["avg_missing_skill_importance_baseline"] = base_pool["avg_missing_skill_importance"].values

    for name, opt in options.items():
        rel = opt[0] if opt is not None else None
        alpha = opt[1] if opt is not None else 0.0
        wss, avg_imp, n_miss = recompute_weighted_features(
            pairs_job, pairs_res, job_skill_sets, resume_skill_sets,
            skill_to_idx, depth_imp, rel, default_importance, alpha=alpha,
        )
        new_pool = pool.copy()
        new_pool["weighted_skill_score"] = wss
        new_pool["avg_missing_skill_importance"] = avg_imp
        # num_missing_skills doesn't depend on importance, so leave it
        pair_dfs[name] = new_pool

        # Quick stat print
        n_changed_w = int((np.abs(wss - pool["weighted_skill_score"].values) > 1e-6).sum())
        print(f"    {name:<30s}: weighted_skill_score "
              f"mean={wss.mean():.4f}  std={wss.std():.4f}  "
              f"unique={int(np.unique(np.round(wss, 6)).size):,}  "
              f"changed_rows={n_changed_w:,}")

    # ---- Build the train/test split (same as Phase 4) ----
    print("\n[5] preparing the same Phase 4 train/test split")
    gold = pd.read_csv(GOLD_CSV)
    gold_no_feat = gold[[c for c in gold.columns if c not in ALL_FEATURES]]
    # Use baseline feature values for the split assignment
    gold_full = gold_no_feat.merge(
        pool[["job_id", "resume_id"] + ALL_FEATURES],
        on=["job_id", "resume_id"], how="left",
    ).dropna(subset=ALL_FEATURES).reset_index(drop=True)
    gold_full["majority_label"] = gold_full["majority_label"].astype(int)
    train_df, test_df = train_test_split(
        gold_full, test_size=TEST_FRACTION,
        stratify=gold_full["majority_label"],
        random_state=RANDOM_SEED,
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
    print(f"    train: {len(train_df)}  test: {len(test_df)}")

    # Active labels
    al_df = pd.read_csv(ACTIVE_LBL_CSV)
    al_y = al_df["majority_label"].astype(int).values

    # ---- Evaluate each option ----
    print("\n[6] training LogReg + evaluating each option")
    results = []
    for name, pair_df in pair_dfs.items():
        print(f"    -> {name}")
        result = evaluate_option(name, pair_df, gold_full, train_df, test_df,
                                  al_df, al_y)
        results.append(result)

    cmp_df = pd.DataFrame(results)
    cmp_df.to_csv(OUT_DIR / "comparison.csv", index=False)

    # ---- Print comparison ----
    print("\n" + "=" * 70)
    print(" Side-by-side comparison")
    print("=" * 70)
    cols_to_show = ["Option", "Test F1", "wss_coef", "wss_abs_coef",
                    "wss_permutation_drop", "wss_unique", "avg_missing_imp_coef"]
    print(cmp_df[cols_to_show].to_string(index=False))

    # ---- Plot grouped bars ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # F1 across options
    ax = axes[0]
    ax.bar(cmp_df["Option"], cmp_df["Test F1"], color="#3b82f6", edgecolor="white")
    for i, v in enumerate(cmp_df["Test F1"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Test F1")
    ax.set_title("Model F1 by weighting scheme", fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)

    # weighted_skill_score |coefficient|
    ax = axes[1]
    ax.bar(cmp_df["Option"], cmp_df["wss_abs_coef"], color="#10b981", edgecolor="white")
    for i, v in enumerate(cmp_df["wss_abs_coef"]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("|coefficient| of weighted_skill_score")
    ax.set_title("How much the model relies on weighted_skill_score\n(absolute LogReg coefficient)",
                 fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)

    # weighted_skill_score permutation drop
    ax = axes[2]
    ax.bar(cmp_df["Option"], cmp_df["wss_permutation_drop"], color="#f59e0b", edgecolor="white")
    for i, v in enumerate(cmp_df["wss_permutation_drop"]):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("F1 drop when weighted_skill_score is permuted")
    ax.set_title("Permutation importance of weighted_skill_score",
                 fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "comparison.png", dpi=150)
    plt.close()
    print(f"\n[write] {OUT_DIR / 'comparison.csv'}")
    print(f"[write] {OUT_DIR / 'comparison.png'}")

    # ---- Generate Option C prep CSV (for optional LLM rating) ----
    print("\n[7] generating Option C prep CSV (LLM rating workload)")
    # For each (job, skill) where skill is extracted from that job
    rating_rows = []
    for j in range(len(jobs)):
        skills = job_skill_sets[j]
        for s in sorted(skills):
            rating_rows.append({
                "job_id":   j,
                "skill":    s,
                "position_title":   jobs.iloc[j]["position_title"],
                "job_description":  str(jobs.iloc[j]["job_description"])[:1000],
            })
    if rating_rows:
        c_df = pd.DataFrame(rating_rows)
        c_df["row_id"] = range(1, len(c_df) + 1)
        c_df.to_csv(OUT_DIR / "llm_rating_input.csv", index=False)
        print(f"    [write] {OUT_DIR / 'llm_rating_input.csv'}  "
              f"({len(c_df):,} (job, skill) pairs)")
        print(f"    For Option C: have an LLM rate each (skill, JD) pair 0–5 for relevance.")
        print(f"    Output should have columns: row_id,relevance (0..5).")

    # ---- Run metadata ----
    meta = {
        "alpha_sbert":     ALPHA_SBERT,
        "beta_tfidf":      BETA_TFIDF,
        "n_skills":        len(skills_list),
        "n_jobs":          len(jobs),
        "n_resumes":       len(resumes),
        "n_test":          int(len(test_df)),
        "n_train":         int(len(train_df) + len(al_df)),
        "results":         cmp_df.to_dict(orient="records"),
        "winner":          str(cmp_df.iloc[cmp_df["Test F1"].idxmax()]["Option"]),
        "winner_f1":       float(cmp_df["Test F1"].max()),
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[write] {OUT_DIR / 'run_metadata.json'}")

    # ---- Bottom line ----
    print("\n" + "=" * 70)
    print(" Bottom line")
    print("=" * 70)
    baseline = cmp_df[cmp_df["Option"].str.contains("Baseline")].iloc[0]
    print(f"  Baseline F1                     : {baseline['Test F1']:.4f}  "
          f"(wss |coef| = {baseline['wss_abs_coef']:.3f})")
    for _, row in cmp_df.iterrows():
        if row["Option"] == baseline["Option"]:
            continue
        delta_f1 = row["Test F1"] - baseline["Test F1"]
        delta_coef = row["wss_abs_coef"] - baseline["wss_abs_coef"]
        print(f"  {row['Option']:<30s}: F1 = {row['Test F1']:.4f} "
              f"({delta_f1:+.4f})   wss |coef| = {row['wss_abs_coef']:.3f} "
              f"({delta_coef:+.3f})")

    winner = cmp_df.iloc[cmp_df["Test F1"].idxmax()]
    print(f"\n  Highest F1: {winner['Option']}")
    winner_wss = cmp_df.iloc[cmp_df["wss_abs_coef"].idxmax()]
    print(f"  Highest weighted_skill_score |coef|: {winner_wss['Option']}")


if __name__ == "__main__":
    main()
