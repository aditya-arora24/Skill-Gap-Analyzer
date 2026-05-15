# `ml_ready_dataset.parquet` — How we built it, what's in it, and is it any good

This document covers the current state of `data/proccessed again/processed/ml_ready_dataset.parquet` after the ESCO vocabulary expansion pass. It records how the dataset was constructed, the labeling rule, full per-feature statistics, and an honest assessment including a structural problem with the current setup.

## 1. How we got here (compressed timeline)

The dataset went through five rounds of construction. Each round addressed a problem the previous one created or revealed.

**Round 1 — v1 baseline.** Course-project pipeline. 130 resumes paired with top-6 jobs each via SBERT cosine similarity → 9,495 pairs. Five features: `semantic_similarity`, `tfidf_similarity`, `num_resume_skills`, `num_job_skills`, `skill_imbalance`. Weak labels generated via 70/30 quantile cut on `skill_coverage`, with `skill_coverage` held out from training features to prevent leakage.

**Round 2 — v2 redesign.** Larger corpus (2,484 resumes × top-50 jobs = 42,650 pairs), richer features (13 total). Three preprocessing fixes were applied:
- JD-side years-of-experience pulled from the structured `model_response` field instead of the free-text JD (raised positive YoE on jobs from 0% to 59%).
- Ambiguous single-token tech skills (`r`, `go`, `swift`, `less`, etc.) gated behind a raw-case capitalization + tech-context check.
- Resume "title" derived from the first non-empty line of `Resume_str` (granular role text like "HR DIRECTOR") instead of the broad `Category` column.

**Round 3 — variant comparison and labeling-rule overhaul.** Initial composite labeling formula (`0.4·embedding + 0.3·weighted_skill + 0.2·title + 0.1·education`) with all four features held out from training. This produced a leakage-safe dataset where the trained classifier *underperformed single-feature baselines* on the gold set (LogReg F1 0.59, Title-Only baseline F1 0.66) — the holdout starved the model. We tested three partitions (strict / skill-only / no-holdout) and settled on a label rule that uses just the skill channel.

**Round 4 — ESCO vocabulary expansion.** `skill_overlap` was 64% zero because the curated tech vocabulary only had 355 entries. Most jobs had ≤2 extracted skills, so few resumes overlapped. Added the ESCO `skillType=knowledge` taxonomy (3,221 concepts × `preferredLabel` + `altLabels` + `hiddenLabels` = 19,929 entries after a length / stopword / blocklist filter). Combined vocabulary jumped to 20,253. After re-extraction: `skill_overlap > 0` rose from 36% to 68%.

**Round 5 — ESCO depth as importance weight.** After Round 4, `weighted_skill_score` was 81% identical to `skill_overlap` because every ESCO skill got the same default fallback importance (2.59). Added a depth-based importance map: each ESCO concept's depth in the `broaderRelationsSkillPillar` hierarchy maps to `(depth / max_depth) × 5`, giving deeper (more specialized) skills higher weight. Pending re-run as of the version this document describes; current `ml_ready_dataset.parquet` reflects state-after-Round-4.

## 2. Final dataset specification

**Path:** `data/proccessed again/processed/ml_ready_dataset.parquet`

| Field | Value |
|---|---|
| Source pairs | 42,650 (top-50 retrieval over 853 jobs × 2,484 resumes) |
| Rows kept after labeling cut | 25,590 |
| Rows dropped (middle band, label = NaN) | 17,060 |
| Positive / Negative split | 12,795 / 12,795 (perfectly balanced by construction) |
| Columns | 11 (2 keys + 1 label + 8 features) |
| Null values | 0 across all columns |

**Schema:**

| # | Column | Dtype | Role |
|---|---|---|---|
| 1 | `job_id` | int32 | key (row index in cleaned_jobs.parquet) |
| 2 | `resume_id` | int32 | key (row index in cleaned_resumes.parquet) |
| 3 | `label` | int8 | weak label (0 or 1) |
| 4 | `embedding_similarity` | float32 | feature |
| 5 | `tfidf_similarity` | float32 | feature |
| 6 | `skill_overlap` | float32 | feature |
| 7 | `weighted_skill_score` | float32 | feature |
| 8 | `num_missing_skills` | float32 | feature |
| 9 | `avg_missing_skill_importance` | float32 | feature |
| 10 | `years_of_experience` | float32 | feature |
| 11 | `title_similarity` | float32 | feature |

**Three features dropped from this dataset** (still present in `pair_features.parquet` for ablation): `experience_gap` (collinear -0.94 with years_of_experience), `experience_relevance_score` (low standalone signal), `education_match` (boolean, originally part of the abandoned 4-feature label rule).

## 3. The weak labeling rule

```
weak_score = 0.7 × weighted_skill_score + 0.3 × title_similarity
```

Quantile cut on the weak_score distribution:

| Cut | Threshold | Action |
|---|---|---|
| 30th percentile | 0.155 | rows with weak_score ≤ 0.155 → label = 0 |
| 70th percentile | 0.340 | rows with weak_score ≥ 0.340 → label = 1 |
| middle 40% | (0.155, 0.340) | dropped (label = NaN) |

The 30/70 cut is what produces the perfectly balanced 12,795/12,795 split — that balance is by construction, not a property of the data.

**What this rule says:** "A pair is a positive match if it has high importance-weighted skill overlap AND/OR high resume-title-vs-job-title similarity. A pair is a negative match if both signals are weak."

The 0.7/0.3 weights mean the skill channel dominates: a pair with `weighted_skill_score = 0.5` and `title_similarity = 0` already scores 0.35, into the upper band. A pair with `weighted_skill_score = 0` and `title_similarity = 1.0` only scores 0.30 — barely below the upper threshold. Skill match is the primary axis; title is the tiebreaker.

## 4. Feature statistics (current run, post-ESCO Round 4)

| Feature | Min | Max | Mean | Std | Notes |
|---|---|---|---|---|---|
| `embedding_similarity` | 0.131 | 0.890 | 0.645 | 0.091 | Smooth Gaussian-ish; the strongest signal column |
| `tfidf_similarity` | 0.000 | 0.310 | 0.078 | 0.032 | Right-skewed; lexical overlap is genuinely sparse here |
| `skill_overlap` | 0.000 | 1.000 | 0.266 | 0.244 | After ESCO expansion: ~30% of rows are 0, rest spread across [0.05, 1.0] |
| `weighted_skill_score` | 0.000 | 1.000 | 0.266 | 0.244 | **Currently ~81% identical to `skill_overlap`** — see §5 |
| `num_missing_skills` | 0 | 36 | 3.65 | 4.17 | Right-skewed; a few jobs need many skills |
| `avg_missing_skill_importance` | 0.00 | 3.60 | 2.24 | 0.79 | Was 28% stuck at 2.59; varies meaningfully now after Round 4 |
| `years_of_experience` | 0 | 50 | 5.25 | 7.58 | 51% non-zero (was 39% before YoE regex expansion); cap at 50 |
| `title_similarity` | -0.106 | 1.000 | 0.383 | 0.175 | 31,254 unique values across 25,590 rows (was 7,286 — Category collapse fixed) |

**Feature health verdict:**
- `embedding_similarity`, `title_similarity`: both vary smoothly across the full range. Strong.
- `tfidf_similarity`, `years_of_experience`, `num_missing_skills`: right-skewed but real signal.
- `skill_overlap`, `weighted_skill_score`: lifted significantly by ESCO but currently redundant with each other (Round 5 fixes that).
- `avg_missing_skill_importance`: unstuck after Round 4, will further improve after Round 5.

## 5. The leakage problem (you asked about this)

**This is the most important section.** The user explicitly asked: "we used features that we'll train on also for weak labeling". That observation is correct, and it's a real problem with the current dataset.

### What's happening

The labeling rule is `weak_score = 0.7 × weighted_skill_score + 0.3 × title_similarity`. Both of those features are also present in the training feature set:

```
Label inputs: weighted_skill_score, title_similarity
Training features: embedding_similarity, tfidf_similarity, skill_overlap,
                   weighted_skill_score, num_missing_skills,
                   avg_missing_skill_importance, years_of_experience,
                   title_similarity
                   ↑ wss and title_sim are in BOTH lists
```

A model trained on these eight features can recover the label deterministically. Specifically: `predict_label(x) = 1 if 0.7·x[wss] + 0.3·x[title_sim] >= 0.34 else 0` is the labeling rule itself, and any decent learner will rediscover it.

### Symptoms you would see

- Training F1 ≈ 1.0 (the rule is literally a function of two columns the model can read).
- Cross-validation F1 ≈ 1.0 (same reason — the rule is in every fold).
- Gold-set F1 substantially lower (because the human-labeled gold pairs aren't generated by this rule, so recovering the rule doesn't help with gold).

The gap between train-CV and gold isn't because the model is overfitting the data — it's because the model is overfitting the *labeling function*. That's leakage.

### How v1 handled this

In the v1 pipeline, `skill_coverage` was the single labeling-rule input, and it was deliberately **excluded** from the model's feature set. The README of the v1 build called this out as the most critical design constraint of the project. The same discipline got dropped during the move to the v2 composite labeling rule.

### What "good" looks like

A leakage-safe version of this dataset would do one of three things:

**Option A — match v1's discipline (recommended).** Hold out `weighted_skill_score` and `title_similarity` from the training features. Train on the remaining six: `embedding_similarity`, `tfidf_similarity`, `skill_overlap`, `num_missing_skills`, `avg_missing_skill_importance`, `years_of_experience`. The model has to predict the label without seeing the columns the label was computed from. This is essentially the Variant B partition we tested earlier.

**Option B — change the labeling source.** Use a labeler that doesn't share columns with the training features. Example: have an LLM judge produce labels (the planned 3-LLM gold standard), then the entire pair-features matrix becomes available for training, no holdout needed.

**Option C — keep the leakage, document it, use only as a diagnostic.** The numbers will look great but won't transfer to gold. Useful as "what would performance look like if we had perfect labels?" but not as a real classifier.

### What you have now

Effectively Option C without the disclaimer. If you train on this dataset and report results, the headline numbers will be inflated relative to what you'll see on real gold-set evaluation.

## 6. Will this be a good dataset?

Direct answer: **the data quality is good; the partition is wrong.**

What's good:
- 25,590 rows, balanced 50/50, zero nulls. No data hygiene problems.
- All eight features have meaningful spread (no more stuck-at-default columns once Round 5 lands).
- The pipeline that produced it has been verified end-to-end with the precision spot-check (matches are mostly relevant skills, not noise).
- ESCO expansion lifted the skill-channel features from "mostly zero" to "useful signal".
- 31k unique title-similarity values means the title channel is actually distinguishing roles.

What's wrong:
- The labeling formula uses two of the same columns the model trains on. Any classifier trained on this will be measuring its ability to rediscover the labeling rule, not its ability to predict resume-job alignment.

What I'd do before training:
1. **Apply the holdout (Option A).** Modify `make_ml_ready.py` to drop `weighted_skill_score` and `title_similarity` from the output, leaving 6 training features. Re-run. Result: a leakage-safe dataset where the trained classifier's gold-set F1 is meaningful.
2. **Run the planned 3-LLM gold expansion.** Once you have a real gold set built independently of the weak labeler, you can revisit Option B and recover those two features as training inputs.
3. **Run Round 5 (depth-based importance).** It's already wired up; just needs the three commands run. This unsticks `weighted_skill_score` so even when held out it's a stronger signal for the label rule.

Bottom line: ship Round 5, then apply the holdout. After that, the dataset is suitable for training and gold-set evaluation. As-is today, it's still in the diagnostic/upper-bound state.
