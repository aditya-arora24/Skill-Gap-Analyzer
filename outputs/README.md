# Variant Comparison — Run Instructions

This directory holds the outputs from the three-variant comparison run.

**Important: numbers from the Cowork sandbox are SMOKE-TEST ONLY.** The
sandbox could not install `sentence-transformers` (pytorch.org wheel index
firewalled, and torch from PyPI exceeds the 1.8 GB free-disk limit). Variant
A was run there to verify the code path works end-to-end; the title-similarity
feature in that run uses **stale cached SBERT embeddings encoded against the
old `Category` source**, so fix 4c is partially reverted for that one feature
in the sandbox run only. All other features are correctly produced.

**For canonical numbers, run all three variants on a machine with
sentence-transformers installed.** The script auto-detects SBERT and re-encodes
the new resume titles when available.

---

## How to run all three variants (canonical, with SBERT)

From the project root, in the same Python environment that has
`sentence-transformers` installed (the one used for the v2 preprocessing
pipeline):

```bash
cd src

# Phase 1: build the shared pair feature table (does this once;
# all three variants read from it).
python build_final_dataset.py --phase pairs

# Phase 2: build each variant's labels + gold set
python build_final_dataset.py --phase variant --config ../configs/variant_A.json
python build_final_dataset.py --phase variant --config ../configs/variant_B.json
python build_final_dataset.py --phase variant --config ../configs/variant_C.json

# Train + evaluate each variant
python evaluate_models.py --config ../configs/variant_A.json
python evaluate_models.py --config ../configs/variant_B.json
python evaluate_models.py --config ../configs/variant_C.json

# Stack metrics CSVs and produce the unified comparison
python compare_variants.py
```

When you run phase 1 with SBERT installed, you should see this in the log:

```
[titles] re-encoding with SBERT (fix 4c applied)
```

If you instead see:

```
[titles] !!! sentence-transformers NOT installed -- using STALE cached SBERT title embeddings (Category-based).
```

…then `sentence-transformers` is not on the path you're running with — the
title features will not reflect fix 4c, and numbers won't be canonical.

---

## Decision-point checks the sandbox already triggered

These are flagged from the smoke run. Decide before / during your canonical
run.

### Decision Point 2 — `experience_gap` data quality

After the JD YoE fix, `experience_gap` distribution on the full pair table:

```
mean   -2.544
std     7.915
min   -50.000
50%     0.000
max    21.000   (was 0 before fix 4a — fix is working)
```

Distribution by sign: 37% > 0, 27% = 0, 36% < 0.

`max > 0` confirms the "always ≤ 0" bug is fixed. `mean < 0` is now just a
property of the data: resumes typically have more experience than JDs require,
which is reasonable.

**However:** correlation between `experience_gap` and `years_of_experience` is
**-0.94**, near-perfect collinearity (because for jobs with no parsed YoE the
gap collapses to `-resume_yoe`). For tree-based models this means the model
will treat them as interchangeable; for LogReg it inflates standard errors
without helping prediction.

Your spec said "drop experience_gap if mean ≤ 0 or it's collinear with
years_of_experience". Both conditions trigger. **You may want to drop
`experience_gap` from all three variant configs** before the canonical run.
The variant config files have it currently included.

### Decision Point 4 — label balance

Variant A label split (sandbox run): **12,811 pos / 12,811 neg** — exactly
50/50 by construction (quantile cut). No imbalance flag.

### Decision Point 1 — SBERT delta

Cannot answer in sandbox. Compare canonical Variant A F1 to the previous
TF-IDF-fallback run (LogReg=0.59, GBM=0.58). If delta > 0.05 the original
spec said to pause and report.

### Decision Point 3 — Variant B vs single-feature baselines

Cannot answer in sandbox. Per the spec, if Variant B still loses to
single-feature baselines, do NOT proceed to Variant C — pause and discuss.

---

## File layout produced by the run

```
configs/
  variant_A.json            # strict holdout, 7 training features
  variant_B.json            # skill-only holdout, 9 training features
  variant_C.json            # no holdout (diagnostic), 11 training features

data/processed/
  pair_features_final.parquet          # shared across variants
  variant_A/
    ml_ready_dataset.parquet
    gold_standard_final.parquet
    summary.json
  variant_B/
    …
  variant_C/
    …

models/
  variant_A/{scaler,logreg_model,gbm_model}.pkl
  variant_B/…
  variant_C/…

outputs/
  variant_A/{metrics_A.csv, feature_importance_A.png}
  variant_B/…
  variant_C/…
  feature_importance_A.png             # also at top level for the report
  feature_importance_B.png
  variant_comparison.csv               # produced by compare_variants.py
  variant_comparison.png
```

---

## Sandbox smoke-test results (Variant A only, NOT canonical)

These numbers are from the sandbox run with stale SBERT title embeddings
(based on old `Category`). They confirm the code path works end-to-end. They
are **not** suitable for the report.

```
Variant A — gold-set results, sorted by F1
  SBERT Only         th=0.65   F1=0.6545
  GBM (tuned)        th=0.35   F1=0.6034
  LogReg (tuned)     th=0.35   F1=0.5946
  Title Similarity   th=0.30   F1=0.5421
  TF-IDF Only        th=0.10   F1=0.5238
  Composite Label    th=0.49   F1=0.4444
  Weighted Skills    th=0.30   F1=0.3733
```

Note the Title Similarity baseline dropped to 0.54 here vs 0.66 in the
previous TF-IDF char-n-gram run. That's the stale-cache effect: SBERT on
broad `Category` strings (e.g., "HR") has less separating power than TF-IDF
char-n-grams on the granular first-line titles ("HR DIRECTOR"). Once you
re-run with SBERT installed, title encoding will use the granular titles
and Title Similarity will likely climb.
